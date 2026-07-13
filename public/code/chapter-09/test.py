"""
第9章テスト: 実写データからの再構成

- COLMAPバイナリ読込（合成データで検証）
- 点群からの初期化（K近傍距離・色の正規化）
- L1損失単調減少（小規模合成シーン）
- 小規模合成シーンでPSNR >= 15dB
- 回帰テスト（前章機能の軽量確認）

COLMAPの実データが無い環境でも走るように、テスト内で
小さな合成COLMAPファイルを作って検証する。
"""

import numpy as np
import os
import struct
import sys
import tempfile
from PIL import Image


# ----- 合成COLMAPデータ生成ヘルパ -----

def _write_synthetic_colmap(base_path, num_cameras=3, num_points=50,
                             img_w=32, img_h=32):
    """合成COLMAPデータ（cameras.bin, images.bin, points3D.bin）を書き出す。

    3方向からZ=5 付近の点群を撮影する設定のミニマルなシーンを作る。
    実写ではなく合成なので、点群の色はグレー寄り、カメラはZ軸負方向を向く。

    Args:
        base_path: 出力先ルート。base_path/sparse/0/ に bin ファイルを置く。
                   base_path/images_8/ にPNG画像を置く。
        num_cameras: カメラ数
        num_points: 3D点数
        img_w, img_h: 画像サイズ

    Returns:
        (num_cameras, num_points)
    """
    sparse_dir = os.path.join(base_path, "sparse", "0")
    images_dir = os.path.join(base_path, "images_8")
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # cameras.bin: 1つのPINHOLEカメラを共有
    with open(os.path.join(sparse_dir, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", 1))  # num_cameras
        f.write(struct.pack("<I", 1))  # camera_id
        f.write(struct.pack("<I", 1))  # model_id = PINHOLE
        f.write(struct.pack("<Q", img_w))  # width
        f.write(struct.pack("<Q", img_h))  # height
        # params: fx, fy, cx, cy
        f.write(struct.pack("<4d", 30.0, 30.0, img_w / 2.0, img_h / 2.0))

    # images.bin: num_cameras個の画像
    rng = np.random.default_rng(0)
    image_names = []
    with open(os.path.join(sparse_dir, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", num_cameras))
        for i in range(num_cameras):
            image_id = i + 1
            # 単位クォータニオン（回転なし）
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            # カメラ位置をZ=5付近で少しずらす
            tx = 0.3 * (i - num_cameras / 2.0)
            ty = 0.0
            tz = 5.0
            camera_id = 1
            name = f"img_{i:02d}.png"
            image_names.append(name)

            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<4d", qw, qx, qy, qz))
            f.write(struct.pack("<3d", tx, ty, tz))
            f.write(struct.pack("<I", camera_id))
            f.write(name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))  # num_points2D = 0

    # points3D.bin: 原点付近のランダム点群
    positions = rng.uniform(-0.5, 0.5, size=(num_points, 3))
    colors = rng.integers(80, 200, size=(num_points, 3), dtype=np.int32)
    with open(os.path.join(sparse_dir, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", num_points))
        for i in range(num_points):
            f.write(struct.pack("<Q", i + 1))  # point3D_id
            f.write(struct.pack("<3d", *positions[i]))
            f.write(struct.pack("<3B", int(colors[i, 0]),
                                int(colors[i, 1]),
                                int(colors[i, 2])))
            f.write(struct.pack("<d", 0.5))  # error
            f.write(struct.pack("<Q", 0))  # num_track = 0

    # ダミー画像: グレー背景に点群中心近傍だけ少し明るく
    for name in image_names:
        arr = np.full((img_h, img_w, 3), 120, dtype=np.uint8)
        # 中央を少し明るく
        arr[img_h // 2 - 4:img_h // 2 + 4,
            img_w // 2 - 4:img_w // 2 + 4] = 200
        Image.fromarray(arr).save(os.path.join(images_dir, name))

    return num_cameras, num_points


# ----- テスト関数 -----

def test_read_colmap_binary():
    """合成COLMAPデータを読み込み、カメラ数・点群数が一致することを確認。"""
    from data_loader import (
        read_cameras_binary, read_images_binary, read_points3D_binary,
    )

    with tempfile.TemporaryDirectory() as tmp:
        num_cameras, num_points = _write_synthetic_colmap(
            tmp, num_cameras=4, num_points=30
        )
        sparse_dir = os.path.join(tmp, "sparse", "0")

        cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
        images = read_images_binary(os.path.join(sparse_dir, "images.bin"))
        positions, colors = read_points3D_binary(
            os.path.join(sparse_dir, "points3D.bin")
        )

        assert len(cameras) == 1, f"Expected 1 camera, got {len(cameras)}"
        assert len(images) == num_cameras, \
            f"Expected {num_cameras} images, got {len(images)}"
        assert positions.shape == (num_points, 3), \
            f"Expected ({num_points}, 3), got {positions.shape}"
        assert colors.shape == (num_points, 3), \
            f"Expected ({num_points}, 3), got {colors.shape}"

    print("  read_colmap_binary: OK")


def test_colmap_dataset():
    """ColmapDataset が画像+カメラを対応付けて読み込めることを確認。"""
    from data_loader import ColmapDataset

    with tempfile.TemporaryDirectory() as tmp:
        num_cameras, num_points = _write_synthetic_colmap(
            tmp, num_cameras=3, num_points=20
        )
        dataset = ColmapDataset(tmp, images_folder="images_8")

        assert len(dataset) == num_cameras, \
            f"Expected {num_cameras} cameras in dataset, got {len(dataset)}"
        assert dataset.points3D.shape == (num_points, 3)
        assert dataset.point_colors.shape == (num_points, 3)
        # 画像が [0, 1] に正規化されていること
        for img in dataset.images:
            assert img.min() >= 0.0 and img.max() <= 1.0

    print("  colmap_dataset: OK")


def test_compute_knn_distances():
    """K近傍距離が正しく計算されることを確認。"""
    from initialization import compute_knn_distances

    # 等間隔格子状の点（1次元的）
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ])
    dists = compute_knn_distances(positions, k=2)
    # 端点: 最近傍2つ → 1と2 → 平均=1.5
    # 中間点: 最近傍2つ → どちらも1 → 平均=1.0
    assert dists.shape == (4,)
    assert abs(dists[0] - 1.5) < 1e-9
    assert abs(dists[1] - 1.0) < 1e-9
    assert abs(dists[2] - 1.0) < 1e-9
    assert abs(dists[3] - 1.5) < 1e-9

    print("  compute_knn_distances: OK")


def test_initialize_gaussians():
    """点群から Gaussian3D を初期化できることを確認。"""
    from initialization import initialize_gaussians

    positions = np.random.randn(20, 3).astype(np.float64)
    colors = np.random.randint(0, 256, size=(20, 3), dtype=np.int32)

    gaussians = initialize_gaussians(positions, colors, k_nearest=3)
    assert len(gaussians) == 20

    # 最初のガウシアンをチェック
    g = gaussians[0]
    # 位置は点座標
    np.testing.assert_allclose(g.position.data, positions[0])
    # 色は [0, 1] に正規化された固定色
    np.testing.assert_allclose(g.color.data, colors[0] / 255.0)
    # クォータニオンは単位
    np.testing.assert_allclose(g.quaternion_raw.data,
                                np.array([1.0, 0.0, 0.0, 0.0]))
    # 不透明度は sigmoid_inv(0.1)
    expected_opacity_raw = np.log(0.1 / 0.9)
    assert abs(float(g.opacity_raw.data) - expected_opacity_raw) < 1e-9

    print("  initialize_gaussians: OK")


def test_trainer_l1_decreases():
    """合成シーンで L1 損失が単調に減少することを確認。

    小規模（合成ガウシアン群を教師画像に合わせる）ため、
    50 iteration でも十分に下がるはず。
    """
    from gaussian3d import Gaussian3D
    from camera import Camera
    from render import render_3d
    from optim import Adam
    from trainer import GaussianTrainer

    np.random.seed(0)

    # 教師ガウシアン（固定シーン）
    target_gaussians = [
        Gaussian3D(
            position=[-0.3, 0.0, 0.0],
            scale_raw=[-1.5, -1.5, -1.5],
            quaternion_raw=[1.0, 0.0, 0.0, 0.0],
            opacity_raw=2.0,
            color=[1.0, 0.3, 0.3],
        ),
        Gaussian3D(
            position=[0.3, 0.0, 0.0],
            scale_raw=[-1.5, -1.5, -1.5],
            quaternion_raw=[1.0, 0.0, 0.0, 0.0],
            opacity_raw=2.0,
            color=[0.3, 1.0, 0.3],
        ),
    ]

    cam = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, 3.0]),
        fx=50.0, fy=50.0, cx=16.0, cy=16.0,
        width=32, height=32,
    )
    target_image = render_3d(target_gaussians, cam, bg_color=(0, 0, 0)).data

    # 学習対象: 位置と色をランダムにずらした初期ガウシアン
    init_gaussians = [
        Gaussian3D(
            position=[-0.1, 0.1, 0.0],
            scale_raw=[-1.5, -1.5, -1.5],
            quaternion_raw=[1.0, 0.0, 0.0, 0.0],
            opacity_raw=2.0,
            color=[0.5, 0.5, 0.5],
        ),
        Gaussian3D(
            position=[0.1, -0.1, 0.0],
            scale_raw=[-1.5, -1.5, -1.5],
            quaternion_raw=[1.0, 0.0, 0.0, 0.0],
            opacity_raw=2.0,
            color=[0.5, 0.5, 0.5],
        ),
    ]

    params = []
    for g in init_gaussians:
        params.extend(g.params)
    optimizer = Adam(params, lr=0.05)

    trainer = GaussianTrainer(
        gaussians=init_gaussians,
        cameras=[cam],
        targets=[target_image],
        optimizer=optimizer,
        bg_color=(0, 0, 0),
    )

    losses = trainer.train(n_iters=50, log_interval=1000)

    # 初期損失と最終損失を比較
    initial_loss = np.mean(losses[:5])
    final_loss = np.mean(losses[-5:])
    assert final_loss < initial_loss * 0.7, \
        f"Loss did not decrease enough: initial={initial_loss:.4f}, " \
        f"final={final_loss:.4f}"

    print(f"  trainer_l1_decreases: OK (initial={initial_loss:.4f}, "
          f"final={final_loss:.4f})")


def test_trainer_psnr_min_bar():
    """小規模合成シーン・1カメラで PSNR >= 15dB を達成できることを確認。

    実写COLMAPでの 108x70 / 200 ガウシアン目標（18-22dB）の
    代替として、合成・小規模での最低ライン検証。
    """
    from gaussian3d import Gaussian3D
    from camera import Camera
    from render import render_3d
    from optim import Adam
    from trainer import GaussianTrainer

    np.random.seed(1)

    # 教師ガウシアン
    target_gaussians = [
        Gaussian3D(
            position=[0.0, 0.0, 0.0],
            scale_raw=[-1.2, -1.2, -1.2],
            quaternion_raw=[1.0, 0.0, 0.0, 0.0],
            opacity_raw=2.0,
            color=[0.8, 0.4, 0.1],
        ),
    ]

    cam = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, 3.0]),
        fx=40.0, fy=40.0, cx=16.0, cy=16.0,
        width=32, height=32,
    )
    target_image = render_3d(target_gaussians, cam, bg_color=(0, 0, 0)).data

    # 学習対象: 初期は位置がずれている
    init_gaussians = [
        Gaussian3D(
            position=[0.2, 0.1, 0.0],
            scale_raw=[-1.2, -1.2, -1.2],
            quaternion_raw=[1.0, 0.0, 0.0, 0.0],
            opacity_raw=2.0,
            color=[0.5, 0.5, 0.5],
        ),
    ]

    params = []
    for g in init_gaussians:
        params.extend(g.params)
    optimizer = Adam(params, lr=0.05)

    trainer = GaussianTrainer(
        gaussians=init_gaussians,
        cameras=[cam],
        targets=[target_image],
        optimizer=optimizer,
        bg_color=(0, 0, 0),
    )
    trainer.train(n_iters=100, log_interval=1000)

    # 学習後のレンダリングと目標画像のMSE → PSNR
    rendered = render_3d(init_gaussians, cam, bg_color=(0, 0, 0)).data
    mse = np.mean((rendered - target_image) ** 2)
    # 値域は [0, 1] を想定、ピーク = 1
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-12))

    assert psnr >= 15.0, f"PSNR below floor: {psnr:.2f} dB"
    print(f"  trainer_psnr_min_bar: OK (PSNR={psnr:.2f} dB)")


def test_regression_render_3d():
    """回帰テスト: render_3d が画像を生成できることを確認（第8章）。"""
    from gaussian3d import Gaussian3D
    from camera import Camera
    from render import render_3d

    g = Gaussian3D(
        position=[0.0, 0.0, 0.0],
        scale_raw=[-0.5, -0.5, -0.5],
        quaternion_raw=[1.0, 0.0, 0.0, 0.0],
        opacity_raw=2.0,
        color=[1.0, 0.0, 0.0],
    )

    cam = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
        fx=100.0, fy=100.0, cx=32.0, cy=32.0,
        width=64, height=64,
    )
    image = render_3d([g], cam, bg_color=(0, 0, 0))
    assert image.data.shape == (64, 64, 3)

    print("  regression_render_3d: OK")


def test_regression_l1_loss():
    """回帰テスト: l1_loss の backward が流れることを確認（第5章）。"""
    from autograd import Tensor
    from loss import l1_loss

    pred = Tensor(np.array([0.3, 0.6, 0.9]), requires_grad=True)
    target = Tensor(np.array([0.5, 0.5, 0.5]))
    loss = l1_loss(pred, target)
    loss.backward()

    assert pred.grad is not None
    assert pred.grad.shape == (3,)

    print("  regression_l1_loss: OK")


if __name__ == "__main__":
    np.random.seed(42)
    print("=== 第9章テスト ===\n")

    tests = [
        ("read_colmap_binary", test_read_colmap_binary),
        ("colmap_dataset", test_colmap_dataset),
        ("compute_knn_distances", test_compute_knn_distances),
        ("initialize_gaussians", test_initialize_gaussians),
        ("trainer_l1_decreases", test_trainer_l1_decreases),
        ("trainer_psnr_min_bar", test_trainer_psnr_min_bar),
        ("regression: render_3d", test_regression_render_3d),
        ("regression: l1_loss", test_regression_l1_loss),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  {name}: FAIL - {e}")
            failed += 1

    print(f"\n=== 結果: {passed}/{passed+failed} passed ===")
    if failed > 0:
        sys.exit(1)
