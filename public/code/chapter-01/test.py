"""
第1章テスト: 2Dガウシアンの描画を検証する。
"""

import tempfile
import numpy as np
from gaussian2d import Gaussian2D, build_covariance_2d, evaluate_gaussian
from render import render_gaussians_weighted_sum


def test_single_gaussian():
    """単一ガウシアンが中心に丸いブロブを描くことを確認する。"""
    cov = build_covariance_2d(sigma_x=10.0, sigma_y=10.0, theta=0.0)
    g = Gaussian2D(mean=[64, 64], covariance=cov, color=[1, 1, 1], opacity=1.0)

    image = render_gaussians_weighted_sum([g], H=128, W=128)

    # 中心ピクセルの値が最も高い（白に近い）
    center_val = image[64, 64]
    assert np.allclose(center_val, [1.0, 1.0, 1.0]), \
        f"中心ピクセルが白でない: {center_val}"

    # 加重和方式では、角のピクセルはガウシアン値がほぼゼロのため黒になる
    corner_val = image[0, 0]
    assert np.allclose(corner_val, [0.0, 0.0, 0.0], atol=1e-6), \
        f"角のピクセルが黒でない: {corner_val}"

    # ガウシアン値自体が中心ほど高いことを直接検証する。
    cov_inv = np.linalg.inv(cov)
    val_center = evaluate_gaussian(np.array([[64.0, 64.0]]), g.mean, cov_inv)
    val_corner = evaluate_gaussian(np.array([[0.0, 0.0]]), g.mean, cov_inv)
    assert val_center[0] > val_corner[0], \
        f"中心({val_center[0]:.6f})が角({val_corner[0]:.6f})より小さい"
    assert val_corner[0] < 1e-6, \
        f"角のガウシアン値が十分小さくない: {val_corner[0]:.6e}"

    print("[PASS] test_single_gaussian: 中心に白いブロブを確認")


def test_elliptical_gaussian():
    """sigma_x != sigma_y で楕円形のガウシアンになることを確認する。"""
    # 横長の楕円: sigma_x > sigma_y
    cov = build_covariance_2d(sigma_x=20.0, sigma_y=5.0, theta=0.0)
    cov_inv = np.linalg.inv(cov)
    mean = np.array([64.0, 64.0])

    # ガウシアン値を直接比較（加重平均は単一ガウシアンで色が一定になるため）
    # x方向（水平方向）に離れた点
    val_right = evaluate_gaussian(np.array([[84.0, 64.0]]), mean, cov_inv)
    # y方向（垂直方向）に同じ距離離れた点
    val_down = evaluate_gaussian(np.array([[64.0, 84.0]]), mean, cov_inv)

    assert val_right[0] > val_down[0], \
        f"横長楕円なのにx方向({val_right[0]:.4f})がy方向({val_down[0]:.4f})より暗い"

    print("[PASS] test_elliptical_gaussian: 横長楕円の形状を確認")


def test_rotated_gaussian():
    """theta=45度で斜めの楕円になることを確認する。"""
    theta = np.pi / 4  # 45度
    cov = build_covariance_2d(sigma_x=20.0, sigma_y=5.0, theta=theta)
    cov_inv = np.linalg.inv(cov)
    mean = np.array([64.0, 64.0])

    # ガウシアン値を直接比較
    # 45度方向（右下）に離れた点: 中心(64,64)から右下に14px
    val_diag = evaluate_gaussian(np.array([[78.0, 78.0]]), mean, cov_inv)
    # 右方向に同じ距離離れた点
    val_right = evaluate_gaussian(np.array([[78.0, 64.0]]), mean, cov_inv)

    # 45度回転なので対角方向のほうがガウシアン値が大きいはず
    assert val_diag[0] > val_right[0], \
        f"45度回転なのに対角方向({val_diag[0]:.4f})が水平方向({val_right[0]:.4f})より暗い"

    print("[PASS] test_rotated_gaussian: 45度回転の楕円を確認")


def test_covariance_symmetry():
    """共分散行列が対称であることを確認する。"""
    cov = build_covariance_2d(sigma_x=15.0, sigma_y=8.0, theta=0.7)
    assert np.allclose(cov, cov.T), \
        f"共分散行列が対称でない:\n{cov}"

    print("[PASS] test_covariance_symmetry: 共分散行列の対称性を確認")


def test_evaluate_gaussian_peak():
    """ガウシアン値が中心で最大（=1）であることを確認する。"""
    mean = np.array([50.0, 50.0])
    cov = build_covariance_2d(sigma_x=10.0, sigma_y=10.0, theta=0.0)
    cov_inv = np.linalg.inv(cov)

    # 中心ピクセルでの値
    center_pixel = np.array([[50.0, 50.0]])
    val_center = evaluate_gaussian(center_pixel, mean, cov_inv)
    assert np.isclose(val_center[0], 1.0), \
        f"中心でのガウシアン値が1でない: {val_center[0]}"

    # 離れたピクセルでの値は1より小さい
    far_pixel = np.array([[80.0, 80.0]])
    val_far = evaluate_gaussian(far_pixel, mean, cov_inv)
    assert val_far[0] < val_center[0], \
        f"離れた点({val_far[0]:.6f})が中心({val_center[0]:.6f})以上"

    print("[PASS] test_evaluate_gaussian_peak: 中心でピーク値1を確認")


def test_three_color_blend():
    """3色のガウシアンを重ねてカラフルな画像が生成されることを確認する。"""
    H, W = 128, 128

    gaussians = [
        Gaussian2D(
            mean=[54, 44],
            covariance=build_covariance_2d(24.0, 10.0, np.pi / 3),
            color=[1, 0, 0],  # 赤
            opacity=1.0,
        ),
        Gaussian2D(
            mean=[74, 44],
            covariance=build_covariance_2d(24.0, 10.0, -np.pi / 3),
            color=[0, 1, 0],  # 緑
            opacity=1.0,
        ),
        Gaussian2D(
            mean=[64, 72],
            covariance=build_covariance_2d(24.0, 10.0, np.pi / 2),
            color=[0, 0, 1],  # 青
            opacity=1.0,
        ),
    ]

    image = render_gaussians_weighted_sum(gaussians, H, W)

    # 画像の形状が正しい
    assert image.shape == (H, W, 3), f"画像の形状が不正: {image.shape}"

    # 値が [0, 1] の範囲内
    assert image.min() >= 0.0 and image.max() <= 1.0, \
        f"値が範囲外: [{image.min()}, {image.max()}]"

    # 赤ガウシアンの中心付近は赤い
    red_center = image[44, 54]  # y=44, x=54（赤の中心）
    assert red_center[0] > 0.5, \
        f"赤ガウシアンの中心が赤くない: R={red_center[0]:.4f}"

    # 青ガウシアンは縦向きなので、同じ距離なら上下方向のほうが値が大きい
    blue_cov_inv = np.linalg.inv(gaussians[2].covariance)
    blue_mean = gaussians[2].mean
    val_up = evaluate_gaussian(np.array([[64.0, 60.0]]), blue_mean, blue_cov_inv)
    val_right = evaluate_gaussian(np.array([[76.0, 72.0]]), blue_mean, blue_cov_inv)
    assert val_up[0] > val_right[0], \
        f"青ガウシアンの向きが不正: 上方向={val_up[0]:.4f}, 右方向={val_right[0]:.4f}"

    # 目視確認用の画像は一時ディレクトリにだけ保存する
    try:
        from PIL import Image
        img_uint8 = (image * 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = f"{tmp_dir}/test_color_blend.png"
            Image.fromarray(img_uint8).save(output_path)
            print(f"  -> {output_path} に保存して目視確認できる状態を確認")
    except ImportError:
        print("  -> Pillow未インストールのため画像保存をスキップ")

    print("[PASS] test_three_color_blend: 3色ガウシアンの合成を確認")


def test_inline_build_covariance():
    """本文中の build_covariance_2d 出力例が正しいことを確認する。"""
    # 円形（sigma_x = sigma_y, 回転なし）
    cov1 = build_covariance_2d(10.0, 10.0, 0.0)
    assert np.allclose(cov1, [[100., 0.], [0., 100.]]), \
        f"円形の出力例が不正: {cov1}"

    # 横長楕円（sigma_x > sigma_y, 回転なし）
    cov2 = build_covariance_2d(20.0, 5.0, 0.0)
    assert np.allclose(cov2, [[400., 0.], [0., 25.]]), \
        f"横長楕円の出力例が不正: {cov2}"

    # 30度回転
    cov3 = build_covariance_2d(20.0, 5.0, np.pi / 6)
    assert np.allclose(cov3, [[306.25, 162.37976321], [162.37976321, 118.75]]), \
        f"30度回転の出力例が不正: {cov3}"

    print("[PASS] test_inline_build_covariance: 本文の共分散行列出力例を確認")


def test_inline_evaluate_gaussian():
    """本文中の evaluate_gaussian 出力例が正しいことを確認する。"""
    cov = build_covariance_2d(10.0, 10.0, 0.0)
    cov_inv = np.linalg.inv(cov)
    mean = np.array([64.0, 64.0])

    # 中心での値は1.0
    val_center = evaluate_gaussian(np.array([[64.0, 64.0]]), mean, cov_inv)
    assert np.isclose(val_center[0], 1.0), \
        f"中心の出力例が不正: {val_center[0]}"

    # 遠い点での値
    val_far = evaluate_gaussian(np.array([[0.0, 0.0]]), mean, cov_inv)
    expected_far = 1.62666462e-18
    assert np.isclose(val_far[0], expected_far, rtol=0.0, atol=1e-26), \
        f"遠い点の出力例が不正: {val_far[0]:.6e} (期待: {expected_far:.6e})"

    print("[PASS] test_inline_evaluate_gaussian: 本文のガウシアン値出力例を確認")


def test_inline_gaussian2d():
    """本文中の Gaussian2D 出力例が正しいことを確認する。"""
    g = Gaussian2D(
        mean=[64, 64],
        covariance=build_covariance_2d(15.0, 15.0, 0.0),
        color=[1, 0, 0],
        opacity=0.8,
    )
    assert np.allclose(g.mean, [64., 64.]), f"meanが不正: {g.mean}"
    assert np.allclose(g.color, [1., 0., 0.]), f"colorが不正: {g.color}"
    assert g.opacity == 0.8, f"opacityが不正: {g.opacity}"

    print("[PASS] test_inline_gaussian2d: 本文のGaussian2D出力例を確認")


def test_inline_render():
    """本文中の render_gaussians_weighted_sum 出力例が正しいことを確認する。"""
    g = Gaussian2D(
        mean=[64, 64],
        covariance=build_covariance_2d(10.0, 10.0, 0.0),
        color=[1, 0, 0],  # 赤
        opacity=1.0,
    )
    image = render_gaussians_weighted_sum([g], H=128, W=128)

    assert image.shape == (128, 128, 3), f"形状が不正: {image.shape}"
    # 加重和では中心が赤、角はほぼ黒
    assert np.allclose(image[64, 64], [1., 0., 0.]), \
        f"中心の色が不正: {image[64, 64]}"
    assert np.allclose(image[0, 0], [0., 0., 0.], atol=1e-6), \
        f"角の色が不正: {image[0, 0]}"

    print("[PASS] test_inline_render: 本文のレンダリング出力例を確認")


def test_inline_matplotlib():
    """本文中の matplotlib スニペットがエラーなく実行できることを確認する。"""
    mean = np.array([64.0, 64.0])
    cov = build_covariance_2d(sigma_x=15.0, sigma_y=8.0, theta=np.pi / 6)
    cov_inv = np.linalg.inv(cov)

    H, W = 128, 128
    ys, xs = np.mgrid[0:H, 0:W]
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1)

    values = evaluate_gaussian(pixels, mean, cov_inv)
    image = values.reshape(H, W)

    # 形状と値域のみ確認（plt.showは呼ばない）
    assert image.shape == (H, W), f"形状が不正: {image.shape}"
    assert np.isclose(image[64, 64], 1.0), f"中心値が不正: {image[64, 64]}"
    assert image.min() >= 0.0 and image.max() <= 1.0, \
        f"値域が不正: [{image.min()}, {image.max()}]"

    print("[PASS] test_inline_matplotlib: 本文のmatplotlibスニペットを確認")


if __name__ == "__main__":
    test_single_gaussian()
    test_elliptical_gaussian()
    test_rotated_gaussian()
    test_covariance_symmetry()
    test_evaluate_gaussian_peak()
    test_three_color_blend()
    test_inline_build_covariance()
    test_inline_evaluate_gaussian()
    test_inline_gaussian2d()
    test_inline_render()
    test_inline_matplotlib()
    print("\n全テスト合格!")
