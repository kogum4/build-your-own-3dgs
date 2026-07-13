"""
第7章テスト: カメラモデルと座標変換。
Cameraクラスの動作確認 + 本文出力例の検証 + 回帰テスト。
"""

import numpy as np
import math
import sys

# テスト結果の集計
results = []


def run_test(name, func):
    """テストを実行して結果を記録する。"""
    try:
        ok = func()
        status = "PASS" if ok else "FAIL"
    except Exception as e:
        ok = False
        status = f"ERROR: {e}"
    results.append((name, ok))
    print(f"  [{status}] {name}")
    return ok


# ===================================================================
# Cameraクラスの基本テスト
# ===================================================================

def test_camera_identity():
    """W=I, t=0 の場合、world_to_camera は恒等変換。"""
    from camera import Camera

    cam = Camera(
        W=np.eye(3), t=np.zeros(3),
        fx=100.0, fy=100.0, cx=50.0, cy=50.0,
        width=100, height=100,
    )
    points_w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    points_c = cam.world_to_camera(points_w)
    return np.allclose(points_c, points_w)


def test_camera_translation():
    """並進のみの場合のworld_to_camera。"""
    from camera import Camera

    cam = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
        fx=100.0, fy=100.0, cx=50.0, cy=50.0,
        width=100, height=100,
    )
    points_w = np.array([[0.0, 0.0, 0.0]])
    points_c = cam.world_to_camera(points_w)
    # ワールド原点はカメラ座標で (0, 0, 5)
    return np.allclose(points_c, [[0.0, 0.0, 5.0]])


def test_camera_rotation():
    """回転のみの場合のworld_to_camera。"""
    from camera import Camera

    # Z軸周り90度回転
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    cam = Camera(
        W=R, t=np.zeros(3),
        fx=100.0, fy=100.0, cx=50.0, cy=50.0,
        width=100, height=100,
    )
    points_w = np.array([[1.0, 0.0, 0.0]])
    points_c = cam.world_to_camera(points_w)
    # R @ [1,0,0]^T = [0, 1, 0]^T (Z軸90度回転)
    return np.allclose(points_c, [[0.0, 1.0, 0.0]], atol=1e-10)


def test_project_basic():
    """基本的な透視投影。"""
    from camera import Camera

    cam = Camera(
        W=np.eye(3), t=np.zeros(3),
        fx=100.0, fy=100.0, cx=50.0, cy=50.0,
        width=100, height=100,
    )
    points_c = np.array([[2.0, 1.0, 5.0]])
    pixels, depths = cam.project(points_c)

    # u = 100 * 2/5 + 50 = 90, v = 100 * 1/5 + 50 = 70
    ok1 = np.allclose(pixels, [[90.0, 70.0]])
    ok2 = np.allclose(depths, [5.0])
    return ok1 and ok2


def test_project_perspective():
    """遠いほど小さく写ることの確認（遠近法）。"""
    from camera import Camera

    cam = Camera(
        W=np.eye(3), t=np.zeros(3),
        fx=100.0, fy=100.0, cx=0.0, cy=0.0,
        width=200, height=200,
    )
    # 同じX座標、異なるZ
    points_c = np.array([
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 2.0],
        [1.0, 0.0, 4.0],
    ])
    pixels, _ = cam.project(points_c)

    # u = fx * X/Z: Z=1→100, Z=2→50, Z=4→25
    ok1 = np.isclose(pixels[0, 0], 100.0)
    ok2 = np.isclose(pixels[1, 0], 50.0)
    ok3 = np.isclose(pixels[2, 0], 25.0)
    return ok1 and ok2 and ok3


def test_project_focal_length():
    """焦点距離の変更で投影サイズが変わることの確認。"""
    from camera import Camera

    points_c = np.array([[1.0, 0.0, 5.0]])

    cam1 = Camera(
        W=np.eye(3), t=np.zeros(3),
        fx=100.0, fy=100.0, cx=0.0, cy=0.0,
        width=200, height=200,
    )
    cam2 = Camera(
        W=np.eye(3), t=np.zeros(3),
        fx=200.0, fy=200.0, cx=0.0, cy=0.0,
        width=200, height=200,
    )

    px1, _ = cam1.project(points_c)
    px2, _ = cam2.project(points_c)

    # 焦点距離2倍 → 投影座標も2倍
    return np.isclose(px2[0, 0], px1[0, 0] * 2)


def test_cube_front_projection():
    """立方体の正面投影: 手前の面が奥より大きく写る。"""
    from camera import Camera

    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float64)

    cam = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
        fx=200.0, fy=200.0, cx=150.0, cy=150.0,
        width=300, height=300,
    )
    pc = cam.world_to_camera(vertices)
    px, depths = cam.project(pc)

    # 手前の面（Z_c=4）の投影幅
    front_width = px[:4, 0].max() - px[:4, 0].min()
    # 奥の面（Z_c=6）の投影幅
    back_width = px[4:, 0].max() - px[4:, 0].min()

    # 手前の面の方が大きく写る
    return front_width > back_width


# ===================================================================
# transposeのgrad_check（第6章で追加済み、回帰テスト）
# ===================================================================

def test_transpose_grad_check():
    """transposeのgrad_check（第6章で追加済み）。"""
    from autograd import grad_check

    ok = grad_check(
        lambda ts: ts[0].transpose(1, 0).sum(),
        [np.random.randn(3, 4)]
    )
    return ok


# ===================================================================
# 本文出力例の検証
# ===================================================================

def test_inline_transpose_rotation():
    """本文7.5: 回転行列のtransposeの確認。"""
    from autograd import Tensor

    angle = math.radians(30)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    R = Tensor(np.array([
        [ cos_a, 0, sin_a],
        [     0, 1,     0],
        [-sin_a, 0, cos_a],
    ]), requires_grad=True)

    R_T = R.transpose(1, 0)

    # R^T の (0,2) は -sin(30) = -0.5
    ok1 = np.isclose(R_T.data[0, 2], -0.5)
    # R^T の (2,0) は sin(30) = 0.5
    ok2 = np.isclose(R_T.data[2, 0], 0.5)
    # R^T^T == R
    ok3 = np.allclose(np.transpose(R_T.data), R.data)

    return ok1 and ok2 and ok3


def test_inline_transpose_backward():
    """本文7.5: transposeのbackward確認。"""
    from autograd import Tensor

    A = Tensor(np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]]), requires_grad=True)

    B = A.transpose(1, 0)  # (2, 3) → (3, 2)
    B.sum().backward()

    # gradの形状がAと同じ (2, 3)
    ok1 = A.grad.shape == (2, 3)
    # sum()の勾配は全要素1
    ok2 = np.allclose(A.grad, np.ones((2, 3)))

    return ok1 and ok2


def test_inline_basic_projection():
    """本文: 基本的な投影 (2,1,5) → (90,70)。"""
    from camera import Camera

    cam = Camera(
        W=np.eye(3), t=np.zeros(3),
        fx=100.0, fy=100.0, cx=50.0, cy=50.0,
        width=100, height=100,
    )
    points_w = np.array([[2.0, 1.0, 5.0]])
    points_c = cam.world_to_camera(points_w)

    ok1 = np.allclose(points_c, [[2.0, 1.0, 5.0]])

    pixels, depths = cam.project(points_c)
    ok2 = np.allclose(pixels, [[90.0, 70.0]])
    ok3 = np.allclose(depths, [5.0])

    return ok1 and ok2 and ok3


def test_inline_front_camera():
    """本文: 正面カメラの投影結果。"""
    from camera import Camera

    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float64)

    cam_front = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
        fx=200.0, fy=200.0, cx=150.0, cy=150.0,
        width=300, height=300,
    )
    pc = cam_front.world_to_camera(vertices)
    px, depths = cam_front.project(pc)

    # 頂点0: pixel=(100.0, 100.0), depth=4.0
    ok1 = np.allclose(px[0], [100.0, 100.0], atol=0.1)
    ok2 = np.isclose(depths[0], 4.0)

    # 頂点4: pixel=(116.7, 116.7), depth=6.0
    ok3 = np.allclose(px[4], [116.7, 116.7], atol=0.1)
    ok4 = np.isclose(depths[4], 6.0)

    return ok1 and ok2 and ok3 and ok4


def test_inline_oblique_camera():
    """本文: 斜めカメラのカメラ座標と投影結果。"""
    from camera import Camera

    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float64)

    angle = math.radians(30)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    R_y30 = np.array([
        [ cos_a, 0, sin_a],
        [     0, 1,     0],
        [-sin_a, 0, cos_a],
    ])

    cam_oblique = Camera(
        W=R_y30, t=np.array([0.0, 0.0, 5.0]),
        fx=200.0, fy=200.0, cx=150.0, cy=150.0,
        width=300, height=300,
    )
    pc = cam_oblique.world_to_camera(vertices)

    # 本文: 斜めカメラのカメラ座標（頂点0, 1）
    ok_c0 = np.allclose(pc[0], [-1.3660, -1.0000, 4.6340], atol=0.001)
    ok_c1 = np.allclose(pc[1], [0.3660, -1.0000, 3.6340], atol=0.001)

    px, depths = cam_oblique.project(pc)

    # 頂点0: pixel=(91.0, 106.8), depth=4.63
    ok1 = np.allclose(px[0], [91.0, 106.8], atol=0.2)
    ok2 = np.isclose(depths[0], 4.63, atol=0.01)

    # 頂点1: pixel=(170.1, 95.0), depth=3.63
    ok3 = np.allclose(px[1], [170.1, 95.0], atol=0.2)
    ok4 = np.isclose(depths[1], 3.63, atol=0.01)

    return ok_c0 and ok_c1 and ok1 and ok2 and ok3 and ok4


def test_inline_focal_length():
    """本文: 焦点距離の比較。"""
    from camera import Camera

    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    ], dtype=np.float64)

    cam_tele = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
        fx=400.0, fy=400.0, cx=150.0, cy=150.0,
        width=300, height=300,
    )
    cam_wide = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
        fx=80.0, fy=80.0, cx=150.0, cy=150.0,
        width=300, height=300,
    )

    pc_t = cam_tele.world_to_camera(vertices)
    px_t, _ = cam_tele.project(pc_t)
    w_tele = px_t[:, 0].max() - px_t[:, 0].min()

    pc_w = cam_wide.world_to_camera(vertices)
    px_w, _ = cam_wide.project(pc_w)
    w_wide = px_w[:, 0].max() - px_w[:, 0].min()

    ok1 = np.isclose(w_tele, 200.0, atol=0.1)
    ok2 = np.isclose(w_wide, 40.0, atol=0.1)

    return ok1 and ok2


def test_inline_perspective():
    """本文: 遠近法の確認 (X=1, Z=1,2,4)。"""
    from camera import Camera

    cam = Camera(
        W=np.eye(3), t=np.zeros(3),
        fx=100.0, fy=100.0, cx=0.0, cy=0.0,
        width=200, height=200,
    )
    test_points = np.array([
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 2.0],
        [1.0, 0.0, 4.0],
    ])
    px, _ = cam.project(test_points)

    # Z=1 → u=100, Z=2 → u=50, Z=4 → u=25
    ok1 = np.isclose(px[0, 0], 100.0)
    ok2 = np.isclose(px[1, 0], 50.0)
    ok3 = np.isclose(px[2, 0], 25.0)

    return ok1 and ok2 and ok3


# ===================================================================
# 回帰スモークテスト
# ===================================================================

def test_regression_ch6_covariance():
    """回帰テスト: Ch.6の共分散行列構築が動作すること。"""
    from autograd import Tensor
    from gaussian3d import build_covariance_3d

    scale_raw = Tensor(np.array([0.0, 0.0, 0.0]), requires_grad=True)
    quat_raw = Tensor(np.array([1.0, 0.0, 0.0, 0.0]), requires_grad=True)
    cov = build_covariance_3d(scale_raw, quat_raw)

    ok1 = np.allclose(cov.data, np.eye(3), atol=1e-10)

    # backwardも動作確認
    cov.sum().backward()
    ok2 = scale_raw.grad is not None

    return ok1 and ok2


def test_regression_ch5_l1_loss():
    """回帰テスト: Ch.5のL1損失が動作すること。"""
    from autograd import Tensor
    from loss import l1_loss

    predicted = Tensor(np.array([0.3, 0.7, 0.5]), requires_grad=True)
    target = Tensor(np.array([0.0, 1.0, 0.5]))
    loss = l1_loss(predicted, target)

    ok1 = np.isclose(loss.data, 0.2)
    loss.backward()
    expected_grad = np.array([1/3, -1/3, 0.0])
    ok2 = np.allclose(predicted.grad, expected_grad)

    return ok1 and ok2


def test_regression_ch4_basic_ops():
    """回帰テスト: Ch.4の基本演算が動作すること。"""
    from autograd import Tensor

    a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
    c = (a * b + a).sum()
    c.backward()

    ok1 = np.allclose(c.data, np.array(28.0))
    ok2 = np.allclose(a.grad, np.array([5.0, 6.0]))
    ok3 = np.allclose(b.grad, np.array([2.0, 3.0]))

    return ok1 and ok2 and ok3


if __name__ == "__main__":
    np.random.seed(42)
    print("=== 第7章テスト: カメラモデルと座標変換 ===\n")

    print("--- Cameraクラス基本テスト ---")
    run_test("Camera 恒等変換", test_camera_identity)
    run_test("Camera 並進のみ", test_camera_translation)
    run_test("Camera 回転のみ", test_camera_rotation)
    run_test("project 基本投影", test_project_basic)
    run_test("project 遠近法", test_project_perspective)
    run_test("project 焦点距離", test_project_focal_length)
    run_test("立方体 正面投影", test_cube_front_projection)

    print("\n--- grad_check ---")
    run_test("transpose grad_check (Ch.6回帰)", test_transpose_grad_check)

    print("\n--- 本文出力例の検証 ---")
    run_test("inline transpose回転行列", test_inline_transpose_rotation)
    run_test("inline transpose backward", test_inline_transpose_backward)
    run_test("inline 基本投影", test_inline_basic_projection)
    run_test("inline 正面カメラ", test_inline_front_camera)
    run_test("inline 斜めカメラ", test_inline_oblique_camera)
    run_test("inline 焦点距離比較", test_inline_focal_length)
    run_test("inline 遠近法確認", test_inline_perspective)

    print("\n--- 回帰スモークテスト ---")
    run_test("Ch.6 共分散行列", test_regression_ch6_covariance)
    run_test("Ch.5 L1損失", test_regression_ch5_l1_loss)
    run_test("Ch.4 基本演算", test_regression_ch4_basic_ops)

    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"結果: {passed}/{total} テスト通過")

    if passed < total:
        print("\n失敗したテスト:")
        for name, ok in results:
            if not ok:
                print(f"  - {name}")
        sys.exit(1)
    else:
        print("全テスト通過!")
