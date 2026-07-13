"""
第8章テスト: EWA Splatting
- 球状ガウシアン: どの角度からでも円に射影
- 細長い楕円体: 視点によって射影形状が変わる
- カリング（画面外・カメラ背面除外）の動作確認
- 本文出力例の検証
- 回帰テスト（前章機能の軽量確認）
"""

import numpy as np
import math
import sys


def test_regression_getitem_slice():
    """回帰テスト: getitem(スライス)のgrad_check（第6章）。"""
    from autograd import grad_check

    # 複数軸スライス
    ok1 = grad_check(
        lambda ts: ts[0][..., :2, :2].sum(),
        [np.random.randn(3, 3)]
    )
    assert ok1, "getitem(slice) 2D grad_check failed"

    # バッチ + 複数軸スライス
    ok2 = grad_check(
        lambda ts: ts[0][..., :2, :2].sum(),
        [np.random.randn(4, 3, 3)]
    )
    assert ok2, "getitem(slice) 3D batch grad_check failed"

    print("  regression_getitem_slice: OK")


def test_compute_jacobian():
    """透視投影ヤコビアンの数値を検証。"""
    from projection import compute_jacobian
    from autograd import Tensor

    # 点 (0, 0, 5), fx=fy=100
    point = Tensor(np.array([0.0, 0.0, 5.0]))
    J = compute_jacobian(point, fx=100.0, fy=100.0)

    expected = np.array([[20.0, 0.0, 0.0],
                          [0.0, 20.0, 0.0]])
    np.testing.assert_allclose(J.data, expected, atol=1e-10)

    # 点 (1, 2, 5), fx=fy=100
    point2 = Tensor(np.array([1.0, 2.0, 5.0]))
    J2 = compute_jacobian(point2, fx=100.0, fy=100.0)

    expected2 = np.array([[20.0, 0.0, -4.0],
                           [0.0, 20.0, -8.0]])
    np.testing.assert_allclose(J2.data, expected2, atol=1e-10)

    print("  compute_jacobian: OK")


def test_sphere_projection():
    """球状ガウシアン: どの角度からでも円に射影される。"""
    from gaussian3d import Gaussian3D
    from camera import Camera
    from projection import project_gaussians

    g = Gaussian3D(
        position=[0.0, 0.0, 0.0],
        scale_raw=[0.0, 0.0, 0.0],
        quaternion_raw=[1.0, 0.0, 0.0, 0.0],
        opacity_raw=0.0,
        color=[1.0, 1.0, 1.0],
    )

    for angle_deg in [0, 30, 45, 60, 90]:
        angle = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = np.array([
            [ cos_a, 0, sin_a],
            [     0, 1,     0],
            [-sin_a, 0, cos_a],
        ])
        cam = Camera(
            W=R, t=np.array([0.0, 0.0, 5.0]),
            fx=100.0, fy=100.0, cx=32.0, cy=32.0,
            width=64, height=64,
        )
        _, covs2d, _, _, _, _ = project_gaussians([g], cam)
        cov = covs2d[0].data

        # 対角要素が等しい（比が1.0）
        ratio = cov[0, 0] / cov[1, 1]
        assert abs(ratio - 1.0) < 1e-6, \
            f"Sphere projection at {angle_deg}deg: ratio={ratio:.6f} != 1.0"

        # 非対角要素がゼロに近い
        assert abs(cov[0, 1]) < 1e-6, \
            f"Sphere projection at {angle_deg}deg: off-diagonal={cov[0,1]:.6f}"

    print("  sphere_projection: OK")


def test_elongated_view_dependence():
    """細長い楕円体: 視点によって射影形状が変わる。"""
    from gaussian3d import Gaussian3D
    from camera import Camera
    from projection import project_gaussians

    g = Gaussian3D(
        position=[0.0, 0.0, 0.0],
        scale_raw=[1.0, -1.0, -1.0],  # X方向に細長い
        quaternion_raw=[1.0, 0.0, 0.0, 0.0],
        opacity_raw=0.0,
        color=[1.0, 0.0, 0.0],
    )

    sigmas_u = []
    for angle_deg in [0, 45, 90]:
        angle = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = np.array([
            [ cos_a, 0, sin_a],
            [     0, 1,     0],
            [-sin_a, 0, cos_a],
        ])
        cam = Camera(
            W=R, t=np.array([0.0, 0.0, 5.0]),
            fx=100.0, fy=100.0, cx=32.0, cy=32.0,
            width=64, height=64,
        )
        _, covs2d, _, _, _, _ = project_gaussians([g], cam)
        sigmas_u.append(np.sqrt(covs2d[0].data[0, 0]))

    # 正面(0度)と横(90度)で射影幅が異なる
    assert sigmas_u[0] != sigmas_u[2], \
        f"Expected different sigma_u at 0deg and 90deg, got {sigmas_u[0]:.2f} and {sigmas_u[2]:.2f}"

    print("  elongated_view_dependence: OK")


def test_culling_behind_camera():
    """カメラ背面のガウシアンがカリングされる。"""
    from gaussian3d import Gaussian3D
    from camera import Camera
    from projection import project_gaussians

    g = Gaussian3D(
        position=[0.0, 0.0, 0.0],
        scale_raw=[0.0, 0.0, 0.0],
        quaternion_raw=[1.0, 0.0, 0.0, 0.0],
        opacity_raw=0.0,
        color=[1.0, 1.0, 1.0],
    )

    # ガウシアンがカメラの後ろにある設定
    cam = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, -5.0]),
        fx=100.0, fy=100.0, cx=32.0, cy=32.0,
        width=64, height=64,
    )

    _, _, _, _, _, indices = project_gaussians([g], cam)
    assert len(indices) == 0, f"Expected 0 visible, got {len(indices)}"

    print("  culling_behind_camera: OK")


def test_culling_outside_screen():
    """画面外のガウシアンがカリングされる。"""
    from gaussian3d import Gaussian3D
    from camera import Camera
    from projection import project_gaussians

    # 画面外に配置（X方向に大きくずらす）
    g = Gaussian3D(
        position=[100.0, 0.0, 0.0],
        scale_raw=[0.0, 0.0, 0.0],
        quaternion_raw=[1.0, 0.0, 0.0, 0.0],
        opacity_raw=0.0,
        color=[1.0, 1.0, 1.0],
    )

    cam = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
        fx=100.0, fy=100.0, cx=32.0, cy=32.0,
        width=64, height=64,
    )

    _, _, _, _, _, indices = project_gaussians([g], cam)
    assert len(indices) == 0, f"Expected 0 visible, got {len(indices)}"

    print("  culling_outside_screen: OK")


def test_render_3d():
    """render_3d が画像を正しく生成する。"""
    from gaussian3d import Gaussian3D
    from camera import Camera
    from render import render_3d

    gaussians = [
        Gaussian3D(
            position=[-1.0, 0.0, 0.0],
            scale_raw=[-0.5, -0.5, -0.5],
            quaternion_raw=[1.0, 0.0, 0.0, 0.0],
            opacity_raw=2.0,
            color=[1.0, 0.0, 0.0],
        ),
        Gaussian3D(
            position=[0.0, 0.0, 0.0],
            scale_raw=[-0.5, -0.5, -0.5],
            quaternion_raw=[1.0, 0.0, 0.0, 0.0],
            opacity_raw=2.0,
            color=[0.0, 1.0, 0.0],
        ),
        Gaussian3D(
            position=[1.0, 0.0, 0.0],
            scale_raw=[-0.5, -0.5, -0.5],
            quaternion_raw=[1.0, 0.0, 0.0, 0.0],
            opacity_raw=2.0,
            color=[0.0, 0.0, 1.0],
        ),
    ]

    cam = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
        fx=100.0, fy=100.0, cx=32.0, cy=32.0,
        width=64, height=64,
    )

    image = render_3d(gaussians, cam, bg_color=(1, 1, 1))

    assert image.data.shape == (64, 64, 3), \
        f"Expected (64, 64, 3), got {image.data.shape}"
    assert image.data.min() >= 0.0, "Image has negative values"
    assert image.data.max() <= 1.5, "Image values too large"

    print("  render_3d: OK")


def test_render_3d_empty():
    """全ガウシアンがカリングされた場合に黒画像が返る。"""
    from gaussian3d import Gaussian3D
    from camera import Camera
    from render import render_3d

    # カメラ背面
    gaussians = [
        Gaussian3D(
            position=[0.0, 0.0, 10.0],
            scale_raw=[0.0, 0.0, 0.0],
            quaternion_raw=[1.0, 0.0, 0.0, 0.0],
            opacity_raw=0.0,
            color=[1.0, 1.0, 1.0],
        ),
    ]

    cam = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, -20.0]),
        fx=100.0, fy=100.0, cx=32.0, cy=32.0,
        width=64, height=64,
    )

    image = render_3d(gaussians, cam)
    assert image.data.shape == (64, 64, 3)
    np.testing.assert_allclose(image.data, 0.0)

    print("  render_3d_empty: OK")


def test_text_output_jacobian():
    """本文出力例の検証: ヤコビアン計算。"""
    from projection import compute_jacobian
    from autograd import Tensor

    # 点 (0, 0, 5)
    point = Tensor(np.array([0.0, 0.0, 5.0]))
    J = compute_jacobian(point, fx=100.0, fy=100.0).data
    # fx/Z = 100/5 = 20, -fx*X/Z^2 = 0
    assert abs(J[0, 0] - 20.0) < 1e-10
    assert abs(J[0, 2] - 0.0) < 1e-10
    assert abs(J[1, 1] - 20.0) < 1e-10

    # 点 (1, 2, 5)
    point2 = Tensor(np.array([1.0, 2.0, 5.0]))
    J2 = compute_jacobian(point2, fx=100.0, fy=100.0).data
    # -fx*X/Z^2 = -100*1/25 = -4
    assert abs(J2[0, 2] - (-4.0)) < 1e-10
    # -fy*Y/Z^2 = -100*2/25 = -8
    assert abs(J2[1, 2] - (-8.0)) < 1e-10

    print("  text_output_jacobian: OK")


def test_text_output_sphere():
    """本文出力例の検証: 球状ガウシアン射影の対角比。"""
    from gaussian3d import Gaussian3D
    from camera import Camera
    from projection import project_gaussians

    g = Gaussian3D(
        position=[0.0, 0.0, 0.0],
        scale_raw=[0.0, 0.0, 0.0],
        quaternion_raw=[1.0, 0.0, 0.0, 0.0],
        opacity_raw=0.0,
        color=[1.0, 1.0, 1.0],
    )
    cam = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
        fx=100.0, fy=100.0, cx=32.0, cy=32.0,
        width=64, height=64,
    )
    _, covs2d, _, _, _, _ = project_gaussians([g], cam)
    ratio = covs2d[0].data[0, 0] / covs2d[0].data[1, 1]
    assert abs(ratio - 1.0) < 1e-6, f"Expected ratio=1.0, got {ratio}"

    print("  text_output_sphere: OK")


def test_regression_matmul():
    """回帰テスト: matmulのgrad_check（第6章）。"""
    from autograd import grad_check

    def f(inputs):
        return (inputs[0] @ inputs[1]).sum()

    ok = grad_check(f, [np.random.randn(3, 4), np.random.randn(4, 2)])
    assert ok, "matmul grad_check failed"

    print("  regression_matmul: OK")


def test_regression_normalize():
    """回帰テスト: normalizeのgrad_check（第6章）。"""
    from autograd import grad_check

    def f(inputs):
        return inputs[0].normalize().sum()

    ok = grad_check(f, [np.random.randn(4)])
    assert ok, "normalize grad_check failed"

    print("  regression_normalize: OK")


def test_regression_transpose():
    """回帰テスト: transposeのgrad_check（第6章）。"""
    from autograd import grad_check

    def f(inputs):
        return inputs[0].transpose(1, 0).sum()

    ok = grad_check(f, [np.random.randn(3, 4)])
    assert ok, "transpose grad_check failed"

    print("  regression_transpose: OK")


def test_regression_build_covariance_3d():
    """回帰テスト: build_covariance_3dのgrad_check（第6章）。"""
    from autograd import grad_check

    def f(inputs):
        from gaussian3d import build_covariance_3d
        return build_covariance_3d(inputs[0], inputs[1]).sum()

    ok = grad_check(f, [np.random.randn(3), np.array([1.0, 0.0, 0.3, 0.1])])
    assert ok, "build_covariance_3d grad_check failed"

    print("  regression_build_covariance_3d: OK")


def test_ewa_gradient_flow():
    """EWA Splatting全体で勾配が流れることを確認。"""
    from gaussian3d import Gaussian3D
    from camera import Camera
    from projection import project_gaussians

    g = Gaussian3D(
        position=[0.0, 0.0, 0.0],
        scale_raw=[0.0, 0.0, 0.0],
        quaternion_raw=[1.0, 0.0, 0.0, 0.0],
        opacity_raw=0.0,
        color=[1.0, 0.5, 0.0],
    )

    cam = Camera(
        W=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
        fx=100.0, fy=100.0, cx=32.0, cy=32.0,
        width=64, height=64,
    )

    means2d, covs2d, _, _, _, _ = project_gaussians([g], cam)

    # covs2dのsumでbackward
    loss = covs2d[0].sum()
    loss.backward()

    # scale_rawとquaternion_rawに勾配が流れていることを確認
    assert g.scale_raw.grad is not None, "scale_raw.grad is None"
    assert not np.allclose(g.scale_raw.grad, 0.0), \
        "scale_raw.grad is all zeros"
    assert g.quaternion_raw.grad is not None, "quaternion_raw.grad is None"

    print("  ewa_gradient_flow: OK")


if __name__ == "__main__":
    np.random.seed(42)
    print("=== 第8章テスト ===\n")

    tests = [
        ("compute_jacobian", test_compute_jacobian),
        ("sphere_projection", test_sphere_projection),
        ("elongated_view_dependence", test_elongated_view_dependence),
        ("culling_behind_camera", test_culling_behind_camera),
        ("culling_outside_screen", test_culling_outside_screen),
        ("render_3d", test_render_3d),
        ("render_3d_empty", test_render_3d_empty),
        ("text_output_jacobian", test_text_output_jacobian),
        ("text_output_sphere", test_text_output_sphere),
        ("EWA gradient flow", test_ewa_gradient_flow),
        ("regression: matmul", test_regression_matmul),
        ("regression: normalize", test_regression_normalize),
        ("regression: transpose", test_regression_transpose),
        ("regression: getitem(slice)", test_regression_getitem_slice),
        ("regression: build_covariance_3d", test_regression_build_covariance_3d),
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
