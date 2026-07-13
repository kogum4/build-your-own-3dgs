"""
第5章テスト: 2Dガウシアン画像フィッティング。
"""

import numpy as np
import time
from gaussian2d import Gaussian2D, build_covariance_2d
from render import (render_gaussians_weighted_sum, render_gaussians_alpha_composite,
                    render_gaussians_alpha_composite_tensor, evaluate_gaussian_tensor,
                    invert_2x2_tensor)
from autograd import Value, scalar_grad_check, Tensor, grad_check, stack, unbind
from autograd import _unbroadcast
from loss import l1_loss
from optim import SGD, Adam


# ===== 第1-2章回帰テスト（スモーク） =====

def test_ch1_single_gaussian():
    """第1章: 単一ガウシアンが中心に丸いブロブを描くことを確認する。"""
    cov = build_covariance_2d(sigma_x=10.0, sigma_y=10.0, theta=0.0)
    g = Gaussian2D(mean=[64, 64], covariance=cov, color=[1, 1, 1], opacity=1.0)
    image = render_gaussians_weighted_sum([g], H=128, W=128)

    assert np.allclose(image[64, 64], [1.0, 1.0, 1.0]), \
        f"中心ピクセルが白でない: {image[64, 64]}"

    print("[PASS] test_ch1_single_gaussian: 第1章回帰テスト合格")


def test_ch2_alpha_composite():
    """第2章: アルファ合成が正しく動作することを確認する。"""
    cov = build_covariance_2d(15.0, 15.0, 0.0)
    front = Gaussian2D(mean=[64, 64], covariance=cov, color=[1, 0, 0],
                       opacity=1.0, depth=1.0)
    back = Gaussian2D(mean=[64, 64], covariance=cov, color=[0, 0, 1],
                      opacity=1.0, depth=2.0)
    image = render_gaussians_alpha_composite([front, back], H=128, W=128)
    center = image[64, 64]
    assert np.allclose(center, [1.0, 0.0, 0.0]), \
        f"遮蔽が機能していない: {center}"

    print("[PASS] test_ch2_alpha_composite: 第2章回帰テスト合格")


# ===== 第3章回帰テスト（スモーク） =====

def test_ch3_value_basic():
    """第3章: Valueクラスの基本動作を確認する。"""
    a = Value(2.0)
    x = Value(3.0)
    b = Value(1.0)
    y = a * x + b
    y.backward()
    assert a.grad == 3.0 and x.grad == 2.0 and b.grad == 1.0
    print("[PASS] test_ch3_value_basic: 第3章回帰テスト合格")


def test_ch3_scalar_grad_check():
    """第3章: scalar_grad_checkがまだ動くことを確認する。"""
    ok = scalar_grad_check(lambda v: (v[0] ** 2 + 2 * v[0]).exp(), [1.0])
    assert ok
    print("[PASS] test_ch3_scalar_grad_check: 第3章回帰テスト合格")


# ===== 第4章回帰テスト（スモーク） =====

def test_ch4_tensor_basic():
    """第4章: Tensorの基本演算が動くことを確認する。"""
    a = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([4.0, 5.0, 6.0]), requires_grad=True)
    c = (a * b + a ** 2).sum()
    c.backward()
    # d/da(a*b + a^2) = b + 2a = [4+2, 5+4, 6+6] = [6, 9, 12]
    assert np.allclose(a.grad, [6.0, 9.0, 12.0])
    assert np.allclose(b.grad, [1.0, 2.0, 3.0])
    print("[PASS] test_ch4_tensor_basic: 第4章回帰テスト合格")


def test_ch4_abs_grad_check():
    """第4章: abs のgrad_checkを確認する。"""
    ok = grad_check(lambda t: t[0].abs().sum(), [np.array([-2.0, 1.0, -3.0])])
    assert ok, "abs grad_check失敗"
    print("[PASS] test_ch4_abs_grad_check: 第4章回帰テスト合格")


# ===== 第5章: L1損失テスト =====

def test_l1_loss_basic():
    """L1損失の基本計算を確認する。"""
    pred = Tensor(np.array([0.3, 0.7, 0.5]), requires_grad=True)
    target = Tensor(np.array([0.0, 1.0, 0.5]))
    loss = l1_loss(pred, target)
    assert np.allclose(loss.data, 0.2), f"L1損失が不正: {loss.data}"
    print("[PASS] test_l1_loss_basic")


def test_l1_loss_backward():
    """L1損失のbackwardを確認する。"""
    pred = Tensor(np.array([0.3, 0.7, 0.5]), requires_grad=True)
    target = Tensor(np.array([0.0, 1.0, 0.5]))
    loss = l1_loss(pred, target)
    loss.backward()
    expected_grad = np.array([1.0/3, -1.0/3, 0.0])
    assert np.allclose(pred.grad, expected_grad, atol=1e-7), \
        f"L1勾配が不正: {pred.grad}"
    print("[PASS] test_l1_loss_backward")


def test_l1_loss_2d():
    """2D画像に対するL1損失を確認する。"""
    pred = Tensor(np.ones((4, 4, 3)) * 0.5, requires_grad=True)
    target = Tensor(np.zeros((4, 4, 3)))
    loss = l1_loss(pred, target)
    assert np.allclose(loss.data, 0.5), f"2D L1損失が不正: {loss.data}"
    print("[PASS] test_l1_loss_2d")


# ===== 第5章: SGDオプティマイザテスト =====

def test_sgd_step():
    """SGDの1ステップを確認する。"""
    p = Tensor(np.array([5.0, 3.0]), requires_grad=True)
    p.grad = np.array([1.0, -2.0])
    sgd = SGD([p], lr=0.1)
    sgd.step()
    assert np.allclose(p.data, [4.9, 3.2]), f"SGD更新が不正: {p.data}"
    print("[PASS] test_sgd_step")


def test_sgd_zero_grad():
    """SGDのzero_gradを確認する。"""
    p = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    p.grad = np.array([3.0, 4.0])
    sgd = SGD([p], lr=0.1)
    sgd.zero_grad()
    assert np.allclose(p.grad, [0.0, 0.0]), f"zero_gradが不正: {p.grad}"
    print("[PASS] test_sgd_zero_grad")


# ===== 第5章: Adamオプティマイザテスト =====

def test_adam_step():
    """Adamの1ステップを確認する。"""
    p = Tensor(np.array([5.0, 5.0]), requires_grad=True)
    adam = Adam([p], lr=0.01)
    p.grad = np.array([1.0, 2.0])
    adam.step()
    # Adam の1ステップ後、パラメータは減少しているはず
    assert p.data[0] < 5.0 and p.data[1] < 5.0, f"Adam更新が不正: {p.data}"
    print("[PASS] test_adam_step")


def test_adam_multiple_steps():
    """Adamの複数ステップを確認する。"""
    p = Tensor(np.array([10.0]), requires_grad=True)
    adam = Adam([p], lr=0.1)
    # 一定の勾配で数ステップ
    for _ in range(10):
        p.grad = np.array([1.0])
        adam.step()
    # パラメータは減少しているはず
    assert p.data[0] < 10.0 - 0.5, f"Adam複数ステップが不正: {p.data}"
    print("[PASS] test_adam_multiple_steps")


def test_adam_zero_grad():
    """Adamのzero_gradを確認する。"""
    p = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    p.grad = np.array([3.0, 4.0])
    adam = Adam([p], lr=0.01)
    adam.zero_grad()
    assert np.allclose(p.grad, [0.0, 0.0])
    print("[PASS] test_adam_zero_grad")


# ===== 第5章: 2x2逆行列 Tensor 演算テスト =====

def test_invert_2x2_tensor():
    """invert_2x2_tensor が数値的に正しい逆行列を返すことを確認する。"""
    rng = np.random.default_rng(0)
    for _ in range(5):
        # 可逆な 2x2 行列を生成（対角優位で det が 0 から離れるようにする）
        m = rng.normal(size=(2, 2))
        m += 3.0 * np.eye(2)

        mat = Tensor(m, requires_grad=True)
        inv = invert_2x2_tensor(mat)

        # 数値: mat @ inv が I に近い
        prod = mat.data @ inv.data
        assert np.allclose(prod, np.eye(2), atol=1e-10), \
            f"mat @ inv が単位行列でない: {prod}"

        # NumPy の逆行列と一致
        assert np.allclose(inv.data, np.linalg.inv(m), atol=1e-10), \
            f"NumPy 逆行列と不一致: {inv.data}"

    # grad_check: 逆行列の要素和を backward
    def f(tensors):
        return invert_2x2_tensor(tensors[0]).sum()

    m = np.array([[3.0, 1.0], [0.5, 2.0]])
    ok = grad_check(f, [m])
    assert ok, "invert_2x2_tensor の grad_check 失敗"
    print("[PASS] test_invert_2x2_tensor")


def test_evaluate_gaussian_tensor_cov_grad():
    """evaluate_gaussian_tensor で cov にも勾配が流れることを確認する。"""
    H, W = 8, 8
    ys, xs = np.mgrid[0:H, 0:W]
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)

    mean = Tensor(np.array([4.0, 4.0]), requires_grad=True)
    cov = Tensor(build_covariance_2d(3, 3, 0), requires_grad=True)

    gauss_val = evaluate_gaussian_tensor(pixels, mean, cov)
    gauss_val.sum().backward()

    # mean の勾配が非ゼロ
    assert not np.allclose(mean.grad, 0.0), \
        f"mean 勾配がゼロ: {mean.grad}"
    # cov の勾配が非ゼロ
    assert not np.allclose(cov.grad, 0.0), \
        f"cov 勾配がゼロ: {cov.grad}"
    print("[PASS] test_evaluate_gaussian_tensor_cov_grad")


# ===== 第5章: Tensor版レンダラーテスト =====

def test_tensor_render_matches_numpy():
    """Tensor版レンダラーがNumPy版と一致することを確認する。"""
    H, W = 16, 16
    cov = build_covariance_2d(5, 5, 0)

    # NumPy版
    g1 = Gaussian2D(mean=[8, 8], covariance=cov, color=[1, 0, 0],
                    opacity=0.9, depth=0)
    g2 = Gaussian2D(mean=[10, 10], covariance=cov, color=[0, 0, 1],
                    opacity=0.9, depth=1)
    image_np = render_gaussians_alpha_composite([g1, g2], H, W)

    # Tensor版
    means = [Tensor(np.array([8.0, 8.0])), Tensor(np.array([10.0, 10.0]))]
    covs = [Tensor(cov), Tensor(cov)]
    colors = [Tensor(np.array([1.0, 0.0, 0.0])),
              Tensor(np.array([0.0, 0.0, 1.0]))]
    opacities = [Tensor(np.array(0.9)), Tensor(np.array(0.9))]
    depths = [0.0, 1.0]
    image_tensor = render_gaussians_alpha_composite_tensor(
        means, covs, colors, opacities, depths, H, W
    )

    diff = np.abs(image_np - image_tensor.data).max()
    assert diff < 1e-10, f"NumPy版とTensor版が不一致: max diff={diff}"
    print("[PASS] test_tensor_render_matches_numpy")


def test_tensor_render_backward():
    """Tensor版レンダラーのbackwardでmean勾配が流れることを確認する。"""
    H, W = 8, 8
    cov = build_covariance_2d(3, 3, 0)

    means = [Tensor(np.array([4.0, 4.0]), requires_grad=True)]
    covs = [Tensor(cov)]
    colors = [Tensor(np.array([1.0, 0.0, 0.0]), requires_grad=True)]
    opacities = [Tensor(np.array(0.9), requires_grad=True)]
    depths = [0.0]

    target = Tensor(np.zeros((H, W, 3)))
    pred = render_gaussians_alpha_composite_tensor(
        means, covs, colors, opacities, depths, H, W
    )
    loss = l1_loss(pred, target)
    loss.backward()

    # mean の勾配が非ゼロであることを確認
    assert not np.allclose(means[0].grad, 0.0), \
        f"mean勾配がゼロ: {means[0].grad}"
    # color の勾配が非ゼロであることを確認
    assert not np.allclose(colors[0].grad, 0.0), \
        f"color勾配がゼロ: {colors[0].grad}"
    # opacity の勾配が非ゼロであることを確認
    assert not np.allclose(opacities[0].grad, 0.0), \
        f"opacity勾配がゼロ: {opacities[0].grad}"
    print("[PASS] test_tensor_render_backward")


# ===== 第5章: 学習テスト =====

def test_fitting_3_gaussians():
    """3ガウシアンで赤青緑の円を近似: 500イテレーションで損失が初期値の10%以下。"""
    np.random.seed(42)
    H, W = 32, 32

    # 目標画像
    target_gaussians = [
        Gaussian2D(mean=[10, 10], covariance=build_covariance_2d(5, 5, 0),
                   color=[1, 0, 0], opacity=0.9, depth=0),
        Gaussian2D(mean=[20, 16], covariance=build_covariance_2d(5, 5, 0),
                   color=[0, 1, 0], opacity=0.9, depth=1),
        Gaussian2D(mean=[16, 24], covariance=build_covariance_2d(5, 5, 0),
                   color=[0, 0, 1], opacity=0.9, depth=2),
    ]
    target_np = render_gaussians_alpha_composite(target_gaussians, H, W)

    N = 3
    means = [Tensor(np.random.rand(2) * W, requires_grad=True) for _ in range(N)]
    covs = [Tensor(build_covariance_2d(5, 5, 0)) for _ in range(N)]
    colors = [Tensor(np.random.rand(3), requires_grad=True) for _ in range(N)]
    opacities = [Tensor(np.array(0.5), requires_grad=True) for _ in range(N)]
    depths = [float(i) for i in range(N)]

    params = means + colors + opacities
    optimizer = Adam(params, lr=0.05)
    target = Tensor(target_np)

    initial_loss = None
    for step in range(500):
        optimizer.zero_grad()
        pred = render_gaussians_alpha_composite_tensor(
            means, covs, colors, opacities, depths, H, W
        )
        loss = l1_loss(pred, target)
        loss.backward()
        optimizer.step()
        if initial_loss is None:
            initial_loss = loss.data

    final_loss = loss.data
    ratio = final_loss / initial_loss
    assert ratio < 0.10, \
        f"500イテレーション後の損失比が10%以上: {ratio:.4f}"
    print(f"[PASS] test_fitting_3_gaussians: "
          f"損失比 = {ratio:.4f} (< 0.10)")


# ===== 第5章: 本文出力例の検証 =====

def test_inline_l1_loss():
    """本文中のL1損失出力例を確認する。"""
    pred = Tensor(np.array([0.3, 0.7, 0.5]), requires_grad=True)
    target = Tensor(np.array([0.0, 1.0, 0.5]))
    loss = l1_loss(pred, target)
    assert f"{loss.data:.1f}" == "0.2", f"L1損失出力例が不正: {loss.data}"
    print("[PASS] test_inline_l1_loss")


def test_inline_l1_loss_grad():
    """本文中のL1損失勾配の出力例を確認する。"""
    pred = Tensor(np.array([0.3, 0.7, 0.5]), requires_grad=True)
    target = Tensor(np.array([0.0, 1.0, 0.5]))
    loss = l1_loss(pred, target)
    loss.backward()
    expected = np.array([1.0/3, -1.0/3, 0.0])
    assert np.allclose(pred.grad, expected, atol=1e-7), \
        f"L1勾配出力例が不正: {pred.grad}"
    print("[PASS] test_inline_l1_loss_grad")


def test_inline_sgd_step():
    """本文中のSGDステップ出力例を確認する。"""
    p = Tensor(np.array([5.0, 3.0]), requires_grad=True)
    p.grad = np.array([1.0, -2.0])
    sgd = SGD([p], lr=0.1)
    sgd.step()
    assert np.allclose(p.data, [4.9, 3.2]), \
        f"SGD出力例が不正: {p.data}"
    print("[PASS] test_inline_sgd_step")


def test_inline_tensor_render_match():
    """本文中のNumPy版とTensor版の一致確認出力例を検証する。"""
    H, W = 16, 16
    cov = build_covariance_2d(5, 5, 0)

    g1 = Gaussian2D(mean=[8, 8], covariance=cov, color=[1, 0, 0],
                    opacity=0.9, depth=0)
    g2 = Gaussian2D(mean=[10, 10], covariance=cov, color=[0, 0, 1],
                    opacity=0.9, depth=1)
    image_np = render_gaussians_alpha_composite([g1, g2], H, W)

    means = [Tensor(np.array([8.0, 8.0])), Tensor(np.array([10.0, 10.0]))]
    covs_t = [Tensor(cov), Tensor(cov)]
    colors = [Tensor(np.array([1.0, 0.0, 0.0])),
              Tensor(np.array([0.0, 0.0, 1.0]))]
    opacities = [Tensor(np.array(0.9)), Tensor(np.array(0.9))]
    depths = [0.0, 1.0]
    image_tensor = render_gaussians_alpha_composite_tensor(
        means, covs_t, colors, opacities, depths, H, W
    )
    diff = np.abs(image_np - image_tensor.data).max()
    assert diff < 1e-10, f"出力例不一致: {diff}"
    print("[PASS] test_inline_tensor_render_match")


if __name__ == "__main__":
    # 第1-2章回帰テスト
    test_ch1_single_gaussian()
    test_ch2_alpha_composite()

    # 第3章回帰テスト
    test_ch3_value_basic()
    test_ch3_scalar_grad_check()

    # 第4章回帰テスト
    test_ch4_tensor_basic()
    test_ch4_abs_grad_check()

    # 第5章: L1損失
    test_l1_loss_basic()
    test_l1_loss_backward()
    test_l1_loss_2d()

    # 第5章: SGD
    test_sgd_step()
    test_sgd_zero_grad()

    # 第5章: Adam
    test_adam_step()
    test_adam_multiple_steps()
    test_adam_zero_grad()

    # 第5章: 2x2逆行列 Tensor 演算
    test_invert_2x2_tensor()
    test_evaluate_gaussian_tensor_cov_grad()

    # 第5章: Tensor版レンダラー
    test_tensor_render_matches_numpy()
    test_tensor_render_backward()

    # 第5章: 学習テスト
    test_fitting_3_gaussians()

    # 本文出力例の検証
    test_inline_l1_loss()
    test_inline_l1_loss_grad()
    test_inline_sgd_step()
    test_inline_tensor_render_match()

    print("\n全テスト合格!")
