"""
第4章テスト: テンソル自動微分エンジンを検証する。
"""

import numpy as np
from gaussian2d import Gaussian2D, build_covariance_2d
from render import render_gaussians_weighted_sum, render_gaussians_alpha_composite
from autograd import Value, scalar_grad_check, Tensor, grad_check, stack, unbind
from autograd import _unbroadcast


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


# ===== 第4章: Tensorクラスの基本テスト =====

def test_tensor_creation():
    """Tensorの生成と初期値を確認する。"""
    t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    assert t.shape == (3,), f"shapeが不正: {t.shape}"
    assert np.allclose(t.grad, [0.0, 0.0, 0.0]), f"初期gradが不正: {t.grad}"
    assert t.requires_grad is True

    t2 = Tensor(np.array([[1, 2], [3, 4]]))
    assert t2.shape == (2, 2)
    assert t2.grad is None  # requires_grad=False
    assert t2.requires_grad is False

    print("[PASS] test_tensor_creation: Tensor生成を確認")


def test_tensor_repr():
    """Tensorの文字列表現を確認する。"""
    t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    s = repr(t)
    assert "Tensor" in s
    assert "data" in s
    assert "grad" in s
    print("[PASS] test_tensor_repr: Tensor reprを確認")


def test_zero_grad():
    """zero_gradが勾配をリセットすることを確認する。"""
    t = Tensor([1.0, 2.0], requires_grad=True)
    t.grad = np.array([5.0, 6.0])
    t.zero_grad()
    assert np.allclose(t.grad, [0.0, 0.0])
    print("[PASS] test_zero_grad: zero_gradを確認")


# ===== 第4章: _unbroadcast テスト =====

def test_unbroadcast_dim_reduction():
    """次元数の差分に対する_unbroadcastを確認する。"""
    grad = np.ones((3, 4))
    result = _unbroadcast(grad, (4,))
    assert result.shape == (4,)
    assert np.allclose(result, [3.0, 3.0, 3.0, 3.0])
    print("[PASS] test_unbroadcast_dim_reduction")


def test_unbroadcast_size1_axis():
    """サイズ1軸に対する_unbroadcastを確認する。"""
    grad = np.ones((2, 3, 3))
    result = _unbroadcast(grad, (2, 1, 3))
    assert result.shape == (2, 1, 3)
    assert np.allclose(result, [[[3.0, 3.0, 3.0]], [[3.0, 3.0, 3.0]]])
    print("[PASS] test_unbroadcast_size1_axis")


def test_unbroadcast_no_change():
    """ブロードキャストなしの場合、_unbroadcastが何もしないことを確認する。"""
    grad = np.ones((3, 4))
    result = _unbroadcast(grad, (3, 4))
    assert result.shape == (3, 4)
    assert np.allclose(result, np.ones((3, 4)))
    print("[PASS] test_unbroadcast_no_change")


# ===== 第4章: M1 要素単位演算 =====

def test_add_forward():
    """加算のforwardを確認する。"""
    a = Tensor(np.array([1.0, 2.0]))
    b = Tensor(np.array([3.0, 4.0]))
    c = a + b
    assert np.allclose(c.data, [4.0, 6.0])
    print("[PASS] test_add_forward")


def test_add_broadcast_backward():
    """ブロードキャスト付き加算のbackwardを確認する。"""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([10.0, 20.0]), requires_grad=True)
    c = (a + b).sum()
    c.backward()
    assert np.allclose(a.grad, [[1.0, 1.0], [1.0, 1.0]])
    assert np.allclose(b.grad, [2.0, 2.0])
    print("[PASS] test_add_broadcast_backward")


def test_sub_backward():
    """減算のbackwardを確認する。"""
    a = Tensor(np.array([5.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    c = (a - b).sum()
    c.backward()
    assert np.allclose(a.grad, [1.0, 1.0])
    assert np.allclose(b.grad, [-1.0, -1.0])
    print("[PASS] test_sub_backward")


def test_mul_backward():
    """乗算のbackwardを確認する。"""
    a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    b = Tensor(np.array([4.0, 5.0]), requires_grad=True)
    c = (a * b).sum()
    c.backward()
    assert np.allclose(a.grad, [4.0, 5.0])
    assert np.allclose(b.grad, [2.0, 3.0])
    print("[PASS] test_mul_backward")


def test_div_backward():
    """除算のbackwardを確認する。"""
    a = Tensor(np.array([6.0, 4.0]), requires_grad=True)
    b = Tensor(np.array([2.0, 4.0]), requires_grad=True)
    c = (a / b).sum()
    c.backward()
    # da = 1/b = [0.5, 0.25]
    assert np.allclose(a.grad, [0.5, 0.25])
    # db = -a/b^2 = [-6/4, -4/16] = [-1.5, -0.25]
    assert np.allclose(b.grad, [-1.5, -0.25])
    print("[PASS] test_div_backward")


def test_neg_backward():
    """否定のbackwardを確認する。"""
    a = Tensor(np.array([1.0, -2.0, 3.0]), requires_grad=True)
    c = (-a).sum()
    c.backward()
    assert np.allclose(a.grad, [-1.0, -1.0, -1.0])
    print("[PASS] test_neg_backward")


def test_pow_backward():
    """べき乗のbackwardを確認する。"""
    a = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    c = (a ** 3).sum()
    c.backward()
    # d(x^3)/dx = 3x^2
    assert np.allclose(a.grad, [12.0, 27.0])
    print("[PASS] test_pow_backward")


def test_radd_rmul():
    """radd/rmulが正しく動作することを確認する。"""
    a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    c = (2 + a).sum()
    c.backward()
    assert np.allclose(a.grad, [1.0, 1.0])

    b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    d = (3 * b).sum()
    d.backward()
    assert np.allclose(b.grad, [3.0, 3.0])

    print("[PASS] test_radd_rmul")


def test_rsub_rtruediv():
    """rsub/rtruedivが正しく動作することを確認する。"""
    a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    c = (5 - a).sum()
    c.backward()
    assert np.allclose(a.grad, [-1.0, -1.0])

    b = Tensor(np.array([2.0, 4.0]), requires_grad=True)
    d = (8 / b).sum()
    d.backward()
    # d(8/b)/db = -8/b^2
    assert np.allclose(b.grad, [-2.0, -0.5])

    print("[PASS] test_rsub_rtruediv")


# ===== 第4章: M1 grad_check =====

def test_grad_check_m1_add():
    ok = grad_check(lambda t: (t[0] + t[1]).sum(),
                    [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5])])
    assert ok, "add grad_check失敗"
    print("[PASS] test_grad_check_m1_add")


def test_grad_check_m1_sub():
    ok = grad_check(lambda t: (t[0] - t[1]).sum(),
                    [np.array([5.0, 3.0]), np.array([1.0, 2.0])])
    assert ok, "sub grad_check失敗"
    print("[PASS] test_grad_check_m1_sub")


def test_grad_check_m1_mul():
    ok = grad_check(lambda t: (t[0] * t[1]).sum(),
                    [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, 1.5])])
    assert ok, "mul grad_check失敗"
    print("[PASS] test_grad_check_m1_mul")


def test_grad_check_m1_div():
    ok = grad_check(lambda t: (t[0] / t[1]).sum(),
                    [np.array([6.0, 4.0]), np.array([2.0, 3.0])])
    assert ok, "div grad_check失敗"
    print("[PASS] test_grad_check_m1_div")


def test_grad_check_m1_neg():
    ok = grad_check(lambda t: (-t[0]).sum(), [np.array([1.0, -2.0, 3.0])])
    assert ok, "neg grad_check失敗"
    print("[PASS] test_grad_check_m1_neg")


def test_grad_check_m1_pow():
    ok = grad_check(lambda t: (t[0] ** 3).sum(), [np.array([2.0, 3.0])])
    assert ok, "pow grad_check失敗"
    print("[PASS] test_grad_check_m1_pow")


# ===== 第4章: M2 数学関数 =====

def test_exp_forward():
    """expのforwardを確認する。"""
    t = Tensor(np.array([0.0, 1.0]), requires_grad=True)
    y = t.exp()
    assert np.allclose(y.data, [1.0, np.e])
    print("[PASS] test_exp_forward")


def test_log_forward():
    """logのforwardを確認する。"""
    t = Tensor(np.array([1.0, np.e]), requires_grad=True)
    y = t.log()
    assert np.allclose(y.data, [0.0, 1.0])
    print("[PASS] test_log_forward")


def test_sigmoid_forward():
    """sigmoidのforwardを確認する。"""
    t = Tensor(np.array([0.0]), requires_grad=True)
    y = t.sigmoid()
    assert np.allclose(y.data, [0.5])
    print("[PASS] test_sigmoid_forward")


def test_abs_forward():
    """absのforwardを確認する。"""
    t = Tensor(np.array([-2.0, 3.0, -1.0]))
    y = t.abs()
    assert np.allclose(y.data, [2.0, 3.0, 1.0])
    print("[PASS] test_abs_forward")


def test_grad_check_m2_exp():
    ok = grad_check(lambda t: t[0].exp().sum(), [np.array([0.5, 1.0, 1.5])])
    assert ok, "exp grad_check失敗"
    print("[PASS] test_grad_check_m2_exp")


def test_grad_check_m2_log():
    ok = grad_check(lambda t: t[0].log().sum(), [np.array([1.0, 2.0, 3.0])])
    assert ok, "log grad_check失敗"
    print("[PASS] test_grad_check_m2_log")


def test_grad_check_m2_sigmoid():
    ok = grad_check(lambda t: t[0].sigmoid().sum(), [np.array([-1.0, 0.0, 1.0])])
    assert ok, "sigmoid grad_check失敗"
    print("[PASS] test_grad_check_m2_sigmoid")


def test_grad_check_m2_abs():
    ok = grad_check(lambda t: t[0].abs().sum(), [np.array([-2.0, 1.0, -3.0])])
    assert ok, "abs grad_check失敗"
    print("[PASS] test_grad_check_m2_abs")


# ===== 第4章: M3 集約・形状操作 =====

def test_sum_axis():
    """axis指定のsumを確認する。"""
    a = Tensor(np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]]), requires_grad=True)
    s = a.sum(axis=0)
    assert np.allclose(s.data, [5.0, 7.0, 9.0])

    total = s.sum()
    total.backward()
    assert np.allclose(a.grad, np.ones((2, 3)))
    print("[PASS] test_sum_axis")


def test_mean():
    """meanを確認する。"""
    a = Tensor(np.array([2.0, 4.0, 6.0]), requires_grad=True)
    m = a.mean()
    assert np.allclose(m.data, 4.0)
    m.backward()
    assert np.allclose(a.grad, [1/3, 1/3, 1/3])
    print("[PASS] test_mean")


def test_reshape():
    """reshapeを確認する。"""
    a = Tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), requires_grad=True)
    b = a.reshape(2, 3)
    assert b.shape == (2, 3)
    c = b.sum()
    c.backward()
    assert np.allclose(a.grad, np.ones(6))
    print("[PASS] test_reshape")


def test_reshape_tuple():
    """reshape((2, 3)) の形式も動くことを確認する。"""
    a = Tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), requires_grad=True)
    b = a.reshape((2, 3))
    assert b.shape == (2, 3)
    print("[PASS] test_reshape_tuple")


def test_stack():
    """stackのforward/backwardを確認する。"""
    a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    c = Tensor(np.array([5.0, 6.0]), requires_grad=True)
    s = stack([a, b, c])
    assert s.shape == (3, 2)
    assert np.allclose(s.data, [[1, 2], [3, 4], [5, 6]])

    total = s.sum()
    total.backward()
    assert np.allclose(a.grad, [1.0, 1.0])
    assert np.allclose(b.grad, [1.0, 1.0])
    assert np.allclose(c.grad, [1.0, 1.0])
    print("[PASS] test_stack")


def test_unbind():
    """unbindのforward/backwardを確認する。"""
    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), requires_grad=True)
    parts = unbind(t, axis=0)
    assert len(parts) == 3
    assert np.allclose(parts[0].data, [1.0, 2.0])
    assert np.allclose(parts[1].data, [3.0, 4.0])

    y = (parts[0] + parts[1] * 2 + parts[2] * 3).sum()
    y.backward()
    assert np.allclose(t.grad, [[1, 1], [2, 2], [3, 3]])
    print("[PASS] test_unbind")


def test_grad_check_m3_sum():
    ok = grad_check(lambda t: t[0].sum(axis=1).sum(),
                    [np.array([[1.0, 2.0], [3.0, 4.0]])])
    assert ok, "sum(axis) grad_check失敗"
    print("[PASS] test_grad_check_m3_sum")


def test_grad_check_m3_mean():
    ok = grad_check(lambda t: t[0].mean(),
                    [np.array([[1.0, 2.0], [3.0, 4.0]])])
    assert ok, "mean grad_check失敗"
    print("[PASS] test_grad_check_m3_mean")


def test_grad_check_m3_reshape():
    ok = grad_check(lambda t: (t[0].reshape(2, 3) ** 2).sum(),
                    [np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])])
    assert ok, "reshape grad_check失敗"
    print("[PASS] test_grad_check_m3_reshape")


def test_grad_check_m3_stack():
    def f_stack(t):
        return stack([t[0], t[1]]).sum()
    ok = grad_check(f_stack,
                    [np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    assert ok, "stack grad_check失敗"
    print("[PASS] test_grad_check_m3_stack")


def test_grad_check_m3_unbind():
    def f_unbind(t):
        parts = unbind(t[0], axis=0)
        return (parts[0] * 2 + parts[1] * 3).sum()
    ok = grad_check(f_unbind,
                    [np.array([[1.0, 2.0], [3.0, 4.0]])])
    assert ok, "unbind grad_check失敗"
    print("[PASS] test_grad_check_m3_unbind")


# ===== 第4章: M3 getitem =====

def test_getitem_int_index():
    """getitem(Ellipsis+整数)の基本動作を確認する。"""
    t = Tensor(np.array([10.0, 20.0, 30.0, 40.0]))
    result = t[..., 2]
    assert np.isclose(result.data, 30.0), f"取り出し結果が不正: {result.data}"
    print("[PASS] test_getitem_int_index")


def test_getitem_ellipsis_int_grad_check():
    """getitem(Ellipsis+整数)のgrad_check。"""
    # 1D
    ok1 = grad_check(lambda t: t[0][..., 0].sum(),
                     [np.random.randn(4)])
    assert ok1, "1D getitem grad_check失敗"

    # 2D
    ok2 = grad_check(lambda t: t[0][..., 1].sum(),
                     [np.random.randn(3, 4)])
    assert ok2, "2D getitem grad_check失敗"
    print("[PASS] test_getitem_ellipsis_int_grad_check")


def test_getitem_slice_grad_check():
    """getitem(スライス)のgrad_check。同じコードでスライスにも対応することを確認する。"""
    # 複数軸スライス: (3, 3) → [..., :2, :2]
    ok1 = grad_check(lambda t: t[0][..., :2, :2].sum(),
                     [np.random.randn(3, 3)])
    assert ok1, "(3,3)[...,:2,:2] grad_check失敗"

    # バッチ + 複数軸スライス: (4, 3, 3) → [..., :2, :2]
    ok2 = grad_check(lambda t: t[0][..., :2, :2].sum(),
                     [np.random.randn(4, 3, 3)])
    assert ok2, "(4,3,3)[...,:2,:2] grad_check失敗"

    # 単一軸スライス: (5, 3) → [:3, :]
    ok3 = grad_check(lambda t: t[0][:3, :].sum(),
                     [np.random.randn(5, 3)])
    assert ok3, "(5,3)[:3,:] grad_check失敗"
    print("[PASS] test_getitem_slice_grad_check")


# ===== 第4章: ブロードキャスト パターン =====

def test_broadcast_3d():
    """(2,1,3)*(1,3,3) のブロードキャストを確認する。"""
    ok = grad_check(lambda t: (t[0] * t[1]).sum(),
                    [np.random.randn(2, 1, 3), np.random.randn(1, 3, 3)])
    assert ok, "(2,1,3)*(1,3,3) grad_check失敗"
    print("[PASS] test_broadcast_3d")


def test_broadcast_scalar():
    """テンソルとスカラーのブロードキャストを確認する。"""
    ok = grad_check(lambda t: (t[0] * t[1]).sum(),
                    [np.random.randn(3, 4), np.array(2.0)])
    assert ok, "スカラーブロードキャスト grad_check失敗"
    print("[PASS] test_broadcast_scalar")


# ===== 第4章: 合成関数 =====

def test_composite_function():
    """合成関数のgrad_checkを確認する。"""
    def f_composite(t):
        a, b = t[0], t[1]
        return (a * b + a ** 2).exp().mean()

    ok = grad_check(f_composite,
                    [np.array([0.5, 1.0]), np.array([0.3, 0.7])])
    assert ok, "合成関数 grad_check失敗"
    print("[PASS] test_composite_function")


# ===== 本文出力例の検証 =====

def test_inline_tensor_creation():
    """本文中のTensor生成の出力例を確認する。"""
    t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    s = repr(t)
    assert "Tensor" in s
    assert t.shape == (3,)
    assert np.allclose(t.grad, [0.0, 0.0, 0.0])
    print("[PASS] test_inline_tensor_creation")


def test_inline_unbroadcast():
    """本文中の_unbroadcast出力例を確認する。"""
    grad = np.ones((3, 4))
    result = _unbroadcast(grad, (4,))
    assert np.allclose(result, [3.0, 3.0, 3.0, 3.0])

    grad = np.ones((2, 3, 3))
    result = _unbroadcast(grad, (2, 1, 3))
    assert result.shape == (2, 1, 3)
    assert np.allclose(result, [[[3.0, 3.0, 3.0]], [[3.0, 3.0, 3.0]]])
    print("[PASS] test_inline_unbroadcast")


def test_inline_add_broadcast():
    """本文中のブロードキャスト加算の出力例を確認する。"""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([10.0, 20.0]), requires_grad=True)
    c = a + b
    assert np.allclose(c.data, [[11.0, 22.0], [13.0, 24.0]])
    print("[PASS] test_inline_add_broadcast")


def test_inline_m1_backward():
    """本文中のM1 backward確認の出力例を検証する。"""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([10.0, 20.0]), requires_grad=True)
    c = (a + b).sum()
    c.backward()
    assert np.allclose(a.grad, [[1.0, 1.0], [1.0, 1.0]])
    assert np.allclose(b.grad, [2.0, 2.0])
    print("[PASS] test_inline_m1_backward")


def test_inline_exp_backward():
    """本文中のexp backward出力例を確認する。"""
    t = Tensor(np.array([0.0, 1.0, 2.0]), requires_grad=True)
    y = t.exp().sum()
    y.backward()
    assert np.allclose(t.grad, [1.0, np.e, np.e**2])
    print("[PASS] test_inline_exp_backward")


def test_inline_sigmoid():
    """本文中のsigmoid出力例を確認する。"""
    t2 = Tensor(np.array([0.0]), requires_grad=True)
    y2 = t2.sigmoid()
    assert np.allclose(y2.data, [0.5])
    print("[PASS] test_inline_sigmoid")


def test_inline_sum_axis():
    """本文中のsum(axis=0)出力例を確認する。"""
    a = Tensor(np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]]), requires_grad=True)
    s = a.sum(axis=0)
    assert np.allclose(s.data, [5.0, 7.0, 9.0])
    total = s.sum()
    total.backward()
    assert np.allclose(a.grad, np.ones((2, 3)))
    print("[PASS] test_inline_sum_axis")


def test_inline_mean():
    """本文中のmean出力例を確認する。"""
    a = Tensor(np.array([2.0, 4.0, 6.0]), requires_grad=True)
    m = a.mean()
    assert np.allclose(m.data, 4.0)
    m.backward()
    assert np.allclose(a.grad, [1/3, 1/3, 1/3])
    print("[PASS] test_inline_mean")


def test_inline_reshape():
    """本文中のreshape出力例を確認する。"""
    a = Tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), requires_grad=True)
    b = a.reshape(2, 3)
    assert np.allclose(b.data, [[1, 2, 3], [4, 5, 6]])
    c = b.sum()
    c.backward()
    assert np.allclose(a.grad, np.ones(6))
    print("[PASS] test_inline_reshape")


def test_inline_getitem():
    """本文中のgetitem出力例を確認する。"""
    t = Tensor(np.array([10.0, 20.0, 30.0, 40.0]), requires_grad=True)
    v = t[..., 2]
    assert np.isclose(v.data, 30.0)
    loss = v * 3.0
    loss.backward()
    assert np.allclose(t.grad, [0.0, 0.0, 3.0, 0.0])
    print("[PASS] test_inline_getitem")


def test_inline_stack_unbind():
    """本文中のstack/unbind出力例を確認する。"""
    # stack
    a = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    b = Tensor(np.array([3.0, 4.0]), requires_grad=True)
    c = Tensor(np.array([5.0, 6.0]), requires_grad=True)
    s = stack([a, b, c])
    assert np.allclose(s.data, [[1, 2], [3, 4], [5, 6]])
    total = s.sum()
    total.backward()
    assert np.allclose(a.grad, [1.0, 1.0])

    # unbind
    t = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), requires_grad=True)
    parts = unbind(t, axis=0)
    assert np.allclose(parts[0].data, [1.0, 2.0])
    assert np.allclose(parts[1].data, [3.0, 4.0])
    y = (parts[0] + parts[1] * 2 + parts[2] * 3).sum()
    y.backward()
    assert np.allclose(t.grad, [[1, 1], [2, 2], [3, 3]])
    print("[PASS] test_inline_stack_unbind")


if __name__ == "__main__":
    # 第1-2章回帰テスト
    test_ch1_single_gaussian()
    test_ch2_alpha_composite()

    # 第3章回帰テスト
    test_ch3_value_basic()
    test_ch3_scalar_grad_check()

    # 第4章: 基本テスト
    test_tensor_creation()
    test_tensor_repr()
    test_zero_grad()

    # 第4章: _unbroadcast
    test_unbroadcast_dim_reduction()
    test_unbroadcast_size1_axis()
    test_unbroadcast_no_change()

    # 第4章: M1 要素単位演算
    test_add_forward()
    test_add_broadcast_backward()
    test_sub_backward()
    test_mul_backward()
    test_div_backward()
    test_neg_backward()
    test_pow_backward()
    test_radd_rmul()
    test_rsub_rtruediv()

    # 第4章: M1 grad_check
    test_grad_check_m1_add()
    test_grad_check_m1_sub()
    test_grad_check_m1_mul()
    test_grad_check_m1_div()
    test_grad_check_m1_neg()
    test_grad_check_m1_pow()

    # 第4章: M2 数学関数
    test_exp_forward()
    test_log_forward()
    test_sigmoid_forward()
    test_abs_forward()
    test_grad_check_m2_exp()
    test_grad_check_m2_log()
    test_grad_check_m2_sigmoid()
    test_grad_check_m2_abs()

    # 第4章: M3 集約・形状操作
    test_sum_axis()
    test_mean()
    test_reshape()
    test_reshape_tuple()
    test_stack()
    test_unbind()
    test_grad_check_m3_sum()
    test_grad_check_m3_mean()
    test_grad_check_m3_reshape()
    test_grad_check_m3_stack()
    test_grad_check_m3_unbind()

    # 第4章: M3 getitem
    test_getitem_int_index()
    test_getitem_ellipsis_int_grad_check()
    test_getitem_slice_grad_check()

    # 第4章: ブロードキャスト
    test_broadcast_3d()
    test_broadcast_scalar()

    # 第4章: 合成関数
    test_composite_function()

    # 本文出力例テスト
    test_inline_tensor_creation()
    test_inline_unbroadcast()
    test_inline_add_broadcast()
    test_inline_m1_backward()
    test_inline_exp_backward()
    test_inline_sigmoid()
    test_inline_sum_axis()
    test_inline_mean()
    test_inline_reshape()
    test_inline_getitem()
    test_inline_stack_unbind()

    print("\n全テスト合格!")
