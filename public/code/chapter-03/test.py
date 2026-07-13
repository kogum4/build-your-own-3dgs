"""
第3章テスト: スカラー自動微分エンジンを検証する。
"""

import numpy as np
from gaussian2d import Gaussian2D, build_covariance_2d
from render import render_gaussians_weighted_sum, render_gaussians_alpha_composite
from autograd import Value, scalar_grad_check


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


# ===== 第3章: Valueクラスの基本テスト =====

def test_value_creation():
    """Valueの生成と初期値を確認する。"""
    v = Value(3.0)
    assert v.data == 3.0, f"dataが不正: {v.data}"
    assert v.grad == 0.0, f"初期gradが不正: {v.grad}"
    assert v._prev == set(), f"初期_prevが空でない"

    print("[PASS] test_value_creation: Value生成を確認")


def test_value_repr():
    """Valueの文字列表現を確認する。"""
    v = Value(3.0)
    s = repr(v)
    assert "3.0000" in s, f"reprにdataが含まれない: {s}"

    print("[PASS] test_value_repr: Value reprを確認")


def test_add():
    """加算のforward/backwardを確認する。"""
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0, f"加算結果が不正: {c.data}"

    c.backward()
    assert a.grad == 1.0, f"a.gradが不正: {a.grad}"
    assert b.grad == 1.0, f"b.gradが不正: {b.grad}"

    print("[PASS] test_add: 加算を確認")


def test_mul():
    """乗算のforward/backwardを確認する。"""
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0, f"乗算結果が不正: {c.data}"

    c.backward()
    assert a.grad == 3.0, f"a.gradが不正: {a.grad} (期待: 3.0)"
    assert b.grad == 2.0, f"b.gradが不正: {b.grad} (期待: 2.0)"

    print("[PASS] test_mul: 乗算を確認")


def test_pow():
    """べき乗のforward/backwardを確認する。"""
    x = Value(3.0)
    y = x ** 2
    assert y.data == 9.0, f"べき乗結果が不正: {y.data}"

    y.backward()
    # d(x^2)/dx = 2x = 6.0
    assert x.grad == 6.0, f"x.gradが不正: {x.grad} (期待: 6.0)"

    print("[PASS] test_pow: べき乗を確認")


def test_exp():
    """exp関数のforward/backwardを確認する。"""
    import math
    x = Value(1.0)
    y = x.exp()
    assert abs(y.data - math.e) < 1e-10, f"exp結果が不正: {y.data}"

    y.backward()
    assert abs(x.grad - math.e) < 1e-10, f"x.gradが不正: {x.grad}"

    print("[PASS] test_exp: exp関数を確認")


def test_relu():
    """ReLU関数のforward/backwardを確認する。"""
    # 正の入力
    x = Value(3.0)
    y = x.relu()
    assert y.data == 3.0
    y.backward()
    assert x.grad == 1.0

    # 負の入力
    x2 = Value(-2.0)
    y2 = x2.relu()
    assert y2.data == 0.0
    y2.backward()
    assert x2.grad == 0.0

    print("[PASS] test_relu: ReLU関数を確認")


def test_sub():
    """減算を確認する。"""
    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    assert c.data == 2.0, f"減算結果が不正: {c.data}"

    c.backward()
    assert a.grad == 1.0, f"a.gradが不正: {a.grad}"
    assert b.grad == -1.0, f"b.gradが不正: {b.grad}"

    print("[PASS] test_sub: 減算を確認")


def test_div():
    """除算を確認する。"""
    a = Value(6.0)
    b = Value(3.0)
    c = a / b
    assert abs(c.data - 2.0) < 1e-10, f"除算結果が不正: {c.data}"

    c.backward()
    # d(a/b)/da = 1/b = 1/3
    # d(a/b)/db = -a/b^2 = -6/9 = -2/3
    assert abs(a.grad - 1/3) < 1e-10, f"a.gradが不正: {a.grad}"
    assert abs(b.grad - (-2/3)) < 1e-10, f"b.gradが不正: {b.grad}"

    print("[PASS] test_div: 除算を確認")


def test_div_int():
    """整数での除算を確認する。"""
    a = Value(6.0)
    c = a / 3
    assert abs(c.data - 2.0) < 1e-10, f"整数除算結果が不正: {c.data}"
    c.backward()
    assert abs(a.grad - 1/3) < 1e-10, f"a.gradが不正: {a.grad}"
    print("[PASS] test_div_int: 整数除算を確認")


def test_neg():
    """否定を確認する。"""
    x = Value(4.0)
    y = -x
    assert y.data == -4.0, f"否定結果が不正: {y.data}"

    y.backward()
    assert x.grad == -1.0, f"x.gradが不正: {x.grad}"

    print("[PASS] test_neg: 否定を確認")


def test_radd_rmul():
    """Python数値とValueの演算（radd/rmul）を確認する。"""
    x = Value(3.0)
    y = 2 + x  # __radd__
    assert y.data == 5.0

    x2 = Value(4.0)
    y2 = 3 * x2  # __rmul__
    assert y2.data == 12.0

    print("[PASS] test_radd_rmul: radd/rmulを確認")


def test_composite_y_eq_ax_plus_b():
    """y = a*x + b の計算グラフと勾配を確認する。"""
    a = Value(2.0)
    x = Value(3.0)
    b = Value(1.0)
    y = a * x + b  # y = 2*3 + 1 = 7

    assert y.data == 7.0, f"y.dataが不正: {y.data}"

    y.backward()
    # dy/da = x = 3.0
    # dy/dx = a = 2.0
    # dy/db = 1.0
    assert a.grad == 3.0, f"a.gradが不正: {a.grad} (期待: 3.0)"
    assert x.grad == 2.0, f"x.gradが不正: {x.grad} (期待: 2.0)"
    assert b.grad == 1.0, f"b.gradが不正: {b.grad} (期待: 1.0)"

    print("[PASS] test_composite_y_eq_ax_plus_b: y=ax+bの勾配を確認")


def test_same_variable_used_twice():
    """同一変数が複数箇所で使われる場合の勾配累積を確認する。"""
    x = Value(3.0)
    y = x + x  # 同じ変数を2回使用
    y.backward()
    # dy/dx = 2 (勾配が累積される)
    assert x.grad == 2.0, f"x.gradが不正: {x.grad} (期待: 2.0)"

    x2 = Value(3.0)
    y2 = x2 * x2  # x^2を掛け算で実現
    y2.backward()
    # dy/dx = 2x = 6
    assert x2.grad == 6.0, f"x2.gradが不正: {x2.grad} (期待: 6.0)"

    print("[PASS] test_same_variable_used_twice: 勾配累積を確認")


# ===== 第3章: grad_check =====

def test_grad_check_add():
    """加算のgrad_checkを確認する。"""
    ok = scalar_grad_check(lambda v: v[0] + v[1], [2.0, 3.0])
    assert ok, "加算のgrad_checkが失敗"
    print("[PASS] test_grad_check_add")


def test_grad_check_mul():
    """乗算のgrad_checkを確認する。"""
    ok = scalar_grad_check(lambda v: v[0] * v[1], [2.0, 3.0])
    assert ok, "乗算のgrad_checkが失敗"
    print("[PASS] test_grad_check_mul")


def test_grad_check_sub():
    """減算のgrad_checkを確認する。"""
    ok = scalar_grad_check(lambda v: v[0] - v[1], [5.0, 3.0])
    assert ok, "減算のgrad_checkが失敗"
    print("[PASS] test_grad_check_sub")


def test_grad_check_div():
    """除算のgrad_checkを確認する。"""
    ok = scalar_grad_check(lambda v: v[0] / v[1], [6.0, 3.0])
    assert ok, "除算のgrad_checkが失敗"
    print("[PASS] test_grad_check_div")


def test_grad_check_pow():
    """べき乗のgrad_checkを確認する。"""
    ok = scalar_grad_check(lambda v: v[0] ** 3, [2.0])
    assert ok, "べき乗のgrad_checkが失敗"
    print("[PASS] test_grad_check_pow")


def test_grad_check_exp():
    """expのgrad_checkを確認する。"""
    ok = scalar_grad_check(lambda v: v[0].exp(), [1.5])
    assert ok, "expのgrad_checkが失敗"
    print("[PASS] test_grad_check_exp")


def test_grad_check_relu():
    """ReLUのgrad_checkを確認する。"""
    # 正の入力
    ok = scalar_grad_check(lambda v: v[0].relu(), [2.0])
    assert ok, "ReLU(正)のgrad_checkが失敗"
    # 負の入力
    ok = scalar_grad_check(lambda v: v[0].relu(), [-2.0])
    assert ok, "ReLU(負)のgrad_checkが失敗"
    print("[PASS] test_grad_check_relu")


def test_grad_check_neg():
    """否定のgrad_checkを確認する。"""
    ok = scalar_grad_check(lambda v: -v[0], [3.0])
    assert ok, "否定のgrad_checkが失敗"
    print("[PASS] test_grad_check_neg")


def test_grad_check_composite():
    """合成関数 f(x)=exp(x^2+2x) のgrad_checkを確認する。"""
    def f(v):
        x = v[0]
        return (x ** 2 + 2 * x).exp()

    ok = scalar_grad_check(f, [1.0])
    assert ok, "合成関数のgrad_checkが失敗"
    print("[PASS] test_grad_check_composite")


def test_grad_check_complex_chain():
    """複雑な合成: f(a,b) = (a*b + a^2).exp() / b のgrad_checkを確認する。"""
    def f(v):
        a, b = v[0], v[1]
        return (a * b + a ** 2).exp() / b

    ok = scalar_grad_check(f, [0.5, 2.0])
    assert ok, "複雑な合成関数のgrad_checkが失敗"
    print("[PASS] test_grad_check_complex_chain")


# ===== 第3章: 勾配降下法 =====

def test_gradient_descent_convergence():
    """f(x)=(x-3)^2 の勾配降下法が x=3 に収束することを確認する。"""
    lr = 0.1
    x_val = 0.0

    for _ in range(100):
        x = Value(x_val)
        loss = (x - 3) ** 2
        loss.backward()
        x_val -= lr * x.grad

    assert abs(x_val - 3.0) < 1e-6, f"収束していない: x={x_val:.6f} (期待: 3.0)"

    print("[PASS] test_gradient_descent_convergence: 勾配降下法の収束を確認")


# ===== 本文出力例の検証 =====

def test_inline_value_creation():
    """本文中のValue生成の出力例を確認する。"""
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert repr(c) == "Value(data=5.0000, grad=0.0000)", \
        f"reprが不正: {repr(c)}"

    print("[PASS] test_inline_value_creation: Value生成の出力例を確認")


def test_inline_ax_plus_b():
    """本文中の y=a*x+b の出力例を確認する。"""
    a = Value(2.0)
    x = Value(3.0)
    b = Value(1.0)
    y = a * x + b

    assert y.data == 7.0, f"y.dataが不正: {y.data}"

    y.backward()
    assert a.grad == 3.0, f"a.gradが不正: {a.grad}"
    assert x.grad == 2.0, f"x.gradが不正: {x.grad}"
    assert b.grad == 1.0, f"b.gradが不正: {b.grad}"

    print("[PASS] test_inline_ax_plus_b: y=ax+b出力例を確認")


def test_inline_pow_exp_relu_forward():
    """本文中のべき乗・exp・ReLU forward確認スニペットの出力例を検証する。"""
    v = Value(3.0) ** 2
    assert repr(v) == "Value(data=9.0000, grad=0.0000)", f"べき乗repr不正: {repr(v)}"

    v = Value(1.0).exp()
    assert repr(v) == "Value(data=2.7183, grad=0.0000)", f"exp repr不正: {repr(v)}"

    v = Value(3.0).relu()
    assert repr(v) == "Value(data=3.0000, grad=0.0000)", f"ReLU(正) repr不正: {repr(v)}"

    v = Value(-2.0).relu()
    assert repr(v) == "Value(data=0.0000, grad=0.0000)", f"ReLU(負) repr不正: {repr(v)}"

    print("[PASS] test_inline_pow_exp_relu_forward: べき乗・exp・ReLU forward出力例を確認")


def test_inline_backward_all_ops():
    """本文中の全演算backward確認スニペットの出力例を検証する。"""
    # 加算
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    assert a.grad == 1.0 and b.grad == 1.0

    # 乗算
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    assert a.grad == 3.0 and b.grad == 2.0

    # べき乗
    x = Value(3.0)
    y = x ** 2
    y.backward()
    assert x.grad == 6.0

    # exp
    x = Value(1.0)
    y = x.exp()
    y.backward()
    assert abs(x.grad - 2.7183) < 0.001

    # ReLU (正)
    x = Value(3.0)
    y = x.relu()
    y.backward()
    assert x.grad == 1.0

    # ReLU (負)
    x = Value(-2.0)
    y = x.relu()
    y.backward()
    assert x.grad == 0.0

    print("[PASS] test_inline_backward_all_ops: 全演算backward出力例を確認")


def test_inline_grad_check_output():
    """本文中のgrad_check出力例を確認する。"""
    def f(v):
        x = v[0]
        return (x ** 2 + 2 * x).exp()

    ok = scalar_grad_check(f, [1.0])
    assert ok, "本文中のgrad_check例が失敗"

    print("[PASS] test_inline_grad_check_output: grad_check出力例を確認")


def test_inline_gradient_descent():
    """本文中の勾配降下法の出力例を確認する。"""
    lr = 0.1
    x_val = 0.0

    for step in range(100):
        x = Value(x_val)
        loss = (x - 3) ** 2
        loss.backward()
        x_val -= lr * x.grad

    # 最終的に x ≈ 3.0
    assert abs(x_val - 3.0) < 1e-6, f"x={x_val} (期待: ≈3.0)"

    print("[PASS] test_inline_gradient_descent: 勾配降下法出力例を確認")


def test_inline_gradient_descent_steps():
    """本文中の勾配降下法の途中経過を確認する。"""
    lr = 0.1
    x_val = 0.0
    steps_to_check = {0: 0.0, 1: 0.6}

    for step in range(100):
        x = Value(x_val)
        loss = (x - 3) ** 2
        loss.backward()

        if step in steps_to_check:
            # ステップ0: x=0, loss=9, grad=-6
            if step == 0:
                assert abs(loss.data - 9.0) < 1e-10, f"step0 loss不正: {loss.data}"
                assert abs(x.grad - (-6.0)) < 1e-10, f"step0 grad不正: {x.grad}"

        x_val -= lr * x.grad

        if step == 0:
            # ステップ1のx値: 0 - 0.1*(-6) = 0.6
            assert abs(x_val - 0.6) < 1e-10, f"step1 x不正: {x_val}"

    print("[PASS] test_inline_gradient_descent_steps: 勾配降下法の途中経過を確認")


def test_inline_learning_rate_experiment():
    """本文中の学習率実験の出力例を確認する。"""
    expected = {
        0.01: (1.3635, 2.677978),
        0.1: (2.9963, 0.000014),
        0.5: (3.0000, 0.000000),
    }
    for lr, (exp_x, exp_loss) in expected.items():
        x_val = 0.0
        for step in range(30):
            x = Value(x_val)
            loss = (x - 3) ** 2
            loss.backward()
            x_val -= lr * x.grad
        actual_loss = (x_val - 3) ** 2
        assert abs(x_val - exp_x) < 0.001, \
            f"lr={lr}: x={x_val:.4f} (期待: {exp_x})"
        assert abs(actual_loss - exp_loss) < 0.001, \
            f"lr={lr}: loss={actual_loss:.6f} (期待: {exp_loss})"

    print("[PASS] test_inline_learning_rate_experiment: 学習率実験の出力例を確認")


if __name__ == "__main__":
    # 第1-2章回帰テスト
    test_ch1_single_gaussian()
    test_ch2_alpha_composite()

    # 第3章: Valueクラス基本テスト
    test_value_creation()
    test_value_repr()
    test_add()
    test_mul()
    test_pow()
    test_exp()
    test_relu()
    test_sub()
    test_div()
    test_div_int()
    test_neg()
    test_radd_rmul()
    test_composite_y_eq_ax_plus_b()
    test_same_variable_used_twice()

    # 第3章: grad_check
    test_grad_check_add()
    test_grad_check_mul()
    test_grad_check_sub()
    test_grad_check_div()
    test_grad_check_pow()
    test_grad_check_exp()
    test_grad_check_relu()
    test_grad_check_neg()
    test_grad_check_composite()
    test_grad_check_complex_chain()

    # 第3章: 勾配降下法
    test_gradient_descent_convergence()

    # 本文出力例テスト
    test_inline_value_creation()
    test_inline_ax_plus_b()
    test_inline_pow_exp_relu_forward()
    test_inline_backward_all_ops()
    test_inline_grad_check_output()
    test_inline_gradient_descent()
    test_inline_gradient_descent_steps()
    test_inline_learning_rate_experiment()

    print("\n全テスト合格!")
