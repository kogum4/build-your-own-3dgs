"""
第6章テスト: 3Dガウシアン表現。
matmul, normalize, getitem のgrad_check + 共分散行列の正定値性 + 回転行列の直交性。
"""

import numpy as np
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


def test_matmul_grad_check():
    """matmulのgrad_check。"""
    from autograd import Tensor, grad_check

    # 2D x 2D
    ok1 = grad_check(
        lambda ts: ts[0].matmul(ts[1]).sum(),
        [np.random.randn(3, 4), np.random.randn(4, 2)]
    )

    # バッチ matmul
    ok2 = grad_check(
        lambda ts: ts[0].matmul(ts[1]).sum(),
        [np.random.randn(2, 3, 4), np.random.randn(2, 4, 5)]
    )

    # ブロードキャスト matmul
    ok3 = grad_check(
        lambda ts: ts[0].matmul(ts[1]).sum(),
        [np.random.randn(2, 3, 4), np.random.randn(4, 5)]
    )

    return ok1 and ok2 and ok3


def test_matmul_operator():
    """@ 演算子のテスト。"""
    from autograd import Tensor

    A = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    B = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
    C = A @ B
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    return np.allclose(C.data, expected)


def test_normalize_grad_check():
    """normalizeのgrad_check。"""
    from autograd import Tensor, grad_check

    # 1Dベクトル
    ok1 = grad_check(
        lambda ts: ts[0].normalize().sum(),
        [np.random.randn(4) + 0.1]
    )

    # 2Dバッチ
    ok2 = grad_check(
        lambda ts: ts[0].normalize().sum(),
        [np.random.randn(3, 4) + 0.1]
    )

    return ok1 and ok2


def test_normalize_unit_length():
    """normalize結果が単位ベクトルであることの確認。"""
    from autograd import Tensor

    x = Tensor(np.array([3.0, 4.0]))
    y = x.normalize()
    norm = np.sqrt(np.sum(y.data ** 2))
    return np.isclose(norm, 1.0)


def test_transpose_grad_check():
    """transposeのgrad_check。"""
    from autograd import Tensor, grad_check

    ok = grad_check(
        lambda ts: ts[0].transpose(1, 0).sum(),
        [np.random.randn(3, 4)]
    )
    return ok


def test_covariance_positive_definite():
    """ランダムパラメータから構築した共分散行列が正定値であることの確認。"""
    from gaussian3d import build_covariance_3d
    from autograd import Tensor

    np.random.seed(42)
    all_pd = True
    for _ in range(10):
        scale_raw = Tensor(np.random.randn(3), requires_grad=True)
        quat_raw = Tensor(np.random.randn(4), requires_grad=True)
        cov = build_covariance_3d(scale_raw, quat_raw)
        eigvals = np.linalg.eigvalsh(cov.data)
        if np.any(eigvals < -1e-10):
            print(f"  負の固有値: {eigvals}")
            all_pd = False

    return all_pd


def test_rotation_orthogonality():
    """クォータニオン→回転行列→直交性の検証（R@R^T ≈ I）。"""
    from gaussian3d import quaternion_to_rotation_matrix
    from autograd import Tensor

    np.random.seed(123)
    all_ok = True
    for _ in range(10):
        q_raw = np.random.randn(4)
        q_normalized = q_raw / np.linalg.norm(q_raw)
        q = Tensor(q_normalized)
        R = quaternion_to_rotation_matrix(q)
        RRT = R.data @ R.data.T
        I = np.eye(3)
        if not np.allclose(RRT, I, atol=1e-10):
            print(f"  R@R^T != I: max diff = {np.abs(RRT - I).max():.2e}")
            all_ok = False

        # det(R) ≈ 1 の確認
        det = np.linalg.det(R.data)
        if not np.isclose(det, 1.0, atol=1e-10):
            print(f"  det(R) = {det:.6f}")
            all_ok = False

    return all_ok


def test_covariance_backward():
    """共分散行列構築のbackward（grad_check）。"""
    from autograd import Tensor, grad_check

    # scale_rawへの勾配
    ok1 = grad_check(
        lambda ts: build_covariance_3d_wrapper(ts).sum(),
        [np.random.randn(3), np.random.randn(4)]
    )

    return ok1


def build_covariance_3d_wrapper(ts):
    from gaussian3d import build_covariance_3d
    return build_covariance_3d(ts[0], ts[1])


def test_gaussian3d_class():
    """Gaussian3Dクラスの基本動作確認。"""
    from gaussian3d import Gaussian3D

    g = Gaussian3D(
        position=[1.0, 2.0, 3.0],
        scale_raw=[0.0, 0.0, 0.0],
        quaternion_raw=[1.0, 0.0, 0.0, 0.0],
        opacity_raw=0.0,
        color=[1.0, 0.0, 0.0],
    )

    # パラメータ数の確認
    assert len(g.params) == 5

    # 不透明度: sigmoid(0) = 0.5
    opacity = g.get_opacity()
    assert np.isclose(opacity.data, 0.5), f"opacity={opacity.data}"

    # スケール: exp(0) = 1
    scale = g.get_scale()
    assert np.allclose(scale.data, [1.0, 1.0, 1.0])

    # 共分散: 単位クォータニオン + 単位スケール → 単位行列
    cov = g.get_covariance()
    assert np.allclose(cov.data, np.eye(3), atol=1e-10), \
        f"cov=\n{cov.data}"

    return True


def test_inline_matmul():
    """本文中のmatmulインライン確認スニペットの検証。"""
    from autograd import Tensor

    A = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    B = Tensor(np.array([[5.0, 6.0], [7.0, 8.0]]), requires_grad=True)
    C = A @ B
    loss = C.sum()
    loss.backward()

    expected_C = np.array([[19., 22.], [43., 50.]])
    expected_gradA = np.array([[11., 15.], [11., 15.]])
    expected_gradB = np.array([[4., 4.], [6., 6.]])

    ok1 = np.allclose(C.data, expected_C)
    ok2 = np.allclose(A.grad, expected_gradA)
    ok3 = np.allclose(B.grad, expected_gradB)

    return ok1 and ok2 and ok3


def test_inline_normalize():
    """本文中のnormalizeインライン確認スニペットの検証。"""
    from autograd import Tensor

    v = Tensor(np.array([3.0, 4.0]))
    n = v.normalize()
    expected = np.array([0.6, 0.8])
    return np.allclose(n.data, expected)


def test_inline_normalize_grad_check():
    """本文中のnormalize grad_checkインライン確認スニペットの検証。"""
    from autograd import Tensor, grad_check

    np.random.seed(0)
    ok = grad_check(
        lambda ts: ts[0].normalize().sum(),
        [np.random.randn(4)]
    )
    return ok


def test_inline_transpose():
    """本文中のtransposeインライン確認スニペットの検証。"""
    from autograd import Tensor

    m = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    mt = m.transpose(1, 0)
    expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    ok1 = m.data.shape == (2, 3)
    ok2 = np.allclose(mt.data, expected)
    return ok1 and ok2


def test_inline_getitem():
    """本文中のgetitemインライン確認スニペットの検証。"""
    from autograd import Tensor

    q = Tensor(np.array([1.0, 0.0, 0.0, 0.0]))
    w = q[..., 0]
    return np.isclose(w.data, 1.0)


def test_inline_quaternion():
    """本文中のクォータニオン→回転行列の確認。"""
    from gaussian3d import quaternion_to_rotation_matrix
    from autograd import Tensor

    # 単位クォータニオン → 単位行列
    q = Tensor(np.array([1.0, 0.0, 0.0, 0.0]))
    R = quaternion_to_rotation_matrix(q)
    return np.allclose(R.data, np.eye(3), atol=1e-10)


def test_inline_quaternion_z90():
    """本文中のz軸周り90度回転の検証。"""
    import math
    from gaussian3d import quaternion_to_rotation_matrix
    from autograd import Tensor

    angle = math.pi / 2
    q_z90 = Tensor(np.array([math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)]))
    R_z90 = quaternion_to_rotation_matrix(q_z90)
    expected = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    return np.allclose(R_z90.data, expected, atol=1e-10)


def test_inline_covariance():
    """本文中の共分散行列構築の確認。"""
    from gaussian3d import build_covariance_3d
    from autograd import Tensor

    scale_raw = Tensor(np.array([0.0, 0.0, 0.0]))
    quat_raw = Tensor(np.array([1.0, 0.0, 0.0, 0.0]))
    cov = build_covariance_3d(scale_raw, quat_raw)

    # exp([0,0,0]) = [1,1,1], 単位クォータニオン → 単位行列
    return np.allclose(cov.data, np.eye(3), atol=1e-10)


def test_inline_gaussian3d():
    """本文中のGaussian3Dクラスの確認。"""
    from gaussian3d import Gaussian3D

    g = Gaussian3D(
        position=[1.0, 2.0, 3.0],
        scale_raw=[0.0, 0.0, 0.0],
        quaternion_raw=[1.0, 0.0, 0.0, 0.0],
        opacity_raw=0.0,
        color=[1.0, 0.0, 0.0],
    )

    ok1 = np.isclose(g.get_opacity().data, 0.5)
    ok2 = np.allclose(g.get_scale().data, [1.0, 1.0, 1.0])
    ok3 = np.allclose(g.get_covariance().data, np.eye(3), atol=1e-10)

    return ok1 and ok2 and ok3


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
    print("=== 第6章テスト: 3Dガウシアン表現 ===\n")

    print("--- grad_check ---")
    run_test("matmul grad_check", test_matmul_grad_check)
    run_test("normalize grad_check", test_normalize_grad_check)
    run_test("transpose grad_check", test_transpose_grad_check)
    run_test("covariance backward grad_check", test_covariance_backward)

    print("\n--- 基本動作確認 ---")
    run_test("matmul @ 演算子", test_matmul_operator)
    run_test("normalize 単位ベクトル", test_normalize_unit_length)
    run_test("共分散行列 正定値", test_covariance_positive_definite)
    run_test("回転行列 直交性", test_rotation_orthogonality)
    run_test("Gaussian3D クラス", test_gaussian3d_class)

    print("\n--- 本文出力例の検証 ---")
    run_test("inline matmul", test_inline_matmul)
    run_test("inline normalize", test_inline_normalize)
    run_test("inline normalize grad_check", test_inline_normalize_grad_check)
    run_test("inline transpose", test_inline_transpose)
    run_test("inline getitem", test_inline_getitem)
    run_test("inline quaternion→回転行列", test_inline_quaternion)
    run_test("inline quaternion z軸90度回転", test_inline_quaternion_z90)
    run_test("inline 共分散行列構築", test_inline_covariance)
    run_test("inline Gaussian3D", test_inline_gaussian3d)

    print("\n--- 回帰スモークテスト ---")
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
