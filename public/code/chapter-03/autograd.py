"""
スカラー自動微分エンジン: Valueクラス。
第3章: スカラー自動微分エンジン

micrograd方式のスカラー自動微分を実装する。
Valueクラスの構造（_backward、_prev、backward()）は
第4章のTensorクラスにそのまま引き継がれる。
"""

import math


class Value:
    """計算グラフのノード。スカラー値と勾配を保持する。

    Attributes:
        data: スカラー値（float）
        grad: 逆伝播で計算された勾配（float）
        _backward: 勾配を伝播させるクロージャ
        _prev: このノードの入力ノードの集合
    """

    def __init__(self, data):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None  # デフォルトは何もしない
        self._prev = set()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- 加算 ---

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            # d(a + b)/da = 1, d(a + b)/db = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    # --- 乗算 ---

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            # d(a * b)/da = b, d(a * b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    # --- 逆伝播 ---

    def backward(self):
        """トポロジカルソートで計算グラフを逆順走査し、勾配を伝播する。"""
        # トポロジカルソート（深さ優先探索）
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # 出力ノードの勾配を1に設定
        self.grad = 1.0

        # 逆順に_backwardを呼び出す
        for v in reversed(topo):
            v._backward()

    # --- べき乗・指数関数・ReLU ---

    def __pow__(self, n):
        assert isinstance(n, (int, float)), "べき指数はint/floatのみ"
        out = Value(self.data ** n)
        out._prev = {self}

        def _backward():
            # d(x^n)/dx = n * x^(n-1)
            self.grad += n * (self.data ** (n - 1)) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        val = math.exp(self.data)
        out = Value(val)
        out._prev = {self}

        def _backward():
            # d(exp(x))/dx = exp(x)
            self.grad += val * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data))
        out._prev = {self}

        def _backward():
            # d(relu(x))/dx = 1 if x > 0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # --- 残りの四則演算 ---

    def __sub__(self, other):
        # a - b = a + (-b)
        return self + (-other)

    def __rsub__(self, other):
        # other - self = (-self) + other
        return (-self) + other

    def __truediv__(self, other):
        # a / b = a * b^(-1)
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** (-1)

    def __rtruediv__(self, other):
        # other / self = other * self^(-1)
        other = other if isinstance(other, Value) else Value(other)
        return other * self ** (-1)

    def __neg__(self):
        return self * (-1)


def scalar_grad_check(f, inputs, eps=1e-5):
    """数値微分と自動微分の勾配を比較する。

    中心差分法で数値勾配を計算し、自動微分の結果と比較します。

    Args:
        f: 検証したい演算を行う関数。Value のリストを受け取り Value を返す
        inputs: float のリスト（各入力の値）
        eps: 数値微分の摂動量

    Returns:
        True: 全ての勾配が一致（相対誤差 < 1e-5）
    """
    # --- ステップ1: 自動微分で勾配を計算 ---
    values = [Value(x) for x in inputs]
    out = f(values)
    out.backward()
    auto_grads = [v.grad for v in values]

    # --- ステップ2: 数値微分で勾配を計算（中心差分法） ---
    num_grads = []
    for i in range(len(inputs)):
        # +eps
        inputs_plus = list(inputs)
        inputs_plus[i] += eps
        values_plus = [Value(x) for x in inputs_plus]
        out_plus = f(values_plus)

        # -eps
        inputs_minus = list(inputs)
        inputs_minus[i] -= eps
        values_minus = [Value(x) for x in inputs_minus]
        out_minus = f(values_minus)

        # 中心差分: (f(x+eps) - f(x-eps)) / (2*eps)
        num_grad = (out_plus.data - out_minus.data) / (2 * eps)
        num_grads.append(num_grad)

    # --- ステップ3: 比較 ---
    all_ok = True
    for i, (ag, ng) in enumerate(zip(auto_grads, num_grads)):
        abs_err = abs(ag - ng)
        denom = max(abs(ag), abs(ng), 1e-8)
        rel_err = abs_err / denom
        if abs_err > 1e-5 and rel_err > 1e-5:
            print(f"  入力[{i}]: 自動微分={ag:.6f}, 数値微分={ng:.6f}, "
                  f"相対誤差={rel_err:.2e} [FAIL]")
            all_ok = False

    return all_ok
