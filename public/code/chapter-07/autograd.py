"""
自動微分エンジン: ValueクラスとTensorクラス。
第7章: カメラモデルと座標変換

Valueクラス（第3章）はスカラー専用の参照実装として残す。
Tensorクラスは data を np.ndarray に拡張し、
ブロードキャスト対応の backward を実装する。
第6章で matmul, normalize, transpose, getitem を追加。
"""

import math
import sys

import numpy as np

# 自動微分の計算グラフが深くなると再帰的トポロジカルソートが
# Pythonのデフォルト再帰上限（1000）に達するため、上限を引き上げる
sys.setrecursionlimit(10000)


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

    # --- 四則演算 ---

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

    def __sub__(self, other):
        # a - b = a + (-b)
        return self + (-other)

    def __rsub__(self, other):
        # other - self = (-self) + other
        return (-self) + other

    def __truediv__(self, other):
        # a / b = a * b^(-1)
        return self * other ** (-1)

    def __rtruediv__(self, other):
        # other / self = other * self^(-1)
        return other * self ** (-1)

    def __neg__(self):
        return self * (-1)

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


def scalar_grad_check(f, inputs, eps=1e-5):
    """数値微分と自動微分の勾配を比較する。

    中心差分法で数値勾配を計算し、自動微分の結果と比較します。

    Args:
        f: Value のリストを受け取り、Value を返す関数
        inputs: float のリスト（各入力の値）
        eps: 数値微分の摂動量

    Returns:
        True: 全ての勾配が一致（相対誤差 < 1e-5）
    """
    # 自動微分で勾配を計算
    values = [Value(x) for x in inputs]
    out = f(values)
    out.backward()
    auto_grads = [v.grad for v in values]

    # 数値微分で勾配を計算（中心差分法）
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

    # 比較
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


# ===================================================================
# Tensor クラス（第4章）
# ===================================================================

def _unbroadcast(grad, shape):
    """ブロードキャストされた勾配を元の形状に集約する。

    forward でブロードキャストにより形状が広がった場合、
    backward では広がった軸方向に sum して元の形状に戻す。

    Args:
        grad: 逆伝播で流れてきた勾配（np.ndarray）
        shape: 元のテンソルの形状（tuple）

    Returns:
        元の形状に集約された勾配（np.ndarray）
    """
    # 次元数の差分だけ先頭に軸が追加されたケースに対応
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # サイズ1の軸（ブロードキャストで広がった軸）を集約
    for i, s in enumerate(shape):
        if s == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class Tensor:
    """計算グラフのノード。N次元配列と勾配を保持する。

    Valueクラスのテンソル版。構造（_backward、_prev、backward()）は
    Valueと同一で、data と grad が np.ndarray に変わっただけ。

    Attributes:
        data: N次元配列（np.ndarray）
        grad: 逆伝播で計算された勾配（np.ndarray、同形状）
        requires_grad: 勾配計算が必要かどうか
        _backward: 勾配を伝播させるクロージャ
        _prev: このノードの入力ノードの集合
    """

    def __init__(self, data, requires_grad=False):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def zero_grad(self):
        """勾配をゼロクリアする。"""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)

    # --- 要素単位演算 ---

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(out.grad, other.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += _unbroadcast(out.grad, self.shape)
            other.grad += _unbroadcast(-out.grad, other.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += _unbroadcast(out.grad / other.data, self.shape)
            other.grad += _unbroadcast(
                -out.grad * self.data / (other.data ** 2), other.shape
            )

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other.__truediv__(self)

    def __neg__(self):
        out = Tensor(-self.data)
        out._prev = {self}

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __pow__(self, n):
        assert isinstance(n, (int, float)), "べき指数はint/floatのみ"
        out = Tensor(self.data ** n)
        out._prev = {self}

        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad

        out._backward = _backward
        return out

    # --- 数学関数 ---

    def exp(self):
        out = Tensor(np.exp(self.data))
        out._prev = {self}

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data))
        out._prev = {self}

        def _backward():
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def sigmoid(self):
        sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(sig)
        out._prev = {self}

        def _backward():
            self.grad += out.grad * sig * (1.0 - sig)

        out._backward = _backward
        return out

    def abs(self):
        out = Tensor(np.abs(self.data))
        out._prev = {self}

        def _backward():
            self.grad += out.grad * np.sign(self.data)

        out._backward = _backward
        return out

    # --- 集約演算 ---

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))
        out._prev = {self}

        def _backward():
            g = out.grad
            # keepdims=False の場合、集約した軸を復元してブロードキャスト
            if axis is not None and not keepdims:
                if isinstance(axis, int):
                    g = np.expand_dims(g, axis=axis)
                else:
                    for ax in sorted(axis):
                        g = np.expand_dims(g, axis=ax)
            # ブロードキャストで元の形状に広げる
            self.grad += np.broadcast_to(g, self.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))
        out._prev = {self}

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                if isinstance(axis, int):
                    g = np.expand_dims(g, axis=axis)
                else:
                    for ax in sorted(axis):
                        g = np.expand_dims(g, axis=ax)
            # 平均の勾配: 1/N をかけてブロードキャスト
            if axis is None:
                n = self.data.size
            elif isinstance(axis, int):
                n = self.data.shape[axis]
            else:
                n = 1
                for ax in axis:
                    n *= self.data.shape[ax]
            self.grad += np.broadcast_to(g, self.shape) / n

        out._backward = _backward
        return out

    # --- 形状操作 ---

    def reshape(self, *shape):
        # reshape(2, 3) と reshape((2, 3)) の両方に対応
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape))
        out._prev = {self}

        def _backward():
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    # --- 行列演算 ---

    def matmul(self, other):
        """行列積 self @ other を計算する。

        バッチ次元にも対応。最後の2次元で行列積を行い、
        先頭の次元はブロードキャストされる。

        backward:
            dL/d(self) = dL/d(out) @ other^T
            dL/d(other) = self^T @ dL/d(out)
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data)
        out._prev = {self, other}

        def _backward():
            # dL/dA = grad @ B^T
            g_self = out.grad @ np.swapaxes(other.data, -2, -1)
            self.grad += _unbroadcast(g_self, self.shape)

            # dL/dB = A^T @ grad
            g_other = np.swapaxes(self.data, -2, -1) @ out.grad
            other.grad += _unbroadcast(g_other, other.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def normalize(self, axis=-1):
        """L2正規化: x / ||x|| を計算する。

        backward: 勾配から半径方向成分を除去し、接線方向だけ残す。
            grad_x = (grad_out - x_hat * dot(x_hat, grad_out)) / ||x||
        """
        # ノルムを計算（keepdimsで形状を維持）
        norm = np.sqrt(np.sum(self.data ** 2, axis=axis, keepdims=True))
        norm = np.maximum(norm, 1e-12)  # ゼロ除算防止
        x_hat = self.data / norm
        out = Tensor(x_hat)
        out._prev = {self}

        def _backward():
            # 半径方向成分: x_hat * dot(x_hat, grad_out)
            dot = np.sum(x_hat * out.grad, axis=axis, keepdims=True)
            # 接線方向のみ残す
            self.grad += (out.grad - x_hat * dot) / norm

        out._backward = _backward
        return out

    def transpose(self, *axes):
        """軸を入れ替える。np.transpose と同じインターフェース。"""
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        out = Tensor(np.transpose(self.data, axes))
        out._prev = {self}

        # 逆順の軸を計算
        inv_axes = [0] * len(axes)
        for i, a in enumerate(axes):
            inv_axes[a] = i

        def _backward():
            self.grad += np.transpose(out.grad, inv_axes)

        out._backward = _backward
        return out

    # --- インデクシング ---

    def __getitem__(self, idx):
        """インデクシング演算（Ellipsis + 整数）。

        クォータニオンの成分抽出 q[..., 0] などに使用。
        backwardの統一パターン:
            grad_input = np.zeros(orig_shape)
            np.add.at(grad_input, idx, grad_output)
        """
        out = Tensor(self.data[idx])
        out._prev = {self}

        def _backward():
            g = np.zeros_like(self.data)
            np.add.at(g, idx, out.grad)
            self.grad += g

        out._backward = _backward
        return out

    # --- 逆伝播 ---

    def backward(self):
        """トポロジカルソートで計算グラフを逆順走査し、勾配を伝播する。"""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # 全ノードの勾配をゼロ初期化（中間ノードも含む）
        for v in topo:
            v.grad = np.zeros_like(v.data)

        # 出力ノードの勾配を1に設定
        self.grad = np.ones_like(self.data)

        # 逆順に_backwardを呼び出す
        for v in reversed(topo):
            v._backward()


def clear_graph(tensor):
    """backward後に計算グラフの参照を解放してメモリリークを防ぐ。

    Args:
        tensor: backward() を呼んだ後の出力Tensor（通常は損失）
    """
    visited = set()

    def _clear(v):
        if v not in visited:
            visited.add(v)
            for p in v._prev:
                _clear(p)
            v._backward = lambda: None
            v._prev = set()

    _clear(tensor)

def stack(tensors, axis=0):
    """テンソルのリストを指定軸で結合する。

    Args:
        tensors: Tensor のリスト（全て同じ形状）
        axis: 結合する軸

    Returns:
        結合されたTensor
    """
    data = np.stack([t.data for t in tensors], axis=axis)
    out = Tensor(data)
    out._prev = set(tensors)

    def _backward():
        grads = np.split(out.grad, len(tensors), axis=axis)
        for t, g in zip(tensors, grads):
            t.grad += g.squeeze(axis=axis) if g.shape[axis] == 1 else g

    out._backward = _backward
    return out


def unbind(tensor, axis=0):
    """テンソルを指定軸でスライスし、リストとして返す。

    Args:
        tensor: 分割するTensor
        axis: 分割する軸

    Returns:
        Tensor のリスト
    """
    slices = []
    n = tensor.data.shape[axis]
    for i in range(n):
        # 指定軸の i 番目を取り出す（その軸は消える）
        idx = [slice(None)] * tensor.ndim
        idx[axis] = i
        s = Tensor(tensor.data[tuple(idx)])
        s._prev = {tensor}
        slices.append(s)

    def make_backward(slice_tensor, index):
        def _backward():
            idx_put = [slice(None)] * tensor.ndim
            idx_put[axis] = index
            tensor.grad[tuple(idx_put)] += slice_tensor.grad
        return _backward

    for i, s in enumerate(slices):
        s._backward = make_backward(s, i)

    return slices


def grad_check(f, inputs, eps=1e-5):
    """テンソル対応の数値微分チェック。

    各入力テンソルの各要素を微小量ずらして数値微分を計算し、
    backwardの結果と比較する。

    Args:
        f: 検証したい演算を行う関数。Tensor のリストを受け取り Tensor を返す
        inputs: np.ndarray のリスト（各入力の値）
        eps: 数値微分の摂動量

    Returns:
        True: 全ての勾配が一致（相対誤差 < 1e-5）
    """
    # 自動微分で勾配を計算
    tensors = [Tensor(x, requires_grad=True) for x in inputs]
    out = f(tensors)
    out.backward()
    auto_grads = [t.grad.copy() for t in tensors]

    # 数値微分で勾配を計算
    all_ok = True
    for i in range(len(inputs)):
        num_grad = np.zeros_like(inputs[i], dtype=np.float64)
        it = np.nditer(inputs[i], flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index

            # +eps
            inputs_plus = [x.copy() for x in inputs]
            inputs_plus[i][idx] += eps
            tensors_plus = [Tensor(x, requires_grad=True) for x in inputs_plus]
            out_plus = f(tensors_plus)

            # -eps
            inputs_minus = [x.copy() for x in inputs]
            inputs_minus[i][idx] -= eps
            tensors_minus = [Tensor(x, requires_grad=True) for x in inputs_minus]
            out_minus = f(tensors_minus)

            # 中心差分
            num_grad[idx] = (out_plus.data.sum() - out_minus.data.sum()) / (2 * eps)
            it.iternext()

        # 比較
        ag = auto_grads[i]
        ng = num_grad
        abs_err = np.abs(ag - ng)
        denom = np.maximum(np.abs(ag), np.abs(ng))
        denom = np.maximum(denom, 1e-8)
        rel_err = abs_err / denom

        mask = (abs_err > 1e-5) & (rel_err > 1e-5)
        if np.any(mask):
            max_rel = rel_err[mask].max()
            print(f"  入力[{i}]: 最大相対誤差={max_rel:.2e} [FAIL]")
            all_ok = False

    return all_ok
