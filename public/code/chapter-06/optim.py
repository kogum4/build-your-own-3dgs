"""
オプティマイザ: SGD と Adam。
第5章: 2Dガウシアン画像フィッティング
"""

import numpy as np


class SGD:
    """確率的勾配降下法（SGD）。

    最も基本的な最適化手法。パラメータを勾配の逆方向に更新します。

        param.data -= lr * param.grad

    Args:
        params: 最適化対象の Tensor のリスト
        lr: 学習率（デフォルト 0.01）
    """

    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        """パラメータを1ステップ更新する。"""
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        """全パラメータの勾配をゼロクリアする。"""
        for p in self.params:
            p.zero_grad()


class Adam:
    """Adam オプティマイザ。

    勾配の1次モーメント（移動平均）と2次モーメント（移動二乗平均）を
    追跡し、パラメータごとに適応的な学習率で更新します。

    リスト方式: パラメータの増減を想定しないシンプルな設計。
    各パラメータに対応するモーメントをリストで保持します。

    Args:
        params: 最適化対象の Tensor のリスト
        lr: 学習率（デフォルト 0.001）
        betas: モーメントの減衰率 (beta1, beta2)（デフォルト (0.9, 0.999)）
        eps: ゼロ除算防止の微小値（デフォルト 1e-8）
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0  # タイムステップ（全パラメータ共通）
        # 1次モーメント（勾配の移動平均）
        self.m = [np.zeros_like(p.data) for p in params]
        # 2次モーメント（勾配の二乗の移動平均）
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        """パラメータを1ステップ更新する。"""
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            # 1次モーメント更新: m = beta1 * m + (1 - beta1) * g
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

            # 2次モーメント更新: v = beta2 * v + (1 - beta2) * g^2
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

            # バイアス補正: 初期のゼロバイアスを修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # パラメータ更新
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """全パラメータの勾配をゼロクリアする。"""
        for p in self.params:
            p.zero_grad()
