"""
3Dガウシアン表現。
第6章: 3Dガウシアン表現

スケールベクトルとクォータニオンから共分散行列を安全に構築する。
sigmoid/exp活性化で制約付きパラメータを扱う。
"""

import numpy as np
from autograd import Tensor, stack


def quaternion_to_rotation_matrix(q):
    """クォータニオンから3x3回転行列を構築する。

    入力はnormalize済みの単位クォータニオン (w, x, y, z) を想定。
    呼び出し元（build_covariance_3d）でnormalize()を適用してから渡す。

    Args:
        q: (4,) のTensor。クォータニオン [w, x, y, z]

    Returns:
        (3, 3) の回転行列Tensor
    """
    # 成分を取り出す
    w = q[0]  # スカラー部
    x = q[1]
    y = q[2]
    z = q[3]

    # 回転行列の9要素を計算
    # 第1行
    r00 = 1.0 - (y * y + z * z) * 2.0
    r01 = (x * y - w * z) * 2.0
    r02 = (x * z + w * y) * 2.0

    # 第2行
    r10 = (x * y + w * z) * 2.0
    r11 = 1.0 - (x * x + z * z) * 2.0
    r12 = (y * z - w * x) * 2.0

    # 第3行
    r20 = (x * z - w * y) * 2.0
    r21 = (y * z + w * x) * 2.0
    r22 = 1.0 - (x * x + y * y) * 2.0

    # stack で (3, 3) の行列を組み立てる
    row0 = stack([r00, r01, r02], axis=0)  # (3,)
    row1 = stack([r10, r11, r12], axis=0)  # (3,)
    row2 = stack([r20, r21, r22], axis=0)  # (3,)
    R = stack([row0, row1, row2], axis=0)   # (3, 3)

    return R


def build_covariance_3d(scale_raw, quaternion_raw):
    """スケールとクォータニオンから3D共分散行列を構築する。

    共分散行列 Sigma = R @ S @ S^T @ R^T
    ここで S = diag(exp(scale_raw))、R はクォータニオンから構築した回転行列。

    exp活性化によりスケールは常に正、normalize()により
    クォータニオンは常に単位クォータニオンとなり、
    共分散行列は常に正定値が保証される。

    Args:
        scale_raw: (3,) のTensor。exp前のスケールパラメータ
        quaternion_raw: (4,) のTensor。正規化前のクォータニオン

    Returns:
        (3, 3) の共分散行列Tensor（正定値保証）
    """
    # スケール: exp で正値を保証
    scale = scale_raw.exp()  # (3,)

    # 対角行列 S を構築（np.diagは勾配が流れないためstackで組み立てる）
    s0, s1, s2 = scale[0], scale[1], scale[2]
    zero = Tensor(np.array(0.0))
    S_row0 = stack([s0, zero, zero], axis=0)
    S_row1 = stack([zero, s1, zero], axis=0)
    S_row2 = stack([zero, zero, s2], axis=0)
    S = stack([S_row0, S_row1, S_row2], axis=0)  # (3, 3)

    # クォータニオン正規化 → 回転行列
    q = quaternion_raw.normalize()  # 単位クォータニオン
    R = quaternion_to_rotation_matrix(q)    # (3, 3)

    # 共分散行列: Sigma = R @ S @ S^T @ R^T
    # S は対角行列なので S^T = S、よって R @ S @ S^T @ R^T = (RS)(RS)^T
    RS = R @ S            # (3, 3)
    RS_T = RS.transpose(1, 0)  # (RS)^T: (3, 3)
    cov = RS @ RS_T            # (3, 3)

    return cov


class Gaussian3D:
    """3Dガウシアンを表現するクラス。

    最適化に適した生パラメータ（requires_grad=True）と、
    活性化関数を通した制約付きパラメータを持つ。

    生パラメータ（最適化対象）:
        position: (3,) 3D位置座標
        scale_raw: (3,) exp前のスケール。exp(scale_raw) が実際のスケール
        quaternion_raw: (4,) 正規化前のクォータニオン
        opacity_raw: () sigmoid前の不透明度。sigmoid(opacity_raw) が実際の不透明度
        color: (3,) RGB色 [0, 1]

    Attributes:
        params: 全ての学習パラメータのリスト
    """

    def __init__(self, position, scale_raw, quaternion_raw,
                 opacity_raw, color):
        """
        Args:
            position: (3,) の配列またはリスト
            scale_raw: (3,) の配列またはリスト
            quaternion_raw: (4,) の配列またはリスト
            opacity_raw: スカラー値
            color: (3,) の配列またはリスト
        """
        self.position = Tensor(position, requires_grad=True)
        self.scale_raw = Tensor(scale_raw, requires_grad=True)
        self.quaternion_raw = Tensor(quaternion_raw, requires_grad=True)
        self.opacity_raw = Tensor(opacity_raw, requires_grad=True)
        # 色は第10章で球面調和関数に拡張するため、この段階では生パラメータのまま扱う
        self.color = Tensor(color, requires_grad=True)

        # 全パラメータをリストにまとめる
        self.params = [
            self.position,
            self.scale_raw,
            self.quaternion_raw,
            self.opacity_raw,
            self.color,
        ]

    def get_covariance(self):
        """共分散行列を計算する。正定値が保証される。"""
        return build_covariance_3d(self.scale_raw, self.quaternion_raw)

    def get_opacity(self):
        """不透明度を [0, 1] の範囲で返す。"""
        return self.opacity_raw.sigmoid()

    def get_scale(self):
        """正のスケール値を返す。"""
        return self.scale_raw.exp()
