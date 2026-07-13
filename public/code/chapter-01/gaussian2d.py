"""
2Dガウシアン関数の定義と評価。
第1章: 2Dガウシアン
"""

import numpy as np


class Gaussian2D:
    """2Dガウシアンを表現するクラス。

    1つのガウシアンは「どこに」「どんな形で」「何色で」「どれくらい濃く」
    存在するかを4つのパラメータで表現します。

    Attributes:
        mean: (2,) 中心座標 [x, y]
        covariance: (2, 2) 共分散行列（楕円の形状を決定）
        color: (3,) RGB色 [0, 1]
        opacity: 不透明度 [0, 1]
    """

    def __init__(self, mean, covariance, color=None, opacity=1.0):
        self.mean = np.array(mean, dtype=np.float64)              # (2,)
        self.covariance = np.array(covariance, dtype=np.float64)  # (2, 2)
        self.color = np.array(
            color if color is not None else [1.0, 1.0, 1.0],
            dtype=np.float64,
        )  # (3,)
        self.opacity = float(opacity)


def build_covariance_2d(sigma_x, sigma_y, theta):
    """回転角+スケールから2x2共分散行列を構築する。

    共分散行列を Sigma = R @ Lambda @ R^T で計算します。
    ここで Lambda = diag(sigma_x^2, sigma_y^2) は分散の対角行列です。
    R は角度 theta（ラジアン）の回転行列です。

    この分解の意味:
      1. diag で各軸方向の「広がり」を決め、
      2. R で楕円全体を回転させる。

    Args:
        sigma_x: x方向の標準偏差（広がりの大きさ）
        sigma_y: y方向の標準偏差（広がりの大きさ）
        theta: 回転角（ラジアン）。反時計回りが正

    Returns:
        (2, 2) の共分散行列
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 回転行列: 2Dの反時計回り回転
    R = np.array([
        [cos_t, -sin_t],
        [sin_t,  cos_t],
    ])

    # 分散の対角行列Λ: 各軸方向の広がりの二乗
    Lambda_ = np.array([
        [sigma_x ** 2, 0.0],
        [0.0, sigma_y ** 2],
    ])

    # 共分散行列: 「広がり → 回転」の順で変換を適用
    covariance = R @ Lambda_ @ R.T
    return covariance


def evaluate_gaussian(pixels, mean, cov_inv):
    """全ピクセルでガウシアン値を一括計算する。

    数式: G(x) = exp(-0.5 * (x - mu)^T Sigma^{-1} (x - mu))

    この式の意味:
      - (x - mu): 各ピクセルから中心までの差分ベクトル
      - Sigma^{-1} を挟んだ二次形式: 共分散を考慮した「距離の二乗」
        （マハラノビス距離の二乗）
      - exp(-0.5 * ...): 距離が大きいほど急速にゼロへ近づく

    Args:
        pixels: (H*W, 2) ピクセル座標の配列
        mean: (2,) ガウシアンの中心座標
        cov_inv: (2, 2) 共分散行列の逆行列

    Returns:
        (H*W,) 各ピクセルでのガウシアン値（0〜1の範囲）
    """
    # 中心からの差分ベクトル: (H*W, 2)
    diff = pixels - mean

    # マハラノビス距離の二乗を計算
    # 手順: diff @ cov_inv で (H*W, 2) を得て、
    #       diff との要素ごとの積をとり、行方向に合計する。
    # これは各行ベクトル d について d^T @ cov_inv @ d を計算するのと等価。
    mahal = np.sum(diff @ cov_inv * diff, axis=1)  # (H*W,)

    return np.exp(-0.5 * mahal)
