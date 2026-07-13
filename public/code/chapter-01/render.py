"""
レンダラー v1: 加重和によるガウシアン描画。
第1章: 2Dガウシアン
"""

import numpy as np
from gaussian2d import evaluate_gaussian


def render_gaussians_weighted_sum(gaussians, H, W):
    """ガウシアン群を加重和で1枚の画像に描画する。

    各ピクセルの色を次の加重和で決定します:

        色 = Σ (opacity_i * G_i(x) * color_i)

    ガウシアン値は中心から離れると急速にゼロへ近づくため、
    どのガウシアンの影響も届かないピクセルは自然に黒になります。
    複数のガウシアンが重なる場所では色が加算され、明るくなります。

    Args:
        gaussians: Gaussian2D オブジェクトのリスト
        H: 画像の高さ（ピクセル）
        W: 画像の幅（ピクセル）

    Returns:
        (H, W, 3) のRGB画像（値域 [0, 1]）
    """
    # --- ピクセル座標グリッドを生成 ---
    # mgrid は [行インデックス, 列インデックス] の配列を返す。
    # 画像座標では x=列, y=行 なので、xs と ys を分けて取得する。
    ys, xs = np.mgrid[0:H, 0:W]  # それぞれ (H, W)

    # (H*W, 2) に整形: 各行が1ピクセルの [x, y] 座標
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1)

    # --- 加重和を累積 ---
    image = np.zeros((H * W, 3), dtype=np.float64)

    for g in gaussians:
        # 共分散行列の逆行列を計算
        cov_inv = np.linalg.inv(g.covariance)  # (2, 2)

        # 全ピクセルでガウシアン値を一括計算
        gauss_val = evaluate_gaussian(pixels, g.mean, cov_inv)  # (H*W,)

        # α_i = 不透明度 × ガウシアン値
        alpha = g.opacity * gauss_val  # (H*W,)

        # 加重和を蓄積
        # alpha[:, np.newaxis] で (H*W, 1) に変換し、色 (3,) とブロードキャスト
        image += alpha[:, np.newaxis] * g.color  # (H*W, 3)

    # 値を [0, 1] にクリップして (H, W, 3) にリシェイプ
    image = np.clip(image, 0.0, 1.0)
    image = image.reshape(H, W, 3)

    return image
