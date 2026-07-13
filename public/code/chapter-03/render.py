"""
レンダラー v2: 加重和 + アルファ合成によるガウシアン描画。
第2章: アルファ合成
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
    ys, xs = np.mgrid[0:H, 0:W]
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1)

    # --- 加重和を累積 ---
    image = np.zeros((H * W, 3), dtype=np.float64)

    for g in gaussians:
        cov_inv = np.linalg.inv(g.covariance)
        gauss_val = evaluate_gaussian(pixels, g.mean, cov_inv)
        alpha = g.opacity * gauss_val
        image += alpha[:, np.newaxis] * g.color

    image = np.clip(image, 0.0, 1.0)
    image = image.reshape(H, W, 3)

    return image


def render_gaussians_alpha_composite(gaussians, H, W, bg_color=(0, 0, 0)):
    """[v2] 深度順ソート + front-to-back アルファ合成。

    ガウシアンを深度（depth属性）の昇順にソートし、手前から奥に向かって
    アルファ合成します。手前のガウシアンが奥のガウシアンを遮蔽する
    自然な前後関係を表現できます。

    合成式:
        C += c_i * alpha_i * T
        T *= (1 - alpha_i)

    ここで alpha_i = opacity_i * G_i(x) です。
    T（累積透過率）が 1e-4 未満になったピクセルは早期に打ち切ります。

    Args:
        gaussians: Gaussian2D オブジェクトのリスト（各要素にdepth属性が必要）
        H: 画像の高さ（ピクセル）
        W: 画像の幅（ピクセル）
        bg_color: 背景色 (R, G, B)。値域 [0, 1]

    Returns:
        (H, W, 3) のRGB画像（値域 [0, 1]）
    """
    # --- ピクセル座標グリッドを生成 ---
    ys, xs = np.mgrid[0:H, 0:W]
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1)  # (H*W, 2)

    # --- 深度順にソート（小さい = 手前） ---
    sorted_gaussians = sorted(gaussians, key=lambda g: g.depth)

    # --- front-to-back アルファ合成 ---
    image = np.zeros((H * W, 3), dtype=np.float64)       # 蓄積される色
    transmittance = np.ones(H * W, dtype=np.float64)      # 累積透過率 T

    for g in sorted_gaussians:
        # T が十分小さいピクセルは全て処理済みなら打ち切り
        if np.max(transmittance) < 1e-4:
            break

        cov_inv = np.linalg.inv(g.covariance)
        gauss_val = evaluate_gaussian(pixels, g.mean, cov_inv)  # (H*W,)

        # alpha_i = opacity * ガウシアン値
        alpha = g.opacity * gauss_val  # (H*W,)

        # C += c_i * alpha_i * T
        image += (alpha * transmittance)[:, np.newaxis] * g.color

        # T *= (1 - alpha_i)
        transmittance *= (1.0 - alpha)

    # 残った透過率で背景色を加算
    bg = np.array(bg_color, dtype=np.float64)
    image += transmittance[:, np.newaxis] * bg

    image = np.clip(image, 0.0, 1.0)
    image = image.reshape(H, W, 3)

    return image
