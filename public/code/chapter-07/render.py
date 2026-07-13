"""
レンダラー v3: 加重和 + アルファ合成 + Tensor版アルファ合成。
第5章: 2Dガウシアン画像フィッティング
"""

import numpy as np
from gaussian2d import evaluate_gaussian
from autograd import Tensor, stack


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


def invert_2x2_tensor(mat):
    """2x2 Tensor行列の逆行列をTensor演算で計算する。

    NumPyのnp.linalg.invを使うと勾配パスが切れるため、
    Tensor演算で逆行列を構築して勾配を流す。

    2x2行列 [[a, b], [c, d]] の逆行列は:
        1/(ad - bc) * [[d, -b], [-c, a]]
    """
    a = mat[0, 0]
    b = mat[0, 1]
    c = mat[1, 0]
    d = mat[1, 1]

    det = a * d - b * c

    row0 = stack([d / det, -b / det], axis=0)
    row1 = stack([-c / det, a / det], axis=0)
    return stack([row0, row1], axis=0)


def evaluate_gaussian_tensor(pixels, mean, cov):
    """Tensor版ガウシアン評価関数。

    第1章で使った距離計算（マハラノビス距離）をTensor演算で計算して
    ガウシアン値を返します。2x2逆行列もTensor演算で構築するので、
    mean と cov の両方に勾配が自動微分で流れます。

    Args:
        pixels: (H*W, 2) ピクセル座標の np.ndarray（定数）
        mean: (2,) 中心座標の Tensor
        cov: (2, 2) 共分散行列の Tensor

    Returns:
        (H*W,) 各ピクセルでのガウシアン値の Tensor
    """
    P = pixels.shape[0]

    # 2x2 逆行列を Tensor 演算で計算し、共分散まで勾配を流す
    cov_inv = invert_2x2_tensor(cov)

    # 差分ベクトル: (H*W, 2)
    diff = Tensor(pixels) - mean.reshape(1, 2)  # ブロードキャスト

    # マハラノビス距離の二乗: diff @ cov_inv @ diff（各行ベクトルごと）
    # diff を (H*W, 2, 1) に、cov_inv を (1, 2, 2) にreshapeして
    # ブロードキャスト乗算 → axis=1 で合計すると (H*W, 2) の行列積になる
    diff_col = diff.reshape(P, 2, 1)     # (H*W, 2, 1)
    cov_inv_b = cov_inv.reshape(1, 2, 2)  # (1, 2, 2)
    transformed = (diff_col * cov_inv_b).sum(axis=1)  # (H*W, 2)
    mahal = (diff * transformed).sum(axis=1)           # (H*W,)

    return (mahal * (-0.5)).exp()


def render_gaussians_alpha_composite_tensor(means, covs, colors, opacities, depths,
                                  H, W, bg_color=(0, 0, 0)):
    """[v3] Tensor版アルファ合成レンダラー。

    全ての計算がTensor演算で行われるため、backwardで勾配を計算できます。
    形状は (N, H*W) を標準として、ガウシアン軸（axis=0）でループします。

    Args:
        means: Tensor のリスト。各要素は (2,) の中心座標
        covs: Tensor のリスト。各要素は (2, 2) の共分散行列
        colors: Tensor のリスト。各要素は (3,) のRGB色
        opacities: Tensor のリスト。各要素は () のスカラー不透明度
        depths: float のリスト。深度値（ソート用、勾配不要）
        H: 画像の高さ（ピクセル）
        W: 画像の幅（ピクセル）
        bg_color: 背景色 (R, G, B)。値域 [0, 1]

    Returns:
        (H, W, 3) のRGB画像 Tensor
    """
    # ピクセル座標グリッドを生成（定数）
    ys, xs = np.mgrid[0:H, 0:W]
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    P = H * W  # ピクセル数

    # 深度順にソート（インデックスを返す）
    order = sorted(range(len(depths)), key=lambda i: depths[i])

    # front-to-back アルファ合成（ループ版）
    image = Tensor(np.zeros((P, 3)))  # 蓄積される色
    T = Tensor(np.ones(P))            # 累積透過率

    for i in order:
        # ガウシアン値を計算: (H*W,)
        gauss_val = evaluate_gaussian_tensor(pixels, means[i], covs[i])

        # alpha_i = opacity * ガウシアン値: (H*W,)
        alpha = opacities[i] * gauss_val

        # 色の寄与: c_i * alpha_i * T
        # alpha * T の形状は (H*W,)、reshape で (H*W, 1) にして色 (3,) とブロードキャスト
        weight = (alpha * T).reshape(P, 1)  # (H*W, 1)
        image = image + weight * colors[i]  # (H*W, 3)

        # 累積透過率の更新: T *= (1 - alpha)
        T = T * (Tensor(np.ones(P)) - alpha)

    # 背景色を加算
    bg = Tensor(np.array(bg_color, dtype=np.float64))
    image = image + T.reshape(P, 1) * bg

    return image.reshape(H, W, 3)
