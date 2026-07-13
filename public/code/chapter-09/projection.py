"""
射影モジュール: 3Dガウシアンを2Dに射影する。
第8章: EWA Splatting

3Dガウシアンの平均・共分散をカメラパラメータを使って
2D画像平面上の平均・共分散に変換する。
EWA Splatting (Sigma' = J W Sigma W^T J^T) を実装。
"""

import numpy as np
from autograd import Tensor, stack


def compute_jacobian(point_c, fx, fy):
    """透視投影のヤコビアンを計算する。

    J = [[fx/Z,    0, -fx*X/Z^2],
         [   0, fy/Z, -fy*Y/Z^2]]

    Args:
        point_c: (3,) カメラ座標の点（Tensor）
        fx: x方向の焦点距離
        fy: y方向の焦点距離

    Returns:
        (2, 3) のヤコビアン（Tensor）。位置への勾配が流れる。
    """
    X = point_c[0]
    Y = point_c[1]
    Z = point_c[2]

    # 各要素を計算
    j00 = fx / Z
    j02 = -fx * X / (Z ** 2)
    j11 = fy / Z
    j12 = -fy * Y / (Z ** 2)

    # 2x3行列を組み立てる。stackにはTensorしか渡せないので
    # 定数の 0 も Tensor にしておく
    zero = Tensor(0.0)
    row0 = stack([j00, zero, j02], axis=0)  # (3,)
    row1 = stack([zero, j11, j12], axis=0)  # (3,)
    J = stack([row0, row1], axis=0)  # (2, 3)

    return J


def project_gaussians(gaussians, camera):
    """3Dガウシアン群をカメラで撮影し、2Dパラメータに変換する。

    各ガウシアンに対して:
    1. 平均をワールド→カメラ座標に変換し、透視投影でピクセル座標を得る
    2. EWA Splatting (Sigma' = J W Sigma W^T J^T) で2D共分散を計算
    3. カメラ背面・画面外のガウシアンをカリングする

    Args:
        gaussians: Gaussian3D オブジェクトのリスト
        camera: Camera オブジェクト

    Returns:
        means2d: Tensor のリスト。各要素は (2,) のピクセル座標
        covs2d: Tensor のリスト。各要素は (2, 2) の2D共分散行列
        depths: float のリスト。深度値（ソート用）
        colors: Tensor のリスト。各要素は (3,) のRGB色
        opacities: Tensor のリスト。各要素は () のスカラー不透明度
        indices: int のリスト。カリング後の元のインデックス
    """
    means2d = []
    covs2d = []
    depths = []
    colors = []
    opacities = []
    indices = []

    # 回転行列 W をTensorに変換（勾配は流さない定数）
    W = Tensor(camera.W)  # (3, 3)

    for i, g in enumerate(gaussians):
        # --- 1. カリング判定（NumPyで高速に行う）---
        pos_np = g.position.data  # (3,)
        # ここでの @ はNumPyの行列積演算子です
        # （後のEWA計算ではTensorの matmul として使います）
        point_c = camera.W @ pos_np + camera.t  # (3,)

        # カメラ背面チェック: Z <= 0.1 のガウシアンは除外
        # Z=0だと透視投影の除算でゼロ除算になるのを防ぎ、
        # カメラに極端に近いガウシアンも除外する
        if point_c[2] <= 0.1:
            continue

        # 透視投影でピクセル座標を計算（カリング判定用）
        u = camera.fx * point_c[0] / point_c[2] + camera.cx
        v = camera.fy * point_c[1] / point_c[2] + camera.cy

        # 画面外チェック: マージンを持たせて判定
        # ガウシアンの裾は中心から離れた位置まで広がるため、
        # 中心が画面外でも裾が画面内に入る可能性がある
        margin = 100
        if (u < -margin or u > camera.width + margin or
                v < -margin or v > camera.height + margin):
            continue

        # --- 2. 2D平均をTensor演算で計算(位置への勾配を流すため)---
        pos_col = g.position.reshape(3, 1)               # (3,) → (3, 1)
        rotated = W.matmul(pos_col)                      # (3, 3) @ (3, 1) = (3, 1)
        pos_cam = rotated.reshape(3) + Tensor(camera.t)  # (3,) に戻して平行移動
        z = pos_cam[2]
        mean_u = camera.fx * pos_cam[0] / z + camera.cx
        mean_v = camera.fy * pos_cam[1] / z + camera.cy
        mean2d = stack([mean_u, mean_v], axis=0)         # (2,)

        # --- 3. 共分散の射影（EWA Splatting）---
        cov3d = g.get_covariance()  # (3, 3) Tensor

        # ヤコビアン（pos_camからTensor演算で計算し、位置への勾配を流す）
        J = compute_jacobian(pos_cam, camera.fx, camera.fy)  # (2, 3)

        # W^T と J^T
        W_T = W.transpose(1, 0)  # (3, 3)
        J_T = J.transpose(1, 0)  # (3, 2)

        # Sigma' = J @ W @ Sigma @ W^T @ J^T（4ステップのmatmul連鎖）
        t1 = J @ W          # (2, 3) @ (3, 3) = (2, 3)
        t2 = t1 @ cov3d     # (2, 3) @ (3, 3) = (2, 3)
        t3 = t2 @ W_T       # (2, 3) @ (3, 3) = (2, 3)
        t4 = t3 @ J_T       # (2, 3) @ (3, 2) = (2, 2)

        cov2d = t4  # (2, 2) Tensor

        # --- 4. 結果を蓄積 ---
        means2d.append(mean2d)
        covs2d.append(cov2d)
        depths.append(float(point_c[2]))
        colors.append(g.color)
        opacities.append(g.get_opacity())
        indices.append(i)

    return means2d, covs2d, depths, colors, opacities, indices
