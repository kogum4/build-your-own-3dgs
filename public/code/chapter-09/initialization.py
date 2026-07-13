"""
点群からガウシアンを初期化する。
第9章: 実写データからの再構成

COLMAPの3D点群から3Dガウシアンの初期パラメータを設定する。
位置=点座標、色=RGBを[0,1]に正規化した固定値、
スケール=K近傍点の平均距離、クォータニオン=単位。
"""

import numpy as np
from gaussian3d import Gaussian3D


def compute_knn_distances(positions, k=3):
    """K近傍点の平均距離を計算する。

    各点について、最も近いK個の点との距離の平均を返す。
    この距離はガウシアンの初期スケールの目安になる。
    点が密集している場所は小さなガウシアン、疎な場所は大きなガウシアンとなり、
    シーンの密度に適応した初期化ができる。

    scipy等は使わず、NumPyの距離行列で素朴に計算する。

    Args:
        positions: (N, 3) の点群座標
        k: 近傍点数（デフォルト 3）

    Returns:
        (N,) の平均距離配列
    """
    N = positions.shape[0]

    # 距離行列を計算: (N, N)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a @ b^T
    sq_norms = np.sum(positions ** 2, axis=1)  # (N,)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * positions @ positions.T
    # 数値誤差で負になることを防ぐ
    dist_sq = np.maximum(dist_sq, 0.0)
    distances = np.sqrt(dist_sq)  # (N, N)

    # 自分自身との距離を大きな値に設定（除外するため）
    np.fill_diagonal(distances, np.inf)

    # 各点について最近傍K個のインデックスを取得
    # kがN-1より大きい場合に対応
    actual_k = min(k, N - 1)
    # argsortで全ソート → 先頭K個を取得（素朴だが確実）
    sorted_indices = np.argsort(distances, axis=1)  # (N, N)
    knn_indices = sorted_indices[:, :actual_k]  # (N, k)

    # K近傍の距離の平均
    # 各点 i について knn_indices[i] が指す K 個の位置から距離を取り出し、平均する
    mean_dists = np.zeros(N)
    for i in range(N):
        mean_dists[i] = np.mean(distances[i, knn_indices[i]])

    return mean_dists


def initialize_gaussians(positions, colors, n_gaussians=None, k_nearest=3):
    """点群から3Dガウシアンを初期化する。

    各3D点を1つのガウシアンに変換する。

    初期化の方針:
    - 位置: 点座標をそのまま使用
    - スケール: K近傍点の平均距離を使い、log(平均距離)をscale_rawに設定
      （exp(scale_raw) が実際のスケールになるため）
    - 回転: 単位クォータニオン [1, 0, 0, 0]（回転なし）
    - 不透明度: sigmoid_inv(0.1) ≈ -2.197（ほぼ透明な状態から開始）
    - 色: RGBを [0, 1] に正規化した固定色。視点依存色は第10章のSHで導入する

    Args:
        positions: (M, 3) の点群座標
        colors: (M, 3) のRGB色（0-255の整数）
        n_gaussians: 使用する点の数。Noneなら全点を使用
        k_nearest: KNN近傍数（デフォルト 3）

    Returns:
        Gaussian3D オブジェクトのリスト
    """
    M = positions.shape[0]

    # 点数制限がある場合、ランダムにサンプリング
    if n_gaussians is not None and n_gaussians < M:
        indices = np.random.choice(M, n_gaussians, replace=False)
        positions = positions[indices]
        colors = colors[indices]
        M = n_gaussians

    # K近傍平均距離を計算
    mean_dists = compute_knn_distances(positions, k=k_nearest)

    # 距離が0の場合（重複点）に備えて下限を設定
    mean_dists = np.maximum(mean_dists, 1e-6)

    # スケールの初期値: log(平均距離)
    scale_raw = np.log(mean_dists)  # (M,)

    # 色の初期値: RGB [0, 255] を [0, 1] に正規化した固定色
    colors_normalized = colors.astype(np.float64) / 255.0  # (M, 3)

    # 不透明度の初期値: sigmoid_inv(0.1)
    opacity_raw = np.log(0.1 / 0.9)  # ≈ -2.197

    # ガウシアンを作成
    gaussians = []
    for i in range(M):
        g = Gaussian3D(
            position=positions[i],
            scale_raw=np.full(3, scale_raw[i]),  # 等方的スケール
            quaternion_raw=[1.0, 0.0, 0.0, 0.0],
            opacity_raw=opacity_raw,
            color=colors_normalized[i],
        )
        gaussians.append(g)

    return gaussians
