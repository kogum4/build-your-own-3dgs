"""
第2章まとめ: 加重和 vs アルファ合成の比較画像を生成する。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams["font.family"] = "Noto Sans JP"
from gaussian2d import Gaussian2D, build_covariance_2d
from render import render_gaussians_weighted_sum, render_gaussians_alpha_composite

H, W = 128, 128

# 3つのガウシアンを前後関係付きで配置
# 赤（手前）、緑（中間）、青（奥）
gaussians = [
    Gaussian2D(
        mean=[54, 64],
        covariance=build_covariance_2d(18.0, 18.0, 0.0),
        color=[1, 0, 0],  # 赤
        opacity=0.9,
        depth=1.0,  # 手前
    ),
    Gaussian2D(
        mean=[64, 64],
        covariance=build_covariance_2d(18.0, 18.0, 0.0),
        color=[0, 1, 0],  # 緑
        opacity=0.9,
        depth=2.0,  # 中間
    ),
    Gaussian2D(
        mean=[74, 64],
        covariance=build_covariance_2d(18.0, 18.0, 0.0),
        color=[0, 0, 1],  # 青
        opacity=0.9,
        depth=3.0,  # 奥
    ),
]

# 加重和でレンダリング
image_weighted = render_gaussians_weighted_sum(gaussians, H, W)

# アルファ合成でレンダリング
image_alpha = render_gaussians_alpha_composite(gaussians, H, W)

# 2枚を横に並べて保存
output_path = Path(__file__).with_name("figures").joinpath("fig-02-04-weighted-vs-alpha.png")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(image_weighted)
ax1.set_title("加重和")
ax1.set_xlabel("x (px)")
ax1.set_ylabel("y (px)")
ax2.imshow(image_alpha)
ax2.set_title("アルファ合成")
ax2.set_xlabel("x (px)")
fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"{output_path.name} を保存しました: {output_path}")
