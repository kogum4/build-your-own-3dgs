"""
第1章まとめ: 3色ガウシアンを描いてPNG保存する。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from gaussian2d import Gaussian2D, build_covariance_2d
from render import render_gaussians_weighted_sum

H, W = 128, 128

# 赤・緑・青の楕円ガウシアンを中心から放射状に配置
gaussians = [
    Gaussian2D(
        mean=[54, 44],
        covariance=build_covariance_2d(24.0, 10.0, np.pi / 3),
        color=[1, 0, 0],  # 赤
        opacity=1.0,
    ),
    Gaussian2D(
        mean=[74, 44],
        covariance=build_covariance_2d(24.0, 10.0, -np.pi / 3),
        color=[0, 1, 0],  # 緑
        opacity=1.0,
    ),
    Gaussian2D(
        mean=[64, 72],
        covariance=build_covariance_2d(24.0, 10.0, np.pi / 2),
        color=[0, 0, 1],  # 青
        opacity=1.0,
    ),
]

# レンダリング
image = render_gaussians_weighted_sum(gaussians, H, W)

# PNG保存
output_path = Path(__file__).with_name("figures").joinpath("fig-01-04-weighted-sum-result.png")
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(image)
ax.set_xlabel("x (px)")
ax.set_ylabel("y (px)")
fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"{output_path.name} を保存しました: {output_path}")
