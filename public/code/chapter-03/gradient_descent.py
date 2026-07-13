"""
第3章まとめ: 勾配降下法で f(x)=(x-3)^2 を最小化し、軌跡をプロットする。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from autograd import Value

# matplotlibで日本語を表示するにはフォントの指定が必要です。
# 利用可能なフォントを順に試し、見つかったものを使います。
for font in ["Hiragino Sans", "Noto Sans CJK JP", "IPAexGothic"]:
    try:
        matplotlib.font_manager.findfont(font, fallback_to_default=False)
        plt.rcParams["font.family"] = font
        break
    except ValueError:
        continue

# --- 勾配降下法 ---
lr = 0.1
x_val = 0.0
trajectory = [(x_val, (x_val - 3) ** 2)]

for step in range(30):
    x = Value(x_val)
    loss = (x - 3) ** 2
    loss.backward()
    x_val -= lr * x.grad
    trajectory.append((x_val, (x_val - 3) ** 2))

# --- プロット ---
xs = np.linspace(-1, 5, 200)
ys = (xs - 3) ** 2

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.plot(xs, ys, color="#4C72B0", linewidth=1.5, label="$f(x) = (x-3)^2$")

# 軌跡をプロット
traj_x = [t[0] for t in trajectory]
traj_y = [t[1] for t in trajectory]
ax.plot(traj_x, traj_y, "o-", color="#DD8452", markersize=4, linewidth=1.0,
        label="勾配降下法")
ax.plot(traj_x[0], traj_y[0], "o", color="#C44E52", markersize=8, label="初期点")
ax.plot(traj_x[-1], traj_y[-1], "*", color="#55A868", markersize=12, label="最小点")

ax.set_xlabel("x", fontsize=11)
ax.set_ylabel("f(x)", fontsize=11)
ax.set_title("勾配降下法による最小値探索: $f(x) = (x-3)^2$", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gradient_descent.png")
print("gradient_descent.png をカレントディレクトリに保存しました")
