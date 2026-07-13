"""第7章 章末まとめ: カメラモデルと座標変換。"""

import numpy as np
import math
from camera import Camera

print("=== カメラモデルと座標変換 ===\n")

# --- 立方体の8頂点 ---
vertices = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1],
], dtype=np.float64)

# --- 1. 正面カメラ ---
print("--- 1. 正面カメラ ---")
cam_front = Camera(
    R=np.eye(3),
    t=np.array([0.0, 0.0, 5.0]),
    fx=200.0, fy=200.0,
    cx=150.0, cy=150.0,
    width=300, height=300,
)
pc = cam_front.world_to_camera(vertices)
px, depths = cam_front.project(pc)
for i, (p, d) in enumerate(zip(px, depths)):
    print(f"  頂点{i}: ({p[0]:.1f}, {p[1]:.1f}), Z={d:.1f}")

# --- 2. 斜めカメラ（Y軸30度回転）---
print("\n--- 2. 斜めカメラ（Y軸30度回転）---")
angle = math.radians(30)
cos_a, sin_a = math.cos(angle), math.sin(angle)
R_y30 = np.array([
    [ cos_a, 0, sin_a],
    [     0, 1,     0],
    [-sin_a, 0, cos_a],
])
cam_oblique = Camera(
    R=R_y30,
    t=np.array([0.0, 0.0, 5.0]),
    fx=200.0, fy=200.0,
    cx=150.0, cy=150.0,
    width=300, height=300,
)
pc = cam_oblique.world_to_camera(vertices)
px, depths = cam_oblique.project(pc)
for i, (p, d) in enumerate(zip(px, depths)):
    print(f"  頂点{i}: ({p[0]:.1f}, {p[1]:.1f}), Z={d:.1f}")

# --- 3. 焦点距離の比較 ---
print("\n--- 3. 焦点距離の比較 ---")
front_verts = vertices[:4]
for fx_val in [80.0, 200.0, 400.0]:
    cam = Camera(
        R=np.eye(3), t=np.array([0.0, 0.0, 5.0]),
        fx=fx_val, fy=fx_val, cx=150.0, cy=150.0,
        width=300, height=300,
    )
    pc = cam.world_to_camera(front_verts)
    px, _ = cam.project(pc)
    w = px[:, 0].max() - px[:, 0].min()
    print(f"  fx={fx_val:.0f}: 投影幅 = {w:.1f} px")

# --- 4. 遠近法の確認 ---
print("\n--- 4. 遠近法の確認 ---")
cam = Camera(
    R=np.eye(3), t=np.zeros(3),
    fx=100.0, fy=100.0, cx=0.0, cy=0.0,
    width=200, height=200,
)
# 同じX座標、異なるZ
test_points = np.array([
    [1.0, 0.0, 1.0],
    [1.0, 0.0, 2.0],
    [1.0, 0.0, 4.0],
])
px, _ = cam.project(test_points)
for p, z in zip(px, [1, 2, 4]):
    print(f"  Z={z}: u = {p[0]:.1f} (X=1, fx=100 → fx*X/Z = {100*1/z:.1f})")

print("\n完了!")
