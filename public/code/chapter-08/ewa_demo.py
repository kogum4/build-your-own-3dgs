import os
import numpy as np
import math
import matplotlib.pyplot as plt
from gaussian3d import Gaussian3D
from camera import Camera
from render import render_3d

# 楕円: 長軸 0.3 / 短軸 0.15 (アスペクト比 2:1)
s_long = math.log(0.3)
s_short = math.log(0.15)

# 赤・青の傾き: Z軸まわり ±25° (画像内で斜めに見える)
half = math.radians(12.5)
q_tilt_pos = [math.cos(half), 0.0, 0.0, math.sin(half)]
q_tilt_neg = [math.cos(half), 0.0, 0.0, -math.sin(half)]
q_id = [1.0, 0.0, 0.0, 0.0]

gaussians = [
    # 赤: 斜め(右下がり)に伸びた楕円、後方左
    Gaussian3D(
        position=[-0.6, 0.0, 0.4],
        scale_raw=[s_long, s_short, s_short],
        quaternion_raw=q_tilt_pos,
        opacity_raw=4.0,
        color=[1.0, 0.0, 0.0],
    ),
    # 緑: 横に少し長い楕円、前方中央
    Gaussian3D(
        position=[0.0, 0.0, -0.4],
        scale_raw=[s_long, s_short, s_short],
        quaternion_raw=q_id,
        opacity_raw=4.0,
        color=[0.0, 1.0, 0.0],
    ),
    # 青: 斜め(右上がり)に伸びた楕円、後方右
    Gaussian3D(
        position=[0.6, 0.0, 0.4],
        scale_raw=[s_long, s_short, s_short],
        quaternion_raw=q_tilt_neg,
        opacity_raw=4.0,
        color=[0.0, 0.0, 1.0],
    ),
]

# 10視点をレンダリングして 2行5列で並べて保存
theta_list = list(range(0, 91, 10))
fig, axes = plt.subplots(2, 5, figsize=(13.5, 6.2))
for ax, theta_deg in zip(axes.flatten(), theta_list):
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    W = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    cam = Camera(
        W=W, t=np.array([0.0, 0.0, 5.0]),
        fx=200.0, fy=200.0, cx=64.0, cy=64.0,
        width=128, height=128,
    )
    image = render_3d(gaussians, cam, bg_color=(1, 1, 1))
    ax.imshow(image.data)
    ax.set_title(rf'$\theta = {theta_deg}°$', fontsize=16)
    ax.axis('off')
plt.tight_layout()
plt.subplots_adjust(hspace=0.35)

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, 'figures', 'fig-08-06-viewpoint-rendered.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'{os.path.basename(save_path)} を保存しました')
