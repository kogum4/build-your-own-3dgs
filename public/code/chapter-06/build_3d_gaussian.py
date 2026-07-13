"""第6章 章末まとめ: 3Dガウシアン表現と共分散行列の正定値性検証。"""

import numpy as np
from autograd import Tensor
from gaussian3d import Gaussian3D, build_covariance_3d

print("=== 3Dガウシアン表現 ===\n")

# --- 1. 基本的なGaussian3Dの作成 ---
g = Gaussian3D(
    position=[1.0, 2.0, 3.0],
    scale_raw=[0.0, 0.0, 0.0],
    quaternion_raw=[1.0, 0.0, 0.0, 0.0],
    opacity_raw=0.0,
    color=[1.0, 0.0, 0.0],
)
print("--- 基本パラメータ ---")
print(f"position: {g.position.data}")
print(f"opacity: {g.get_opacity().data:.1f}")
print(f"scale: {g.get_scale().data}")
print(f"covariance:\n{g.get_covariance().data}\n")

# --- 2. ランダムパラメータで正定値性を検証 ---
print("--- 正定値性の検証（10個のランダムガウシアン） ---")
np.random.seed(42)
all_pd = True
for i in range(10):
    scale_raw = Tensor(np.random.randn(3), requires_grad=True)
    quat_raw = Tensor(np.random.randn(4), requires_grad=True)
    cov = build_covariance_3d(scale_raw, quat_raw)
    eigvals = np.linalg.eigvalsh(cov.data)
    if np.any(eigvals < -1e-10):
        print(f"  ガウシアン {i}: 負の固有値 {eigvals} [FAIL]")
        all_pd = False

if all_pd:
    print("  全て正定値 [OK]")
