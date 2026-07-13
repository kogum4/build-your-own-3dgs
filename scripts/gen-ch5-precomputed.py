"""ch5-live-training ウィジェット用の録画済み学習アニメーションを生成する。

実行: uv run --with numpy --with pillow python scripts/gen-ch5-precomputed.py
出力: public/data/ch5/precomputed.json
  (既定設定 N=64, 48x48, 200ステップ の学習を10ステップごとに記録)
"""

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "public" / "code" / "chapter-05"))

from autograd import Tensor, clear_graph  # noqa: E402
from gaussian2d import build_covariance_2d  # noqa: E402
from render import render_gaussians_alpha_composite_tensor  # noqa: E402
from loss import l1_loss  # noqa: E402
from optim import Adam  # noqa: E402
from PIL import Image  # noqa: E402

H = W = 48
N = 64
STEPS = 200
RECORD_EVERY = 10

target_img = (
    Image.open(REPO / "public" / "data" / "ch5" / "target.png")
    .convert("RGB")
    .resize((W, H), Image.LANCZOS)
)
target_image = np.array(target_img).astype(np.float64) / 255.0

np.random.seed(42)
means = [Tensor(np.random.rand(2) * np.array([W, H]), requires_grad=True) for _ in range(N)]
covs = [Tensor(build_covariance_2d(4, 4, 0)) for _ in range(N)]
colors = [Tensor(np.random.rand(3), requires_grad=True) for _ in range(N)]
opacities = [Tensor(np.array(0.5), requires_grad=True) for _ in range(N)]
depths = [float(i) for i in range(N)]

params = means + colors + opacities
optimizer = Adam(params, lr=0.05)
target = Tensor(target_image)

frames = []
losses = []
t0 = time.time()
for step in range(STEPS + 1):
    optimizer.zero_grad()
    pred = render_gaussians_alpha_composite_tensor(means, covs, colors, opacities, depths, H, W)
    loss = l1_loss(pred, target)
    loss.backward()
    clear_graph(loss)
    optimizer.step()
    losses.append(float(loss.data))

    if step % RECORD_EVERY == 0:
        img = np.clip(pred.data.reshape(H, W, 3), 0, 1)
        raw = (img * 255).astype(np.uint8).tobytes()
        frames.append({"step": step, "loss": float(loss.data), "rgb": base64.b64encode(raw).decode()})
        print(f"step {step:4d}  loss={float(loss.data):.5f}  {time.time() - t0:.1f}s", flush=True)

out = {
    "config": {"N": N, "H": H, "W": W, "steps": STEPS, "lr": 0.05, "seed": 42},
    "losses": losses,
    "frames": frames,
}
(REPO / "public" / "data" / "ch5" / "precomputed.json").write_text(json.dumps(out), encoding="utf-8")
print("saved: public/data/ch5/precomputed.json")
