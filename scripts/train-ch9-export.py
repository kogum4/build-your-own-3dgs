"""第9章の garden 学習をローカルで実行し、ch9-garden ウィジェット用アセットを出力する。

実行: uv run --with numpy --with pillow python scripts/train-ch9-export.py
出力:
  public/data/ch9/cameras.json        train/test カメラパラメータ
  public/data/ch9/test-views/*.png    test 視点の正解画像 (108x70)
  public/data/ch9/trained-params.json 学習済みガウシアン (150イテレーションごとに上書き)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "public" / "code" / "chapter-09"))

from data_loader import ColmapDataset  # noqa: E402
from initialization import initialize_gaussians  # noqa: E402
from trainer import GaussianTrainer  # noqa: E402
from optim import Adam  # noqa: E402
from render import render_3d  # noqa: E402
from PIL import Image  # noqa: E402

DATA = "F:/Obsidian/PersonalMemo/build-own-3dgs-workspace/output/data/colmap_garden"
OUT = REPO / "public" / "data" / "ch9"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "test-views").mkdir(exist_ok=True)

N_ITERS = 3000
EXPORT_EVERY = 150

np.random.seed(42)

print("データセットを読み込み中...", flush=True)
dataset = ColmapDataset(DATA, resize=(108, 70))

# train_garden.py と同じ分割
n_total = len(dataset)
stride = n_total // 50
train_idx = [i * stride for i in range(50)]
remaining = [i for i in range(n_total) if i not in set(train_idx)]
test_stride = len(remaining) // 10
test_idx = [remaining[i * test_stride] for i in range(10)]

train_cams = [dataset.cameras[i] for i in train_idx]
train_imgs = [dataset.images[i] for i in train_idx]
test_cams = [dataset.cameras[i] for i in test_idx]
test_imgs = [dataset.images[i] for i in test_idx]


def cam_dict(cam):
    return {
        "W": np.asarray(cam.W).tolist(),
        "t": np.asarray(cam.t).tolist(),
        "fx": float(cam.fx),
        "fy": float(cam.fy),
        "cx": float(cam.cx),
        "cy": float(cam.cy),
        "width": int(cam.width),
        "height": int(cam.height),
    }


# --- カメラと正解画像のエクスポート (学習前に完了させる) ---
cameras_json = {
    "train": [cam_dict(c) for c in train_cams],
    "test": [cam_dict(c) for c in test_cams],
}
(OUT / "cameras.json").write_text(json.dumps(cameras_json), encoding="utf-8")
for i, img in enumerate(test_imgs):
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
        OUT / "test-views" / f"test-{i:02d}.png"
    )
print(f"カメラ {len(train_cams)}+{len(test_cams)} 視点、正解画像 {len(test_imgs)} 枚を出力", flush=True)

# --- 学習 ---
print("ガウシアンを初期化中...", flush=True)
gaussians = initialize_gaussians(dataset.points3D, dataset.point_colors, n_gaussians=200)

params = []
for g in gaussians:
    params.extend(g.params)
optimizer = Adam(params, lr=0.005)
trainer = GaussianTrainer(
    gaussians=gaussians,
    cameras=train_cams,
    targets=train_imgs,
    optimizer=optimizer,
    bg_color=(0, 0, 0),
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def export_params(iteration, losses):
    data = {
        "iteration": iteration,
        "losses": [float(x) for x in losses],
        "positions": [g.position.data.tolist() for g in gaussians],
        "scales": [np.exp(g.scale_raw.data).tolist() for g in gaussians],
        "quaternions": [
            (g.quaternion_raw.data / np.linalg.norm(g.quaternion_raw.data)).tolist()
            for g in gaussians
        ],
        "opacities": [float(sigmoid(g.opacity_raw.data)) for g in gaussians],
        "colors": [np.clip(g.color.data, 0, 1).tolist() for g in gaussians],
    }
    tmp = OUT / "trained-params.json.tmp"
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(OUT / "trained-params.json")


def mean_psnr(cams, imgs):
    psnrs = []
    for cam, tgt in zip(cams, imgs):
        rendered = render_3d(gaussians, cam, bg_color=(0, 0, 0)).data
        mse = np.mean((rendered - tgt) ** 2)
        psnrs.append(-10 * np.log10(mse) if mse > 0 else float("inf"))
    return float(np.mean(psnrs))


print("学習開始...", flush=True)
losses = []
t0 = time.time()
for it in range(N_ITERS):
    cam_idx = it % len(train_cams)
    loss = trainer.train_step(cam_idx)
    losses.append(float(loss))
    if (it + 1) % EXPORT_EVERY == 0 or it == N_ITERS - 1:
        export_params(it + 1, losses)
        elapsed = time.time() - t0
        print(
            f"iter {it + 1:5d}/{N_ITERS}  loss={loss:.5f}  "
            f"{elapsed / (it + 1):.2f}s/iter  経過{elapsed / 60:.1f}分",
            flush=True,
        )

print("評価中...", flush=True)
print(f"train PSNR: {mean_psnr(train_cams, train_imgs):.2f} dB", flush=True)
print(f"test  PSNR: {mean_psnr(test_cams, test_imgs):.2f} dB", flush=True)
print("完了", flush=True)
