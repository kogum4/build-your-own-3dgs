"""gardenデータセットでL1学習を実行する。"""

import numpy as np

from data_loader import ColmapDataset
from initialization import initialize_gaussians
from trainer import GaussianTrainer
from optim import Adam


np.random.seed(42)

# --- 1. データセット読み込み ---
print("データセットを読み込み中...")
dataset = ColmapDataset("data/colmap_garden", resize=(108, 70))

# gardenシーンの全 185 視点を train 50 / test 10 に分割する。
# 等間隔に train を抜き、その合間から 10 視点を test に回す。
# test 視点は学習中に一度も見せず、最後の評価でだけ使う。
# 学習に使っていない視点で成立するかを確認するのが目的（novel view 評価）。
n_total = len(dataset)
stride = n_total // 50  # 185 // 50 = 3
train_idx = [i * stride for i in range(50)]
remaining = [i for i in range(n_total) if i not in set(train_idx)]
test_stride = len(remaining) // 10
test_idx = [remaining[i * test_stride] for i in range(10)]

train_cams = [dataset.cameras[i] for i in train_idx]
train_imgs = [dataset.images[i] for i in train_idx]
test_cams = [dataset.cameras[i] for i in test_idx]
test_imgs = [dataset.images[i] for i in test_idx]
print(f"  train: {len(train_cams)}視点, test: {len(test_cams)}視点, "
      f"点群: {dataset.points3D.shape[0]}")

# --- 2. ガウシアン初期化 ---
print("ガウシアンを初期化中...")
gaussians = initialize_gaussians(
    dataset.points3D, dataset.point_colors,
    n_gaussians=200,
)
print(f"  ガウシアン数: {len(gaussians)}")

# --- 3. オプティマイザ設定 ---
params = []
for g in gaussians:
    params.extend(g.params)
optimizer = Adam(params, lr=0.005)

# --- 4. トレーナー設定 ---
trainer = GaussianTrainer(
    gaussians=gaussians,
    cameras=train_cams,
    targets=train_imgs,
    optimizer=optimizer,
    bg_color=(0, 0, 0),
)

# --- 5. 学習実行 ---
print("学習開始...")
losses = trainer.train(n_iters=3000, log_interval=150)

# --- 6. 評価 ---
# 学習に使った train 視点と、一度も見せていない test 視点の
# 両方で PSNR を測る。test の方が大きく下がるようなら過学習。
from render import render_3d
from PIL import Image


def mean_psnr(cams, imgs):
    psnrs = []
    for cam, tgt in zip(cams, imgs):
        rendered = render_3d(gaussians, cam, bg_color=(0, 0, 0)).data
        mse = np.mean((rendered - tgt) ** 2)
        psnrs.append(-10 * np.log10(mse) if mse > 0 else float("inf"))
    return float(np.mean(psnrs)), psnrs


print("\n評価中...")
train_mean, _ = mean_psnr(train_cams, train_imgs)
test_mean, test_psnrs = mean_psnr(test_cams, test_imgs)
print(f"  train 平均 PSNR: {train_mean:.2f} dB  ({len(train_cams)} 視点)")
print(f"  test  平均 PSNR: {test_mean:.2f} dB  ({len(test_cams)} 視点)")
for i, p in enumerate(test_psnrs[:4]):
    print(f"    test[{i}] PSNR = {p:.2f} dB")

# test 先頭 4 視点の target / rendered を並べて保存
for i in range(min(4, len(test_cams))):
    tgt = test_imgs[i]
    rendered = np.clip(render_3d(gaussians, test_cams[i],
                                  bg_color=(0, 0, 0)).data, 0, 1)
    side_by_side = np.concatenate([tgt, rendered], axis=1)
    img_u8 = (side_by_side * 255.0).astype(np.uint8)
    Image.fromarray(img_u8).save(f"_garden_test{i}_target_vs_rendered.png")

print("\n学習完了。test 視点の target/rendered を _garden_test*.png に出力。")
