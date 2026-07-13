"""Novel view synthesis 実験スクリプト。

garden データセットを train/test に等間隔で分割し、train 視点のみで学習、
test 視点は学習中に一度も見せずに最後の評価と可視化に使う。
章本文の図 9.1（成果の証拠）と図 9.4（損失曲線 + iter 進行）の素材を生成する。
"""

import os
import time
import numpy as np
from PIL import Image

from data_loader import ColmapDataset
from initialization import initialize_gaussians
from trainer import GaussianTrainer
from optim import Adam
from render import render_3d


# ========== 実験パラメータ ==========
N_TRAIN = 50
N_TEST = 10
N_GAUSSIANS = 200
N_ITERS = 3000
RESIZE = (108, 70)
LR = 0.005
SEED = 42

# iter 進行可視化のスナップショット iter 番号（iter=0 は学習前）
PROGRESS_ITERS_TARGET = [0, N_ITERS // 3, 2 * N_ITERS // 3, N_ITERS]

# 保存先
OUT_DIR = os.path.join("chapters", "chapter-09", "figures", "_experiment")
os.makedirs(OUT_DIR, exist_ok=True)


def psnr_from_mse(mse):
    return -10 * np.log10(mse) if mse > 0 else float("inf")


def evaluate(gaussians, cameras, images, bg_color=(0, 0, 0)):
    """各視点の PSNR を計算して (mean, per_view list, rendered list) を返す。"""
    psnrs = []
    rendered_list = []
    for cam, tgt in zip(cameras, images):
        rendered = render_3d(gaussians, cam, bg_color=bg_color).data
        mse = np.mean((rendered - tgt) ** 2)
        psnrs.append(psnr_from_mse(mse))
        rendered_list.append(np.clip(rendered, 0, 1))
    return float(np.mean(psnrs)), psnrs, rendered_list


def save_png(arr01, path):
    img_u8 = (np.clip(arr01, 0, 1) * 255.0).astype(np.uint8)
    Image.fromarray(img_u8).save(path)


def main():
    np.random.seed(SEED)

    # ---- 1. データ読み込み ----
    print("データセットを読み込み中...")
    dataset = ColmapDataset("data/colmap_garden", resize=RESIZE)
    n_total = len(dataset)
    print(f"  全視点数: {n_total}, 点群数: {dataset.points3D.shape[0]}")

    # ---- 2. train/test 分割 ----
    # train 50 視点を等間隔に抽出、test 10 視点は train の「合間」に入れて
    # novel view（内挿）として働かせる
    stride = n_total // N_TRAIN  # 185 // 50 = 3
    train_idx = [i * stride for i in range(N_TRAIN)]
    # 残ったインデックスから等間隔に test を抽出（train と重複しない）
    remaining = [i for i in range(n_total) if i not in set(train_idx)]
    test_stride = len(remaining) // N_TEST
    test_idx = [remaining[i * test_stride] for i in range(N_TEST)]
    print(f"  train: {N_TRAIN}視点 (最初/最後: {train_idx[0]}, {train_idx[-1]})")
    print(f"  test:  {N_TEST}視点 (最初/最後: {test_idx[0]}, {test_idx[-1]})")

    train_cams = [dataset.cameras[i] for i in train_idx]
    train_imgs = [dataset.images[i] for i in train_idx]
    test_cams = [dataset.cameras[i] for i in test_idx]
    test_imgs = [dataset.images[i] for i in test_idx]

    # ---- 3. ガウシアン初期化 ----
    print("ガウシアンを初期化中...")
    gaussians = initialize_gaussians(
        dataset.points3D, dataset.point_colors,
        n_gaussians=N_GAUSSIANS,
    )
    print(f"  ガウシアン数: {len(gaussians)}")

    # ---- 4. オプティマイザとトレーナー ----
    params = []
    for g in gaussians:
        params.extend(g.params)
    optimizer = Adam(params, lr=LR)
    trainer = GaussianTrainer(
        gaussians=gaussians,
        cameras=train_cams,
        targets=train_imgs,
        optimizer=optimizer,
        bg_color=(0, 0, 0),
    )

    # iter 進行可視化用に、test[0] を固定視点として iter=0 の時点を保存
    progress_cam = test_cams[0]
    progress_tgt = test_imgs[0]

    print("\niter=0 時点の進行視点をレンダリング...")
    rendered0 = render_3d(gaussians, progress_cam, bg_color=(0, 0, 0)).data
    mse0 = np.mean((rendered0 - progress_tgt) ** 2)
    print(f"  iter=0 test[0] PSNR = {psnr_from_mse(mse0):.2f} dB")
    save_png(rendered0, os.path.join(OUT_DIR, "progress_iter_0000.png"))
    save_png(progress_tgt, os.path.join(OUT_DIR, "progress_target.png"))

    # ---- 5. 学習 ----
    progress_snaps = {0: rendered0}
    snap_targets = set(PROGRESS_ITERS_TARGET)
    snap_targets.discard(0)

    print("\n学習開始...")
    t0 = time.time()
    losses = []
    n_cams = len(train_cams)
    log_interval = max(100, N_ITERS // 20)

    for it in range(1, N_ITERS + 1):
        camera_idx = np.random.randint(n_cams)
        loss_val = trainer.train_step(camera_idx)
        losses.append(loss_val)
        if it % log_interval == 0:
            avg = np.mean(losses[-log_interval:])
            elapsed = time.time() - t0
            print(f"  Iteration {it}/{N_ITERS}, Loss: {avg:.4f}, elapsed: {elapsed:.1f}s")
        if it in snap_targets:
            r = render_3d(gaussians, progress_cam, bg_color=(0, 0, 0)).data
            progress_snaps[it] = r
            save_png(r, os.path.join(OUT_DIR, f"progress_iter_{it:04d}.png"))

    wall_time = time.time() - t0
    print(f"\n学習完了: wall time = {wall_time:.1f} s ({wall_time/60:.1f} min)")

    # ---- 6. 評価 ----
    print("\nTrain 視点の評価...")
    train_mean_psnr, train_psnrs, _ = evaluate(gaussians, train_cams, train_imgs)
    print(f"  train mean PSNR = {train_mean_psnr:.2f} dB")

    print("Test 視点の評価（novel views）...")
    test_mean_psnr, test_psnrs, test_rendered = evaluate(
        gaussians, test_cams, test_imgs
    )
    print(f"  test mean PSNR = {test_mean_psnr:.2f} dB")
    for i, p in enumerate(test_psnrs):
        print(f"    test[{i}] idx={test_idx[i]:3d}  PSNR = {p:.2f} dB")

    # ---- 7. 図素材の保存 ----
    # fig-09-01 用: test の先頭 4 視点の target / rendered
    print("\nfig-09-01 用素材保存...")
    n_panels = min(4, len(test_cams))
    for i in range(n_panels):
        save_png(test_imgs[i], os.path.join(OUT_DIR, f"novel_target_{i}.png"))
        save_png(test_rendered[i], os.path.join(OUT_DIR, f"novel_rendered_{i}.png"))

    # 損失曲線用
    np.save(os.path.join(OUT_DIR, "losses.npy"), np.array(losses))

    # サマリ書き出し
    summary = {
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_gaussians": N_GAUSSIANS,
        "n_iters": N_ITERS,
        "resize": list(RESIZE),
        "lr": LR,
        "seed": SEED,
        "wall_time_sec": wall_time,
        "train_mean_psnr": train_mean_psnr,
        "test_mean_psnr": test_mean_psnr,
        "test_psnrs_first4": test_psnrs[:n_panels],
        "test_psnrs_all": test_psnrs,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "progress_iters": sorted(progress_snaps.keys()),
        "loss_first": float(losses[0]),
        "loss_last_100_mean": float(np.mean(losses[-100:])),
    }
    import json
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nsummary.json を保存しました。")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
