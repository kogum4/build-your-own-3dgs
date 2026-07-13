"""
ガウシアントレーナー。
第9章: 実写データからの再構成

素朴版ラスタライザ（第8章の render_3d）と L1 損失を使って
3Dガウシアン群を学習する、素直な学習ループ。

差し替え抽象やコールバックは入れない。以降の章では、
このファイルを直接編集して機能を足していく方針。
"""

import numpy as np
from autograd import Tensor, clear_graph
from loss import l1_loss
from render import render_3d


class GaussianTrainer:
    """3Dガウシアンの学習を管理するクラス。

    Attributes:
        gaussians: Gaussian3D オブジェクトのリスト
        cameras: Camera オブジェクトのリスト
        targets: 目標画像のリスト。各要素は (H, W, 3) のNumPy配列
        optimizer: Adam オプティマイザ
        bg_color: 背景色 (R, G, B)
    """

    def __init__(self, gaussians, cameras, targets, optimizer,
                 bg_color=(0, 0, 0)):
        """
        Args:
            gaussians: Gaussian3D オブジェクトのリスト
            cameras: Camera オブジェクトのリスト
            targets: 目標画像のリスト（[0, 1] のfloat配列）
            optimizer: Adam オプティマイザ
            bg_color: 背景色 (R, G, B)
        """
        self.gaussians = gaussians
        self.cameras = cameras
        self.targets = targets
        self.optimizer = optimizer
        self.bg_color = bg_color

    def train_step(self, camera_idx):
        """1ステップの学習を実行する。

        手順:
        1. 勾配ゼロクリア
        2. カメラと目標画像を選択
        3. render_3d でレンダリング
        4. l1_loss で損失計算
        5. backward + 計算グラフ解放
        6. オプティマイザ更新

        Args:
            camera_idx: 使用するカメラのインデックス

        Returns:
            損失値（float）
        """
        # 1. 勾配ゼロクリア
        self.optimizer.zero_grad()

        # 2. カメラと目標画像を選択
        camera = self.cameras[camera_idx]
        target_image = self.targets[camera_idx]

        # 3. レンダリング（素朴版ラスタライザ）
        rendered = render_3d(self.gaussians, camera, bg_color=self.bg_color)

        # 4. 損失計算（L1）
        target = Tensor(target_image)
        loss = l1_loss(rendered, target)
        loss_val = float(loss.data)

        # 5. backward + 計算グラフ解放（メモリリーク防止）
        loss.backward()
        clear_graph(loss)

        # 6. オプティマイザ更新
        self.optimizer.step()

        return loss_val

    def train(self, n_iters, log_interval=100):
        """学習ループを実行する。

        各イテレーションでランダムにカメラを選択し、train_stepを呼ぶ。

        Args:
            n_iters: イテレーション数
            log_interval: ログ出力間隔

        Returns:
            losses: 損失値のリスト
        """
        n_cameras = len(self.cameras)
        losses = []

        for i in range(n_iters):
            # ランダムにカメラを選択
            camera_idx = np.random.randint(n_cameras)
            loss_val = self.train_step(camera_idx)
            losses.append(loss_val)

            if (i + 1) % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                print(f"Iteration {i + 1}/{n_iters}, "
                      f"Loss: {avg_loss:.4f}")

        return losses
