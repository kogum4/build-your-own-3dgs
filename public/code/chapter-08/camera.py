"""
カメラモデルと座標変換。
第7章: カメラモデルと座標変換

ピンホールカメラモデルを実装する。
外部パラメータ（W, t）でワールド→カメラ座標変換を、
内部パラメータ（fx, fy, cx, cy）でカメラ座標→ピクセル座標変換を行う。
"""

import numpy as np


class Camera:
    """ピンホールカメラモデル。

    外部パラメータ（カメラの位置と向き）と内部パラメータ（レンズの性質）を
    保持し、ワールド座標系の3D点をピクセル座標に変換する。

    Attributes:
        W: (3, 3) 回転行列（ワールド→カメラ）
        t: (3,) 並進ベクトル（ワールド→カメラ）
        fx: x方向の焦点距離（ピクセル単位）
        fy: y方向の焦点距離（ピクセル単位）
        cx: 主点のx座標（ピクセル単位）
        cy: 主点のy座標（ピクセル単位）
        width: 画像の幅（ピクセル）
        height: 画像の高さ（ピクセル）
    """

    def __init__(self, W, t, fx, fy, cx, cy, width, height):
        """
        Args:
            W: (3, 3) 回転行列（ワールド→カメラ）
            t: (3,) 並進ベクトル（ワールド→カメラ）
            fx: x方向の焦点距離（ピクセル単位）
            fy: y方向の焦点距離（ピクセル単位）
            cx: 主点のx座標（ピクセル単位）
            cy: 主点のy座標（ピクセル単位）
            width: 画像の幅（ピクセル）
            height: 画像の高さ（ピクセル）
        """
        self.W = np.array(W, dtype=np.float64)   # (3, 3)
        self.t = np.array(t, dtype=np.float64)   # (3,)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)

    def world_to_camera(self, points_w):
        """ワールド座標をカメラ座標に変換する。

        カメラ座標 = W @ ワールド座標 + t

        Args:
            points_w: (N, 3) ワールド座標の点群

        Returns:
            (N, 3) カメラ座標の点群
        """
        # (N, 3) @ (3, 3)^T + (3,) = (N, 3)
        points_c = points_w @ self.W.T + self.t
        return points_c

    def project(self, points_c):
        """カメラ座標をピクセル座標に投影する。

        透視投影: u = fx * X/Z + cx, v = fy * Y/Z + cy

        Args:
            points_c: (N, 3) カメラ座標の点群

        Returns:
            pixels: (N, 2) ピクセル座標 [u, v]
            depths: (N,) 深度値 Z
        """
        X = points_c[:, 0]  # (N,)
        Y = points_c[:, 1]  # (N,)
        Z = points_c[:, 2]  # (N,)

        # 透視投影: Zで割ることで遠近法を実現
        u = self.fx * X / Z + self.cx  # (N,)
        v = self.fy * Y / Z + self.cy  # (N,)

        pixels = np.stack([u, v], axis=1)  # (N, 2)
        depths = Z                          # (N,)

        return pixels, depths
