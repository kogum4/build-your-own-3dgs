"""
COLMAPデータ読み込み。
第9章: 実写データからの再構成

COLMAPのバイナリ出力ファイル（cameras.bin, images.bin, points3D.bin）を読み込み、
カメラパラメータ・画像外部パラメータ・3D点群を取得する。
"""

import struct
import os
import numpy as np
from PIL import Image
from camera import Camera


def read_cameras_binary(path):
    """cameras.bin を読み込む。

    COLMAPはカメラの内部パラメータをcameras.binに保存する。
    1つのカメラモデルを複数の画像が共有することが多い。

    バイナリ形式:
        - 先頭8バイト: カメラ数（uint64）
        - 各カメラ: camera_id(uint32) + model_id(uint32) + width(uint64) +
                    height(uint64) + params(float64 × パラメータ数)

    対応カメラモデル:
        - SIMPLE_PINHOLE (id=0): params=[f, cx, cy]
        - PINHOLE (id=1): params=[fx, fy, cx, cy]
        - SIMPLE_RADIAL (id=2): params=[f, cx, cy, k]（kは無視）

    Args:
        path: cameras.bin のファイルパス

    Returns:
        dict: {camera_id: {"model_id", "width", "height", "fx", "fy", "cx", "cy"}}
    """
    cameras = {}

    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            if model_id == 0:
                # SIMPLE_PINHOLE: f, cx, cy
                f_val, cx, cy = struct.unpack("<3d", f.read(24))
                fx, fy = f_val, f_val
            elif model_id == 1:
                # PINHOLE: fx, fy, cx, cy
                fx, fy, cx, cy = struct.unpack("<4d", f.read(32))
            elif model_id == 2:
                # SIMPLE_RADIAL: f, cx, cy, k（kは無視）
                f_val, cx, cy, _k = struct.unpack("<4d", f.read(32))
                fx, fy = f_val, f_val
            else:
                raise ValueError(f"未対応のカメラモデル: model_id={model_id}")

            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }

    return cameras


def quaternion_to_rotation(qw, qx, qy, qz):
    """COLMAPのクォータニオン(w, x, y, z)を回転行列に変換する。

    COLMAPは (w, x, y, z) 順のクォータニオンでカメラの回転を表す。
    これを3x3回転行列に変換する。

    Args:
        qw, qx, qy, qz: クォータニオンの各成分

    Returns:
        (3, 3) の回転行列（NumPy配列）
    """
    # 正規化
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)],
    ])
    return R


def read_images_binary(path):
    """images.bin を読み込む。

    COLMAPは各画像の外部パラメータ（カメラの位置と向き）を
    images.binに保存する。

    バイナリ形式:
        - 先頭8バイト: 画像数（uint64）
        - 各画像: image_id(uint32) + qw,qx,qy,qz(float64×4) +
                  tx,ty,tz(float64×3) + camera_id(uint32) +
                  image_name(null終端文字列) +
                  num_points2D(uint64) + points2D(float64×2 + int64)×num_points2D

    Args:
        path: images.bin のファイルパス

    Returns:
        dict: {image_id: {"R", "t", "camera_id", "name"}}
    """
    images = {}

    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]

            # null終端の画像ファイル名を読み込む
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode("utf-8")

            # 2D特徴点をスキップ
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            # 各点: x(float64) + y(float64) + point3D_id(int64) = 24バイト
            f.read(num_points2D * 24)

            # クォータニオンを回転行列に変換
            R = quaternion_to_rotation(qw, qx, qy, qz)
            t = np.array([tx, ty, tz])

            images[image_id] = {
                "R": R,
                "t": t,
                "camera_id": camera_id,
                "name": name,
            }

    return images


def read_points3D_binary(path):
    """points3D.bin を読み込む。

    COLMAPが三角測量で求めた3D点群を読み込む。
    各点は3D座標・RGB色・再投影誤差・追跡情報を持つ。

    バイナリ形式:
        - 先頭8バイト: 点数（uint64）
        - 各点: point3D_id(uint64) + x,y,z(float64×3) + r,g,b(uint8×3) +
                error(float64) + num_track(uint64) +
                track(uint32×2)×num_track

    Args:
        path: points3D.bin のファイルパス

    Returns:
        positions: (N, 3) の座標配列
        colors: (N, 3) のRGB配列（0-255の整数）
    """
    positions = []
    colors = []

    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_points):
            _point3D_id = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            _error = struct.unpack("<d", f.read(8))[0]
            num_track = struct.unpack("<Q", f.read(8))[0]
            # 各追跡: image_id(uint32) + point2D_idx(uint32) = 8バイト
            f.read(num_track * 8)

            positions.append([x, y, z])
            colors.append([r, g, b])

    positions = np.array(positions, dtype=np.float64)
    colors = np.array(colors, dtype=np.uint8)

    return positions, colors


class ColmapDataset:
    """COLMAPデータセットを管理するクラス。

    COLMAPの出力ファイルと画像を読み込み、学習に必要な
    Cameraオブジェクトと画像データを提供する。

    Attributes:
        cameras: Camera オブジェクトのリスト
        images: (H, W, 3) のNumPy配列のリスト（値域 [0, 1]）
        image_names: 画像ファイル名のリスト
        points3D: (N, 3) の3D点群座標
        point_colors: (N, 3) のRGB色（0-255）
    """

    def __init__(self, base_path, images_folder="images_8",
                 resize=None):
        """COLMAPデータセットを読み込む。

        Args:
            base_path: COLMAPデータのルートディレクトリ
            images_folder: 画像フォルダ名（デフォルト "images_8"）。
                多くのデータセットでは元画像の1/2, 1/4, 1/8縮小版を
                images_2, images_4, images_8 というフォルダに用意する慣習がある。
                1/8版が最も軽量で実験に向いている。
            resize: (width, height) にリサイズ。Noneならリサイズしない
        """
        sparse_path = os.path.join(base_path, "sparse", "0")
        images_path = os.path.join(base_path, images_folder)

        # --- COLMAPバイナリを読み込む ---
        colmap_cameras = read_cameras_binary(
            os.path.join(sparse_path, "cameras.bin")
        )
        colmap_images = read_images_binary(
            os.path.join(sparse_path, "images.bin")
        )
        positions, point_colors = read_points3D_binary(
            os.path.join(sparse_path, "points3D.bin")
        )
        self.points3D = positions
        self.point_colors = point_colors

        # --- 画像とカメラを対応付ける ---
        self.cameras = []
        self.images = []
        self.image_names = []

        # image_id順にソートして順序を固定
        for image_id in sorted(colmap_images.keys()):
            img_info = colmap_images[image_id]
            cam_info = colmap_cameras[img_info["camera_id"]]

            # 画像を読み込む
            img_path = os.path.join(images_path, img_info["name"])
            if not os.path.exists(img_path):
                continue
            pil_img = Image.open(img_path)

            # 元画像とCOLMAPカメラの解像度が異なる場合のスケール比
            orig_w = cam_info["width"]
            orig_h = cam_info["height"]
            actual_w, actual_h = pil_img.size
            scale_x = actual_w / orig_w
            scale_y = actual_h / orig_h

            # 内部パラメータをスケール
            fx = cam_info["fx"] * scale_x
            fy = cam_info["fy"] * scale_y
            cx = cam_info["cx"] * scale_x
            cy = cam_info["cy"] * scale_y
            width = actual_w
            height = actual_h

            # リサイズが指定された場合
            if resize is not None:
                new_w, new_h = resize
                resize_scale_x = new_w / width
                resize_scale_y = new_h / height
                fx *= resize_scale_x
                fy *= resize_scale_y
                cx *= resize_scale_x
                cy *= resize_scale_y
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                width = new_w
                height = new_h

            # NumPy配列に変換（[0, 1] の float64）
            img_array = np.array(pil_img, dtype=np.float64) / 255.0

            # Camera オブジェクトを作成
            camera = Camera(
                W=img_info["R"],
                t=img_info["t"],
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height,
            )

            self.cameras.append(camera)
            self.images.append(img_array)
            self.image_names.append(img_info["name"])

    def __len__(self):
        return len(self.cameras)
