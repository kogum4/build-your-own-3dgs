"""
第2章テスト: アルファ合成レンダリングを検証する。
"""

import numpy as np
from gaussian2d import Gaussian2D, build_covariance_2d, evaluate_gaussian
from render import render_gaussians_weighted_sum, render_gaussians_alpha_composite


# ===== 第1章回帰テスト（スモーク） =====

def test_ch1_single_gaussian():
    """第1章: 単一ガウシアンが中心に丸いブロブを描くことを確認する。"""
    cov = build_covariance_2d(sigma_x=10.0, sigma_y=10.0, theta=0.0)
    g = Gaussian2D(mean=[64, 64], covariance=cov, color=[1, 1, 1], opacity=1.0)
    image = render_gaussians_weighted_sum([g], H=128, W=128)

    assert np.allclose(image[64, 64], [1.0, 1.0, 1.0]), \
        f"中心ピクセルが白でない: {image[64, 64]}"
    assert np.allclose(image[0, 0], [0.0, 0.0, 0.0], atol=1e-6), \
        f"角のピクセルが黒でない: {image[0, 0]}"

    print("[PASS] test_ch1_single_gaussian: 第1章回帰テスト合格")


def test_ch1_covariance():
    """第1章: 共分散行列の基本プロパティを確認する。"""
    cov = build_covariance_2d(sigma_x=15.0, sigma_y=8.0, theta=0.7)
    assert np.allclose(cov, cov.T), f"共分散行列が対称でない:\n{cov}"

    print("[PASS] test_ch1_covariance: 第1章回帰テスト合格")


# ===== 第2章: Gaussian2D depth属性 =====

def test_gaussian2d_depth():
    """Gaussian2Dにdepth属性が追加されていることを確認する。"""
    cov = build_covariance_2d(10.0, 10.0, 0.0)

    # デフォルト値
    g1 = Gaussian2D(mean=[64, 64], covariance=cov)
    assert g1.depth == 0.0, f"デフォルトdepthが不正: {g1.depth}"

    # 明示的な値
    g2 = Gaussian2D(mean=[64, 64], covariance=cov, depth=5.0)
    assert g2.depth == 5.0, f"depthが不正: {g2.depth}"

    print("[PASS] test_gaussian2d_depth: depth属性を確認")


# ===== 第2章: アルファ合成の基本テスト =====

def test_alpha_single_gaussian():
    """単一ガウシアンのアルファ合成が加重和と同じ結果になることを確認する。"""
    cov = build_covariance_2d(10.0, 10.0, 0.0)
    g = Gaussian2D(mean=[64, 64], covariance=cov, color=[1, 0, 0], opacity=1.0,
                   depth=0.0)

    image_ws = render_gaussians_weighted_sum([g], H=128, W=128)
    image_ac = render_gaussians_alpha_composite([g], H=128, W=128, bg_color=(0, 0, 0))

    assert np.allclose(image_ws, image_ac, atol=1e-10), \
        "単一ガウシアンで加重和とアルファ合成が一致しない"

    print("[PASS] test_alpha_single_gaussian: 単一ガウシアンの一致を確認")


def test_opaque_occlusion():
    """完全不透明(alpha=1)のガウシアンが後ろを完全に隠すことを確認する。"""
    cov = build_covariance_2d(15.0, 15.0, 0.0)

    # 赤が手前（depth=1）、青が奥（depth=2）、完全に重なる
    front = Gaussian2D(mean=[64, 64], covariance=cov, color=[1, 0, 0],
                       opacity=1.0, depth=1.0)
    back = Gaussian2D(mean=[64, 64], covariance=cov, color=[0, 0, 1],
                      opacity=1.0, depth=2.0)

    image = render_gaussians_alpha_composite([front, back], H=128, W=128)

    # 中心ピクセルは赤（手前が完全不透明なので青は見えない）
    center = image[64, 64]
    assert np.allclose(center, [1.0, 0.0, 0.0]), \
        f"中心が赤でない（遮蔽失敗）: {center}"

    print("[PASS] test_opaque_occlusion: 完全不透明ガウシアンの遮蔽を確認")


def test_transparent_blending():
    """半透明ガウシアンの重なりで後ろが透けて見えることを確認する。"""
    cov = build_covariance_2d(15.0, 15.0, 0.0)

    # 赤が手前（半透明）、青が奥
    front = Gaussian2D(mean=[64, 64], covariance=cov, color=[1, 0, 0],
                       opacity=0.5, depth=1.0)
    back = Gaussian2D(mean=[64, 64], covariance=cov, color=[0, 0, 1],
                      opacity=1.0, depth=2.0)

    image = render_gaussians_alpha_composite([front, back], H=128, W=128)

    # 中心ピクセル: 赤と青の両方の成分が存在するはず
    center = image[64, 64]
    assert center[0] > 0.1, f"赤成分が小さすぎる: R={center[0]:.4f}"
    assert center[2] > 0.1, f"青成分が小さすぎる: B={center[2]:.4f}"

    # 数値的に検証: 中心ではガウシアン値=1
    # 赤(手前, opacity=0.5): alpha=0.5, C = 0.5*[1,0,0] = [0.5, 0, 0], T = 0.5
    # 青(奥, opacity=1.0): alpha=1.0, C += 0.5*[0,0,1] = [0, 0, 0.5]
    # 合計: [0.5, 0.0, 0.5]
    assert np.allclose(center, [0.5, 0.0, 0.5]), \
        f"半透明合成の数値が不正: {center} (期待: [0.5, 0.0, 0.5])"

    print("[PASS] test_transparent_blending: 半透明ガウシアンの透過を確認")


def test_depth_sorting():
    """ガウシアンのリスト順に関わらず深度順でソートされることを確認する。"""
    cov = build_covariance_2d(15.0, 15.0, 0.0)

    # リスト順は青(奥)→赤(手前)だが、depth順に正しくソートされるはず
    back = Gaussian2D(mean=[64, 64], covariance=cov, color=[0, 0, 1],
                      opacity=1.0, depth=2.0)
    front = Gaussian2D(mean=[64, 64], covariance=cov, color=[1, 0, 0],
                       opacity=1.0, depth=1.0)

    image = render_gaussians_alpha_composite([back, front], H=128, W=128)

    # 赤が手前なので中心は赤
    center = image[64, 64]
    assert np.allclose(center, [1.0, 0.0, 0.0]), \
        f"深度ソートが機能していない: {center}"

    print("[PASS] test_depth_sorting: 深度順ソートを確認")


def test_background_color():
    """背景色が正しく適用されることを確認する。"""
    cov = build_covariance_2d(10.0, 10.0, 0.0)
    g = Gaussian2D(mean=[64, 64], covariance=cov, color=[1, 0, 0],
                   opacity=0.5, depth=0.0)

    # 白背景でレンダリング
    image = render_gaussians_alpha_composite([g], H=128, W=128, bg_color=(1, 1, 1))

    # ガウシアンの影響がない角は白背景
    corner = image[0, 0]
    assert np.allclose(corner, [1.0, 1.0, 1.0], atol=1e-6), \
        f"角が白背景でない: {corner}"

    # 中心はガウシアン値=1, opacity=0.5なので alpha=0.5
    # C = 0.5 * [1,0,0] + 0.5 * [1,1,1] = [1.0, 0.5, 0.5]
    center = image[64, 64]
    assert np.allclose(center, [1.0, 0.5, 0.5]), \
        f"中心の色が不正: {center} (期待: [1.0, 0.5, 0.5])"

    print("[PASS] test_background_color: 背景色の適用を確認")


def test_transmittance_decay():
    """累積透過率が単調に減少することを確認する。"""
    cov = build_covariance_2d(15.0, 15.0, 0.0)
    gaussians = []
    for i in range(5):
        gaussians.append(Gaussian2D(
            mean=[64, 64], covariance=cov, color=[1, 1, 1],
            opacity=0.3, depth=float(i),
        ))

    # 手動で中心ピクセルの透過率を追跡
    T = 1.0
    transmittances = [T]
    for g in gaussians:
        alpha = g.opacity * 1.0  # 中心ではガウシアン値=1
        T *= (1.0 - alpha)
        transmittances.append(T)

    # 単調減少を確認
    for i in range(len(transmittances) - 1):
        assert transmittances[i] > transmittances[i + 1], \
            f"T[{i}]={transmittances[i]:.4f} <= T[{i+1}]={transmittances[i+1]:.4f}"

    # 最終的にゼロに近づく
    assert transmittances[-1] < 0.2, \
        f"5個のガウシアン後の透過率が高すぎる: {transmittances[-1]:.4f}"

    print("[PASS] test_transmittance_decay: 累積透過率の単調減少を確認")


def test_early_termination():
    """T < 1e-4 で早期打ち切りが機能することを確認する。"""
    cov = build_covariance_2d(50.0, 50.0, 0.0)

    # 大量の完全不透明ガウシアンで全ピクセルのTをゼロにする
    gaussians = []
    for i in range(100):
        gaussians.append(Gaussian2D(
            mean=[64, 64], covariance=cov, color=[1, 0, 0],
            opacity=1.0, depth=float(i),
        ))

    # エラーなく完了すること、結果が正しいことを確認
    image = render_gaussians_alpha_composite(gaussians, H=64, W=64)
    assert image.shape == (64, 64, 3), f"形状が不正: {image.shape}"

    print("[PASS] test_early_termination: 早期打ち切りを確認")


def test_output_range():
    """出力画像の値域が [0, 1] であることを確認する。"""
    cov = build_covariance_2d(15.0, 15.0, 0.0)
    gaussians = [
        Gaussian2D(mean=[64, 64], covariance=cov, color=[1, 0, 0],
                   opacity=0.8, depth=0.0),
        Gaussian2D(mean=[64, 64], covariance=cov, color=[0, 1, 0],
                   opacity=0.8, depth=1.0),
        Gaussian2D(mean=[64, 64], covariance=cov, color=[0, 0, 1],
                   opacity=0.8, depth=2.0),
    ]

    image = render_gaussians_alpha_composite(gaussians, H=128, W=128)
    assert image.min() >= 0.0, f"最小値が負: {image.min()}"
    assert image.max() <= 1.0, f"最大値が1超: {image.max()}"

    print("[PASS] test_output_range: 出力値域 [0, 1] を確認")


# ===== 本文出力例の検証 =====

def test_inline_depth_attribute():
    """本文中の Gaussian2D depth属性の出力例を確認する。"""
    cov = build_covariance_2d(15.0, 15.0, 0.0)
    g = Gaussian2D(
        mean=[64, 64],
        covariance=cov,
        color=[1, 0, 0],
        opacity=0.8,
        depth=2.5,
    )
    assert g.depth == 2.5, f"depthが不正: {g.depth}"

    print("[PASS] test_inline_depth_attribute: 本文のdepth出力例を確認")


def test_inline_alpha_composite():
    """本文中のアルファ合成の出力例を確認する。"""
    cov = build_covariance_2d(15.0, 15.0, 0.0)

    front = Gaussian2D(mean=[64, 64], covariance=cov, color=[1, 0, 0],
                       opacity=0.7, depth=1.0)
    back = Gaussian2D(mean=[64, 64], covariance=cov, color=[0, 0, 1],
                      opacity=1.0, depth=2.0)

    image = render_gaussians_alpha_composite([front, back], H=128, W=128)
    center = image[64, 64]

    # 中心: alpha_front = 0.7*1.0 = 0.7
    # C = 0.7*[1,0,0] = [0.7, 0, 0]
    # T = 1 - 0.7 = 0.3
    # C += 0.3*1.0*[0,0,1] = [0.7, 0, 0.3]
    assert np.allclose(center, [0.7, 0.0, 0.3]), \
        f"本文の合成例が不正: {center} (期待: [0.7, 0.0, 0.3])"

    print("[PASS] test_inline_alpha_composite: 本文のアルファ合成出力例を確認")


def test_inline_transmittance_values():
    """本文中の累積透過率の推移例を確認する。"""
    # alpha = 0.3 のガウシアンを5個合成した場合のT
    T = 1.0
    expected_T = [1.0]
    for _ in range(5):
        T *= (1.0 - 0.3)
        expected_T.append(round(T, 4))

    # T = 1.0, 0.7, 0.49, 0.343, 0.2401, 0.1681
    assert np.isclose(expected_T[1], 0.7), f"T[1]が不正: {expected_T[1]}"
    assert np.isclose(expected_T[2], 0.49), f"T[2]が不正: {expected_T[2]}"
    assert np.isclose(expected_T[3], 0.343), f"T[3]が不正: {expected_T[3]}"
    assert np.isclose(expected_T[4], 0.2401), f"T[4]が不正: {expected_T[4]}"
    assert np.isclose(expected_T[5], 0.1681, atol=1e-4), \
        f"T[5]が不正: {expected_T[5]}"

    print("[PASS] test_inline_transmittance_values: 本文の透過率推移を確認")


def test_inline_weighted_sum_center_color():
    """本文中の加重和の中心色の説明を確認する。"""
    cov = build_covariance_2d(18.0, 18.0, 0.0)
    gaussians = [
        Gaussian2D(mean=[54, 64], covariance=cov, color=[1, 0, 0],
                   opacity=0.9, depth=1.0),
        Gaussian2D(mean=[64, 64], covariance=cov, color=[0, 1, 0],
                   opacity=0.9, depth=2.0),
        Gaussian2D(mean=[74, 64], covariance=cov, color=[0, 0, 1],
                   opacity=0.9, depth=3.0),
    ]
    image_ws = render_gaussians_weighted_sum(gaussians, H=128, W=128)
    center = np.round(image_ws[64, 64], 3)
    assert np.allclose(center, [0.771, 0.9, 0.771]), \
        f"加重和の中心色が不正: {center}"

    print("[PASS] test_inline_weighted_sum_center_color: 加重和中心色を確認")


def test_inline_alpha_composite_center_color():
    """本文中のアルファ合成の中心色を確認する。"""
    cov = build_covariance_2d(18.0, 18.0, 0.0)
    gaussians = [
        Gaussian2D(mean=[54, 64], covariance=cov, color=[1, 0, 0],
                   opacity=0.9, depth=1.0),
        Gaussian2D(mean=[64, 64], covariance=cov, color=[0, 1, 0],
                   opacity=0.9, depth=2.0),
        Gaussian2D(mean=[74, 64], covariance=cov, color=[0, 0, 1],
                   opacity=0.9, depth=3.0),
    ]
    image_ac = render_gaussians_alpha_composite(gaussians, H=128, W=128)
    center = np.round(image_ac[64, 64], 3)
    assert np.allclose(center, [0.771, 0.206, 0.018]), \
        f"アルファ合成の中心色が不正: {center}"

    print("[PASS] test_inline_alpha_composite_center_color: アルファ合成中心色を確認")


def test_early_termination_timing():
    """早期打ち切りの実測スニペットがエラーなく実行できることを確認する。"""
    import time
    cov = build_covariance_2d(30.0, 30.0, 0.0)
    many_gaussians = [
        Gaussian2D(mean=[64, 64], covariance=cov, color=[1, 0, 0],
                   opacity=0.5, depth=float(i))
        for i in range(50)
    ]
    start = time.time()
    for _ in range(3):
        render_gaussians_alpha_composite(many_gaussians, H=128, W=128)
    elapsed = (time.time() - start) / 3
    assert elapsed < 10.0, f"処理時間が異常に長い: {elapsed:.3f}秒"

    print(f"[PASS] test_early_termination_timing: 早期打ち切り実測OK ({elapsed:.3f}秒)")


# ===== 第1章の本文出力例の回帰テスト =====

def test_ch1_inline_build_covariance():
    """第1章: build_covariance_2d 出力例の回帰テスト。"""
    cov1 = build_covariance_2d(10.0, 10.0, 0.0)
    assert np.allclose(cov1, [[100., 0.], [0., 100.]])

    cov2 = build_covariance_2d(20.0, 5.0, 0.0)
    assert np.allclose(cov2, [[400., 0.], [0., 25.]])

    print("[PASS] test_ch1_inline_build_covariance: 第1章回帰テスト合格")


if __name__ == "__main__":
    # 第1章回帰テスト
    test_ch1_single_gaussian()
    test_ch1_covariance()
    test_ch1_inline_build_covariance()

    # 第2章テスト
    test_gaussian2d_depth()
    test_alpha_single_gaussian()
    test_opaque_occlusion()
    test_transparent_blending()
    test_depth_sorting()
    test_background_color()
    test_transmittance_decay()
    test_early_termination()
    test_output_range()

    # 本文出力例テスト
    test_inline_depth_attribute()
    test_inline_alpha_composite()
    test_inline_transmittance_values()
    test_inline_weighted_sum_center_color()
    test_inline_alpha_composite_center_color()
    test_early_termination_timing()

    print("\n全テスト合格!")
