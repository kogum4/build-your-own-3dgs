"""
損失関数。
第5章: 2Dガウシアン画像フィッティング
"""

from autograd import Tensor


def l1_loss(predicted, target):
    """L1損失（平均絶対誤差）を計算する。

    L1損失 = mean(|predicted - target|)

    予測画像と目標画像の各ピクセルの差の絶対値を平均します。

    Args:
        predicted: 予測画像のTensor
        target: 目標画像のTensor（同じ形状）

    Returns:
        スカラーTensor（損失値）
    """
    diff = predicted - target
    return diff.abs().mean()
