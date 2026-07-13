# Pyodide Worker 起動時に実行されるブートストラップ。
# stdout/stderr のキャプチャ、matplotlib 図の回収、日本語フォント設定を提供する。
import io
import os
import sys

# 章コードが figures/ へ画像を保存できるように作っておく
os.makedirs("figures", exist_ok=True)

_stdout_buf = None
_stderr_buf = None


def _capture_start():
    global _stdout_buf, _stderr_buf
    _stdout_buf = io.StringIO()
    _stderr_buf = io.StringIO()
    sys.stdout = _stdout_buf
    sys.stderr = _stderr_buf


def _capture_end():
    global _stdout_buf, _stderr_buf
    out = _stdout_buf.getvalue() if _stdout_buf is not None else ""
    err = _stderr_buf.getvalue() if _stderr_buf is not None else ""
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _stdout_buf = None
    _stderr_buf = None
    return [out, err]


def _capture_figures():
    """開いている matplotlib Figure を PNG bytes のリストにして全て閉じる。"""
    if "matplotlib" not in sys.modules:
        return []
    import matplotlib.pyplot as plt

    figures = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        figures.append(buf.getvalue())
    plt.close("all")
    return figures


def _setup_matplotlib(font_path=None):
    """Agg バックエンドと日本語フォントを設定する (matplotlib ロード直後に一度呼ぶ)。"""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["figure.figsize"] = (6.0, 4.5)
    matplotlib.rcParams["axes.unicode_minus"] = False
    if font_path and os.path.exists(font_path):
        import matplotlib.font_manager as fm

        fm.fontManager.addfont(font_path)
        name = fm.FontProperties(fname=font_path).get_name()
        matplotlib.rcParams["font.family"] = name
