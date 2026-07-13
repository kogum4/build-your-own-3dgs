// 第5章: ライブ学習デモ
// 本文と同じ fit_image パイプライン (Tensor自動微分 + Adam) を Pyodide Worker の
// チャンク実行で回し、損失曲線と再構成画像の変化をリアルタイム表示する。
// 初期表示は録画済みアニメーション、ボタンで実ブラウザ学習に切替。
import type { WidgetContext } from './mount';
import { imagePanel, widgetHeader } from './ui/controls';

const BASE = import.meta.env.BASE_URL.endsWith('/')
  ? import.meta.env.BASE_URL
  : `${import.meta.env.BASE_URL}/`;

const SETUP_TEMPLATE = `
import numpy as np
from PIL import Image
from autograd import Tensor, clear_graph
from gaussian2d import build_covariance_2d
from render import render_gaussians_alpha_composite_tensor
from loss import l1_loss
from optim import Adam

_w5 = {}

def _ch5_setup(N, H, W, steps, lr, seed):
    img = Image.open("ch5_target.png").convert("RGB").resize((W, H), Image.LANCZOS)
    target_image = np.array(img).astype(np.float64) / 255.0
    np.random.seed(seed)
    means = [Tensor(np.random.rand(2) * np.array([W, H]), requires_grad=True) for _ in range(N)]
    covs = [Tensor(build_covariance_2d(4, 4, 0)) for _ in range(N)]
    colors = [Tensor(np.random.rand(3), requires_grad=True) for _ in range(N)]
    opacities = [Tensor(np.array(0.5), requires_grad=True) for _ in range(N)]
    depths = [float(i) for i in range(N)]
    params = means + colors + opacities
    _w5.update(
        H=H, W=W, steps=steps, step=0,
        means=means, covs=covs, colors=colors, opacities=opacities, depths=depths,
        optimizer=Adam(params, lr=lr), target=Tensor(target_image), last=None,
    )

def _ch5_chunk(n):
    H, W = _w5["H"], _w5["W"]
    loss_val = 0.0
    pred = None
    for _ in range(n):
        if _w5["step"] >= _w5["steps"]:
            break
        _w5["optimizer"].zero_grad()
        pred = render_gaussians_alpha_composite_tensor(
            _w5["means"], _w5["covs"], _w5["colors"], _w5["opacities"], _w5["depths"], H, W
        )
        loss = l1_loss(pred, _w5["target"])
        loss_val = float(loss.data)
        loss.backward()
        clear_graph(loss)
        _w5["optimizer"].step()
        _w5["step"] += 1
    if pred is not None:
        img = np.clip(pred.data.reshape(H, W, 3), 0, 1)
        _w5["last"] = ((img * 255).astype(np.uint8)).tobytes()
    done = _w5["step"] >= _w5["steps"]
    return (done, _w5["step"], _w5["steps"], loss_val, H, W, _w5["last"])

_ch5_setup(__N__, __H__, __W__, __STEPS__, 0.05, 42)
`;

interface Precomputed {
  config: { N: number; H: number; W: number; steps: number };
  losses: number[];
  frames: { step: number; loss: number; rgb: string }[];
}

export function mount(el: HTMLElement, ctx: WidgetContext) {
  const ja = ctx.locale === 'ja';
  widgetHeader(
    el,
    ja ? 'インタラクティブ: ブラウザ内で学習を動かす' : 'Interactive: train right in your browser',
    ja
      ? '本文と同じ自作autograd+Adamの学習ループが、あなたのブラウザの中で実際に動きます。'
      : 'The same autograd + Adam training loop from this chapter, running live in your browser.',
  );

  const H = 48;
  const W = 48;
  const state = { n: 64, steps: 200, running: false, mode: 'idle' as 'idle' | 'replay' | 'live' };

  // --- レイアウト ---
  const cols = document.createElement('div');
  cols.className = 'w-columns';
  el.appendChild(cols);

  const left = document.createElement('div');
  left.className = 'w-col';
  cols.appendChild(left);

  const right = document.createElement('div');
  right.className = 'w-col';
  cols.appendChild(right);

  // 左: 目標画像 + コントロール
  const targetWrap = document.createElement('div');
  targetWrap.className = 'w-canvas-frame';
  left.appendChild(targetWrap);
  const targetFig = document.createElement('figure');
  targetFig.style.margin = '0';
  targetFig.style.textAlign = 'center';
  const targetImg = document.createElement('img');
  targetImg.src = `${BASE}data/ch5/target.png`;
  targetImg.width = 144;
  targetImg.style.imageRendering = 'pixelated';
  targetImg.alt = '';
  const targetCap = document.createElement('figcaption');
  targetCap.style.cssText = 'font-size: var(--text-xs); color: var(--muted); margin-top: 6px;';
  targetCap.textContent = ja ? '目標画像 (target.png)' : 'Target image (target.png)';
  targetFig.append(targetImg, targetCap);
  targetWrap.appendChild(targetFig);

  const controls = document.createElement('div');
  controls.style.marginTop = '12px';
  left.appendChild(controls);

  // N 選択
  const nRow = document.createElement('div');
  nRow.style.cssText = 'display:flex; gap:8px; align-items:center; margin-bottom:10px;';
  const nLabel = document.createElement('span');
  nLabel.style.cssText = 'font-size: var(--text-xs); color: var(--muted);';
  nLabel.textContent = ja ? 'ガウシアン数 N:' : 'Gaussians N:';
  nRow.appendChild(nLabel);
  const nButtons: HTMLButtonElement[] = [];
  for (const n of [32, 64, 128]) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'w-button' + (n === state.n ? ' primary' : '');
    b.textContent = String(n);
    b.addEventListener('click', () => {
      if (state.running) return;
      state.n = n;
      nButtons.forEach((x) => x.classList.toggle('primary', x === b));
    });
    nButtons.push(b);
    nRow.appendChild(b);
  }
  controls.appendChild(nRow);

  // 開始/停止ボタン
  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display:flex; gap:8px; align-items:center; flex-wrap:wrap;';
  const startBtn = document.createElement('button');
  startBtn.type = 'button';
  startBtn.className = 'w-button primary';
  startBtn.textContent = ja ? '▶ 自分のブラウザで学習する' : '▶ Train in your browser';
  const stopBtn = document.createElement('button');
  stopBtn.type = 'button';
  stopBtn.className = 'w-button';
  stopBtn.textContent = ja ? '⏹ 停止' : '⏹ Stop';
  stopBtn.disabled = true;
  btnRow.append(startBtn, stopBtn);
  controls.appendChild(btnRow);

  const statusLine = document.createElement('p');
  statusLine.style.cssText = 'margin:10px 0 0; font-size: var(--text-xs); color: var(--muted);';
  controls.appendChild(statusLine);

  // 進捗バー
  const progressWrap = document.createElement('div');
  progressWrap.style.cssText =
    'height:6px; background:var(--border-subtle); border-radius:999px; margin-top:8px; overflow:hidden;';
  const progressBar = document.createElement('div');
  progressBar.style.cssText =
    'height:100%; width:0%; background:var(--accent); border-radius:999px; transition:width .15s;';
  progressWrap.appendChild(progressBar);
  controls.appendChild(progressWrap);

  // 右: 再構成画像 + 損失曲線
  const renderWrap = document.createElement('div');
  renderWrap.className = 'w-canvas-frame';
  right.appendChild(renderWrap);
  const renderFig = document.createElement('div');
  renderFig.style.textAlign = 'center';
  renderWrap.appendChild(renderFig);
  const panel = imagePanel(renderFig, W, H, 144);
  const renderCap = document.createElement('p');
  renderCap.style.cssText = 'font-size: var(--text-xs); color: var(--muted); margin:6px 0 0;';
  renderCap.textContent = ja ? '再構成画像' : 'Reconstruction';
  renderFig.appendChild(renderCap);

  const lossCanvas = document.createElement('canvas');
  lossCanvas.width = 560;
  lossCanvas.height = 120;
  lossCanvas.style.cssText = 'width:100%; margin-top:10px; background:#fff; border:1px solid var(--border-subtle); border-radius:6px;';
  right.appendChild(lossCanvas);
  const lossCtx = lossCanvas.getContext('2d')!;

  const losses: number[] = [];

  function drawLossCurve() {
    const w = lossCanvas.width;
    const h = lossCanvas.height;
    lossCtx.clearRect(0, 0, w, h);
    lossCtx.fillStyle = '#57606a';
    lossCtx.font = '20px system-ui';
    lossCtx.fillText(ja ? 'L1損失' : 'L1 loss', 10, 24);
    if (losses.length < 2) return;
    const max = Math.max(...losses);
    const min = Math.min(...losses);
    const range = max - min || 1;
    lossCtx.strokeStyle = '#0969da';
    lossCtx.lineWidth = 2.5;
    lossCtx.beginPath();
    losses.forEach((l, i) => {
      const x = (i / (losses.length - 1)) * (w - 20) + 10;
      const y = h - 12 - ((l - min) / range) * (h - 44);
      if (i === 0) lossCtx.moveTo(x, y);
      else lossCtx.lineTo(x, y);
    });
    lossCtx.stroke();
  }
  drawLossCurve();

  // --- 録画済みアニメーションの再生 ---
  let replayTimer: ReturnType<typeof setInterval> | null = null;

  async function startReplay() {
    try {
      const res = await fetch(`${BASE}data/ch5/precomputed.json`);
      if (!res.ok) return;
      const data = (await res.json()) as Precomputed;
      state.mode = 'replay';
      let i = 0;
      const play = () => {
        if (state.mode !== 'replay') return;
        const frame = data.frames[i];
        const raw = Uint8Array.from(atob(frame.rgb), (c) => c.charCodeAt(0));
        panel.draw(raw);
        losses.length = 0;
        losses.push(...data.losses.slice(0, frame.step + 1));
        drawLossCurve();
        progressBar.style.width = `${(frame.step / data.config.steps) * 100}%`;
        statusLine.textContent = ja
          ? `録画済みデモを再生中 (N=${data.config.N}, step ${frame.step}/${data.config.steps})`
          : `Playing recorded demo (N=${data.config.N}, step ${frame.step}/${data.config.steps})`;
        i = (i + 1) % data.frames.length;
      };
      play();
      replayTimer = setInterval(play, 350);
    } catch {
      // 録画データなしでもライブ学習は使える
    }
  }

  function stopReplay() {
    if (replayTimer) {
      clearInterval(replayTimer);
      replayTimer = null;
    }
  }

  // --- ライブ学習 ---
  let session: { cancel: () => void } | null = null;

  startBtn.addEventListener('click', async () => {
    if (state.running) return;
    state.running = true;
    state.mode = 'live';
    stopReplay();
    startBtn.disabled = true;
    stopBtn.disabled = false;
    losses.length = 0;
    statusLine.textContent = ja ? 'Python環境を準備中…' : 'Preparing Python environment…';

    const setup = SETUP_TEMPLATE.replace('__N__', String(state.n))
      .replace('__H__', String(H))
      .replace('__W__', String(W))
      .replace('__STEPS__', String(state.steps));

    try {
      const s = await ctx.pyEnv.startTraining({
        files: [{ url: `${BASE}data/ch5/target.png`, path: 'ch5_target.png' }],
        setup,
        chunk: '_ch5_chunk(5)',
        onProgress: (p) => {
          if (p.image) panel.draw(new Uint8Array(p.image));
          losses.push(p.loss);
          drawLossCurve();
          progressBar.style.width = `${(p.step / p.total) * 100}%`;
          statusLine.textContent = ja
            ? `学習中 … step ${p.step}/${p.total}  loss=${p.loss.toFixed(4)}`
            : `Training … step ${p.step}/${p.total}  loss=${p.loss.toFixed(4)}`;
        },
      });
      session = s;
      const { cancelled } = await s.done;
      statusLine.textContent = cancelled
        ? ja
          ? '停止しました。'
          : 'Stopped.'
        : ja
          ? `学習完了! ${state.n}個のガウシアンが画像を再現しました。`
          : `Done! ${state.n} Gaussians reconstructed the image.`;
    } catch (e) {
      statusLine.textContent = `${ja ? 'エラー' : 'Error'}: ${String(e)}`;
    } finally {
      state.running = false;
      startBtn.disabled = false;
      stopBtn.disabled = true;
      session = null;
    }
  });

  stopBtn.addEventListener('click', () => session?.cancel());

  void startReplay();
}
