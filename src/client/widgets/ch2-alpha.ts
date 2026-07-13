// 第2章: 加重和 vs アルファ合成
// 3つのガウシアンの depth / opacity をスライダーで変え、加重和とアルファ合成を左右比較する。
// 右画像のピクセルをクリックすると compositeSteps による T 減衰の積み上げバーを表示する。
// 計算は本文の式 (2.1)(2.2) と同一: C += c·α·T, T *= (1-α)
import type { WidgetContext } from './mount';
import { covariance2d } from './math/mat';
import {
  renderWeightedSum,
  renderAlphaComposite,
  compositeSteps,
  type Gaussian2DLike,
} from './math/splat';
import { sliderRow, imagePanel, widgetHeader, jsReimplNote } from './ui/controls';

const W = 120;
const H = 120;
const DISPLAY_W = 170;

interface GaussianDef {
  key: 'red' | 'green' | 'blue';
  name: string;
  mean: [number, number];
  color: [number, number, number];
  covariance: ReturnType<typeof covariance2d>;
  params: { depth: number; opacity: number };
}

function cssColor(c: [number, number, number]): string {
  const to255 = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255);
  return `rgb(${to255(c[0])}, ${to255(c[1])}, ${to255(c[2])})`;
}

export function mount(el: HTMLElement, ctx: WidgetContext) {
  const ja = ctx.locale === 'ja';
  widgetHeader(
    el,
    ja ? 'インタラクティブ: 加重和 vs アルファ合成' : 'Interactive: weighted sum vs alpha compositing',
    `${jsReimplNote(ctx.locale)} ${
      ja
        ? '右の画像をクリックすると、そのピクセルの T 減衰を確認できます。'
        : 'Click the right image to inspect the transmittance decay at that pixel.'
    }`,
  );

  const cols = document.createElement('div');
  cols.className = 'w-columns';
  el.appendChild(cols);

  // --- 左: 3ガウシアンの depth / opacity スライダー ---
  const left = document.createElement('div');
  left.className = 'w-col';
  cols.appendChild(left);

  // --- 右: 比較画像 + T減衰パネル ---
  const right = document.createElement('div');
  right.className = 'w-col';
  cols.appendChild(right);

  // 3色が中央付近で重なる配置 (共分散は固定、depth と opacity のみ可変)
  const defs: GaussianDef[] = [
    {
      key: 'red',
      name: ja ? '赤' : 'Red',
      mean: [W * 0.4, H * 0.4],
      color: [1, 0.15, 0.15],
      covariance: covariance2d(20, 13, (25 * Math.PI) / 180),
      params: { depth: 1.0, opacity: 0.8 },
    },
    {
      key: 'green',
      name: ja ? '緑' : 'Green',
      mean: [W * 0.62, H * 0.44],
      color: [0.15, 1, 0.25],
      covariance: covariance2d(16, 16, 0),
      params: { depth: 2.5, opacity: 0.8 },
    },
    {
      key: 'blue',
      name: ja ? '青' : 'Blue',
      mean: [W * 0.5, H * 0.64],
      color: [0.2, 0.35, 1],
      covariance: covariance2d(22, 12, (-35 * Math.PI) / 180),
      params: { depth: 4.0, opacity: 0.8 },
    },
  ];

  for (const def of defs) {
    const group = document.createElement('div');
    group.className = 'w-group';
    group.dataset.color = def.key;
    const title = document.createElement('p');
    title.className = 'w-group-title';
    title.textContent = def.name;
    group.appendChild(title);
    left.appendChild(group);

    sliderRow(group, {
      label: 'depth',
      min: 0.5,
      max: 5.0,
      step: 0.1,
      value: def.params.depth,
      onInput: (v) => {
        def.params.depth = v;
        scheduleRender();
      },
    });
    sliderRow(group, {
      label: 'opacity',
      min: 0.05,
      max: 1.0,
      step: 0.05,
      value: def.params.opacity,
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        def.params.opacity = v;
        scheduleRender();
      },
    });
  }

  const hint = document.createElement('p');
  hint.className = 'w-note';
  hint.textContent = ja
    ? 'depth を動かすと右のアルファ合成だけが変わります (加重和は順序に鈍感)。'
    : 'Moving depth changes only the alpha-composited image (the weighted sum is order-insensitive).';
  left.appendChild(hint);

  // --- 比較画像 (左=加重和 / 右=アルファ合成) ---
  const frame = document.createElement('div');
  frame.className = 'w-canvas-frame';
  frame.style.gap = '14px';
  frame.style.flexWrap = 'wrap';
  right.appendChild(frame);

  function labeledPanel(caption: string) {
    const box = document.createElement('div');
    box.style.display = 'flex';
    box.style.flexDirection = 'column';
    box.style.alignItems = 'center';
    box.style.gap = '6px';
    frame.appendChild(box);
    const panel = imagePanel(box, W, H, DISPLAY_W);
    const cap = document.createElement('p');
    cap.style.margin = '0';
    cap.style.fontSize = 'var(--text-xs)';
    cap.style.color = 'var(--muted)';
    cap.textContent = caption;
    box.appendChild(cap);
    return panel;
  }

  const panelWeighted = labeledPanel(ja ? '加重和 (depth を無視)' : 'Weighted sum (ignores depth)');
  const panelAlpha = labeledPanel(
    ja ? 'アルファ合成 (クリックで画素選択)' : 'Alpha compositing (click to pick a pixel)',
  );
  panelAlpha.canvas.style.cursor = 'crosshair';

  // --- T減衰パネル ---
  const stepsCard = document.createElement('div');
  stepsCard.className = 'w-group';
  stepsCard.style.marginTop = '10px';
  right.appendChild(stepsCard);

  const stepsTitle = document.createElement('p');
  stepsTitle.className = 'w-group-title';
  stepsCard.appendChild(stepsTitle);

  const stepsNote = document.createElement('p');
  stepsNote.className = 'w-note';
  stepsNote.style.margin = '0 0 8px';
  stepsNote.textContent = ja
    ? '各ステップ: C += c·α·T のあと T *= (1−α)。手前の α が大きいほど奥の寄与は小さくなります。'
    : 'Each step: C += c·α·T, then T *= (1−α). The larger the front α, the smaller the rear contributions.';
  stepsCard.appendChild(stepsNote);

  const stepsBody = document.createElement('div');
  stepsCard.appendChild(stepsBody);

  // 選択ピクセル (初期値は3色が重なるあたり)
  const sel = { x: Math.round(W * 0.5), y: Math.round(H * 0.48) };

  panelAlpha.canvas.addEventListener('click', (e) => {
    const rect = panelAlpha.canvas.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * W);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * H);
    sel.x = Math.min(W - 1, Math.max(0, x));
    sel.y = Math.min(H - 1, Math.max(0, y));
    scheduleRender();
  });

  let raf = 0;
  function scheduleRender() {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      renderNow();
    });
  }

  function renderNow() {
    const items = defs.map((d) => ({
      def: d,
      g: {
        mean: d.mean,
        covariance: d.covariance,
        color: d.color,
        opacity: d.params.opacity,
        depth: d.params.depth,
      } satisfies Gaussian2DLike,
    }));
    const gaussians = items.map((it) => it.g);

    panelWeighted.draw(renderWeightedSum(gaussians, W, H));
    const { image } = renderAlphaComposite(gaussians, W, H);
    panelAlpha.draw(image);
    drawCrossMarker();
    updateSteps(items);
  }

  /** 選択ピクセル位置の十字マーカー (putImageData の後に上描き) */
  function drawCrossMarker() {
    const g2 = panelAlpha.canvas.getContext('2d')!;
    const cx = sel.x + 0.5;
    const cy = sel.y + 0.5;
    const arm = 5;
    const cross = () => {
      g2.beginPath();
      g2.moveTo(cx - arm, cy);
      g2.lineTo(cx + arm, cy);
      g2.moveTo(cx, cy - arm);
      g2.lineTo(cx, cy + arm);
      g2.stroke();
    };
    g2.save();
    g2.strokeStyle = 'rgba(0, 0, 0, 0.85)';
    g2.lineWidth = 3;
    cross();
    g2.strokeStyle = '#ffffff';
    g2.lineWidth = 1;
    cross();
    g2.restore();
  }

  function updateSteps(items: { def: GaussianDef; g: Gaussian2DLike }[]) {
    const steps = compositeSteps(
      items.map((it) => it.g),
      sel.x,
      sel.y,
    );
    const tFinal = steps.length > 0 ? steps[steps.length - 1].tAfter : 1;

    // タイトル (合成後のピクセル色スウォッチ付き。背景は黒なので C がそのまま画素色)
    stepsTitle.textContent = ja
      ? `選択ピクセル (${sel.x}, ${sel.y}) の合成ステップ (手前 → 奥) `
      : `Compositing steps at pixel (${sel.x}, ${sel.y}) (front to back) `;
    const composed: [number, number, number] = [0, 0, 0];
    for (const s of steps) {
      composed[0] += s.weight * s.gaussian.color[0];
      composed[1] += s.weight * s.gaussian.color[1];
      composed[2] += s.weight * s.gaussian.color[2];
    }
    const swatch = document.createElement('span');
    swatch.style.display = 'inline-block';
    swatch.style.width = '11px';
    swatch.style.height = '11px';
    swatch.style.borderRadius = '2px';
    swatch.style.border = '1px solid var(--border-subtle)';
    swatch.style.verticalAlign = 'middle';
    swatch.style.background = cssColor(composed);
    swatch.title = ja ? '合成結果 C' : 'composited C';
    stepsTitle.appendChild(swatch);

    stepsBody.innerHTML = '';

    // --- 積み上げバー: 幅1.0 = 光の全量。各セグメント = weight (α·T)、残り = T ---
    const stack = document.createElement('div');
    stack.style.display = 'flex';
    stack.style.height = '16px';
    stack.style.border = '1px solid var(--border-subtle)';
    stack.style.borderRadius = '4px';
    stack.style.overflow = 'hidden';
    stack.style.background = 'repeating-linear-gradient(45deg, #f0f2f5 0 4px, #fafbfc 4px 8px)';
    for (const s of steps) {
      const seg = document.createElement('div');
      seg.style.flex = `0 0 ${(s.weight * 100).toFixed(2)}%`;
      seg.style.background = cssColor(s.gaussian.color);
      const item = items.find((it) => it.g === s.gaussian);
      seg.title = `${item ? item.def.name : '?'}: α·T = ${s.weight.toFixed(3)}`;
      stack.appendChild(seg);
    }
    stepsBody.appendChild(stack);

    const stackCap = document.createElement('p');
    stackCap.style.margin = '4px 0 10px';
    stackCap.style.fontSize = 'var(--text-xs)';
    stackCap.style.color = 'var(--faint)';
    stackCap.style.fontFamily = 'var(--font-mono)';
    stackCap.textContent = ja
      ? `色付き = 各ガウシアンの寄与 α·T / 斜線 = 残り T = ${tFinal.toFixed(3)}`
      : `colored = per-Gaussian contribution α·T / hatched = remaining T = ${tFinal.toFixed(3)}`;
    stepsBody.appendChild(stackCap);

    // --- ステップごとの行: 名前 + α + α·T + T (before → after) + 寄与バー ---
    for (let i = 0; i < steps.length; i++) {
      const s = steps[i];
      const item = items.find((it) => it.g === s.gaussian);
      const name = item ? item.def.name : '?';

      const head = document.createElement('div');
      head.style.display = 'flex';
      head.style.flexWrap = 'wrap';
      head.style.alignItems = 'center';
      head.style.gap = '10px';
      head.style.marginTop = i === 0 ? '0' : '8px';
      head.style.fontFamily = 'var(--font-mono)';
      head.style.fontSize = 'var(--text-xs)';

      const dot = document.createElement('span');
      dot.style.display = 'inline-block';
      dot.style.width = '10px';
      dot.style.height = '10px';
      dot.style.borderRadius = '2px';
      dot.style.background = cssColor(s.gaussian.color);
      head.appendChild(dot);

      const label = document.createElement('span');
      label.textContent = `${i + 1}. ${name} (depth=${(s.gaussian.depth ?? 0).toFixed(1)})`;
      head.appendChild(label);

      const alphaSpan = document.createElement('span');
      alphaSpan.textContent = `α=${s.alpha.toFixed(3)}`;
      head.appendChild(alphaSpan);

      const weightSpan = document.createElement('span');
      weightSpan.textContent = `α·T=${s.weight.toFixed(3)}`;
      head.appendChild(weightSpan);

      const tSpan = document.createElement('span');
      tSpan.style.color = 'var(--muted)';
      tSpan.textContent = `T: ${s.tBefore.toFixed(3)} → ${s.tAfter.toFixed(3)}`;
      head.appendChild(tSpan);

      stepsBody.appendChild(head);

      const track = document.createElement('div');
      track.style.height = '8px';
      track.style.marginTop = '3px';
      track.style.borderRadius = '4px';
      track.style.background = 'var(--bg-subtle, #f6f8fa)';
      track.style.overflow = 'hidden';
      const fill = document.createElement('div');
      fill.style.height = '100%';
      fill.style.width = `${(s.weight * 100).toFixed(2)}%`;
      fill.style.borderRadius = '4px';
      fill.style.background = cssColor(s.gaussian.color);
      track.appendChild(fill);
      stepsBody.appendChild(track);
    }
  }

  renderNow();
}
