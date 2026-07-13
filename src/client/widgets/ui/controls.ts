// ウィジェット共通の UI 部品 (フレームワーク非依存の素の DOM)

export interface SliderOptions {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  /** 値の表示フォーマッタ (既定は小数1桁) */
  format?: (v: number) => string;
  onInput: (v: number) => void;
}

export function sliderRow(parent: HTMLElement, opts: SliderOptions) {
  const row = document.createElement('label');
  row.className = 'w-slider-row';

  const label = document.createElement('span');
  label.className = 'w-slider-label';
  label.textContent = opts.label;

  const input = document.createElement('input');
  input.type = 'range';
  input.min = String(opts.min);
  input.max = String(opts.max);
  input.step = String(opts.step);
  input.value = String(opts.value);

  const value = document.createElement('span');
  value.className = 'w-slider-value';
  const format = opts.format ?? ((v: number) => v.toFixed(1));
  value.textContent = format(opts.value);

  input.addEventListener('input', () => {
    const v = Number(input.value);
    value.textContent = format(v);
    opts.onInput(v);
  });

  row.append(label, input, value);
  parent.appendChild(row);

  return {
    el: row,
    get value() {
      return Number(input.value);
    },
    set value(v: number) {
      input.value = String(v);
      value.textContent = format(v);
    },
  };
}

export interface CanvasPanelOptions {
  width: number;
  height: number;
  /** CSS 上の表示倍率 (ピクセルアート用) */
  className?: string;
}

/** devicePixelRatio 対応の Canvas パネル */
export function canvasPanel(parent: HTMLElement, opts: CanvasPanelOptions) {
  const canvas = document.createElement('canvas');
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = opts.width * dpr;
  canvas.height = opts.height * dpr;
  canvas.style.width = `${opts.width}px`;
  canvas.style.height = `${opts.height}px`;
  if (opts.className) canvas.className = opts.className;
  parent.appendChild(canvas);
  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);
  return { canvas, ctx, width: opts.width, height: opts.height };
}

/** n×m の行列を数値表示するカード */
export function matrixCard(parent: HTMLElement, title: string, rows: number, cols: number) {
  const card = document.createElement('div');
  card.className = 'w-matrix-card';

  const h = document.createElement('div');
  h.className = 'w-matrix-title';
  h.innerHTML = title;
  card.appendChild(h);

  const grid = document.createElement('div');
  grid.className = 'w-matrix-grid';
  grid.style.gridTemplateColumns = `repeat(${cols}, auto)`;
  const cells: HTMLElement[] = [];
  for (let i = 0; i < rows * cols; i++) {
    const cell = document.createElement('span');
    cell.textContent = '0';
    grid.appendChild(cell);
    cells.push(cell);
  }
  card.appendChild(grid);
  parent.appendChild(card);

  return {
    el: card,
    update(values: number[][], digits = 3) {
      let i = 0;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          cells[i++].textContent = formatNum(values[r]?.[c] ?? 0, digits);
        }
      }
    },
  };
}

function formatNum(v: number, digits: number): string {
  if (Object.is(v, -0)) v = 0;
  const s = v.toFixed(digits);
  return s === '-' + '0.'.padEnd(digits + 2, '0') ? s.slice(1) : s;
}

/** H×W×3 の Float32(0..1) / Uint8 画像を Canvas に等倍描画する (ピクセル拡大表示) */
export function imagePanel(parent: HTMLElement, W: number, H: number, displayWidth: number) {
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  canvas.className = 'w-image-panel';
  canvas.style.width = `${displayWidth}px`;
  canvas.style.imageRendering = 'pixelated';
  parent.appendChild(canvas);
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.createImageData(W, H);

  return {
    canvas,
    draw(rgb: Float32Array | Uint8Array | Uint8ClampedArray) {
      const px = imageData.data;
      const isFloat = rgb instanceof Float32Array;
      for (let i = 0; i < W * H; i++) {
        px[i * 4] = isFloat ? Math.round(Math.min(1, Math.max(0, rgb[i * 3])) * 255) : rgb[i * 3];
        px[i * 4 + 1] = isFloat
          ? Math.round(Math.min(1, Math.max(0, rgb[i * 3 + 1])) * 255)
          : rgb[i * 3 + 1];
        px[i * 4 + 2] = isFloat
          ? Math.round(Math.min(1, Math.max(0, rgb[i * 3 + 2])) * 255)
          : rgb[i * 3 + 2];
        px[i * 4 + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
    },
  };
}

/** ウィジェットの見出しと説明文 */
export function widgetHeader(parent: HTMLElement, title: string, note?: string) {
  const head = document.createElement('div');
  head.className = 'w-header';
  const h = document.createElement('p');
  h.className = 'w-title';
  h.textContent = title;
  head.appendChild(h);
  if (note) {
    const n = document.createElement('p');
    n.className = 'w-note';
    n.textContent = note;
    head.appendChild(n);
  }
  parent.appendChild(head);
}

/** JS再実装の注記 (教材の Python コードとの二重実装の混乱を防ぐ) */
export function jsReimplNote(locale: 'ja' | 'en'): string {
  return locale === 'ja'
    ? 'この可視化はJSによる再実装です (数式は本文と同一)。'
    : 'This visualization is a JS reimplementation (same math as the text).';
}
