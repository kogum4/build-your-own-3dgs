// 第1章: 3色ガウシアンミキサー
// 赤・緑・青のガウシアンの σx, σy, θ をスライダーで変えて加重和レンダリングを即時再描画する。
// (プロトタイプのウィジェットを Pyodide → JS 移植して即応化)
import type { WidgetContext } from './mount';
import { covariance2d } from './math/mat';
import { renderWeightedSum, type Gaussian2DLike } from './math/splat';
import { sliderRow, imagePanel, widgetHeader, jsReimplNote } from './ui/controls';

const W = 120;
const H = 120;

interface ColorParams {
  sigmaX: number;
  sigmaY: number;
  thetaDeg: number;
}

export function mount(el: HTMLElement, ctx: WidgetContext) {
  const ja = ctx.locale === 'ja';
  widgetHeader(
    el,
    ja ? 'インタラクティブ: 3色ガウシアンの加重和' : 'Interactive: weighted sum of three Gaussians',
    jsReimplNote(ctx.locale),
  );

  const cols = document.createElement('div');
  cols.className = 'w-columns';
  el.appendChild(cols);

  const left = document.createElement('div');
  left.className = 'w-col';
  cols.appendChild(left);

  const right = document.createElement('div');
  right.className = 'w-col';
  cols.appendChild(right);
  const frame = document.createElement('div');
  frame.className = 'w-canvas-frame';
  right.appendChild(frame);
  const panel = imagePanel(frame, W, H, 300);

  // 本文 draw_gaussians.py と同じ配置 (中心を正三角形状に)
  const defs: {
    key: 'red' | 'green' | 'blue';
    label: string;
    mean: [number, number];
    color: [number, number, number];
    params: ColorParams;
  }[] = [
    {
      key: 'red',
      label: ja ? '赤' : 'Red',
      mean: [W * 0.38, H * 0.36],
      color: [1, 0.15, 0.15],
      params: { sigmaX: 18, sigmaY: 10, thetaDeg: 25 },
    },
    {
      key: 'green',
      label: ja ? '緑' : 'Green',
      mean: [W * 0.64, H * 0.42],
      color: [0.15, 1, 0.25],
      params: { sigmaX: 14, sigmaY: 14, thetaDeg: 0 },
    },
    {
      key: 'blue',
      label: ja ? '青' : 'Blue',
      mean: [W * 0.5, H * 0.66],
      color: [0.2, 0.35, 1],
      params: { sigmaX: 22, sigmaY: 9, thetaDeg: -40 },
    },
  ];

  for (const def of defs) {
    const group = document.createElement('div');
    group.className = 'w-group';
    group.dataset.color = def.key;
    const title = document.createElement('p');
    title.className = 'w-group-title';
    title.textContent = def.label;
    group.appendChild(title);
    left.appendChild(group);

    sliderRow(group, {
      label: 'sigma_x',
      min: 4,
      max: 32,
      step: 0.5,
      value: def.params.sigmaX,
      onInput: (v) => {
        def.params.sigmaX = v;
        scheduleRender();
      },
    });
    sliderRow(group, {
      label: 'sigma_y',
      min: 4,
      max: 32,
      step: 0.5,
      value: def.params.sigmaY,
      onInput: (v) => {
        def.params.sigmaY = v;
        scheduleRender();
      },
    });
    sliderRow(group, {
      label: 'theta',
      min: -180,
      max: 180,
      step: 1,
      value: def.params.thetaDeg,
      format: (v) => `${v.toFixed(0)}°`,
      onInput: (v) => {
        def.params.thetaDeg = v;
        scheduleRender();
      },
    });
  }

  let raf = 0;
  function scheduleRender() {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      renderNow();
    });
  }

  function renderNow() {
    const gaussians: Gaussian2DLike[] = defs.map((d) => ({
      mean: d.mean,
      covariance: covariance2d(d.params.sigmaX, d.params.sigmaY, (d.params.thetaDeg * Math.PI) / 180),
      color: d.color,
      opacity: 0.9,
    }));
    panel.draw(renderWeightedSum(gaussians, W, H));
  }

  renderNow();
}
