// 第1章: 共分散行列エクスプローラ
// σx, σy, θ → Σ = R Λ R^T を計算し、1σ楕円と固有ベクトルを Canvas に描画する。
// (プロトタイプ ch1-covariance-ellipse.html の移植。固有値分解は JS 閉形式)
import type { WidgetContext } from './mount';
import { covariance2d, rot2, eigh2, type Mat2 } from './math/mat';
import { sliderRow, canvasPanel, matrixCard, widgetHeader, jsReimplNote } from './ui/controls';

export function mount(el: HTMLElement, ctx: WidgetContext) {
  const ja = ctx.locale === 'ja';
  widgetHeader(
    el,
    ja ? 'インタラクティブ: 共分散行列と楕円' : 'Interactive: covariance matrices and ellipses',
    jsReimplNote(ctx.locale),
  );

  const cols = document.createElement('div');
  cols.className = 'w-columns';
  el.appendChild(cols);

  // --- 左: スライダー + 行列 ---
  const left = document.createElement('div');
  left.className = 'w-col';
  cols.appendChild(left);

  const state = { sx: 1.4, sy: 0.7, thetaDeg: 30 };

  sliderRow(left, {
    label: 'sigma_x',
    min: 0.2,
    max: 2.5,
    step: 0.05,
    value: state.sx,
    format: (v) => v.toFixed(2),
    onInput: (v) => {
      state.sx = v;
      update();
    },
  });
  sliderRow(left, {
    label: 'sigma_y',
    min: 0.2,
    max: 2.5,
    step: 0.05,
    value: state.sy,
    format: (v) => v.toFixed(2),
    onInput: (v) => {
      state.sy = v;
      update();
    },
  });
  sliderRow(left, {
    label: 'theta',
    min: -180,
    max: 180,
    step: 1,
    value: state.thetaDeg,
    format: (v) => `${v.toFixed(0)}°`,
    onInput: (v) => {
      state.thetaDeg = v;
      update();
    },
  });

  const matrixRow = document.createElement('div');
  matrixRow.className = 'w-matrix-row';
  left.appendChild(matrixRow);
  const mR = matrixCard(matrixRow, 'R', 2, 2);
  const mL = matrixCard(matrixRow, 'Λ', 2, 2);
  const mS = matrixCard(matrixRow, 'Σ = RΛRᵀ', 2, 2);
  const mE = matrixCard(matrixRow, ja ? '固有値' : 'eigenvalues', 1, 2);

  // --- 右: Canvas ---
  const right = document.createElement('div');
  right.className = 'w-col';
  cols.appendChild(right);
  const frame = document.createElement('div');
  frame.className = 'w-canvas-frame';
  right.appendChild(frame);
  const { ctx: g, width: CW, height: CH } = canvasPanel(frame, { width: 300, height: 300 });

  const RANGE = 3.2; // 表示範囲 ±RANGE
  const toPx = (x: number, y: number): [number, number] => [
    ((x + RANGE) / (2 * RANGE)) * CW,
    ((RANGE - y) / (2 * RANGE)) * CH,
  ];

  function drawAxes() {
    g.clearRect(0, 0, CW, CH);
    g.strokeStyle = '#eef1f4';
    g.lineWidth = 1;
    for (let v = -3; v <= 3; v++) {
      const [x1, y1] = toPx(v, -RANGE);
      const [, y2] = toPx(-RANGE, v);
      g.beginPath();
      g.moveTo(x1, 0);
      g.lineTo(x1, CH);
      g.stroke();
      g.beginPath();
      g.moveTo(0, y2);
      g.lineTo(CW, y2);
      g.stroke();
      void y1;
    }
    g.strokeStyle = '#c9d1d9';
    const [ox, oy] = toPx(0, 0);
    g.beginPath();
    g.moveTo(0, oy);
    g.lineTo(CW, oy);
    g.stroke();
    g.beginPath();
    g.moveTo(ox, 0);
    g.lineTo(ox, CH);
    g.stroke();
  }

  function drawEllipse(Sigma: Mat2) {
    const { eigvals, eigvecs } = eigh2(Sigma);
    const r1 = Math.sqrt(Math.max(0, eigvals[0]));
    const r2 = Math.sqrt(Math.max(0, eigvals[1]));
    const angle = Math.atan2(eigvecs[0][1], eigvecs[0][0]);

    const [ox, oy] = toPx(0, 0);
    const sx = CW / (2 * RANGE);

    g.save();
    g.translate(ox, oy);
    g.rotate(-angle);
    g.beginPath();
    g.ellipse(0, 0, r1 * sx, r2 * sx, 0, 0, Math.PI * 2);
    g.fillStyle = 'rgba(9, 105, 218, 0.15)';
    g.fill();
    g.strokeStyle = 'rgba(9, 105, 218, 0.9)';
    g.lineWidth = 2;
    g.stroke();
    g.restore();

    // 固有ベクトル矢印 (長さ = sqrt(固有値))
    drawArrow(eigvecs[0][0] * r1, eigvecs[0][1] * r1, '#cf222e');
    drawArrow(eigvecs[1][0] * r2, eigvecs[1][1] * r2, '#1a7f37');
  }

  function drawArrow(x: number, y: number, color: string) {
    const [ox, oy] = toPx(0, 0);
    const [tx, ty] = toPx(x, y);
    g.strokeStyle = color;
    g.fillStyle = color;
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(ox, oy);
    g.lineTo(tx, ty);
    g.stroke();
    const ang = Math.atan2(ty - oy, tx - ox);
    g.beginPath();
    g.moveTo(tx, ty);
    g.lineTo(tx - 9 * Math.cos(ang - 0.4), ty - 9 * Math.sin(ang - 0.4));
    g.lineTo(tx - 9 * Math.cos(ang + 0.4), ty - 9 * Math.sin(ang + 0.4));
    g.closePath();
    g.fill();
  }

  function update() {
    const theta = (state.thetaDeg * Math.PI) / 180;
    const Sigma = covariance2d(state.sx, state.sy, theta);
    const R = rot2(theta);
    const { eigvals } = eigh2(Sigma);

    mR.update(R);
    mL.update([
      [state.sx * state.sx, 0],
      [0, state.sy * state.sy],
    ]);
    mS.update(Sigma);
    mE.update([eigvals]);

    drawAxes();
    drawEllipse(Sigma);
  }

  update();
}
