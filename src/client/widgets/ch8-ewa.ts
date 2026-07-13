// 第8章: EWA Splatting エクスプローラ
// 3Dガウシアン Σ₃D をカメラで観察したときの 2D 共分散 Σ' = J W Σ Wᵀ Jᵀ を可視化する。
// J の式・カリング・座標規約は public/code/chapter-08/projection.py と同一:
//   J = [[fx/Z, 0, -fx*X/Z^2], [0, fy/Z, -fy*Y/Z^2]]
//   point_c = W @ pos + t, u = fx*X/Z + cx, v = fy*Y/Z + cy
//   カリング: Z <= 0.1 (背面) / 画面外 margin = 100
// カメラ配置は ewa_demo.py と同じ規約 (W = rotY(yaw), t = (0, 0, dist) の周回カメラ)。
import type { WidgetContext } from './mount';
import {
  eigh2,
  mul3,
  transpose3,
  mulVec3,
  diag3,
  rotY,
  type Mat2,
  type Mat3,
  type Vec3,
} from './math/mat';
import { sliderRow, canvasPanel, matrixCard, widgetHeader, jsReimplNote } from './ui/controls';

// ---------- 2x3 行列演算 (ウィジェット内ローカル実装) ----------

type Mat23 = [[number, number, number], [number, number, number]];

/** projection.py の compute_jacobian と同じ式 */
function computeJacobian(pointC: Vec3, fx: number, fy: number): Mat23 {
  const X = pointC[0];
  const Y = pointC[1];
  const Z = pointC[2];
  return [
    [fx / Z, 0, (-fx * X) / (Z * Z)],
    [0, fy / Z, (-fy * Y) / (Z * Z)],
  ];
}

/** (2x3) @ (3x3) = (2x3) */
function mul23x3(A: Mat23, B: Mat3): Mat23 {
  const C: Mat23 = [
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 2; r++)
    for (let c = 0; c < 3; c++)
      C[r][c] = A[r][0] * B[0][c] + A[r][1] * B[1][c] + A[r][2] * B[2][c];
  return C;
}

/** (2x3) @ (3x2) = (2x2)。B23 の転置を右から掛ける (A @ B23ᵀ) */
function mulABt(A: Mat23, B23: Mat23): Mat2 {
  const C: Mat2 = [
    [0, 0],
    [0, 0],
  ];
  for (let r = 0; r < 2; r++)
    for (let c = 0; c < 2; c++)
      C[r][c] = A[r][0] * B23[c][0] + A[r][1] * B23[c][1] + A[r][2] * B23[c][2];
  return C;
}

// ---------- 定数 (projection.py / ewa_demo.py の規約) ----------

const IMG_W = 240; // camera.width
const IMG_H = 180; // camera.height
const CX = IMG_W / 2; // 主点は中央
const CY = IMG_H / 2;
const CULL_MARGIN = 100; // projection.py: margin = 100
const Z_NEAR = 0.1; // projection.py: point_c[2] <= 0.1 で背面カリング

/** ガウシアンの中心 (原点付近)。X・Y をわずかにずらして J の第3列 -f·X/Z², -f·Y/Z² を非ゼロにする */
const MU: Vec3 = [0.35, 0.45, 0.0];

interface ProjResult {
  Sigma3: Mat3;
  posCam: Vec3;
  culled: 'back' | 'offscreen' | null;
  u: number;
  v: number;
  J: Mat23;
  Sigma2: Mat2;
}

export function mount(el: HTMLElement, ctx: WidgetContext): void {
  const ja = ctx.locale === 'ja';
  widgetHeader(
    el,
    ja
      ? 'インタラクティブ: EWA Splatting (Σ′ = J W Σ Wᵀ Jᵀ)'
      : 'Interactive: EWA splatting (Σ′ = J W Σ Wᵀ Jᵀ)',
    jsReimplNote(ctx.locale),
  );

  const cols = document.createElement('div');
  cols.className = 'w-columns';
  el.appendChild(cols);

  // --- 左: スライダー + 行列パネル ---
  const left = document.createElement('div');
  left.className = 'w-col';
  cols.appendChild(left);

  const state = {
    sx: 0.8,
    sy: 0.3,
    sz: 0.15,
    rotDeg: 30, // ガウシアンの Y 軸周り回転
    yawDeg: 0, // カメラ方位角
    dist: 4, // ガウシアンまでの距離 Z
    focal: 200, // fx = fy
  };

  const scaleOpts = { min: 0.1, max: 1.5, step: 0.05, format: (v: number) => v.toFixed(2) };
  sliderRow(left, {
    label: 's_x',
    ...scaleOpts,
    value: state.sx,
    onInput: (v) => {
      state.sx = v;
      update();
    },
  });
  sliderRow(left, {
    label: 's_y',
    ...scaleOpts,
    value: state.sy,
    onInput: (v) => {
      state.sy = v;
      update();
    },
  });
  sliderRow(left, {
    label: 's_z',
    ...scaleOpts,
    value: state.sz,
    onInput: (v) => {
      state.sz = v;
      update();
    },
  });
  sliderRow(left, {
    label: ja ? '回転 (Y軸)' : 'rot (Y axis)',
    min: -90,
    max: 90,
    step: 1,
    value: state.rotDeg,
    format: (v) => `${v.toFixed(0)}°`,
    onInput: (v) => {
      state.rotDeg = v;
      update();
    },
  });
  sliderRow(left, {
    label: ja ? 'カメラ yaw' : 'camera yaw',
    min: -60,
    max: 60,
    step: 1,
    value: state.yawDeg,
    format: (v) => `${v.toFixed(0)}°`,
    onInput: (v) => {
      state.yawDeg = v;
      update();
    },
  });
  sliderRow(left, {
    label: ja ? '距離 Z' : 'distance Z',
    min: 2,
    max: 12,
    step: 0.1,
    value: state.dist,
    format: (v) => v.toFixed(1),
    onInput: (v) => {
      state.dist = v;
      update();
    },
  });
  sliderRow(left, {
    label: 'fx = fy',
    min: 100,
    max: 400,
    step: 5,
    value: state.focal,
    format: (v) => v.toFixed(0),
    onInput: (v) => {
      state.focal = v;
      update();
    },
  });

  const matrixRow = document.createElement('div');
  matrixRow.className = 'w-matrix-row';
  left.appendChild(matrixRow);
  const mS3 = matrixCard(matrixRow, 'Σ₃D (3×3)', 3, 3);
  const mJ = matrixCard(matrixRow, 'J (2×3)', 2, 3);
  const mS2 = matrixCard(matrixRow, 'Σ′ = JWΣWᵀJᵀ (2×2)', 2, 2);
  const mUV = matrixCard(matrixRow, 'μ′ = (u, v)', 1, 2);

  // --- 右: 俯瞰図 Canvas + 画像平面 Canvas ---
  const right = document.createElement('div');
  right.className = 'w-col';
  cols.appendChild(right);

  const topLabel = document.createElement('p');
  topLabel.className = 'w-note';
  topLabel.textContent = ja
    ? '俯瞰図: XZ平面を真上から (1σ楕円の XZ 断面とカメラ)'
    : 'Top view: XZ plane from above (1σ XZ cross-section and camera)';
  right.appendChild(topLabel);
  const topFrame = document.createElement('div');
  topFrame.className = 'w-canvas-frame';
  right.appendChild(topFrame);
  const top = canvasPanel(topFrame, { width: 300, height: 300 });

  const imgLabel = document.createElement('p');
  imgLabel.className = 'w-note';
  imgLabel.style.marginTop = '10px';
  imgLabel.textContent = ja
    ? `画像平面: 投影された1σ楕円 (${IMG_W}×${IMG_H} px, 主点は中央)`
    : `Image plane: projected 1σ ellipse (${IMG_W}×${IMG_H} px, principal point at center)`;
  right.appendChild(imgLabel);
  const imgFrame = document.createElement('div');
  imgFrame.className = 'w-canvas-frame';
  right.appendChild(imgFrame);
  const img = canvasPanel(imgFrame, { width: 300, height: 225 });
  const IMG_SCALE = img.width / IMG_W; // 画像px → canvas px (= 1.25)

  // --- キャプション ---
  const caption = document.createElement('p');
  caption.className = 'w-note';
  el.appendChild(caption);

  // ---------- 射影の計算 (projection.py の project_gaussians と同じ手順) ----------

  function project(): ProjResult {
    const rot = (state.rotDeg * Math.PI) / 180;
    const yaw = (state.yawDeg * Math.PI) / 180;
    const fx = state.focal;
    const fy = state.focal;

    // Σ₃D = R S Sᵀ Rᵀ (R: Y軸回転, S = diag(s_x, s_y, s_z))
    const R = rotY(rot);
    const Sigma3 = mul3(
      mul3(R, diag3(state.sx * state.sx, state.sy * state.sy, state.sz * state.sz)),
      transpose3(R),
    );

    // カメラ外部パラメータ (ewa_demo.py と同じ): W = rotY(yaw), t = (0, 0, dist)
    const W = rotY(yaw);
    const t: Vec3 = [0, 0, state.dist];

    // point_c = W @ pos + t
    const wp = mulVec3(W, MU);
    const posCam: Vec3 = [wp[0] + t[0], wp[1] + t[1], wp[2] + t[2]];

    // 透視投影 (u = fx*X/Z + cx, v = fy*Y/Z + cy)
    const u = (fx * posCam[0]) / posCam[2] + CX;
    const v = (fy * posCam[1]) / posCam[2] + CY;

    // ヤコビアンと Σ' = J W Σ Wᵀ Jᵀ (t1 = J@W, Σ' = t1 @ Σ @ t1ᵀ)
    const J = computeJacobian(posCam, fx, fy);
    const t1 = mul23x3(J, W);
    const t2 = mul23x3(t1, Sigma3);
    const Sigma2 = mulABt(t2, t1);

    // カリング判定 (projection.py と同一条件)
    let culled: ProjResult['culled'] = null;
    if (posCam[2] <= Z_NEAR) {
      culled = 'back';
    } else if (
      u < -CULL_MARGIN ||
      u > IMG_W + CULL_MARGIN ||
      v < -CULL_MARGIN ||
      v > IMG_H + CULL_MARGIN
    ) {
      culled = 'offscreen';
    }

    return { Sigma3, posCam, culled, u, v, J, Sigma2 };
  }

  // ---------- 俯瞰図 (XZ平面) ----------

  function drawTop(res: ProjResult) {
    const g = top.ctx;
    const CW = top.width;
    const CH = top.height;
    const yaw = (state.yawDeg * Math.PI) / 180;
    const halfRange = Math.max(3, state.dist * 1.15);
    const k = (CW / 2 - 14) / halfRange;
    const toPx = (x: number, z: number): [number, number] => [CW / 2 + x * k, CH / 2 - z * k];

    g.clearRect(0, 0, CW, CH);

    // 軸 (x: 右, z: 上)
    g.strokeStyle = '#c9d1d9';
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(0, CH / 2);
    g.lineTo(CW, CH / 2);
    g.moveTo(CW / 2, 0);
    g.lineTo(CW / 2, CH);
    g.stroke();
    g.fillStyle = '#8b949e';
    g.font = '10px ui-monospace, monospace';
    g.fillText('x', CW - 12, CH / 2 - 5);
    g.fillText('z', CW / 2 + 6, 12);

    // ガウシアンの XZ 断面 (Σ₃D の {x,z} ブロックを固有値分解)
    const Sxz: Mat2 = [
      [res.Sigma3[0][0], res.Sigma3[0][2]],
      [res.Sigma3[2][0], res.Sigma3[2][2]],
    ];
    const eg = eigh2(Sxz);
    const r1 = Math.sqrt(Math.max(0, eg.eigvals[0]));
    const r2 = Math.sqrt(Math.max(0, eg.eigvals[1]));
    const ang = Math.atan2(eg.eigvecs[0][1], eg.eigvecs[0][0]);
    const [gx, gy] = toPx(MU[0], MU[2]);
    g.save();
    g.translate(gx, gy);
    g.rotate(-ang); // canvas は z 軸 (上向き) が反転
    g.beginPath();
    g.ellipse(0, 0, r1 * k, r2 * k, 0, 0, Math.PI * 2);
    g.fillStyle = 'rgba(9, 105, 218, 0.15)';
    g.fill();
    g.strokeStyle = 'rgba(9, 105, 218, 0.9)';
    g.lineWidth = 2;
    g.stroke();
    g.restore();

    // カメラ位置 C = -Wᵀ t = (dist·sin(yaw), 0, -dist·cos(yaw))、視線 = (-sin(yaw), cos(yaw))
    const camX = state.dist * Math.sin(yaw);
    const camZ = -state.dist * Math.cos(yaw);
    const fwd: [number, number] = [-Math.sin(yaw), Math.cos(yaw)];
    const [px, py] = toPx(camX, camZ);

    // 視野角 (水平): 半角 = atan(cx / fx)
    const halfFov = Math.atan(CX / state.focal);
    const len = state.dist + halfRange * 0.35;
    g.strokeStyle = 'rgba(130, 80, 223, 0.35)';
    g.lineWidth = 1;
    for (const s of [-1, 1]) {
      const a = halfFov * s;
      const dx = fwd[0] * Math.cos(a) - fwd[1] * Math.sin(a);
      const dz = fwd[0] * Math.sin(a) + fwd[1] * Math.cos(a);
      const [qx, qy] = toPx(camX + dx * len, camZ + dz * len);
      g.beginPath();
      g.moveTo(px, py);
      g.lineTo(qx, qy);
      g.stroke();
    }

    // 視線方向の矢印
    const [ax, ay] = toPx(camX + fwd[0] * state.dist * 0.3, camZ + fwd[1] * state.dist * 0.3);
    g.strokeStyle = '#8250df';
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(px, py);
    g.lineTo(ax, ay);
    g.stroke();
    const aang = Math.atan2(ay - py, ax - px);
    g.fillStyle = '#8250df';
    g.beginPath();
    g.moveTo(ax, ay);
    g.lineTo(ax - 8 * Math.cos(aang - 0.4), ay - 8 * Math.sin(aang - 0.4));
    g.lineTo(ax - 8 * Math.cos(aang + 0.4), ay - 8 * Math.sin(aang + 0.4));
    g.closePath();
    g.fill();

    // カメラ本体
    g.fillStyle = '#8250df';
    g.fillRect(px - 4, py - 4, 8, 8);
    g.fillText(ja ? 'カメラ' : 'camera', px + 8, py + 4);
  }

  // ---------- 画像平面 ----------

  function drawImagePlane(res: ProjResult) {
    const g = img.ctx;
    const CW = img.width;
    const CH = img.height;
    g.clearRect(0, 0, CW, CH);

    // 画像枠と主点の十字線
    g.strokeStyle = '#eef1f4';
    g.lineWidth = 1;
    g.beginPath();
    g.moveTo(0, CH / 2);
    g.lineTo(CW, CH / 2);
    g.moveTo(CW / 2, 0);
    g.lineTo(CW / 2, CH);
    g.stroke();
    g.strokeStyle = '#c9d1d9';
    g.strokeRect(0.5, 0.5, CW - 1, CH - 1);

    if (res.culled) {
      g.fillStyle = '#cf222e';
      g.font = '12px sans-serif';
      const msg =
        res.culled === 'back'
          ? ja
            ? 'カリング: カメラ背面 (Z ≤ 0.1)'
            : 'culled: behind camera (Z ≤ 0.1)'
          : ja
            ? 'カリング: 画面外 (margin 100 px)'
            : 'culled: off-screen (margin 100 px)';
      g.fillText(msg, 12, CH / 2 - 8);
      return;
    }

    // Σ' を固有値分解して 1σ 楕円を描画 (v 軸は下向き = canvas と同じ向き)
    const eg = eigh2(res.Sigma2);
    const r1 = Math.sqrt(Math.max(0, eg.eigvals[0]));
    const r2 = Math.sqrt(Math.max(0, eg.eigvals[1]));
    const ang = Math.atan2(eg.eigvecs[0][1], eg.eigvecs[0][0]);
    const cx = res.u * IMG_SCALE;
    const cy = res.v * IMG_SCALE;

    g.save();
    g.translate(cx, cy);
    g.rotate(ang);
    g.beginPath();
    g.ellipse(0, 0, Math.max(0.5, r1 * IMG_SCALE), Math.max(0.5, r2 * IMG_SCALE), 0, 0, Math.PI * 2);
    g.fillStyle = 'rgba(9, 105, 218, 0.15)';
    g.fill();
    g.strokeStyle = 'rgba(9, 105, 218, 0.9)';
    g.lineWidth = 2;
    g.stroke();
    g.restore();

    // 固有ベクトル軸 (長さ = √固有値)
    g.lineWidth = 2;
    g.strokeStyle = '#cf222e';
    g.beginPath();
    g.moveTo(cx, cy);
    g.lineTo(cx + eg.eigvecs[0][0] * r1 * IMG_SCALE, cy + eg.eigvecs[0][1] * r1 * IMG_SCALE);
    g.stroke();
    g.strokeStyle = '#1a7f37';
    g.beginPath();
    g.moveTo(cx, cy);
    g.lineTo(cx + eg.eigvecs[1][0] * r2 * IMG_SCALE, cy + eg.eigvecs[1][1] * r2 * IMG_SCALE);
    g.stroke();

    // 2D 平均 μ'
    g.fillStyle = '#0969da';
    g.beginPath();
    g.arc(cx, cy, 2.5, 0, Math.PI * 2);
    g.fill();
  }

  // ---------- 更新 ----------

  function update() {
    const res = project();

    mS3.update(res.Sigma3, 3);
    if (res.culled) {
      const zero23: number[][] = [
        [0, 0, 0],
        [0, 0, 0],
      ];
      mJ.update(zero23, 2);
      mS2.update(
        [
          [0, 0],
          [0, 0],
        ],
        1,
      );
      mUV.update([[0, 0]], 1);
      caption.textContent = ja
        ? 'このガウシアンは projection.py のカリング条件により除外されました。'
        : 'This Gaussian is culled by the conditions in projection.py.';
    } else {
      mJ.update(res.J, 2);
      mS2.update(res.Sigma2, 1);
      mUV.update([[res.u, res.v]], 1);

      const eg = eigh2(res.Sigma2);
      const r1 = Math.sqrt(Math.max(0, eg.eigvals[0]));
      const zc = res.posCam[2];
      caption.textContent = ja
        ? `深度 Z = ${zc.toFixed(2)} → 1σ長径 ≈ ${r1.toFixed(1)} px。J ∝ 1/Z なので、距離を2倍にすると楕円の半径は約半分になる。`
        : `Depth Z = ${zc.toFixed(2)} → 1σ major radius ≈ ${r1.toFixed(1)} px. Since J ∝ 1/Z, doubling the distance roughly halves the ellipse radii.`;
    }

    drawTop(res);
    drawImagePlane(res);
  }

  update();
}
