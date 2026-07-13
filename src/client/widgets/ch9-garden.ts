// 第9章: 学習済みガーデンの自由視点エクスプローラ
// ローカルで学習した200ガウシアンのパラメータ (trained-params.json) を、
// 本文と同じフルパイプライン (world→camera → EWA射影 → depthソート → α合成) の
// JS移植でリアルタイムレンダリングする。
import type { WidgetContext } from './mount';
import { quatToRot, type Quat } from './math/quat';
import { mul3, mulVec3, transpose3, diag3, eigh2, inv2, type Mat3, type Mat2, type Vec3 } from './math/mat';
import { widgetHeader, sliderRow, imagePanel } from './ui/controls';

const BASE = import.meta.env.BASE_URL.endsWith('/')
  ? import.meta.env.BASE_URL
  : `${import.meta.env.BASE_URL}/`;

interface CameraJson {
  W: number[][];
  t: number[];
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

interface TrainedParams {
  iteration: number;
  positions: number[][];
  scales: number[][];
  quaternions: number[][];
  opacities: number[];
  colors: number[][];
}

interface PreparedGaussian {
  position: Vec3;
  cov3d: Mat3;
  color: [number, number, number];
  opacity: number;
}

export function mount(el: HTMLElement, ctx: WidgetContext) {
  const ja = ctx.locale === 'ja';
  widgetHeader(
    el,
    ja
      ? 'インタラクティブ: 学習済みガーデンを自由視点で見る'
      : 'Interactive: fly around the trained garden scene',
    ja
      ? 'この章の学習を実際に実行して得た200個の3Dガウシアンを、本文と同じ射影・合成パイプラインのJS移植で描画しています。'
      : 'These 200 Gaussians were trained with the exact code from this chapter; rendering uses a JS port of the same projection and compositing pipeline.',
  );

  const body = document.createElement('div');
  el.appendChild(body);
  const loading = document.createElement('p');
  loading.style.cssText = 'color: var(--muted); font-size: var(--text-sm);';
  loading.textContent = ja ? '学習済みパラメータを読み込み中…' : 'Loading trained parameters…';
  body.appendChild(loading);

  void init(body, loading, ja).catch(() => {
    loading.textContent = ja
      ? '学習済みパラメータの読み込みに失敗しました。'
      : 'Failed to load trained parameters.';
  });
}

async function init(body: HTMLElement, loading: HTMLElement, ja: boolean) {
  const [paramsRes, camsRes] = await Promise.all([
    fetch(`${BASE}data/ch9/trained-params.json`),
    fetch(`${BASE}data/ch9/cameras.json`),
  ]);
  if (!paramsRes.ok || !camsRes.ok) {
    loading.textContent = ja
      ? '学習済みパラメータは準備中です。'
      : 'Trained parameters are being prepared.';
    return;
  }
  const params = (await paramsRes.json()) as TrainedParams;
  const cams = (await camsRes.json()) as { train: CameraJson[]; test: CameraJson[] };
  loading.remove();

  // --- ガウシアンの事前計算 (Σ3D = R S² Rᵀ) ---
  const gaussians: PreparedGaussian[] = params.positions.map((pos, i) => {
    const R = quatToRot(params.quaternions[i] as Quat);
    const s = params.scales[i];
    const S2 = diag3(s[0] * s[0], s[1] * s[1], s[2] * s[2]);
    const cov3d = mul3(mul3(R, S2), transpose3(R));
    return {
      position: pos as Vec3,
      cov3d,
      color: params.colors[i] as [number, number, number],
      opacity: params.opacities[i],
    };
  });

  const W = cams.train[0].width;
  const H = cams.train[0].height;

  // --- UI ---
  const cols = document.createElement('div');
  cols.className = 'w-columns';
  body.appendChild(cols);

  const left = document.createElement('div');
  left.className = 'w-col';
  cols.appendChild(left);

  const renderFrame = document.createElement('div');
  renderFrame.className = 'w-canvas-frame';
  left.appendChild(renderFrame);
  const renderBox = document.createElement('div');
  renderBox.style.textAlign = 'center';
  renderFrame.appendChild(renderBox);
  const panel = imagePanel(renderBox, W, H, 324);
  const renderCap = document.createElement('p');
  renderCap.style.cssText = 'margin:6px 0 0; font-size: var(--text-xs); color: var(--muted);';
  renderCap.textContent = ja
    ? `レンダリング (${W}×${H}, 200ガウシアン, 学習${params.iteration}イテレーション)`
    : `Rendered (${W}×${H}, 200 Gaussians, ${params.iteration} iterations)`;
  renderBox.appendChild(renderCap);

  const right = document.createElement('div');
  right.className = 'w-col';
  cols.appendChild(right);

  const gtFrame = document.createElement('div');
  gtFrame.className = 'w-canvas-frame';
  right.appendChild(gtFrame);
  const gtBox = document.createElement('div');
  gtBox.style.textAlign = 'center';
  gtFrame.appendChild(gtBox);
  const gtImg = document.createElement('img');
  gtImg.width = 324;
  gtImg.style.imageRendering = 'pixelated';
  gtImg.alt = '';
  gtImg.style.display = 'none';
  gtBox.appendChild(gtImg);
  const gtEmpty = document.createElement('p');
  gtEmpty.style.cssText = 'margin:24px 12px; font-size: var(--text-xs); color: var(--faint);';
  gtEmpty.textContent = ja
    ? 'test視点に合わせると、学習に使っていない実写と比較できます。'
    : 'Snap to a test view to compare with a real photo the model never saw.';
  gtBox.appendChild(gtEmpty);
  const gtCap = document.createElement('p');
  gtCap.style.cssText = 'margin:6px 0 0; font-size: var(--text-xs); color: var(--muted);';
  gtBox.appendChild(gtCap);

  // カメラパス スライダー
  const slider = sliderRow(left, {
    label: ja ? 'カメラパス' : 'camera path',
    min: 0,
    max: cams.train.length - 1,
    step: 0.05,
    value: 0,
    format: (v) => v.toFixed(2),
    onInput: () => scheduleRender(),
  });

  // test 視点スナップボタン
  const testRow = document.createElement('div');
  testRow.style.cssText = 'display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; align-items:center;';
  const testLabel = document.createElement('span');
  testLabel.style.cssText = 'font-size: var(--text-xs); color: var(--muted);';
  testLabel.textContent = ja ? 'test視点と比較:' : 'Compare with test view:';
  testRow.appendChild(testLabel);
  let activeTest: number | null = null;
  const testButtons: HTMLButtonElement[] = [];
  cams.test.forEach((_, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'w-button';
    b.style.padding = '2px 9px';
    b.textContent = String(i + 1);
    b.addEventListener('click', () => {
      activeTest = activeTest === i ? null : i;
      testButtons.forEach((x, j) => x.classList.toggle('primary', j === activeTest));
      updateGt();
      scheduleRender();
    });
    testButtons.push(b);
    testRow.appendChild(b);
  });
  left.appendChild(testRow);

  const perfLine = document.createElement('p');
  perfLine.style.cssText = 'margin:8px 0 0; font-size: var(--text-xs); color: var(--faint);';
  left.appendChild(perfLine);

  function updateGt() {
    if (activeTest === null) {
      gtImg.style.display = 'none';
      gtEmpty.style.display = '';
      gtCap.textContent = '';
    } else {
      gtImg.src = `${BASE}data/ch9/test-views/test-${String(activeTest).padStart(2, '0')}.png`;
      gtImg.style.display = '';
      gtEmpty.style.display = 'none';
      gtCap.textContent = ja
        ? `実写 (test視点 ${activeTest + 1} — 学習には未使用)`
        : `Photo (test view ${activeTest + 1} — never used in training)`;
    }
  }

  // --- カメラ補間 ---
  function interpolateCamera(pos: number): CameraJson {
    const i = Math.min(Math.floor(pos), cams.train.length - 2);
    const f = Math.min(1, Math.max(0, pos - i));
    const a = cams.train[i];
    const b = cams.train[Math.min(i + 1, cams.train.length - 1)];
    const Wm: number[][] = [];
    for (let r = 0; r < 3; r++) {
      Wm.push([0, 1, 2].map((c) => a.W[r][c] * (1 - f) + b.W[r][c] * f));
    }
    // 行を Gram-Schmidt で再直交化 (行 = カメラの x/y/z 軸)
    const rows = Wm as [number[], number[], number[]];
    normalize(rows[0]);
    subProj(rows[1], rows[0]);
    normalize(rows[1]);
    rows[2] = cross(rows[0], rows[1]);
    return {
      W: rows,
      t: a.t.map((v, k) => v * (1 - f) + b.t[k] * f),
      fx: a.fx,
      fy: a.fy,
      cx: a.cx,
      cy: a.cy,
      width: a.width,
      height: a.height,
    };
  }

  function normalize(v: number[]) {
    const n = Math.hypot(v[0], v[1], v[2]) || 1;
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
  }

  function subProj(v: number[], u: number[]) {
    const d = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
    v[0] -= d * u[0];
    v[1] -= d * u[1];
    v[2] -= d * u[2];
  }

  function cross(a: number[], b: number[]): number[] {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }

  // --- レンダラー (projection.py + render_3d の JS 移植) ---
  const image = new Float32Array(W * H * 3);
  const T = new Float32Array(W * H);

  function render(cam: CameraJson) {
    const t0 = performance.now();
    image.fill(0);
    T.fill(1);

    const camW = cam.W as unknown as Mat3;
    const camWT = transpose3(camW);

    interface Projected {
      u: number;
      v: number;
      covInv: Mat2;
      radius: number;
      depth: number;
      color: [number, number, number];
      opacity: number;
    }
    const projected: Projected[] = [];

    for (const g of gaussians) {
      // world → camera
      const pc = mulVec3(camW, g.position);
      pc[0] += cam.t[0];
      pc[1] += cam.t[1];
      pc[2] += cam.t[2];
      const z = pc[2];
      if (z <= 0.1) continue; // カメラ背面カリング (projection.py と同じ)

      const u = (cam.fx * pc[0]) / z + cam.cx;
      const v = (cam.fy * pc[1]) / z + cam.cy;
      const margin = 100;
      if (u < -margin || u > cam.width + margin || v < -margin || v > cam.height + margin) continue;

      // J W Σ Wᵀ Jᵀ (J は 2×3 なので直接展開して計算)
      const j00 = cam.fx / z;
      const j02 = (-cam.fx * pc[0]) / (z * z);
      const j11 = cam.fy / z;
      const j12 = (-cam.fy * pc[1]) / (z * z);

      const M = mul3(mul3(camW, g.cov3d), camWT); // W Σ Wᵀ (3×3)
      // A = J M (2×3)
      const a00 = j00 * M[0][0] + j02 * M[2][0];
      const a01 = j00 * M[0][1] + j02 * M[2][1];
      const a02 = j00 * M[0][2] + j02 * M[2][2];
      const a10 = j11 * M[1][0] + j12 * M[2][0];
      const a11 = j11 * M[1][1] + j12 * M[2][1];
      const a12 = j11 * M[1][2] + j12 * M[2][2];
      // Σ' = A Jᵀ (2×2)
      const cov2d: Mat2 = [
        [a00 * j00 + a02 * j02, a00 * 0 + a01 * j11 + a02 * j12],
        [a10 * j00 + a12 * j02, a11 * j11 + a12 * j12],
      ];

      const { eigvals } = eigh2(cov2d);
      const radius = 3 * Math.sqrt(Math.max(eigvals[0], 1e-8));
      projected.push({
        u,
        v,
        covInv: inv2(cov2d),
        radius,
        depth: z,
        color: g.color,
        opacity: g.opacity,
      });
    }

    projected.sort((p, q) => p.depth - q.depth);

    for (const p of projected) {
      const x0 = Math.max(0, Math.floor(p.u - p.radius));
      const x1 = Math.min(W - 1, Math.ceil(p.u + p.radius));
      const y0 = Math.max(0, Math.floor(p.v - p.radius));
      const y1 = Math.min(H - 1, Math.ceil(p.v + p.radius));
      if (x0 > x1 || y0 > y1) continue;
      const [[ia, ib], [ic, id]] = p.covInv;
      for (let y = y0; y <= y1; y++) {
        const dy = y - p.v;
        for (let x = x0; x <= x1; x++) {
          const dx = x - p.u;
          const q = dx * (ia * dx + ib * dy) + dy * (ic * dx + id * dy);
          if (q > 18) continue; // 3σ相当の打ち切り
          const idx = y * W + x;
          const alpha = Math.min(0.999, p.opacity * Math.exp(-0.5 * q));
          const w = alpha * T[idx];
          image[idx * 3] += w * p.color[0];
          image[idx * 3 + 1] += w * p.color[1];
          image[idx * 3 + 2] += w * p.color[2];
          T[idx] *= 1 - alpha;
        }
      }
    }

    panel.draw(image);
    const ms = performance.now() - t0;
    perfLine.textContent = ja
      ? `${projected.length}/${gaussians.length} ガウシアンを描画 — ${ms.toFixed(1)} ms`
      : `Rendered ${projected.length}/${gaussians.length} Gaussians — ${ms.toFixed(1)} ms`;
  }

  let raf = 0;
  function scheduleRender() {
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      const cam = activeTest !== null ? cams.test[activeTest] : interpolateCamera(slider.value);
      render(cam);
    });
  }

  updateGt();
  scheduleRender();
}
