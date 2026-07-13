// 第6章: 3Dガウシアン楕円体エクスプローラ
// スケール (s_x, s_y, s_z) と回転軸+回転角から四元数 q を作り、
// R = quat_to_rotmat(q)、Σ = R S Sᵀ Rᵀ を計算して楕円体をワイヤーフレーム描画する。
// (四元数→回転行列の式は gaussian3d.py の quaternion_to_rotation_matrix と同一)
import type { WidgetContext } from './mount';
import { mul3, transpose3, mulVec3, diag3, rotY, rotX, type Mat3, type Vec3 } from './math/mat';
import { normalizeQuat, axisAngleToQuat, quatToRot } from './math/quat';
import { sliderRow, canvasPanel, matrixCard, widgetHeader, jsReimplNote } from './ui/controls';

const DEG = Math.PI / 180;

export function mount(el: HTMLElement, ctx: WidgetContext): void {
  const ja = ctx.locale === 'ja';
  widgetHeader(
    el,
    ja
      ? 'インタラクティブ: スケール + 四元数で決まる3D楕円体'
      : 'Interactive: 3D ellipsoid from scale + quaternion',
    jsReimplNote(ctx.locale),
  );

  const cols = document.createElement('div');
  cols.className = 'w-columns';
  el.appendChild(cols);

  // --- 左: スライダー + 数値パネル ---
  const left = document.createElement('div');
  left.className = 'w-col';
  cols.appendChild(left);

  const state = {
    sx: 1.4,
    sy: 0.7,
    sz: 0.4,
    phiDeg: 40, // 回転軸の方位角 (x-z 平面内、y 軸周り)
    psiDeg: 30, // 回転軸の仰角
    thetaDeg: 50, // 回転角
  };

  const scaleDefs = [
    { key: 'sx' as const, label: 's_x' },
    { key: 'sy' as const, label: 's_y' },
    { key: 'sz' as const, label: 's_z' },
  ];
  for (const def of scaleDefs) {
    sliderRow(left, {
      label: def.label,
      min: 0.2,
      max: 2.0,
      step: 0.05,
      value: state[def.key],
      format: (v) => v.toFixed(2),
      onInput: (v) => {
        state[def.key] = v;
        update();
      },
    });
  }
  sliderRow(left, {
    label: ja ? 'φ 方位角' : 'φ azimuth',
    min: 0,
    max: 360,
    step: 1,
    value: state.phiDeg,
    format: (v) => `${v.toFixed(0)}°`,
    onInput: (v) => {
      state.phiDeg = v;
      update();
    },
  });
  sliderRow(left, {
    label: ja ? 'ψ 仰角' : 'ψ elevation',
    min: -90,
    max: 90,
    step: 1,
    value: state.psiDeg,
    format: (v) => `${v.toFixed(0)}°`,
    onInput: (v) => {
      state.psiDeg = v;
      update();
    },
  });
  sliderRow(left, {
    label: ja ? 'θ 回転角' : 'θ angle',
    min: 0,
    max: 360,
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
  const mQ = matrixCard(matrixRow, ja ? 'q = (w, x, y, z) 正規化済み' : 'q = (w, x, y, z), normalized', 1, 4);
  const mSigma = matrixCard(matrixRow, 'Σ = R S Sᵀ Rᵀ', 3, 3);

  // --- 右: Canvas (オービット視点) ---
  const right = document.createElement('div');
  right.className = 'w-col';
  cols.appendChild(right);
  const frame = document.createElement('div');
  frame.className = 'w-canvas-frame';
  right.appendChild(frame);
  const { canvas, ctx: g, width: CW, height: CH } = canvasPanel(frame, { width: 340, height: 340 });
  const hint = document.createElement('p');
  hint.className = 'w-note';
  hint.textContent = ja ? 'ドラッグで視点を回転できます。' : 'Drag to orbit the view.';
  right.appendChild(hint);

  // --- 単位球のワイヤーフレーム (緯線9本 + 経線12本) をあらかじめ生成 ---
  const NLAT = 9;
  const NLON = 12;
  const spherePolys: Vec3[][] = [];
  for (let i = 1; i <= NLAT; i++) {
    // 緯線: 高さ一定の円
    const lat = -Math.PI / 2 + (Math.PI * i) / (NLAT + 1);
    const r = Math.cos(lat);
    const y = Math.sin(lat);
    const pts: Vec3[] = [];
    for (let k = 0; k <= 48; k++) {
      const t = (2 * Math.PI * k) / 48;
      pts.push([r * Math.cos(t), y, r * Math.sin(t)]);
    }
    spherePolys.push(pts);
  }
  for (let j = 0; j < NLON; j++) {
    // 経線: 極から極への半円
    const lon = (2 * Math.PI * j) / NLON;
    const cl = Math.cos(lon);
    const sl = Math.sin(lon);
    const pts: Vec3[] = [];
    for (let k = 0; k <= 24; k++) {
      const lat = -Math.PI / 2 + (Math.PI * k) / 24;
      const r = Math.cos(lat);
      pts.push([r * cl, Math.sin(lat), r * sl]);
    }
    spherePolys.push(pts);
  }

  // --- 簡易透視投影 (カメラ距離 6、注視点 = 原点) ---
  const CAM_DIST = 6;
  const FOCAL = 380;
  let viewYaw = -0.65;
  let viewPitch = 0.38;

  function viewMatrix(): Mat3 {
    return mul3(rotX(viewPitch), rotY(viewYaw));
  }

  /** ビュー座標 → 画面座標 + カメラからの距離 */
  function project(pv: Vec3): { x: number; y: number; depth: number } {
    const depth = CAM_DIST - pv[2];
    const s = FOCAL / depth;
    return { x: CW / 2 + pv[0] * s, y: CH / 2 - pv[1] * s, depth };
  }

  /** 奥の線ほど薄く (depth ∈ [CAM_DIST-2.4, CAM_DIST+2.4] を alpha に対応付け) */
  function alphaFor(depth: number): number {
    const t = (depth - (CAM_DIST - 2.4)) / 4.8;
    return Math.max(0.15, Math.min(0.95, 0.95 - 0.72 * t));
  }

  function drawSeg(a: Vec3, b: Vec3, rgb: string, lineWidth: number): void {
    const pa = project(a);
    const pb = project(b);
    const alpha = alphaFor((pa.depth + pb.depth) / 2);
    g.strokeStyle = `rgba(${rgb}, ${alpha.toFixed(3)})`;
    g.lineWidth = lineWidth;
    g.beginPath();
    g.moveTo(pa.x, pa.y);
    g.lineTo(pb.x, pb.y);
    g.stroke();
  }

  function drawAxisLabel(pv: Vec3, text: string, color: string): void {
    const p = project(pv);
    g.fillStyle = color;
    g.font = '12px monospace';
    g.fillText(text, p.x + 4, p.y - 4);
  }

  /** 回転軸の単位ベクトル n(φ, ψ) — 方位角は x-z 平面 (y-up)、仰角は y 方向 */
  function rotationAxis(): Vec3 {
    const phi = state.phiDeg * DEG;
    const psi = state.psiDeg * DEG;
    return [Math.cos(psi) * Math.cos(phi), Math.sin(psi), Math.cos(psi) * Math.sin(phi)];
  }

  // 現在の変形行列 A = R·S (draw() が参照)
  let A: Mat3 = diag3(1, 1, 1);

  function draw(): void {
    g.clearRect(0, 0, CW, CH);
    const V = viewMatrix();

    // ワールド座標軸 x/y/z (赤/緑/青)
    const axes: { dir: Vec3; rgb: string; color: string; label: string }[] = [
      { dir: [1, 0, 0], rgb: '207, 34, 46', color: '#cf222e', label: 'x' },
      { dir: [0, 1, 0], rgb: '26, 127, 55', color: '#1a7f37', label: 'y' },
      { dir: [0, 0, 1], rgb: '9, 105, 218', color: '#0969da', label: 'z' },
    ];
    for (const ax of axes) {
      const tip: Vec3 = [ax.dir[0] * 2.6, ax.dir[1] * 2.6, ax.dir[2] * 2.6];
      const neg: Vec3 = [-tip[0] * 0.5, -tip[1] * 0.5, -tip[2] * 0.5];
      drawSeg(mulVec3(V, neg), mulVec3(V, [0, 0, 0]), ax.rgb, 0.6);
      drawSeg(mulVec3(V, [0, 0, 0]), mulVec3(V, tip), ax.rgb, 1.4);
      drawAxisLabel(mulVec3(V, tip), ax.label, ax.color);
    }

    // 回転軸 n (破線・オレンジ)
    const n = rotationAxis();
    const nPos: Vec3 = [n[0] * 2.4, n[1] * 2.4, n[2] * 2.4];
    const nNeg: Vec3 = [-nPos[0], -nPos[1], -nPos[2]];
    g.setLineDash([5, 4]);
    drawSeg(mulVec3(V, nNeg), mulVec3(V, [0, 0, 0]), '188, 76, 0', 1.2);
    drawSeg(mulVec3(V, [0, 0, 0]), mulVec3(V, nPos), '188, 76, 0', 1.2);
    g.setLineDash([]);
    drawAxisLabel(mulVec3(V, nPos), ja ? '回転軸 n' : 'axis n', '#bc4c00');

    // 楕円体ワイヤーフレーム: 単位球の各点を A = R·S で変形 → ビュー変換
    const M = mul3(V, A);
    for (const poly of spherePolys) {
      let prev = mulVec3(M, poly[0]);
      for (let k = 1; k < poly.length; k++) {
        const cur = mulVec3(M, poly[k]);
        drawSeg(prev, cur, '102, 57, 186', 1.1);
        prev = cur;
      }
    }
  }

  function update(): void {
    // gaussian3d.py と同じ手順: q を正規化 → R → S = diag(s) → Σ = (RS)(RS)ᵀ
    const q = normalizeQuat(axisAngleToQuat(rotationAxis(), state.thetaDeg * DEG));
    const R = quatToRot(q);
    const S = diag3(state.sx, state.sy, state.sz);
    A = mul3(R, S); // RS
    const Sigma = mul3(A, transpose3(A)); // (RS)(RS)ᵀ = R S Sᵀ Rᵀ

    mQ.update([[q[0], q[1], q[2], q[3]]]);
    mSigma.update(Sigma);
    draw();
  }

  // --- マウス / タッチのオービット操作 (Pointer Events) ---
  canvas.style.touchAction = 'none';
  canvas.style.cursor = 'grab';
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener('pointerdown', (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
    e.preventDefault();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    viewYaw += (e.clientX - lastX) * 0.01;
    viewPitch += (e.clientY - lastY) * 0.01;
    viewPitch = Math.max(-1.4, Math.min(1.4, viewPitch));
    lastX = e.clientX;
    lastY = e.clientY;
    draw();
  });
  const endDrag = () => {
    dragging = false;
    canvas.style.cursor = 'grab';
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  update();
}
