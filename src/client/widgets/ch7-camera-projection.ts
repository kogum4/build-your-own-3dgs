// 第7章: カメラモデルと座標変換 (world → camera → 透視投影 → 画像座標)
// yaw / pitch / 距離 d で軌道上のカメラを動かし、原点注視の look-at から
// 本文 camera.py と同じ規約の外部パラメータ (W, t) を構成する。
//   p_c = W p_w + t,   u = fx·X/Z + cx,   v = fy·Y/Z + cy
// 座標系は本文の規約 (右手系・Y 下向き・カメラは +Z 前方 = OpenCV/COLMAP) に一致。
// 検証: yaw=30°, pitch=0, d=5 のとき W が project_cube.py の R_y30 と一致し、
// 投影結果 (fx=200, cx=cy=150) も本文の出力表と一致することを確認済み。
import type { WidgetContext } from './mount';
import type { Mat3, Vec3 } from './math/mat';
import { mulVec3, transpose3 } from './math/mat';
import { sliderRow, canvasPanel, matrixCard, widgetHeader, jsReimplNote } from './ui/controls';

// --- 画像 (仮想センサー) の解像度と主点: camera.py の width/height/cx/cy に対応 ---
const IMG_W = 240;
const IMG_H = 180;
const CX = IMG_W / 2;
const CY = IMG_H / 2;
const Z_NEAR = 0.1; // これ以下の深度 (カメラ後方・至近) の頂点はクリップ

// 立方体 (一辺2・中心原点)。project_cube.py と同じ頂点順・辺定義
const VERTS: readonly Vec3[] = [
  [-1, -1, -1],
  [1, -1, -1],
  [1, 1, -1],
  [-1, 1, -1],
  [-1, -1, 1],
  [1, -1, 1],
  [1, 1, 1],
  [-1, 1, 1],
];
const EDGES: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 0],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 4],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7],
];

function norm3(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross3(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/**
 * 原点注視の look-at から world→camera 回転 W を作る。
 * W の行 = カメラの X (右) / Y (下) / Z (前方 = 光軸) 軸の世界座標表現。
 * 本書は世界もカメラも Y 下向き規約なので、世界の「下」= +Y をヒントに直交基底を組む。
 * (yaw=0, pitch=0 でカメラが (0,0,-d) にあるとき W = I になり、本文の正面カメラと一致)
 */
function lookAtW(camPos: Vec3): Mat3 {
  const z = norm3([-camPos[0], -camPos[1], -camPos[2]]); // 光軸 +Z: カメラ → 原点
  const down: Vec3 = [0, 1, 0]; // 世界の下方向 (Y 下向き規約)
  const x = norm3(cross3(down, z)); // カメラ +X (右)
  const y = cross3(z, x); // カメラ +Y (下)
  return [
    [x[0], x[1], x[2]],
    [y[0], y[1], y[2]],
    [z[0], z[1], z[2]],
  ];
}

function line(g: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) {
  g.beginPath();
  g.moveTo(x1, y1);
  g.lineTo(x2, y2);
  g.stroke();
}

export function mount(el: HTMLElement, ctx: WidgetContext) {
  const ja = ctx.locale === 'ja';
  widgetHeader(
    el,
    ja
      ? 'インタラクティブ: カメラ軌道と透視投影'
      : 'Interactive: camera orbit and perspective projection',
    jsReimplNote(ctx.locale),
  );

  const cols = document.createElement('div');
  cols.className = 'w-columns';
  el.appendChild(cols);

  // --- 左: スライダー + 外部パラメータ表示 ---
  const left = document.createElement('div');
  left.className = 'w-col';
  cols.appendChild(left);

  const state = { yawDeg: 30, pitchDeg: 20, dist: 6, f: 120 };

  sliderRow(left, {
    label: 'yaw',
    min: -180,
    max: 180,
    step: 1,
    value: state.yawDeg,
    format: (v) => `${v.toFixed(0)}°`,
    onInput: (v) => {
      state.yawDeg = v;
      update();
    },
  });
  sliderRow(left, {
    label: 'pitch',
    min: -60,
    max: 60,
    step: 1,
    value: state.pitchDeg,
    format: (v) => `${v.toFixed(0)}°`,
    onInput: (v) => {
      state.pitchDeg = v;
      update();
    },
  });
  sliderRow(left, {
    label: ja ? 'd (距離)' : 'd (distance)',
    min: 3,
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
    min: 60,
    max: 400,
    step: 1,
    value: state.f,
    format: (v) => `${v.toFixed(0)} px`,
    onInput: (v) => {
      state.f = v;
      update();
    },
  });

  const matrixRow = document.createElement('div');
  matrixRow.className = 'w-matrix-row';
  left.appendChild(matrixRow);
  const mW = matrixCard(matrixRow, 'W (world→camera)', 3, 3);
  const mT = matrixCard(matrixRow, 't', 3, 1);
  const mPos = matrixCard(matrixRow, ja ? 'カメラ位置 = −Wᵀt' : 'camera pos = −Wᵀt', 3, 1);

  // --- 中央: 俯瞰図 (XZ 平面を上から) ---
  const mid = document.createElement('div');
  mid.className = 'w-col';
  cols.appendChild(mid);
  const midFrame = document.createElement('div');
  midFrame.className = 'w-canvas-frame';
  mid.appendChild(midFrame);
  const { ctx: g1, width: TW, height: TH } = canvasPanel(midFrame, { width: 300, height: 300 });
  const midCaption = document.createElement('p');
  midCaption.className = 'w-note';
  midCaption.style.textAlign = 'center';
  midCaption.textContent = ja
    ? '俯瞰図: XZ平面を上から (pitch は非反映)'
    : 'Top view: X–Z plane from above (pitch not shown)';
  mid.appendChild(midCaption);

  // --- 右: カメラから見た画像平面 ---
  const right = document.createElement('div');
  right.className = 'w-col';
  cols.appendChild(right);
  const rightFrame = document.createElement('div');
  rightFrame.className = 'w-canvas-frame';
  right.appendChild(rightFrame);
  const { ctx: g2, width: IW, height: IH } = canvasPanel(rightFrame, { width: 300, height: 225 });
  const imgCaption = document.createElement('p');
  imgCaption.className = 'w-note';
  imgCaption.style.textAlign = 'center';
  right.appendChild(imgCaption);

  function drawTopView(yaw: number) {
    const viewR = state.dist + 3; // 表示半径 (カメラ + 視錐台が収まる)
    const toPx = (x: number, z: number): [number, number] => [
      TW / 2 + (x / viewR) * (TW / 2),
      TH / 2 - (z / viewR) * (TH / 2),
    ];

    g1.clearRect(0, 0, TW, TH);

    // グリッド (2 単位刻み) と軸
    g1.strokeStyle = '#eef1f4';
    g1.lineWidth = 1;
    for (let v = 2; v <= viewR; v += 2) {
      for (const sgn of [v, -v]) {
        const [gx] = toPx(sgn, 0);
        const [, gz] = toPx(0, sgn);
        line(g1, gx, 0, gx, TH);
        line(g1, 0, gz, TW, gz);
      }
    }
    const [ox, oy] = toPx(0, 0);
    g1.strokeStyle = '#c9d1d9';
    line(g1, 0, oy, TW, oy);
    line(g1, ox, 0, ox, TH);
    g1.fillStyle = '#8b949e';
    g1.font = '10px system-ui, sans-serif';
    g1.fillText('+X', TW - 18, oy - 5);
    g1.fillText('+Z', ox + 5, 12);

    // カメラ位置 (俯瞰図は yaw と距離のみ反映)
    const camX = state.dist * Math.sin(yaw);
    const camZ = -state.dist * Math.cos(yaw);
    const [px, py] = toPx(camX, camZ);

    // 視錐台: 水平半画角 = atan((画像幅/2) / fx)
    const halfFov = Math.atan(IMG_W / 2 / state.f);
    const phi = Math.atan2(-camZ, -camX); // 前方 (原点方向) の XZ 面内角度
    const L = state.dist + 2;
    const STEPS = 24;
    g1.beginPath();
    g1.moveTo(px, py);
    for (let i = 0; i <= STEPS; i++) {
      const a = phi - halfFov + (2 * halfFov * i) / STEPS;
      const [qx, qy] = toPx(camX + L * Math.cos(a), camZ + L * Math.sin(a));
      g1.lineTo(qx, qy);
    }
    g1.closePath();
    g1.fillStyle = 'rgba(9, 105, 218, 0.08)';
    g1.fill();
    g1.strokeStyle = 'rgba(9, 105, 218, 0.45)';
    g1.lineWidth = 1;
    g1.stroke();

    // 視線 (カメラ → 原点、破線)
    g1.save();
    g1.setLineDash([4, 4]);
    g1.strokeStyle = '#c9d1d9';
    line(g1, px, py, ox, oy);
    g1.restore();

    // 立方体 (上面の正方形 x, z ∈ [-1, 1])
    const [ax, ay] = toPx(-1, 1);
    const [bx, by] = toPx(1, -1);
    g1.beginPath();
    g1.rect(ax, ay, bx - ax, by - ay);
    g1.fillStyle = 'rgba(9, 105, 218, 0.15)';
    g1.fill();
    g1.strokeStyle = 'rgba(9, 105, 218, 0.9)';
    g1.lineWidth = 1.5;
    g1.stroke();

    // カメラ (点 + 向き三角形)
    const sAng = Math.atan2(oy - py, ox - px); // 画面上で原点を向く角度
    g1.fillStyle = '#24292f';
    g1.beginPath();
    g1.arc(px, py, 4, 0, Math.PI * 2);
    g1.fill();
    g1.beginPath();
    g1.moveTo(px + 16 * Math.cos(sAng), py + 16 * Math.sin(sAng));
    g1.lineTo(px + 7 * Math.cos(sAng + 2.4), py + 7 * Math.sin(sAng + 2.4));
    g1.lineTo(px + 7 * Math.cos(sAng - 2.4), py + 7 * Math.sin(sAng - 2.4));
    g1.closePath();
    g1.fill();
  }

  function drawImageView(W: Mat3, t: Vec3) {
    const s = IW / IMG_W; // 画像ピクセル → 表示ピクセル
    g2.clearRect(0, 0, IW, IH);
    g2.fillStyle = '#fff';
    g2.fillRect(0, 0, IW, IH);

    // 画像枠と主点 (cx, cy) の十字 (薄く)
    g2.strokeStyle = '#d0d7de';
    g2.lineWidth = 1;
    g2.strokeRect(0.5, 0.5, IW - 1, IH - 1);
    const pcx = CX * s;
    const pcy = CY * s;
    g2.strokeStyle = 'rgba(140, 149, 159, 0.7)';
    line(g2, pcx - 8, pcy, pcx + 8, pcy);
    line(g2, pcx, pcy - 8, pcx, pcy + 8);

    // world → camera → 透視投影 (camera.py と同一の式)
    const pc = VERTS.map((p): Vec3 => {
      const q = mulVec3(W, p); // W @ p_w
      return [q[0] + t[0], q[1] + t[1], q[2] + t[2]]; // + t
    });
    const uv = pc.map((q): [number, number] => [
      state.f * (q[0] / q[2]) + CX, // u = fx·X/Z + cx
      state.f * (q[1] / q[2]) + CY, // v = fy·Y/Z + cy
    ]);
    const zs = pc.map((q) => q[2]);
    const zMin = Math.min(...zs);
    const zMax = Math.max(...zs);

    // 12 辺のワイヤーフレーム (近い辺ほど濃く)。カメラ後方の頂点を含む辺はクリップ
    g2.lineWidth = 2;
    for (const [i, j] of EDGES) {
      if (zs[i] <= Z_NEAR || zs[j] <= Z_NEAR) continue;
      const zAvg = (zs[i] + zs[j]) / 2;
      const nearness = zMax > zMin ? 1 - (zAvg - zMin) / (zMax - zMin) : 1;
      g2.strokeStyle = `rgba(9, 105, 218, ${(0.35 + 0.55 * nearness).toFixed(3)})`;
      line(g2, uv[i][0] * s, uv[i][1] * s, uv[j][0] * s, uv[j][1] * s);
    }
    g2.fillStyle = '#cf222e';
    for (let i = 0; i < VERTS.length; i++) {
      if (zs[i] <= Z_NEAR) continue;
      g2.beginPath();
      g2.arc(uv[i][0] * s, uv[i][1] * s, 2.5, 0, Math.PI * 2);
      g2.fill();
    }
  }

  function update() {
    const yaw = (state.yawDeg * Math.PI) / 180;
    const pitch = (state.pitchDeg * Math.PI) / 180;

    // 軌道上のカメラ位置 (世界座標)。pitch 正 = 上空側 (Y 下向き規約なので -Y)
    const camPos: Vec3 = [
      state.dist * Math.cos(pitch) * Math.sin(yaw),
      -state.dist * Math.sin(pitch),
      -state.dist * Math.cos(pitch) * Math.cos(yaw),
    ];
    const W = lookAtW(camPos);
    const Wc = mulVec3(W, camPos);
    const t: Vec3 = [-Wc[0], -Wc[1], -Wc[2]]; // p_c = W(p_w − C) より t = −W C

    // 表示用: 本文 7.5 節の逆算式 カメラ位置 = −Wᵀ t (camPos と一致するはず)
    const back = mulVec3(transpose3(W), t);

    mW.update(W);
    mT.update([[t[0]], [t[1]], [t[2]]]);
    mPos.update([[-back[0]], [-back[1]], [-back[2]]]);

    drawTopView(yaw);
    drawImageView(W, t);

    const hfovDeg = ((2 * Math.atan(IMG_W / 2 / state.f)) * 180) / Math.PI;
    imgCaption.textContent = ja
      ? `画像平面 ${IMG_W}×${IMG_H} px ・ 水平画角 ${hfovDeg.toFixed(1)}°`
      : `Image plane ${IMG_W}×${IMG_H} px · horizontal FOV ${hfovDeg.toFixed(1)}°`;
  }

  update();
}
