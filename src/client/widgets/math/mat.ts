// 小さな固定サイズ行列演算 (本文の数式をそのまま JS に写したもの)

export type Mat2 = [[number, number], [number, number]];
export type Mat3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number],
];
export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

export function rot2(theta: number): Mat2 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [c, -s],
    [s, c],
  ];
}

/** Σ = R Λ R^T (Λ = diag(sx^2, sy^2)) */
export function covariance2d(sigmaX: number, sigmaY: number, theta: number): Mat2 {
  const [[c, ms], [s, c2]] = rot2(theta);
  // R Λ
  const a = c * sigmaX * sigmaX;
  const b = ms * sigmaY * sigmaY;
  const d = s * sigmaX * sigmaX;
  const e = c2 * sigmaY * sigmaY;
  // (R Λ) R^T
  return [
    [a * c + b * ms, a * s + b * c2],
    [d * c + e * ms, d * s + e * c2],
  ];
}

export function mul2(A: Mat2, B: Mat2): Mat2 {
  return [
    [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
    [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]],
  ];
}

export function transpose2(A: Mat2): Mat2 {
  return [
    [A[0][0], A[1][0]],
    [A[0][1], A[1][1]],
  ];
}

export function inv2(A: Mat2): Mat2 {
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  const d = det === 0 ? 1e-12 : det;
  return [
    [A[1][1] / d, -A[0][1] / d],
    [-A[1][0] / d, A[0][0] / d],
  ];
}

/**
 * 対称 2x2 行列の固有値分解 (閉形式)。
 * 戻り値は固有値の降順で、eigvecs[i] が eigvals[i] に対応する単位固有ベクトル。
 */
export function eigh2(S: Mat2): { eigvals: [number, number]; eigvecs: [Vec2, Vec2] } {
  const a = S[0][0];
  const b = S[0][1];
  const d = S[1][1];
  const tr = a + d;
  const diff = a - d;
  const disc = Math.sqrt(diff * diff + 4 * b * b);
  const l1 = (tr + disc) / 2;
  const l2 = (tr - disc) / 2;

  let v1: Vec2;
  if (Math.abs(b) > 1e-12) {
    v1 = norm2([l1 - d, b]);
  } else {
    v1 = a >= d ? [1, 0] : [0, 1];
  }
  const v2: Vec2 = [-v1[1], v1[0]];
  return { eigvals: [l1, l2], eigvecs: [v1, v2] };
}

function norm2(v: Vec2): Vec2 {
  const n = Math.hypot(v[0], v[1]) || 1;
  return [v[0] / n, v[1] / n];
}

// ---------- 3x3 ----------

export function mul3(A: Mat3, B: Mat3): Mat3 {
  const C: Mat3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j];
  return C;
}

export function transpose3(A: Mat3): Mat3 {
  return [
    [A[0][0], A[1][0], A[2][0]],
    [A[0][1], A[1][1], A[2][1]],
    [A[0][2], A[1][2], A[2][2]],
  ];
}

export function mulVec3(A: Mat3, v: Vec3): Vec3 {
  return [
    A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
    A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
    A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
  ];
}

export function diag3(x: number, y: number, z: number): Mat3 {
  return [
    [x, 0, 0],
    [0, y, 0],
    [0, 0, z],
  ];
}

/** Y軸回転行列 */
export function rotY(theta: number): Mat3 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ];
}

/** X軸回転行列 */
export function rotX(theta: number): Mat3 {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
}
