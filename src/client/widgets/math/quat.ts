// 四元数 → 回転行列 (第6章 gaussian3d.py の quat_to_rotmat と同じ式)
import type { Mat3 } from './mat';

export type Quat = [number, number, number, number]; // (w, x, y, z)

export function normalizeQuat(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/** 回転軸 (単位ベクトル) と回転角から四元数を作る */
export function axisAngleToQuat(axis: [number, number, number], angle: number): Quat {
  const half = angle / 2;
  const s = Math.sin(half);
  const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  return [Math.cos(half), (axis[0] / n) * s, (axis[1] / n) * s, (axis[2] / n) * s];
}

export function quatToRot(q: Quat): Mat3 {
  const [w, x, y, z] = normalizeQuat(q);
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}
