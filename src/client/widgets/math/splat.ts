// ガウシアン評価とレンダリング (第1〜2章の render.py を JS に移植したもの)
import { inv2, type Mat2, type Vec2 } from './mat';

export interface Gaussian2DLike {
  mean: Vec2; // ピクセル座標
  covariance: Mat2;
  color: [number, number, number]; // 0..1
  opacity: number; // 0..1
  depth?: number;
}

/**
 * 1つのガウシアンを全ピクセルで評価して Float32Array (H*W) を返す。
 * exp(-0.5 * d^T Σ^{-1} d)
 */
export function evaluateGaussian(
  g: Gaussian2DLike,
  W: number,
  H: number,
  out?: Float32Array,
): Float32Array {
  const vals = out ?? new Float32Array(W * H);
  const inv = inv2(g.covariance);
  const [a, b] = inv[0];
  const [c, d] = inv[1];
  const [mx, my] = g.mean;
  let i = 0;
  for (let y = 0; y < H; y++) {
    const dy = y - my;
    for (let x = 0; x < W; x++) {
      const dx = x - mx;
      // d^T Σ^{-1} d (Σ^{-1} は対称なので b ≒ c)
      const q = dx * (a * dx + b * dy) + dy * (c * dx + d * dy);
      vals[i++] = Math.exp(-0.5 * q);
    }
  }
  return vals;
}

/** 加重和レンダリング (第1章 render_gaussians_weighted_sum) → RGB Float32Array (H*W*3) */
export function renderWeightedSum(
  gaussians: Gaussian2DLike[],
  W: number,
  H: number,
): Float32Array {
  const image = new Float32Array(W * H * 3);
  const buf = new Float32Array(W * H);
  for (const g of gaussians) {
    evaluateGaussian(g, W, H, buf);
    const [r, gr, b] = g.color;
    for (let i = 0; i < W * H; i++) {
      const w = g.opacity * buf[i];
      image[i * 3] += w * r;
      image[i * 3 + 1] += w * gr;
      image[i * 3 + 2] += w * b;
    }
  }
  // クリップ
  for (let i = 0; i < image.length; i++) image[i] = Math.min(1, image[i]);
  return image;
}

/** front-to-back アルファ合成 (第2章 render_gaussians_alpha_composite) */
export function renderAlphaComposite(
  gaussians: Gaussian2DLike[],
  W: number,
  H: number,
): { image: Float32Array; transmittance: Float32Array } {
  const sorted = [...gaussians].sort((p, q) => (p.depth ?? 0) - (q.depth ?? 0));
  const image = new Float32Array(W * H * 3);
  const T = new Float32Array(W * H).fill(1);
  const buf = new Float32Array(W * H);
  for (const g of sorted) {
    evaluateGaussian(g, W, H, buf);
    const [r, gr, b] = g.color;
    for (let i = 0; i < W * H; i++) {
      const alpha = Math.min(0.999, g.opacity * buf[i]);
      const w = alpha * T[i];
      image[i * 3] += w * r;
      image[i * 3 + 1] += w * gr;
      image[i * 3 + 2] += w * b;
      T[i] *= 1 - alpha;
    }
  }
  return { image, transmittance: T };
}

/** 1ピクセル分のアルファ合成の途中経過 (第2章の T 減衰可視化用) */
export function compositeSteps(
  gaussians: Gaussian2DLike[],
  px: number,
  py: number,
): {
  gaussian: Gaussian2DLike;
  alpha: number;
  weight: number; // alpha * T (このガウシアンの寄与)
  tBefore: number;
  tAfter: number;
}[] {
  const sorted = [...gaussians].sort((p, q) => (p.depth ?? 0) - (q.depth ?? 0));
  const steps = [];
  let T = 1;
  for (const g of sorted) {
    const inv = inv2(g.covariance);
    const dx = px - g.mean[0];
    const dy = py - g.mean[1];
    const q = dx * (inv[0][0] * dx + inv[0][1] * dy) + dy * (inv[1][0] * dx + inv[1][1] * dy);
    const alpha = Math.min(0.999, g.opacity * Math.exp(-0.5 * q));
    const weight = alpha * T;
    steps.push({ gaussian: g, alpha, weight, tBefore: T, tAfter: T * (1 - alpha) });
    T *= 1 - alpha;
  }
  return steps;
}
