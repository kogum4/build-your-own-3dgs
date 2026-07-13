// ウィジェット名 → 動的 import の対応表 (章ごとにチャンク分割される)
import type { WidgetMountFn } from './mount';

type WidgetModule = { mount: WidgetMountFn };

export const registry: Record<string, () => Promise<WidgetModule>> = {
  'ch1-covariance': () => import('./ch1-covariance'),
  'ch1-rgb-mixer': () => import('./ch1-rgb-mixer'),
  'ch2-alpha': () => import('./ch2-alpha'),
  'ch3-autograd-graph': () => import('./ch3-autograd-graph'),
  'ch4-broadcast': () => import('./ch4-broadcast'),
  'ch6-ellipsoid': () => import('./ch6-ellipsoid'),
  'ch7-camera-projection': () => import('./ch7-camera-projection'),
  'ch8-ewa': () => import('./ch8-ewa'),
  'ch5-live-training': () => import('./ch5-live-training'),
  'ch9-garden': () => import('./ch9-garden'),
};
