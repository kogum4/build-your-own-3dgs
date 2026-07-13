// 全16章のメタデータ。サイドバー・章カード・PrevNext・Coming Soon 判定の唯一の真実の源。
// 新しい章を公開するときは status を 'published' に変更し、`pnpm import -- --chapter NN` を実行する。

export type ChapterStatus = 'published' | 'coming-soon';

export interface ChapterMeta {
  number: number;
  /** ゼロ埋め2桁の URL slug ('01' 〜 '16') */
  slug: string;
  status: ChapterStatus;
  title: { ja: string; en: string };
  description: { ja: string; en: string };
}

export const chapterList: ChapterMeta[] = [
  {
    number: 1,
    slug: '01',
    status: 'published',
    title: { ja: '2Dガウシアン', en: '2D Gaussians' },
    description: {
      ja: 'ガウス関数の数式を読み解き、共分散行列で形を操り、NumPyで「ぼんやり光る楕円」を描く出発点の章。',
      en: 'Read the Gaussian formula, shape it with covariance matrices, and render soft glowing ellipses with NumPy.',
    },
  },
  {
    number: 2,
    slug: '02',
    status: 'published',
    title: { ja: 'アルファ合成', en: 'Alpha Compositing' },
    description: {
      ja: '奥行き順に色を重ねるアルファ合成を実装し、累積透過率で「手前が奥を隠す」を再現する。',
      en: 'Implement front-to-back alpha compositing and model occlusion with accumulated transmittance.',
    },
  },
  {
    number: 3,
    slug: '03',
    status: 'published',
    title: { ja: 'スカラー自動微分エンジン', en: 'A Scalar Autograd Engine' },
    description: {
      ja: '小さなValueクラスで計算グラフと連鎖律を実装し、勾配が逆向きに流れる仕組みを理解する。',
      en: 'Build a tiny Value class with a computation graph and the chain rule to see how gradients flow backward.',
    },
  },
  {
    number: 4,
    slug: '04',
    status: 'published',
    title: { ja: 'テンソル自動微分エンジン', en: 'A Tensor Autograd Engine' },
    description: {
      ja: 'スカラーからテンソルへ。行列演算とブロードキャストに対応した自動微分エンジンに拡張する。',
      en: 'From scalars to tensors: extend the autograd engine to matrix operations and broadcasting.',
    },
  },
  {
    number: 5,
    slug: '05',
    status: 'published',
    title: { ja: '2Dガウシアン画像フィッティング', en: 'Fitting Images with 2D Gaussians' },
    description: {
      ja: '自作の自動微分とAdamで数百個の2Dガウシアンを学習させ、1枚の画像を再現する。',
      en: 'Train hundreds of 2D Gaussians with your own autograd and Adam to reconstruct a target image.',
    },
  },
  {
    number: 6,
    slug: '06',
    status: 'published',
    title: { ja: '3Dガウシアン表現', en: 'Representing 3D Gaussians' },
    description: {
      ja: 'ガウシアンを3Dへ。四元数による回転と3D共分散行列で、空間に置ける光のかたまりを定義する。',
      en: 'Take Gaussians into 3D: define blobs of light in space with quaternion rotations and 3D covariance matrices.',
    },
  },
  {
    number: 7,
    slug: '07',
    status: 'published',
    title: { ja: 'カメラモデルと座標変換', en: 'Camera Models and Coordinate Transforms' },
    description: {
      ja: 'ワールド座標からカメラ、そして画像平面へ。透視投影とカメラパラメータを実装する。',
      en: 'From world coordinates to camera to image plane: implement perspective projection and camera parameters.',
    },
  },
  {
    number: 8,
    slug: '08',
    status: 'published',
    title: { ja: 'EWA Splatting', en: 'EWA Splatting' },
    description: {
      ja: '3Dガウシアンをヤコビアンで画像平面の2D楕円に落とし込む、3DGSの心臓部を実装する。',
      en: 'Project 3D Gaussians onto the image plane as 2D ellipses via Jacobians — the heart of 3DGS.',
    },
  },
  {
    number: 9,
    slug: '09',
    status: 'published',
    title: { ja: '実写データからの再構成', en: 'Reconstruction from Real-World Data' },
    description: {
      ja: 'COLMAPの実写データを読み込み、マルチビュー学習で本物のシーンを再構成する。',
      en: 'Load real COLMAP data and reconstruct an actual scene through multi-view training.',
    },
  },
  {
    number: 10,
    slug: '10',
    status: 'coming-soon',
    title: { ja: '球面調和関数', en: 'Spherical Harmonics' },
    description: {
      ja: '球面調和関数で、視線方向によって変わる色を表現する。',
      en: 'Represent view-dependent color with spherical harmonics.',
    },
  },
  {
    number: 11,
    slug: '11',
    status: 'coming-soon',
    title: { ja: 'Tensor演算の拡張', en: 'Extending Tensor Operations' },
    description: {
      ja: 'clampやcumprodなど、ここから先の章で必要になるTensor演算を追加する。',
      en: 'Add clamp, cumprod, and other tensor operations needed by later chapters.',
    },
  },
  {
    number: 12,
    slug: '12',
    status: 'coming-soon',
    title: { ja: 'タイルベースラスタライザ', en: 'A Tile-Based Rasterizer' },
    description: {
      ja: '画面をタイルに分割して高速化する、本家方式のラスタライザを実装する。',
      en: 'Speed up rendering with a tile-based rasterizer, just like the original implementation.',
    },
  },
  {
    number: 13,
    slug: '13',
    status: 'coming-soon',
    title: { ja: 'SSIM損失の追加', en: 'Adding SSIM Loss' },
    description: {
      ja: '構造的類似度SSIMを実装し、L1損失と組み合わせて画質を引き上げる。',
      en: 'Implement structural similarity (SSIM) and combine it with L1 loss to boost image quality.',
    },
  },
  {
    number: 14,
    slug: '14',
    status: 'coming-soon',
    title: { ja: 'Adaptive Density Control', en: 'Adaptive Density Control' },
    description: {
      ja: 'ガウシアンを増やし、消し、分割する。シーンに適応する密度制御を実装する。',
      en: 'Clone, split, and prune Gaussians with adaptive density control.',
    },
  },
  {
    number: 15,
    slug: '15',
    status: 'coming-soon',
    title: { ja: '手動バックワード', en: 'Manual Backward Passes' },
    description: {
      ja: '自動微分を手書きのbackwardに置き換えて、学習を数倍高速化する。',
      en: 'Replace autograd with hand-written backward passes for a multi-fold speedup.',
    },
  },
  {
    number: 16,
    slug: '16',
    status: 'coming-soon',
    title: {
      ja: '統合・評価・インタラクティブビューア',
      en: 'Integration, Evaluation, and an Interactive Viewer',
    },
    description: {
      ja: 'すべてを統合して評価し、インタラクティブビューアで自分の3DGSを動かす。',
      en: 'Integrate everything, evaluate it, and fly through your own 3DGS in an interactive viewer.',
    },
  },
];

export const publishedChapters = chapterList.filter((c) => c.status === 'published');

export function getChapterMeta(slug: string): ChapterMeta | undefined {
  return chapterList.find((c) => c.slug === slug);
}

export function getPrevNext(slug: string): {
  prev: ChapterMeta | undefined;
  next: ChapterMeta | undefined;
} {
  const idx = publishedChapters.findIndex((c) => c.slug === slug);
  return {
    prev: idx > 0 ? publishedChapters[idx - 1] : undefined,
    next: idx >= 0 && idx < publishedChapters.length - 1 ? publishedChapters[idx + 1] : undefined,
  };
}
