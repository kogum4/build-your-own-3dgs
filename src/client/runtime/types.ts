// メインスレッド ↔ Pyodide Worker の RPC メッセージ型

export type BootPhase = 'loading-pyodide' | 'loading-packages' | 'mounting-files' | 'ready';

export interface BootMessage {
  type: 'boot';
  chapter: string;
  baseUrl: string;
}

export interface ExecMessage {
  type: 'exec';
  id: number;
  code: string;
}

export interface TrainStartMessage {
  type: 'train-start';
  id: number;
  /** 事前に Pyodide FS へ配置するデータファイル */
  files?: { url: string; path: string }[];
  /** 一度だけ実行するセットアップコード */
  setup: string;
  /** 繰り返し評価するチャンク式。Python タプル
   * (done, step, total, loss, h, w, image_bytes|None) を返すこと */
  chunk: string;
}

export interface TrainCancelMessage {
  type: 'train-cancel';
  id: number;
}

export type MainToWorker = BootMessage | ExecMessage | TrainStartMessage | TrainCancelMessage;

export interface StatusMessage {
  type: 'status';
  phase: BootPhase;
  detail?: string;
}

export interface ResultMessage {
  type: 'result';
  id: number;
  ok: boolean;
  stdout: string;
  stderr: string;
  errorMessage?: string;
  /** PNG バイト列 (Transferable) */
  figures: ArrayBuffer[];
  durationMs: number;
}

export interface FatalMessage {
  type: 'fatal';
  message: string;
}

export interface TrainProgressMessage {
  type: 'train-progress';
  id: number;
  step: number;
  total: number;
  loss: number;
  /** H*W*3 の uint8 生画像 (Transferable) */
  image?: ArrayBuffer;
  height: number;
  width: number;
}

export interface TrainDoneMessage {
  type: 'train-done';
  id: number;
  cancelled: boolean;
  error?: string;
}

export type WorkerToMain =
  | StatusMessage
  | ResultMessage
  | FatalMessage
  | TrainProgressMessage
  | TrainDoneMessage;

export interface ExecResult {
  ok: boolean;
  stdout: string;
  stderr: string;
  errorMessage?: string;
  figures: ArrayBuffer[];
  durationMs: number;
}
