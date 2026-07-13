// メインスレッド側の Pyodide 環境ファサード (シングルトン)。
// Worker の生成・起動・実行キュー・リセットを管理する。
import type { WorkerToMain, ExecResult, BootPhase, TrainProgressMessage } from './types';

export interface TrainProgress {
  step: number;
  total: number;
  loss: number;
  image?: ArrayBuffer;
  width: number;
  height: number;
}

export interface TrainSession {
  cancel: () => void;
  /** 正常終了/キャンセルで resolve、Pythonエラーで reject */
  done: Promise<{ cancelled: boolean }>;
}

export interface TrainOptions {
  files?: { url: string; path: string }[];
  setup: string;
  chunk: string;
  onProgress: (p: TrainProgress) => void;
}

export type EnvState =
  | { kind: 'idle' }
  | { kind: 'booting'; phase: BootPhase; detail?: string; startedAt: number }
  | { kind: 'ready' }
  | { kind: 'running' }
  | { kind: 'error'; message: string };

type StateListener = (state: EnvState) => void;

class PyEnv {
  private worker: Worker | null = null;
  private chapter = '';
  private seq = 0;
  private pending = new Map<number, { resolve: (r: ExecResult) => void }>();
  private trainings = new Map<
    number,
    {
      onProgress: (p: TrainProgress) => void;
      resolve: (r: { cancelled: boolean }) => void;
      reject: (e: Error) => void;
    }
  >();
  private bootPromise: Promise<void> | null = null;
  private bootResolve: (() => void) | null = null;
  private bootReject: ((e: Error) => void) | null = null;
  private runChain: Promise<unknown> = Promise.resolve();
  private state: EnvState = { kind: 'idle' };
  private listeners = new Set<StateListener>();
  private runningCount = 0;

  getState(): EnvState {
    return this.state;
  }

  onState(listener: StateListener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  private setState(state: EnvState) {
    this.state = state;
    this.listeners.forEach((l) => l(state));
  }

  /** Worker を起動して章スナップショットをマウントする (多重起動は無視) */
  ensureBoot(chapter: string): Promise<void> {
    if (this.bootPromise) return this.bootPromise;
    this.chapter = chapter;
    this.setState({ kind: 'booting', phase: 'loading-pyodide', startedAt: performance.now() });

    this.worker = new Worker(new URL('./pyodide.worker.ts', import.meta.url), { type: 'module' });
    this.worker.addEventListener('message', (e: MessageEvent<WorkerToMain>) => this.onMessage(e.data));
    this.worker.addEventListener('error', (e) => {
      this.setState({ kind: 'error', message: e.message || 'Worker error' });
      this.bootReject?.(new Error(e.message || 'Worker error'));
    });

    this.bootPromise = new Promise<void>((resolve, reject) => {
      this.bootResolve = resolve;
      this.bootReject = reject;
    });

    this.worker.postMessage({
      type: 'boot',
      chapter,
      baseUrl: import.meta.env.BASE_URL,
    });
    return this.bootPromise;
  }

  private onMessage(msg: WorkerToMain) {
    if (msg.type === 'status') {
      if (msg.phase === 'ready') {
        this.setState(this.runningCount > 0 ? { kind: 'running' } : { kind: 'ready' });
        this.bootResolve?.();
      } else {
        const startedAt =
          this.state.kind === 'booting' ? this.state.startedAt : performance.now();
        this.setState({ kind: 'booting', phase: msg.phase, detail: msg.detail, startedAt });
      }
    } else if (msg.type === 'result') {
      const entry = this.pending.get(msg.id);
      if (entry) {
        this.pending.delete(msg.id);
        entry.resolve({
          ok: msg.ok,
          stdout: msg.stdout,
          stderr: msg.stderr,
          errorMessage: msg.errorMessage,
          figures: msg.figures,
          durationMs: msg.durationMs,
        });
      }
    } else if (msg.type === 'train-progress') {
      this.trainings.get(msg.id)?.onProgress(msg as TrainProgressMessage);
    } else if (msg.type === 'train-done') {
      const t = this.trainings.get(msg.id);
      if (t) {
        this.trainings.delete(msg.id);
        if (msg.error) t.reject(new Error(msg.error));
        else t.resolve({ cancelled: msg.cancelled });
      }
      if (this.trainings.size === 0 && this.runningCount === 0 && this.state.kind === 'running') {
        this.setState({ kind: 'ready' });
      }
    } else if (msg.type === 'fatal') {
      this.setState({ kind: 'error', message: msg.message });
      this.bootReject?.(new Error(msg.message));
      this.failAllPending(msg.message);
    }
  }

  /** チャンク学習セッションを開始する (進捗コールバック+ソフトキャンセル付き) */
  async startTraining(opts: TrainOptions): Promise<TrainSession> {
    await this.ensureBoot(this.chapter);
    const id = ++this.seq;
    const done = new Promise<{ cancelled: boolean }>((resolve, reject) => {
      this.trainings.set(id, { onProgress: opts.onProgress, resolve, reject });
    });
    this.setState({ kind: 'running' });
    this.worker!.postMessage({
      type: 'train-start',
      id,
      files: opts.files,
      setup: opts.setup,
      chunk: opts.chunk,
    });
    const finalize = () => {
      if (this.trainings.size === 0 && this.runningCount === 0) {
        this.setState({ kind: 'ready' });
      }
    };
    done.then(finalize, finalize);
    return {
      cancel: () => this.worker?.postMessage({ type: 'train-cancel', id }),
      done,
    };
  }

  private failAllPending(message: string) {
    for (const [, t] of this.trainings) t.reject(new Error(message));
    this.trainings.clear();
    for (const [, entry] of this.pending) {
      entry.resolve({
        ok: false,
        stdout: '',
        stderr: '',
        errorMessage: message,
        figures: [],
        durationMs: 0,
      });
    }
    this.pending.clear();
  }

  /** コードを共有名前空間で実行する (直列化される) */
  run(code: string): Promise<ExecResult> {
    const task = this.runChain.then(async () => {
      await this.ensureBoot(this.chapter);
      this.runningCount++;
      this.setState({ kind: 'running' });
      try {
        return await new Promise<ExecResult>((resolve) => {
          const id = ++this.seq;
          this.pending.set(id, { resolve });
          this.worker!.postMessage({ type: 'exec', id, code });
        });
      } finally {
        this.runningCount--;
        if (this.state.kind === 'running' && this.runningCount === 0) {
          this.setState({ kind: 'ready' });
        }
      }
    });
    // チェーンはエラーでも継続させる
    this.runChain = task.catch(() => undefined);
    return task;
  }

  /** アイドル時にバックグラウンドで起動しておく (読んでいる間に準備完了させる) */
  prefetch(chapter: string) {
    this.chapter = chapter;
    const conn = (navigator as { connection?: { saveData?: boolean } }).connection;
    if (conn?.saveData) return; // データセーバー時はプリフェッチしない
    const start = () => this.ensureBoot(chapter).catch(() => undefined);
    if ('requestIdleCallback' in window) {
      requestIdleCallback(start, { timeout: 8000 });
    } else {
      setTimeout(start, 2500);
    }
  }

  /** 環境を破棄して再起動する (暴走コードからの回復手段) */
  reset() {
    this.worker?.terminate();
    this.worker = null;
    this.bootPromise = null;
    this.bootResolve = null;
    this.bootReject = null;
    this.failAllPending('環境がリセットされました');
    this.runChain = Promise.resolve();
    this.setState({ kind: 'idle' });
    if (this.chapter) {
      void this.ensureBoot(this.chapter);
    }
  }
}

export const pyEnv = new PyEnv();
