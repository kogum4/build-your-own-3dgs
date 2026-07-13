/// <reference lib="webworker" />
// Pyodide 実行 Worker。メインスレッドでは Pyodide を一切動かさない。
// Pyodide 本体は CDN (jsDelivr) から取得し、バージョンは npm パッケージと同期させる。
import { loadPyodide, version as PYODIDE_VERSION, type PyodideInterface } from 'pyodide';
import bootstrapPy from './bootstrap.py?raw';
import type { MainToWorker, WorkerToMain, BootPhase } from './types';

const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// import 名 → Pyodide パッケージ名 (これ以外の外部モジュールは stdlib とみなす)
const PACKAGE_MAP: Record<string, string> = {
  numpy: 'numpy',
  matplotlib: 'matplotlib',
  PIL: 'pillow',
};

let pyodide: PyodideInterface;
let baseUrl = '/';
const loadedPackages = new Set<string>();
/** 章スナップショットのモジュール名 → そのファイルが import するモジュール名一覧 */
const localImports = new Map<string, string[]>();
let matplotlibReady = false;

function post(msg: WorkerToMain, transfer: Transferable[] = []) {
  (self as unknown as Worker).postMessage(msg, transfer);
}

function status(phase: BootPhase, detail?: string) {
  post({ type: 'status', phase, detail });
}

function importsOf(code: string): string[] {
  const names = new Set<string>();
  const re = /^\s*(?:from\s+([A-Za-z_][\w]*)|import\s+([A-Za-z_][\w]*(?:\s*,\s*[A-Za-z_][\w]*)*))/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(code)) !== null) {
    if (m[1]) names.add(m[1]);
    else if (m[2]) m[2].split(',').forEach((n) => names.add(n.trim().split(/\s/)[0]!));
  }
  return [...names];
}

/** コード (と、そこから import されるローカルモジュール) が必要とする外部パッケージをロードする */
async function ensurePackagesFor(code: string) {
  const needed = new Set<string>();
  const seen = new Set<string>();
  const stack = importsOf(code);
  while (stack.length > 0) {
    const mod = stack.pop()!;
    if (seen.has(mod)) continue;
    seen.add(mod);
    const pkg = PACKAGE_MAP[mod];
    if (pkg) needed.add(pkg);
    else if (localImports.has(mod)) stack.push(...(localImports.get(mod) ?? []));
  }
  const toLoad = [...needed].filter((p) => !loadedPackages.has(p));
  if (toLoad.length > 0) {
    status('loading-packages', toLoad.join(', '));
    await pyodide.loadPackage(toLoad);
    toLoad.forEach((p) => loadedPackages.add(p));
  }
  if (loadedPackages.has('matplotlib') && !matplotlibReady) {
    await setupMatplotlib();
  }
}

async function setupMatplotlib() {
  matplotlibReady = true;
  let fontPath: string | null = null;
  try {
    const res = await fetch(`${baseUrl}fonts/NotoSansJP-Regular.ttf`);
    if (res.ok) {
      const buf = new Uint8Array(await res.arrayBuffer());
      pyodide.FS.mkdirTree('/fonts');
      pyodide.FS.writeFile('/fonts/NotoSansJP-Regular.ttf', buf);
      fontPath = '/fonts/NotoSansJP-Regular.ttf';
    }
  } catch {
    // フォントなしでも続行 (日本語ラベルは豆腐になるが機能は損なわない)
  }
  const setup = pyodide.globals.get('_setup_matplotlib');
  try {
    setup(fontPath);
  } finally {
    setup.destroy();
  }
}

async function boot(chapter: string) {
  status('loading-pyodide');
  pyodide = await loadPyodide({ indexURL: INDEX_URL });

  status('loading-packages', 'numpy');
  await pyodide.loadPackage('numpy');
  loadedPackages.add('numpy');

  status('mounting-files');
  const manifestRes = await fetch(`${baseUrl}code/manifest.json`);
  if (!manifestRes.ok) throw new Error(`manifest.json の取得に失敗 (${manifestRes.status})`);
  const manifest = (await manifestRes.json()) as { chapters: Record<string, string[]> };
  const files = manifest.chapters[chapter] ?? [];
  await Promise.all(
    files.map(async (name) => {
      const res = await fetch(`${baseUrl}code/chapter-${chapter}/${name}`);
      if (!res.ok) throw new Error(`${name} の取得に失敗 (${res.status})`);
      const text = await res.text();
      pyodide.FS.writeFile(`/home/pyodide/${name}`, text);
      localImports.set(name.replace(/\.py$/, ''), importsOf(text));
    }),
  );

  pyodide.runPython(bootstrapPy);
  status('ready');
}

function trimTraceback(message: string): string {
  const idx = message.indexOf('File "<exec>"');
  if (idx === -1) return message;
  return 'Traceback (most recent call last):\n  ' + message.slice(idx);
}

async function exec(id: number, code: string) {
  const t0 = performance.now();
  let ok = true;
  let errorMessage: string | undefined;

  try {
    await ensurePackagesFor(code);
  } catch (e) {
    post({
      type: 'result',
      id,
      ok: false,
      stdout: '',
      stderr: '',
      errorMessage: `パッケージの読み込みに失敗しました: ${String(e)}`,
      figures: [],
      durationMs: performance.now() - t0,
    });
    return;
  }

  pyodide.runPython('_capture_start()');
  try {
    await pyodide.runPythonAsync(code);
  } catch (e) {
    ok = false;
    errorMessage = trimTraceback(e instanceof Error ? e.message : String(e));
  }

  const captured = pyodide.runPython('_capture_end()');
  const [stdout, stderr] = captured.toJs() as [string, string];
  captured.destroy();

  let figures: ArrayBuffer[] = [];
  if (matplotlibReady) {
    const figProxy = pyodide.runPython('_capture_figures()');
    const list = figProxy.toJs() as Uint8Array[];
    figProxy.destroy();
    figures = list.map((u8) => u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength) as ArrayBuffer);
  }

  post(
    { type: 'result', id, ok, stdout, stderr, errorMessage, figures, durationMs: performance.now() - t0 },
    figures,
  );
}

// ---------- チャンク学習 (学習ループの進捗報告・ソフトキャンセル) ----------
const cancelledTrainings = new Set<number>();

async function train(msg: { id: number; files?: { url: string; path: string }[]; setup: string; chunk: string }) {
  const { id } = msg;
  try {
    await ensurePackagesFor(msg.setup);
    // データファイルを FS へ配置
    for (const f of msg.files ?? []) {
      const res = await fetch(f.url);
      if (!res.ok) throw new Error(`${f.url} の取得に失敗 (${res.status})`);
      pyodide.FS.writeFile(f.path, new Uint8Array(await res.arrayBuffer()));
    }
    pyodide.runPython('_capture_start()');
    try {
      await pyodide.runPythonAsync(msg.setup);
    } finally {
      pyodide.runPython('_capture_end()');
    }

    // チャンクループ: 各チャンク間でイベントループへ制御を返し、キャンセルを受け付ける
    for (;;) {
      if (cancelledTrainings.has(id)) {
        post({ type: 'train-done', id, cancelled: true });
        return;
      }
      const proxy = await pyodide.runPythonAsync(msg.chunk);
      const [done, step, total, loss, h, w, imageBytes] = proxy.toJs() as [
        boolean,
        number,
        number,
        number,
        number,
        number,
        Uint8Array | undefined,
      ];
      proxy.destroy();
      let image: ArrayBuffer | undefined;
      if (imageBytes) {
        image = imageBytes.buffer.slice(
          imageBytes.byteOffset,
          imageBytes.byteOffset + imageBytes.byteLength,
        ) as ArrayBuffer;
      }
      post(
        { type: 'train-progress', id, step, total, loss, image, height: h, width: w },
        image ? [image] : [],
      );
      if (done) {
        post({ type: 'train-done', id, cancelled: false });
        return;
      }
      // メッセージ処理の機会を作る (キャンセル受付)
      await new Promise((r) => setTimeout(r, 0));
    }
  } catch (e) {
    post({
      type: 'train-done',
      id,
      cancelled: false,
      error: trimTraceback(e instanceof Error ? e.message : String(e)),
    });
  } finally {
    cancelledTrainings.delete(id);
  }
}

self.addEventListener('message', (event: MessageEvent<MainToWorker>) => {
  const msg = event.data;
  if (msg.type === 'boot') {
    baseUrl = msg.baseUrl.endsWith('/') ? msg.baseUrl : `${msg.baseUrl}/`;
    boot(msg.chapter).catch((e) => post({ type: 'fatal', message: String(e) }));
  } else if (msg.type === 'exec') {
    exec(msg.id, msg.code).catch((e) =>
      post({
        type: 'result',
        id: msg.id,
        ok: false,
        stdout: '',
        stderr: '',
        errorMessage: String(e),
        figures: [],
        durationMs: 0,
      }),
    );
  } else if (msg.type === 'train-start') {
    void train(msg);
  } else if (msg.type === 'train-cancel') {
    cancelledTrainings.add(msg.id);
  }
});
