// 章ページ共通クライアントエントリ。
// 実行可能コードブロックのアップグレード、ウィジェットのマウント、Pyodide のプリフェッチを行う。
import '../styles/runtime.css';
import { upgradeExecBlocks, upgradeSampleOutputs } from './exec-block/exec-block';
import { mountStatusPill } from './runtime/status-pill';
import { pyEnv } from './runtime/py-env';
import { mountWidgets } from './widgets/mount';

function init() {
  const main = document.querySelector<HTMLElement>('main[data-chapter]');
  if (!main) return;
  const chapter = main.dataset.chapter!;

  // 期待出力 → details 化 (exec ブロックとの関連付けより先に行う)
  upgradeSampleOutputs(main);

  const hasExec = main.querySelector('pre[data-exec]') !== null;
  if (hasExec) {
    mountStatusPill();
    upgradeExecBlocks(main, chapter);
    // 読んでいる間に裏で Python 環境を準備しておく
    pyEnv.prefetch(chapter);
  }

  // 章別インタラクティブウィジェット
  mountWidgets(main, chapter);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
