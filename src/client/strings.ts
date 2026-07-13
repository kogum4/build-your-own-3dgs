// クライアントランタイム用の UI 文字列 (ページの lang 属性で切替)

const dict = {
  ja: {
    run: '▶ 実行',
    runBoot: '▶ 起動して実行',
    running: '実行中…',
    queued: '待機中…',
    reset: 'リセット',
    edited: '編集済み',
    expectedOutput: '期待される出力',
    noOutput: '(出力なし)',
    bootLoading: 'Python環境を準備中',
    bootPackages: 'パッケージを読み込み中',
    ready: 'Python 準備完了',
    envError: 'Python環境のエラー',
    resetEnv: '環境をリセット',
    envResetDone: '環境をリセットしました。各ブロックは再実行してください。',
    errorLabel: 'エラー',
    seconds: '秒',
  },
  en: {
    run: '▶ Run',
    runBoot: '▶ Boot & run',
    running: 'Running…',
    queued: 'Queued…',
    reset: 'Reset',
    edited: 'edited',
    expectedOutput: 'Expected output',
    noOutput: '(no output)',
    bootLoading: 'Preparing Python environment',
    bootPackages: 'Loading packages',
    ready: 'Python ready',
    envError: 'Python environment error',
    resetEnv: 'Reset environment',
    envResetDone: 'Environment was reset. Re-run blocks as needed.',
    errorLabel: 'Error',
    seconds: 's',
  },
} as const;

export type ClientStrings = { [K in keyof (typeof dict)['ja']]: string };

export function getStrings(): ClientStrings {
  const lang = document.documentElement.lang === 'en' ? 'en' : 'ja';
  return dict[lang];
}
