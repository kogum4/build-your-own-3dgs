// コードフェンスのメタ文字列 (```python exec file=render.py mode=append など) を
// <pre> の data 属性へ変換する Shiki transformer。
// 契約 (docs/runtime-contract.md):
//   exec        → pre[data-exec][data-mode]  … クライアントが CodeMirror 化して実行可能にする
//   file=NAME   → pre[data-file]             … CSS でファイルタブ表示、exec 時は Pyodide FS へ書込み
//   mode=append → pre[data-mode="append"]    … 既存ファイルへの追記ブロック
//   output      → pre[data-sample-output]    … 「期待される出力」。Run 後に折りたたまれる
export function shikiExecTransformer() {
  return {
    name: 'exec-transformer',
    pre(node) {
      const raw = this.options?.meta?.__raw ?? '';
      const tokens = String(raw).trim().split(/\s+/).filter(Boolean);
      if (tokens.length === 0) return;

      // クラスは Astro 内部の astro-code 付与と衝突するため触らない。
      // スタイル・クライアントの走査はすべて data 属性セレクタで行う。
      const props = (node.properties ??= {});

      let exec = false;
      let output = false;
      let file;
      let mode;
      for (const token of tokens) {
        if (token === 'exec') exec = true;
        else if (token === 'output') output = true;
        else if (token.startsWith('file=')) file = token.slice('file='.length);
        else if (token.startsWith('mode=')) mode = token.slice('mode='.length);
      }

      if (file) props['data-file'] = file;
      if (exec) {
        props['data-exec'] = '';
        props['data-mode'] = mode ?? 'new';
      }
      if (output) props['data-sample-output'] = '';
    },
  };
}
