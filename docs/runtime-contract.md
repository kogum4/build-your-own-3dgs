# ランタイム契約 (Markdown → DOM → クライアントJS)

サイト骨格 (Astro/remark/Shiki) とクライアントランタイム (`src/client/`) の間の契約。

## Markdown 正規記法 → ビルド出力

| 記法 | 出力 DOM |
|---|---|
| ```` ```python ```` | `pre.astro-code` (Shiki 静的ハイライトのみ) |
| ```` ```python file=X.py ```` | `pre[data-file="X.py"]` (CSS `::before` でファイルタブ) |
| ```` ```python exec file=X.py [mode=append] ```` | `pre[data-exec][data-mode][data-file]` |
| ```` ```python noexec ```` | exec 判定の明示的オプトアウト (取り込み時に解決) |
| ```` ```text output ```` | `pre[data-sample-output]` (期待される出力) |
| `::widget{name="NAME"}` | `div.widget-mount[data-widget="NAME"]` (子要素 = 非JSフォールバック) |

変換は `src/plugins/shiki-exec-transformer.mjs` と `src/plugins/remark-widget-directive.mjs` が行う。
クラスは Astro 内部の `astro-code` 付与と衝突するため使わず、**data 属性のみ**が契約。

## クライアント側 (src/client/entry.ts)

- `main[data-chapter]` の属性で章 slug を得る
- `pre[data-sample-output]` → `<details class="sample-output-details">` 化
- `pre[data-exec]` → CodeMirror 6 エディタ + Run/Reset + 出力パネル (`exec-block/`)
  - コード原文は `pre.textContent`
  - Run = Worker 上の共有 Python 名前空間で exec (**FS への書込みは行わない**。
    章スナップショットは起動時に完全な状態でマウント済みなので import は常に成功する)
- `div.widget-mount[data-widget]` → `widgets/registry.ts` から動的 import してマウント
  (IntersectionObserver で可視直前に遅延ロード)

## Pyodide Worker (runtime/)

- Pyodide 本体は jsDelivr CDN、バージョンは npm `pyodide` パッケージと同期
- boot: numpy ロード → `public/code/manifest.json` から章の .py を FS へマウント → bootstrap.py
- パッケージは import 文の走査で遅延ロード (numpy / matplotlib / pillow)。
  matplotlib 初回ロード時に `public/fonts/NotoSansJP-Regular.ttf` を登録 (日本語ラベル対策)
- 学習系は `train-start` RPC のチャンク実行 (n ステップごとに進捗report + キャンセルフラグ確認)。
  GitHub Pages では SharedArrayBuffer が使えないため、これがソフトキャンセルの唯一の手段
- 暴走コードからの回復は `worker.terminate()` + 再起動 (ステータスピルの「環境をリセット」)

## データアセット

- `public/code/chapter-NN/*.py` + `manifest.json` — `pnpm import:content` が生成
- `public/data/ch5/` — target.png + precomputed.json (`scripts/gen-ch5-precomputed.py`)
- `public/data/ch9/` — cameras.json + test-views/ + trained-params.json (`scripts/train-ch9-export.py`)
- クライアントの fetch はすべて `import.meta.env.BASE_URL` 起点 (base path 対応)

## 新しい章を公開する手順

1. ソース md (F: ドライブ) に必要なら `::widget{name}` 行やフェンスメタ (`noexec` 等) を追記
2. ウィジェットを `src/client/widgets/chN-*.ts` に実装し `registry.ts` に登録
3. `pnpm import:content -- --chapter NN`
4. `src/data/chapters.ts` の該当章 `status` を `'published'` に変更
5. `pnpm build && pnpm preview` で確認してコミット
