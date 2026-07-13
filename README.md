# Build Your Own 3D Gaussian Splatting

**NumPyだけで、ゼロから作る3D Gaussian Splatting** — インタラクティブWeb教材

https://kogum4.github.io/build-your-own-3dgs/

PyTorchもCUDAも使わず、PythonとNumPyだけで3D Gaussian Splattingをスクラッチ実装する教材です。ブラウザ上でコードを編集・実行しながら（Pyodide）、数式と実装の対応を一歩ずつ学べます。

- 📖 全16章構成（現在 第1〜9章を公開、以降は順次公開）
- ▶️ すべてのコードブロックがブラウザ内で実行・編集可能
- 🎛️ 各章にインタラクティブデモ（共分散楕円、計算グラフ、ライブ学習、自由視点レンダリング等）
- 🌐 日本語 / English（英訳は順次公開）

## 開発

```bash
pnpm install
pnpm dev        # 開発サーバー
pnpm build      # 本番ビルド (dist/)
pnpm preview    # base path 込みのプレビュー (GitHub Pages 相当)
pnpm check      # 型チェック
```

### コンテンツの取り込み

章コンテンツは教材ソース（ローカルのワークスペース）から変換スクリプトで取り込みます:

```bash
pnpm import:content                  # 設定済みの全章
pnpm import:content -- --chapter 03  # 単章
```

変換の詳細・新しい章を公開する手順は [docs/runtime-contract.md](docs/runtime-contract.md) を参照してください。

## 構成

- **Astro 5** — 静的サイト生成、i18n ルーティング
- **remark-math + KaTeX** — 数式（ビルド時レンダリング）
- **Shiki + 自作 transformer** — コードハイライトと実行可能ブロックのマーキング
- **CodeMirror 6 + Pyodide (Web Worker)** — ブラウザ内 Python 実行環境
- **素のTS Canvas/SVGウィジェット** — 章別インタラクティブデモ

## デプロイ

`main` ブランチへの push で GitHub Actions が自動ビルドし GitHub Pages へデプロイします。

## ライセンス

教材テキスト・コードともに、リポジトリ所有者に帰属します。
