# Build Your Own 3D Gaussian Splatting

**NumPyだけで、ゼロから作る3D Gaussian Splatting** — インタラクティブWeb教材

**📖 サイトはこちら: https://kogum4.github.io/build-your-own-3dgs/** ([日本語](https://kogum4.github.io/build-your-own-3dgs/) / [English](https://kogum4.github.io/build-your-own-3dgs/en/))

English README is available at [README.md](README.md).

PyTorchもCUDAも使わず、PythonとNumPyだけで3D Gaussian Splattingをスクラッチ実装する教材です。自動微分エンジンからラスタライザまで、すべて自分の手で作ります。コードブロックはブラウザ上でそのまま編集・実行できるので（Pyodide）、数式と実装の対応を一歩ずつ確かめながら読み進められます。

- 📖 全16章構成（現在 第1〜9章を公開、以降は順次公開）
- ▶️ すべてのコードブロックがブラウザ内で実行・編集可能
- 🎛️ 各章にインタラクティブデモ（共分散楕円、計算グラフの逆伝播、ライブ学習、学習済みシーンの自由視点レンダリング等）
- 🌐 日本語 / English（英訳は順次公開、第1章は翻訳済み）

## フィードバック

本文の誤り・コードの不具合・改善の提案などがあれば、ぜひ [Issue](https://github.com/kogum4/build-your-own-3dgs/issues) を立ててください。内容へのフィードバックを歓迎します（日本語・英語どちらでも）。

## 開発

```bash
pnpm install
pnpm dev        # 開発サーバー
pnpm build      # 本番ビルド (dist/)
pnpm preview    # base path 込みのプレビュー (GitHub Pages 相当)
pnpm check      # 型チェック
```

### コンテンツの取り込み

章コンテンツは教材ソース（ローカルのワークスペース、このリポジトリには含まれない）から変換スクリプトで取り込みます:

```bash
pnpm import:content                  # 設定済みの全章
pnpm import:content -- --chapter 03  # 単章
```

変換の詳細・新しい章を公開する手順は [docs/runtime-contract.md](docs/runtime-contract.md)（英語）を参照してください。

## 構成

- **Astro 5** — 静的サイト生成、i18n ルーティング
- **remark-math + KaTeX** — 数式（ビルド時レンダリング）
- **Shiki + 自作 transformer** — コードハイライトと実行可能ブロックのマーキング
- **CodeMirror 6 + Pyodide (Web Worker)** — ブラウザ内 Python 実行環境
- **素のTS Canvas/SVGウィジェット** — 章別インタラクティブデモ

## ライセンス

- **コード**（サイトのソース・ウィジェット・教材のPythonコード）は [MIT License](LICENSE) です。
- **教材の本文と図**は [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.ja) です。
