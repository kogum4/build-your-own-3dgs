# Build Your Own 3D Gaussian Splatting

**Build 3D Gaussian Splatting from scratch, with nothing but NumPy** — an interactive web textbook

**📖 Read it here: https://kogum4.github.io/build-your-own-3dgs/** ([日本語](https://kogum4.github.io/build-your-own-3dgs/) / [English](https://kogum4.github.io/build-your-own-3dgs/en/))

日本語版のREADMEは [README.ja.md](README.ja.md) にあります。

No PyTorch. No CUDA. This textbook implements 3D Gaussian Splatting from scratch with Python and NumPy — from an autograd engine to a rasterizer. Every code block runs and can be edited right in your browser (via Pyodide), so you can follow the math and the implementation one step at a time.

- 📖 16 chapters planned (chapters 1–9 available now, more on the way)
- ▶️ Every code block is executable and editable in the browser
- 🎛️ Interactive demos in every chapter (covariance ellipse explorer, computation-graph backprop, live training, novel-view rendering of a trained scene, and more)
- 🌐 Japanese / English (translations rolling out; Chapter 1 is fully translated)

## Feedback

Found a mistake in the text, a bug in the code, or have a suggestion? Please [open an issue](https://github.com/kogum4/build-your-own-3dgs/issues) — feedback on the content is very welcome, in English or Japanese.

## Development

```bash
pnpm install
pnpm dev        # dev server
pnpm build      # production build (dist/)
pnpm preview    # preview with the base path (mirrors GitHub Pages)
pnpm check      # type check
```

### Importing content

Chapter content is converted from the source workspace (local, not part of this repo) by an import script:

```bash
pnpm import:content                  # all configured chapters
pnpm import:content -- --chapter 03  # a single chapter
```

See [docs/runtime-contract.md](docs/runtime-contract.md) for the conversion details and the checklist for publishing a new chapter.

## Architecture

- **Astro 5** — static site generation, i18n routing
- **remark-math + KaTeX** — math rendered at build time
- **Shiki + custom transformer** — syntax highlighting and executable-block markup
- **CodeMirror 6 + Pyodide (Web Worker)** — in-browser Python environment
- **Vanilla TS Canvas/SVG widgets** — per-chapter interactive demos

## License

- **Code** (site source, widgets, and the textbook's Python code) is licensed under the [MIT License](LICENSE).
- **Textbook text and figures** are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
