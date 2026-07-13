// 第4章: ブロードキャストと _unbroadcast のインタラクティブ可視化
// forward: 形状の異なる A と B がブロードキャストで出力形状に揃う様子を、
//   実セル (濃色) と複製された仮想セル (半透明) のグリッドで表示する。
// backward: 上流勾配 dC (全要素1) を、コピーされた軸方向に sum して
//   元の形状に潰す (_unbroadcast) 様子を矢印と合計値で表示する。
import type { WidgetContext } from './mount';
import { widgetHeader, jsReimplNote } from './ui/controls';

type Op = '+' | '*';

interface Operand {
  /** ブロードキャスト前の行数 (広がる軸は 1) */
  rows: number;
  /** ブロードキャスト前の列数 (広がる軸は 1) */
  cols: number;
  /** 元の次元数 (0=スカラー, 1=ベクトル, 2=行列)。backward の sum 式表示に使う */
  ndim: 0 | 1 | 2;
  /** rows×cols の実データ */
  data: number[][];
  shapeLabel: string;
}

interface Preset {
  label: string;
  op: Op;
  outRows: number;
  outCols: number;
  a: Operand;
  b: Operand;
}

const CELL = 40; // セルの1辺 (px)

const CSS_TEXT = `
.widget-mount .bc-controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:4px}
.widget-mount .bc-sep{width:1px;height:22px;background:var(--border,#d0d7de);margin:0 4px}
.widget-mount .bc-view{margin-top:12px}
.widget-mount .bc-row{display:flex;flex-wrap:wrap;align-items:center;gap:14px}
.widget-mount .bc-grad-row{align-items:flex-start;gap:28px;margin-top:16px}
.widget-mount .bc-block{display:flex;flex-direction:column;gap:5px}
.widget-mount .bc-title{margin:0;font-family:var(--font-mono);font-size:var(--text-xs,12px);color:var(--muted,#57606a)}
.widget-mount .bc-grid{display:grid;gap:3px;width:max-content}
.widget-mount .bc-cell{width:${CELL}px;height:${CELL}px;display:flex;align-items:center;justify-content:center;border-radius:4px;border:1px solid transparent;font-family:var(--font-mono);font-size:var(--text-xs,12px);font-variant-numeric:tabular-nums;user-select:none}
.widget-mount .bc-grid[data-tone='a'] .bc-cell.real{background:#0969da;color:#fff}
.widget-mount .bc-grid[data-tone='a'] .bc-cell.virtual{background:rgba(9,105,218,.13);color:#0969da;border:1px dashed rgba(9,105,218,.55)}
.widget-mount .bc-grid[data-tone='b'] .bc-cell.real{background:#1a7f37;color:#fff}
.widget-mount .bc-grid[data-tone='b'] .bc-cell.virtual{background:rgba(26,127,55,.13);color:#1a7f37;border:1px dashed rgba(26,127,55,.55)}
.widget-mount .bc-grid[data-tone='c'] .bc-cell.real{background:#57606a;color:#fff}
.widget-mount .bc-grid[data-tone='c'] .bc-cell.virtual{background:rgba(87,96,106,.13);color:#57606a;border:1px dashed rgba(87,96,106,.5)}
.widget-mount .bc-cell.hl-link{outline:2px solid #bf8700;outline-offset:1px}
.widget-mount .bc-cell.hl-src{outline:2px solid #cf222e;outline-offset:1px}
.widget-mount .bc-op{font-family:var(--font-mono);font-size:20px;color:var(--muted,#57606a);padding:0 2px}
.widget-mount .bc-flexv{display:flex;flex-direction:column;gap:4px;width:max-content}
.widget-mount .bc-flexh{display:flex;gap:6px;width:max-content}
.widget-mount .bc-center{display:flex;justify-content:center}
.widget-mount .bc-arrows{display:grid;gap:3px;width:max-content}
.widget-mount .bc-arrow{display:flex;align-items:center;justify-content:center;min-width:16px;min-height:16px;color:#cf222e;font-size:14px;line-height:1}
.widget-mount .bc-arrow-line{width:100%;text-align:center;color:#cf222e;font-family:var(--font-mono);font-size:var(--text-xs,12px)}
.widget-mount .bc-caption{margin:0;font-family:var(--font-mono);font-size:var(--text-xs,12px);color:var(--faint,#8c959f)}
.widget-mount .bc-note{margin:10px 0 0;font-size:var(--text-xs,12px);color:var(--faint,#8c959f)}
.widget-mount .bc-note-top{margin:0 0 12px}
`;

function makePresets(ja: boolean): Preset[] {
  const a34: Operand = {
    rows: 3,
    cols: 4,
    ndim: 2,
    data: [
      [10, 20, 30, 40],
      [50, 60, 70, 80],
      [90, 100, 110, 120],
    ],
    shapeLabel: '(3,4)',
  };
  return [
    {
      label: '(3,1) + (1,4)',
      op: '+',
      outRows: 3,
      outCols: 4,
      a: { rows: 3, cols: 1, ndim: 2, data: [[10], [20], [30]], shapeLabel: '(3,1)' },
      b: { rows: 1, cols: 4, ndim: 2, data: [[1, 2, 3, 4]], shapeLabel: '(1,4)' },
    },
    {
      label: '(3,4) + (1,4)',
      op: '+',
      outRows: 3,
      outCols: 4,
      a: a34,
      b: { rows: 1, cols: 4, ndim: 2, data: [[1, 2, 3, 4]], shapeLabel: '(1,4)' },
    },
    {
      label: ja ? '(3,4) + スカラー' : '(3,4) + scalar',
      op: '+',
      outRows: 3,
      outCols: 4,
      a: a34,
      b: { rows: 1, cols: 1, ndim: 0, data: [[5]], shapeLabel: ja ? 'スカラー' : 'scalar' },
    },
    {
      label: '(2,3) * (3,)',
      op: '*',
      outRows: 2,
      outCols: 3,
      a: {
        rows: 2,
        cols: 3,
        ndim: 2,
        data: [
          [10, 20, 30],
          [40, 50, 60],
        ],
        shapeLabel: '(2,3)',
      },
      b: { rows: 1, cols: 3, ndim: 1, data: [[1, 2, 3]], shapeLabel: '(3,)' },
    },
  ];
}

/** ブロードキャスト後の位置 (i, j) に対応する実データの値 */
function valAt(o: Operand, i: number, j: number): number {
  return o.data[o.rows === 1 ? 0 : i][o.cols === 1 ? 0 : j];
}

/** ブロードキャスト後の位置 (i, j) の複製元 (実セル) の位置 */
function srcOf(o: Operand, i: number, j: number): [number, number] {
  return [o.rows === 1 ? 0 : i, o.cols === 1 ? 0 : j];
}

/** ブロードキャスト後の位置 (i, j) が実セル (複製ではない) かどうか */
function isReal(o: Operand, i: number, j: number): boolean {
  return (o.rows !== 1 || i === 0) && (o.cols !== 1 || j === 0);
}

function apply(op: Op, x: number, y: number): number {
  return op === '+' ? x + y : x * y;
}

function fmt(v: number): string {
  return String(v);
}

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

function buildGrid(rows: number, cols: number, tone: 'a' | 'b' | 'c') {
  const root = h('div', 'bc-grid');
  root.dataset.tone = tone;
  root.style.gridTemplateColumns = `repeat(${cols}, ${CELL}px)`;
  const cells: HTMLElement[][] = [];
  for (let i = 0; i < rows; i++) {
    const rowCells: HTMLElement[] = [];
    for (let j = 0; j < cols; j++) {
      const cell = h('div', 'bc-cell');
      root.appendChild(cell);
      rowCells.push(cell);
    }
    cells.push(rowCells);
  }
  return { root, cells };
}

// --- forward 可視化 ---
function renderForward(view: HTMLElement, p: Preset, ja: boolean): void {
  const { outRows, outCols } = p;
  const allCells: HTMLElement[] = [];
  const clearHl = () => {
    for (const c of allCells) c.classList.remove('hl-src', 'hl-link');
  };

  const row = h('div', 'bc-row');
  row.addEventListener('mouseleave', clearHl);
  view.appendChild(row);

  const buildInput = (name: string, o: Operand, tone: 'a' | 'b') => {
    const bcNote = o.rows === outRows && o.cols === outCols ? '' : ` → (${outRows},${outCols})`;
    const block = h('div', 'bc-block');
    block.appendChild(h('p', 'bc-title', `${name} ${o.shapeLabel}${bcNote}`));
    const { root, cells } = buildGrid(outRows, outCols, tone);
    for (let i = 0; i < outRows; i++) {
      for (let j = 0; j < outCols; j++) {
        const cell = cells[i][j];
        cell.textContent = fmt(valAt(o, i, j));
        cell.classList.add(isReal(o, i, j) ? 'real' : 'virtual');
        allCells.push(cell);
      }
    }
    block.appendChild(root);
    row.appendChild(block);
    return cells;
  };

  const aCells = buildInput('A', p.a, 'a');
  row.appendChild(h('span', 'bc-op', p.op === '*' ? '×' : '+'));
  const bCells = buildInput('B', p.b, 'b');
  row.appendChild(h('span', 'bc-op', '='));

  const cBlock = h('div', 'bc-block');
  cBlock.appendChild(
    h('p', 'bc-title', `C = A ${p.op === '*' ? '×' : '+'} B  (${outRows},${outCols})`),
  );
  const { root: cRoot, cells: cCells } = buildGrid(outRows, outCols, 'c');
  for (let i = 0; i < outRows; i++) {
    for (let j = 0; j < outCols; j++) {
      const cell = cCells[i][j];
      cell.textContent = fmt(apply(p.op, valAt(p.a, i, j), valAt(p.b, i, j)));
      cell.classList.add('real');
      allCells.push(cell);
    }
  }
  cBlock.appendChild(cRoot);
  row.appendChild(cBlock);

  // 入力グリッドのホバー: 複製元の実セル (赤枠) と同じ実セル由来のコピー (黄枠) を強調
  const wireInput = (cells: HTMLElement[][], o: Operand) => {
    for (let i = 0; i < outRows; i++) {
      for (let j = 0; j < outCols; j++) {
        cells[i][j].addEventListener('mouseenter', () => {
          clearHl();
          const [si, sj] = srcOf(o, i, j);
          for (let r = 0; r < outRows; r++) {
            for (let c = 0; c < outCols; c++) {
              const [ri, rj] = srcOf(o, r, c);
              if (ri === si && rj === sj) cells[r][c].classList.add('hl-link');
            }
          }
          cells[si][sj].classList.remove('hl-link');
          cells[si][sj].classList.add('hl-src');
        });
      }
    }
  };
  wireInput(aCells, p.a);
  wireInput(bCells, p.b);

  // C のホバー: その要素を作った A・B の対応セル + 複製元をハイライト
  const markSource = (cells: HTMLElement[][], o: Operand, i: number, j: number) => {
    const [si, sj] = srcOf(o, i, j);
    if (si !== i || sj !== j) cells[i][j].classList.add('hl-link');
    cells[si][sj].classList.add('hl-src');
  };
  for (let i = 0; i < outRows; i++) {
    for (let j = 0; j < outCols; j++) {
      cCells[i][j].addEventListener('mouseenter', () => {
        clearHl();
        markSource(aCells, p.a, i, j);
        markSource(bCells, p.b, i, j);
      });
    }
  }

  view.appendChild(
    h(
      'p',
      'bc-note',
      ja
        ? '濃色のセルが実データ、半透明のセルがブロードキャストで複製された仮想コピーです。セルにホバーすると複製元の実セルが赤枠、同じ実セル由来のコピーが黄枠で光ります。C のセルにホバーすると対応する A・B のセルが光ります。'
        : 'Solid cells hold real data; translucent cells are virtual copies made by broadcasting. Hover a cell to highlight its source cell (red) and sibling copies (yellow). Hover a C cell to see the corresponding A and B cells.',
    ),
  );
}

// --- backward 可視化 ---

/** _unbroadcast に対応する sum の式 (章の実装と同じ規則: 次元差は sum(axis=0)、サイズ1軸は keepdims=True) */
function sumExpr(op: Op, otherName: string, target: Operand): string {
  const base = op === '+' ? 'dC' : `(dC × ${otherName})`;
  if (target.ndim === 0) return `${base}.sum()`;
  if (target.ndim === 1) return `${base}.sum(axis=0)`;
  return target.rows === 1
    ? `${base}.sum(axis=0, keepdims=True)`
    : `${base}.sum(axis=1, keepdims=True)`;
}

function gradBlock(
  name: string,
  target: Operand,
  otherName: string,
  other: Operand,
  tone: 'a' | 'b',
  p: Preset,
  ja: boolean,
): HTMLElement {
  const { outRows, outCols, op } = p;
  const block = h('div', 'bc-block');
  block.appendChild(h('p', 'bc-title', `∂L/∂${name} = ${name}.grad`));

  // ブロードキャスト形状での勾配 (dC=1 なので、加算はそのまま1、乗算は相手側の値)
  const expanded: number[][] = [];
  for (let i = 0; i < outRows; i++) {
    const rowVals: number[] = [];
    for (let j = 0; j < outCols; j++) rowVals.push(op === '+' ? 1 : valAt(other, i, j));
    expanded.push(rowVals);
  }

  const sumRows = target.rows === 1 && outRows > 1; // axis=0 で潰す
  const sumCols = target.cols === 1 && outCols > 1; // axis=1 で潰す

  // _unbroadcast: 広がった軸方向に sum して元の形状に集約
  const result: number[][] = [];
  for (let i = 0; i < target.rows; i++) {
    result.push(Array.from({ length: target.cols }, () => 0));
  }
  for (let i = 0; i < outRows; i++) {
    for (let j = 0; j < outCols; j++) {
      result[target.rows === 1 ? 0 : i][target.cols === 1 ? 0 : j] += expanded[i][j];
    }
  }

  block.appendChild(
    h(
      'p',
      'bc-caption',
      op === '+'
        ? ja
          ? `局所勾配は 1 → dC がそのまま流れる  (${outRows},${outCols})`
          : `local grad is 1 → dC flows through  (${outRows},${outCols})`
        : ja
          ? `乗算則: dC × ${otherName} の値  (${outRows},${outCols})`
          : `product rule: dC × ${otherName}  (${outRows},${outCols})`,
    ),
  );

  const exp = buildGrid(outRows, outCols, tone);
  const collapsing = sumRows || sumCols;
  for (let i = 0; i < outRows; i++) {
    for (let j = 0; j < outCols; j++) {
      const cell = exp.cells[i][j];
      cell.textContent = fmt(expanded[i][j]);
      cell.classList.add(collapsing ? 'virtual' : 'real');
    }
  }

  if (!collapsing) {
    block.appendChild(exp.root);
    block.appendChild(
      h(
        'p',
        'bc-caption',
        ja
          ? `形状が ${target.shapeLabel} と一致 — sum は不要`
          : `already ${target.shapeLabel} — no sum needed`,
      ),
    );
  } else if (sumRows && sumCols) {
    // スカラーへ全要素 sum
    const wrap = h('div', 'bc-flexv');
    wrap.appendChild(exp.root);
    wrap.appendChild(h('div', 'bc-arrow-line', ja ? '↓ sum（全要素）' : '↓ sum (all elements)'));
    const res = buildGrid(1, 1, tone);
    res.cells[0][0].textContent = fmt(result[0][0]);
    res.cells[0][0].classList.add('real');
    const center = h('div', 'bc-center');
    center.appendChild(res.root);
    wrap.appendChild(center);
    block.appendChild(wrap);
    block.appendChild(
      h('p', 'bc-caption', `${name}.grad = ${sumExpr(op, otherName, target)} → ${target.shapeLabel}`),
    );
  } else if (sumRows) {
    // axis=0 方向 (列ごとに縦に合計)
    const wrap = h('div', 'bc-flexv');
    wrap.appendChild(exp.root);
    const arrows = h('div', 'bc-arrows');
    arrows.style.gridTemplateColumns = `repeat(${outCols}, ${CELL}px)`;
    for (let j = 0; j < outCols; j++) arrows.appendChild(h('span', 'bc-arrow', '↓'));
    wrap.appendChild(arrows);
    const res = buildGrid(1, outCols, tone);
    for (let j = 0; j < outCols; j++) {
      res.cells[0][j].textContent = fmt(result[0][j]);
      res.cells[0][j].classList.add('real');
    }
    wrap.appendChild(res.root);
    block.appendChild(wrap);
    block.appendChild(
      h('p', 'bc-caption', `${name}.grad = ${sumExpr(op, otherName, target)} → ${target.shapeLabel}`),
    );
  } else {
    // axis=1 方向 (行ごとに横に合計)
    const wrap = h('div', 'bc-flexh');
    wrap.appendChild(exp.root);
    const arrows = h('div', 'bc-arrows');
    arrows.style.gridTemplateRows = `repeat(${outRows}, ${CELL}px)`;
    for (let i = 0; i < outRows; i++) arrows.appendChild(h('span', 'bc-arrow', '→'));
    wrap.appendChild(arrows);
    const res = buildGrid(outRows, 1, tone);
    for (let i = 0; i < outRows; i++) {
      res.cells[i][0].textContent = fmt(result[i][0]);
      res.cells[i][0].classList.add('real');
    }
    wrap.appendChild(res.root);
    block.appendChild(wrap);
    block.appendChild(
      h('p', 'bc-caption', `${name}.grad = ${sumExpr(op, otherName, target)} → ${target.shapeLabel}`),
    );
  }
  return block;
}

function renderBackward(view: HTMLElement, p: Preset, ja: boolean): void {
  const { outRows, outCols } = p;
  view.appendChild(
    h(
      'p',
      'bc-note bc-note-top',
      ja
        ? 'L = C.sum() とすると、上流勾配 dC = ∂L/∂C は全要素 1 です。ブロードキャストで広がった側は、コピーされた軸方向に勾配を sum して元の形状に潰します（_unbroadcast）。'
        : 'Assuming L = C.sum(), the upstream gradient dC = ∂L/∂C is all ones. For the broadcast side, the gradient is summed along the copied axes back to the original shape (_unbroadcast).',
    ),
  );

  const dcBlock = h('div', 'bc-block');
  dcBlock.appendChild(
    h('p', 'bc-title', ja ? `dC = ∂L/∂C（全要素 1）  (${outRows},${outCols})` : `dC = ∂L/∂C (all ones)  (${outRows},${outCols})`),
  );
  const dc = buildGrid(outRows, outCols, 'c');
  for (let i = 0; i < outRows; i++) {
    for (let j = 0; j < outCols; j++) {
      dc.cells[i][j].textContent = '1';
      dc.cells[i][j].classList.add('real');
    }
  }
  dcBlock.appendChild(dc.root);
  view.appendChild(dcBlock);

  const row = h('div', 'bc-row bc-grad-row');
  row.appendChild(gradBlock('A', p.a, 'B', p.b, 'a', p, ja));
  row.appendChild(gradBlock('B', p.b, 'A', p.a, 'b', p, ja));
  view.appendChild(row);
}

export function mount(el: HTMLElement, ctx: WidgetContext): void {
  const ja = ctx.locale === 'ja';
  widgetHeader(
    el,
    ja
      ? 'インタラクティブ: ブロードキャストと _unbroadcast'
      : 'Interactive: broadcasting and _unbroadcast',
    jsReimplNote(ctx.locale),
  );

  const style = document.createElement('style');
  style.textContent = CSS_TEXT;
  el.appendChild(style);

  const presets = makePresets(ja);
  let presetIdx = 0;
  let showBackward = false;

  const controls = h('div', 'bc-controls');
  el.appendChild(controls);
  const presetButtons = presets.map((preset, idx) => {
    const btn = h('button', 'w-button', preset.label);
    btn.type = 'button';
    btn.addEventListener('click', () => {
      presetIdx = idx;
      render();
    });
    controls.appendChild(btn);
    return btn;
  });
  controls.appendChild(h('span', 'bc-sep'));
  const toggleBtn = h('button', 'w-button');
  toggleBtn.type = 'button';
  toggleBtn.addEventListener('click', () => {
    showBackward = !showBackward;
    render();
  });
  controls.appendChild(toggleBtn);

  const view = h('div', 'bc-view');
  el.appendChild(view);

  function render(): void {
    presetButtons.forEach((btn, i) => btn.classList.toggle('primary', i === presetIdx));
    toggleBtn.classList.toggle('primary', showBackward);
    toggleBtn.setAttribute('aria-pressed', String(showBackward));
    toggleBtn.textContent = showBackward
      ? ja
        ? 'forward を見る'
        : 'Show forward'
      : ja
        ? 'backward を見る'
        : 'Show backward';

    view.innerHTML = '';
    const p = presets[presetIdx];
    if (showBackward) renderBackward(view, p, ja);
    else renderForward(view, p, ja);
  }

  render();
}
