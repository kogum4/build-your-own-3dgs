// 第3章: 計算グラフと backward のステップ実行
// 固定式 f = (a·b + c)² の計算グラフを SVG で描き、章の Value クラスと同じ構造の
// ミニ Value (data / grad / _backward クロージャ / 逆トポロジカル順) で
// 逆伝播を1ノードずつ実行して「局所勾配 × 上流勾配」を数値で確認する。
import type { WidgetContext } from './mount';
import { sliderRow, widgetHeader, jsReimplNote } from './ui/controls';

// --- ミニ Value クラス (章の autograd.py と同じ構造の JS 移植、約40行) ---
class Value {
  data: number;
  grad = 0;
  _backward: () => void = () => {};
  _prev: Value[] = [];

  constructor(data: number) {
    this.data = data;
  }

  add(other: Value): Value {
    const out = new Value(this.data + other.data);
    out._prev = [this, other];
    out._backward = () => {
      // d(a + b)/da = 1, d(a + b)/db = 1
      this.grad += 1.0 * out.grad;
      other.grad += 1.0 * out.grad;
    };
    return out;
  }

  mul(other: Value): Value {
    const out = new Value(this.data * other.data);
    out._prev = [this, other];
    out._backward = () => {
      // d(a * b)/da = b, d(a * b)/db = a
      this.grad += other.data * out.grad;
      other.grad += this.data * out.grad;
    };
    return out;
  }

  pow(n: number): Value {
    const out = new Value(this.data ** n);
    out._prev = [this];
    out._backward = () => {
      // d(x^n)/dx = n * x^(n-1)
      this.grad += n * this.data ** (n - 1) * out.grad;
    };
    return out;
  }

  /** backward() 内の reversed(topo) と同じ「逆トポロジカル順」を返す (DFS) */
  topoReversed(): Value[] {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const p of v._prev) build(p);
      topo.push(v);
    };
    build(this);
    return topo.reverse();
  }
}

type NodeKey = 'a' | 'b' | 'c' | 'u' | 'v' | 'f';
const NODE_KEYS: NodeKey[] = ['a', 'b', 'c', 'u', 'v', 'f'];

interface Step {
  /** このステップで _backward を実行する (処理中としてハイライトする) ノード */
  node: NodeKey;
  /** このステップで grad が確定するノード */
  computes: NodeKey[];
  /** 実行して「局所勾配 × 上流勾配」の説明行を返す */
  run: () => string[];
}

interface NodeView {
  rect: SVGRectElement;
  title: SVGTextElement;
  dataText: SVGTextElement;
  gradText: SVGTextElement;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
let instanceCounter = 0;

function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string>,
): SVGElementTagNameMap[K] {
  const e = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}

/** 数式表示用フォーマッタ (負数は括弧で包む) */
function num(x: number): string {
  const s = (Object.is(x, -0) ? 0 : x).toFixed(2);
  return x < 0 ? `(${s})` : s;
}

export function mount(el: HTMLElement, ctx: WidgetContext): void {
  const ja = ctx.locale === 'ja';
  widgetHeader(
    el,
    ja
      ? 'インタラクティブ: f = (a·b + c)² の計算グラフと逆伝播'
      : 'Interactive: computation graph and backprop for f = (a·b + c)²',
    (ja
      ? 'スライダーで葉ノードの値を変え、backward を1ステップずつ実行して勾配の伝播を追いかけます。'
      : 'Move the leaf sliders, then run backward one step at a time to trace how gradients propagate.') +
      ' ' +
      jsReimplNote(ctx.locale),
  );

  // --- SVG 計算グラフ (viewBox で responsive) ---
  const frame = document.createElement('div');
  frame.className = 'w-canvas-frame';
  el.appendChild(frame);

  const VBW = 496;
  const VBH = 240;
  const NW = 96; // ノード幅
  const NH = 58; // ノード高さ
  const svg = svgEl('svg', {
    viewBox: `0 0 ${VBW} ${VBH}`,
    role: 'img',
    'aria-label': ja ? 'f = (a·b + c)² の計算グラフ' : 'computation graph of f = (a·b + c)²',
  });
  svg.style.width = '100%';
  svg.style.maxWidth = '640px';
  svg.style.height = 'auto';
  frame.appendChild(svg);

  // 矢印マーカー (id はインスタンスごとに一意化)
  const markerId = `ch3-ag-arrow-${++instanceCounter}`;
  const defs = svgEl('defs', {});
  const marker = svgEl('marker', {
    id: markerId,
    viewBox: '0 0 10 10',
    refX: '9',
    refY: '5',
    markerWidth: '7',
    markerHeight: '7',
    orient: 'auto-start-reverse',
  });
  marker.appendChild(svgEl('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: '#8c959f' }));
  defs.appendChild(marker);
  svg.appendChild(defs);

  // エッジ: 左 (葉) → 右 (f)
  const edges = [
    { x1: 104, y1: 36, x2: 136, y2: 66 }, // a -> u
    { x1: 104, y1: 118, x2: 136, y2: 88 }, // b -> u
    { x1: 232, y1: 77, x2: 264, y2: 128 }, // u -> v
    { x1: 104, y1: 200, x2: 264, y2: 150 }, // c -> v
    { x1: 360, y1: 139, x2: 392, y2: 139 }, // v -> f
  ];
  for (const e of edges) {
    svg.appendChild(
      svgEl('line', {
        x1: String(e.x1),
        y1: String(e.y1),
        x2: String(e.x2),
        y2: String(e.y2),
        stroke: '#8c959f',
        'stroke-width': '1.5',
        'marker-end': `url(#${markerId})`,
      }),
    );
  }

  // ノード: 丸角矩形 + 演算子ラベル + data / grad
  const nodeDefs: { key: NodeKey; label: string; x: number; y: number }[] = [
    { key: 'a', label: 'a', x: 8, y: 7 },
    { key: 'b', label: 'b', x: 8, y: 89 },
    { key: 'c', label: 'c', x: 8, y: 171 },
    { key: 'u', label: 'u = a·b', x: 136, y: 48 },
    { key: 'v', label: 'v = u + c', x: 264, y: 110 },
    { key: 'f', label: 'f = v²', x: 392, y: 110 },
  ];
  const views = {} as Record<NodeKey, NodeView>;
  for (const d of nodeDefs) {
    const g = svgEl('g', {});
    const rect = svgEl('rect', {
      x: String(d.x),
      y: String(d.y),
      width: String(NW),
      height: String(NH),
      rx: '8',
      fill: '#f6f8fa',
      stroke: '#d0d7de',
      'stroke-width': '1.5',
    });
    const cx = d.x + NW / 2;
    const mkText = (dy: number, size: number, weight: string) => {
      const t = svgEl('text', {
        x: String(cx),
        y: String(d.y + dy),
        'text-anchor': 'middle',
        'font-family': 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
        'font-size': String(size),
        'font-weight': weight,
      });
      g.appendChild(t);
      return t;
    };
    g.appendChild(rect);
    const title = mkText(17, 13, '700');
    title.textContent = d.label;
    const dataText = mkText(35, 11, '400');
    const gradText = mkText(50, 11, '400');
    svg.appendChild(g);
    views[d.key] = { rect, title, dataText, gradText };
  }

  const legend = document.createElement('p');
  legend.className = 'w-note';
  legend.textContent = ja
    ? '灰色 = grad 未計算 / 青 = grad 計算済み / 橙 = いま _backward を処理中のノード'
    : 'gray = grad not yet computed / blue = grad computed / orange = node whose _backward is running';
  el.appendChild(legend);

  // --- 下段: スライダー + ボタン (左) / ステップログ (右) ---
  const cols = document.createElement('div');
  cols.className = 'w-columns';
  el.appendChild(cols);

  const left = document.createElement('div');
  left.className = 'w-col';
  cols.appendChild(left);

  const right = document.createElement('div');
  right.className = 'w-col';
  cols.appendChild(right);

  // --- 状態 ---
  const params = { a: 2.0, b: -1.5, c: 0.5 };
  let nodes = {} as Record<NodeKey, Value>;
  let steps: Step[] = [];
  let stepIndex = 0; // 次に実行するステップ
  let computed = new Set<NodeKey>();
  let currentNode: NodeKey | null = null;

  for (const key of ['a', 'b', 'c'] as const) {
    sliderRow(left, {
      label: key,
      min: -3.0,
      max: 3.0,
      step: 0.1,
      value: params[key],
      format: (v) => v.toFixed(1),
      onInput: (v) => {
        params[key] = v;
        forward(); // forward 再計算 + backward 状態リセット
      },
    });
  }

  const btnRow = document.createElement('div');
  btnRow.style.display = 'flex';
  btnRow.style.flexWrap = 'wrap';
  btnRow.style.gap = '8px';
  btnRow.style.marginTop = '10px';
  left.appendChild(btnRow);

  const stepBtn = document.createElement('button');
  stepBtn.className = 'w-button primary';
  stepBtn.textContent = ja ? 'backward を1ステップ進める' : 'Step backward once';
  const allBtn = document.createElement('button');
  allBtn.className = 'w-button';
  allBtn.textContent = ja ? '一気に backward' : 'Run full backward';
  const resetBtn = document.createElement('button');
  resetBtn.className = 'w-button';
  resetBtn.textContent = ja ? 'リセット' : 'Reset';
  btnRow.append(stepBtn, allBtn, resetBtn);

  // ステップログパネル
  const panel = document.createElement('div');
  panel.className = 'w-group';
  const panelTitle = document.createElement('p');
  panelTitle.className = 'w-group-title';
  panelTitle.textContent = ja ? '局所勾配 × 上流勾配' : 'local grad × upstream grad';
  panel.appendChild(panelTitle);
  const logBox = document.createElement('div');
  logBox.style.fontFamily = 'var(--font-mono)';
  logBox.style.fontSize = 'var(--text-xs)';
  logBox.style.lineHeight = '1.75';
  panel.appendChild(logBox);
  right.appendChild(panel);

  function addLogEntry(head: string, lines: string[]) {
    const entry = document.createElement('div');
    entry.style.marginBottom = '6px';
    const h = document.createElement('div');
    h.style.fontWeight = '700';
    h.textContent = head;
    entry.appendChild(h);
    for (const line of lines) {
      const p = document.createElement('div');
      p.textContent = line;
      entry.appendChild(p);
    }
    logBox.appendChild(entry);
    // 最新エントリだけ強調、過去はグレー
    [...logBox.children].forEach((c, i) => {
      (c as HTMLElement).style.color =
        i === logBox.children.length - 1 ? 'var(--fg, #24292f)' : 'var(--faint, #8c959f)';
    });
  }

  /** 章の backward() と同じ処理を、逆トポロジカル順に1ノードずつのステップに分解する */
  function buildSteps(): Step[] {
    const n = nodes;
    const list: Step[] = [];

    // ステップ1: 起点 f.grad = 1 (df/df = 1)
    list.push({
      node: 'f',
      computes: ['f'],
      run: () => {
        n.f.grad = 1.0;
        return [
          ja
            ? '起点: f.grad = ∂f/∂f = 1.00（出力ノード自身の勾配は 1 から始める）'
            : 'seed: f.grad = ∂f/∂f = 1.00 (the output node starts with gradient 1)',
        ];
      },
    });

    // 以降: reversed(topo) = [f, v, u, c, b, a] の順に _backward を呼ぶ
    const order = n.f.topoReversed();
    const keyOf = new Map<Value, NodeKey>(NODE_KEYS.map((k): [Value, NodeKey] => [n[k], k]));
    for (const value of order) {
      const key = keyOf.get(value)!;
      if (key === 'f') {
        // f = v²: d(v^2)/dv = 2v
        list.push({
          node: 'f',
          computes: ['v'],
          run: () => {
            n.f._backward();
            return [
              `v.grad = ∂f/∂v × f.grad = 2v × f.grad = (2×${num(n.v.data)}) × ${num(n.f.grad)} = ${num(n.v.grad)}`,
            ];
          },
        });
      } else if (key === 'v') {
        // v = u + c: d(u+c)/du = 1, d(u+c)/dc = 1
        list.push({
          node: 'v',
          computes: ['u', 'c'],
          run: () => {
            n.v._backward();
            return [
              `u.grad = ∂v/∂u × v.grad = 1 × ${num(n.v.grad)} = ${num(n.u.grad)}`,
              `c.grad = ∂v/∂c × v.grad = 1 × ${num(n.v.grad)} = ${num(n.c.grad)}`,
            ];
          },
        });
      } else if (key === 'u') {
        // u = a·b: d(ab)/da = b, d(ab)/db = a
        list.push({
          node: 'u',
          computes: ['a', 'b'],
          run: () => {
            n.u._backward();
            return [
              `a.grad = ∂u/∂a × u.grad = b × u.grad = ${num(n.b.data)} × ${num(n.u.grad)} = ${num(n.a.grad)}`,
              `b.grad = ∂u/∂b × u.grad = a × u.grad = ${num(n.a.data)} × ${num(n.u.grad)} = ${num(n.b.grad)}`,
            ];
          },
        });
      } else {
        // 葉ノード: _backward は何もしない (デフォルトの no-op)
        list.push({
          node: key,
          computes: [],
          run: () => {
            n[key]._backward();
            return [
              ja
                ? `${key} は葉ノード: _backward は何もしない（${key}.grad = ${num(n[key].grad)} で確定）`
                : `${key} is a leaf: _backward does nothing (${key}.grad = ${num(n[key].grad)} is final)`,
            ];
          },
        });
      }
    }
    return list;
  }

  function paintNodes() {
    for (const key of NODE_KEYS) {
      const view = views[key];
      const isComputed = computed.has(key);
      const isCurrent = currentNode === key;
      view.rect.setAttribute('fill', isCurrent ? '#fff1e5' : isComputed ? '#ddf4ff' : '#f6f8fa');
      view.rect.setAttribute('stroke', isCurrent ? '#bc4c00' : isComputed ? '#0969da' : '#d0d7de');
      view.rect.setAttribute('stroke-width', isCurrent ? '2.5' : '1.5');
      view.title.setAttribute('fill', isComputed || isCurrent ? '#24292f' : '#57606a');
      view.dataText.setAttribute('fill', '#24292f');
      view.dataText.textContent = `data ${nodes[key].data.toFixed(2)}`;
      view.gradText.setAttribute('fill', isComputed ? '#0969da' : '#8c959f');
      view.gradText.setAttribute('font-weight', isComputed ? '700' : '400');
      view.gradText.textContent = `grad ${(Object.is(nodes[key].grad, -0) ? 0 : nodes[key].grad).toFixed(2)}`;
    }
  }

  function updateButtons() {
    const done = stepIndex >= steps.length;
    stepBtn.disabled = done;
    allBtn.disabled = done;
  }

  /** forward を再計算し、backward の進行状態をリセットする */
  function forward() {
    const a = new Value(params.a);
    const b = new Value(params.b);
    const c = new Value(params.c);
    const u = a.mul(b); // u = a·b
    const v = u.add(c); // v = u + c
    const f = v.pow(2); // f = v²
    nodes = { a, b, c, u, v, f };
    steps = buildSteps();
    stepIndex = 0;
    computed = new Set();
    currentNode = null;
    logBox.innerHTML = '';
    const hint = document.createElement('div');
    hint.style.color = 'var(--faint, #8c959f)';
    hint.textContent = ja
      ? `forward 完了: f = (a·b + c)² = ${f.data.toFixed(2)}。「backward を1ステップ進める」で勾配を伝播します。`
      : `forward done: f = (a·b + c)² = ${f.data.toFixed(2)}. Press "Step backward once" to propagate gradients.`;
    logBox.appendChild(hint);
    paintNodes();
    updateButtons();
  }

  function doStep(repaint: boolean) {
    const s = steps[stepIndex];
    if (!s) return;
    stepIndex++;
    const lines = s.run();
    for (const k of s.computes) computed.add(k);
    currentNode = s.node;
    addLogEntry(
      (ja ? `ステップ ${stepIndex}/${steps.length}: ` : `step ${stepIndex}/${steps.length}: `) +
        (stepIndex === 1 ? 'f.grad = 1' : `${s.node}._backward()`),
      lines,
    );
    if (stepIndex >= steps.length) {
      addLogEntry(
        ja ? 'backward 完了' : 'backward complete',
        [
          `∂f/∂a = ${num(nodes.a.grad)}, ∂f/∂b = ${num(nodes.b.grad)}, ∂f/∂c = ${num(nodes.c.grad)}`,
        ],
      );
    }
    if (repaint) {
      paintNodes();
      updateButtons();
    }
  }

  stepBtn.addEventListener('click', () => doStep(true));
  allBtn.addEventListener('click', () => {
    while (stepIndex < steps.length) doStep(false);
    paintNodes();
    updateButtons();
  });
  resetBtn.addEventListener('click', forward);

  forward();
}
