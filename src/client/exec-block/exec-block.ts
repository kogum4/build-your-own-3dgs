// pre[data-exec] を CodeMirror エディタ + Run/Reset + 出力パネルにアップグレードする
import type { EditorView } from '@codemirror/view';
import { createEditor, setEditorContent } from './cm-setup';
import { pyEnv } from '../runtime/py-env';
import { getStrings } from '../strings';

interface ExecBlock {
  code: string;
  view: EditorView;
  runBtn: HTMLButtonElement;
  resetBtn: HTMLButtonElement;
  editedBadge: HTMLElement;
  timeLabel: HTMLElement;
  output: HTMLElement;
  sampleOutput: HTMLDetailsElement | null;
}

/** pre[data-sample-output] を折りたたみ可能な <details> に変換する */
export function upgradeSampleOutputs(root: ParentNode) {
  const t = getStrings();
  for (const pre of root.querySelectorAll<HTMLPreElement>('pre[data-sample-output]')) {
    const details = document.createElement('details');
    details.className = 'sample-output-details';
    details.open = true;
    const summary = document.createElement('summary');
    summary.textContent = t.expectedOutput;
    details.appendChild(summary);
    pre.replaceWith(details);
    pre.classList.add('in-details');
    details.appendChild(pre);
  }
}

export function upgradeExecBlocks(root: ParentNode, chapter: string) {
  const t = getStrings();
  const pres = [...root.querySelectorAll<HTMLPreElement>('pre[data-exec]')];

  for (const pre of pres) {
    const code = (pre.textContent ?? '').replace(/\n$/, '');
    const file = pre.getAttribute('data-file');
    const mode = pre.getAttribute('data-mode');

    // --- DOM 構築 ---
    const wrapper = document.createElement('div');
    wrapper.className = 'exec-widget';

    if (file) {
      const tab = document.createElement('div');
      tab.className = 'exec-file-tab';
      tab.textContent =
        mode === 'append'
          ? document.documentElement.lang === 'en'
            ? `append to ${file}`
            : `${file} への追記`
          : file;
      wrapper.appendChild(tab);
    }

    const editorHost = document.createElement('div');
    editorHost.className = 'exec-editor';
    wrapper.appendChild(editorHost);

    const toolbar = document.createElement('div');
    toolbar.className = 'exec-toolbar';

    const runBtn = document.createElement('button');
    runBtn.type = 'button';
    runBtn.className = 'exec-run';
    runBtn.textContent = pyEnv.getState().kind === 'ready' ? t.run : t.runBoot;

    const resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.className = 'exec-reset';
    resetBtn.textContent = t.reset;
    resetBtn.hidden = true;

    const editedBadge = document.createElement('span');
    editedBadge.className = 'exec-edited';
    editedBadge.textContent = t.edited;
    editedBadge.hidden = true;

    const timeLabel = document.createElement('span');
    timeLabel.className = 'exec-time';

    toolbar.append(runBtn, resetBtn, editedBadge, timeLabel);
    wrapper.appendChild(toolbar);

    const output = document.createElement('div');
    output.className = 'exec-output';
    output.hidden = true;
    wrapper.appendChild(output);

    // 直後の期待出力 (すでに details 化済み) を関連付ける
    const next = pre.nextElementSibling;
    const sampleOutput =
      next instanceof HTMLDetailsElement && next.classList.contains('sample-output-details')
        ? next
        : null;

    pre.replaceWith(wrapper);

    const block: ExecBlock = {
      code,
      view: createEditor(editorHost, code, (changed) => {
        editedBadge.hidden = !changed;
        resetBtn.hidden = !changed;
      }),
      runBtn,
      resetBtn,
      editedBadge,
      timeLabel,
      output,
      sampleOutput,
    };

    runBtn.addEventListener('click', () => void runBlock(block, chapter));
    resetBtn.addEventListener('click', () => {
      setEditorContent(block.view, block.code);
      block.editedBadge.hidden = true;
      block.resetBtn.hidden = true;
    });
  }

  // 準備完了したらボタンのラベルを通常表示に
  pyEnv.onState((state) => {
    const label = state.kind === 'ready' || state.kind === 'running' ? t.run : t.runBoot;
    for (const btn of document.querySelectorAll<HTMLButtonElement>('.exec-run:not(:disabled)')) {
      btn.textContent = label;
    }
  });
}

async function runBlock(block: ExecBlock, chapter: string) {
  const t = getStrings();
  block.runBtn.disabled = true;
  block.runBtn.textContent = t.running;
  block.runBtn.classList.add('spinning');

  try {
    await pyEnv.ensureBoot(chapter);
    const result = await pyEnv.run(block.view.state.doc.toString());
    renderResult(block, result);
  } catch (e) {
    renderResult(block, {
      ok: false,
      stdout: '',
      stderr: '',
      errorMessage: String(e),
      figures: [],
      durationMs: 0,
    });
  } finally {
    block.runBtn.disabled = false;
    block.runBtn.textContent = t.run;
    block.runBtn.classList.remove('spinning');
  }
}

function renderResult(
  block: ExecBlock,
  result: {
    ok: boolean;
    stdout: string;
    stderr: string;
    errorMessage?: string;
    figures: ArrayBuffer[];
    durationMs: number;
  },
) {
  const t = getStrings();
  const { output } = block;
  output.hidden = false;
  output.innerHTML = '';

  if (result.stdout) {
    const pre = document.createElement('pre');
    pre.className = 'out-stdout';
    pre.textContent = result.stdout;
    output.appendChild(pre);
  }

  for (const buf of result.figures) {
    const img = document.createElement('img');
    img.className = 'out-figure';
    img.alt = '';
    img.src = URL.createObjectURL(new Blob([buf], { type: 'image/png' }));
    output.appendChild(img);
  }

  if (result.stderr) {
    const pre = document.createElement('pre');
    pre.className = 'out-stderr';
    pre.textContent = result.stderr;
    output.appendChild(pre);
  }

  if (!result.ok && result.errorMessage) {
    const pre = document.createElement('pre');
    pre.className = 'out-error';
    pre.textContent = result.errorMessage;
    output.appendChild(pre);
  }

  if (result.ok && !result.stdout && !result.stderr && result.figures.length === 0) {
    const p = document.createElement('p');
    p.className = 'out-empty';
    p.textContent = t.noOutput;
    output.appendChild(p);
  }

  block.timeLabel.textContent = `${Math.round(result.durationMs)} ms`;

  // 実行に成功したら期待出力を折りたたむ (実出力と見比べられるように)
  if (result.ok && block.sampleOutput) {
    block.sampleOutput.open = false;
  }
}
