// CodeMirror 6 の最小構成 (Python 編集用)
import { EditorState } from '@codemirror/state';
import { EditorView, keymap, highlightActiveLine } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import {
  HighlightStyle,
  syntaxHighlighting,
  indentUnit,
  bracketMatching,
} from '@codemirror/language';
import { tags } from '@lezer/highlight';
import { python } from '@codemirror/lang-python';

// 静的コードブロック (Shiki の nord) と揃えた穏やかなダークテーマ
const theme = EditorView.theme(
  {
    '&': {
      fontSize: '0.875rem',
      backgroundColor: 'var(--code-dark-bg, #2e3440)',
      color: 'var(--code-dark-fg, #d8dee9)',
    },
    '.cm-content': {
      fontFamily: 'var(--font-mono)',
      padding: '12px 0',
      lineHeight: '1.6',
      caretColor: '#d8dee9',
    },
    '.cm-line': {
      padding: '0 18px',
    },
    '&.cm-focused': {
      outline: 'none',
    },
    '.cm-cursor': {
      borderLeftColor: '#d8dee9',
    },
    '.cm-activeLine': {
      backgroundColor: 'rgba(216, 222, 233, 0.045)',
    },
    '.cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection': {
      backgroundColor: 'rgba(129, 161, 193, 0.3) !important',
    },
    '.cm-matchingBracket, &.cm-focused .cm-matchingBracket': {
      backgroundColor: 'rgba(136, 192, 208, 0.22)',
      outline: 'none',
    },
  },
  { dark: true },
);

// Nord のトークンカラー (Shiki 側と目視で一致する範囲の簡易マッピング)
const nordHighlight = HighlightStyle.define([
  { tag: [tags.keyword, tags.operator, tags.modifier], color: '#81a1c1' },
  { tag: [tags.string, tags.special(tags.string), tags.regexp], color: '#a3be8c' },
  { tag: [tags.comment, tags.docComment], color: '#616e88' },
  { tag: [tags.number, tags.bool, tags.atom, tags.null], color: '#b48ead' },
  { tag: [tags.function(tags.variableName), tags.function(tags.propertyName)], color: '#88c0d0' },
  { tag: [tags.className, tags.namespace], color: '#8fbcbb' },
  { tag: [tags.definition(tags.variableName), tags.propertyName], color: '#d8dee9' },
  { tag: tags.self, color: '#81a1c1' },
]);

export function createEditor(
  parent: HTMLElement,
  doc: string,
  onDocChanged: (changed: boolean) => void,
): EditorView {
  const original = doc;
  const state = EditorState.create({
    doc,
    extensions: [
      history(),
      bracketMatching(),
      highlightActiveLine(),
      indentUnit.of('    '),
      python(),
      syntaxHighlighting(nordHighlight),
      keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
      theme,
      EditorView.updateListener.of((update) => {
        if (update.docChanged) {
          onDocChanged(update.state.doc.toString() !== original);
        }
      }),
    ],
  });
  return new EditorView({ state, parent });
}

export function setEditorContent(view: EditorView, content: string) {
  view.dispatch({
    changes: { from: 0, to: view.state.doc.length, insert: content },
  });
}
