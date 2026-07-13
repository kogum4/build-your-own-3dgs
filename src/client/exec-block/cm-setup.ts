// CodeMirror 6 の最小構成 (Python 編集用)
import { EditorState } from '@codemirror/state';
import { EditorView, keymap, highlightActiveLine } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import {
  defaultHighlightStyle,
  syntaxHighlighting,
  indentUnit,
  bracketMatching,
} from '@codemirror/language';
import { python } from '@codemirror/lang-python';

const theme = EditorView.theme({
  '&': {
    fontSize: '0.875rem',
    backgroundColor: '#fff',
  },
  '.cm-content': {
    fontFamily: 'var(--font-mono)',
    padding: '12px 0',
    lineHeight: '1.6',
  },
  '.cm-line': {
    padding: '0 18px',
  },
  '&.cm-focused': {
    outline: 'none',
  },
  '.cm-activeLine': {
    backgroundColor: 'rgba(9, 105, 218, 0.04)',
  },
});

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
      syntaxHighlighting(defaultHighlightStyle),
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
