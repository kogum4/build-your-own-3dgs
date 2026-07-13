// 段落の中に画像が1枚だけある場合、<figure> + <figcaption>(alt文字列) に変換する。
// ソース教材の図は alt に「図1.4: キャプション」を持つため、これを可視キャプションにする。
import { visit } from 'unist-util-visit';

export function rehypeFigureCaption() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      if (!parent || index === undefined || node.tagName !== 'p') return;

      const meaningful = node.children.filter(
        (child) => !(child.type === 'text' && child.value.trim() === ''),
      );
      if (meaningful.length !== 1) return;
      const [img] = meaningful;
      if (img.type !== 'element' || img.tagName !== 'img') return;

      const alt = String(img.properties?.alt ?? '').trim();
      const figureChildren = [img];
      if (alt) {
        figureChildren.push({
          type: 'element',
          tagName: 'figcaption',
          properties: {},
          children: [{ type: 'text', value: alt }],
        });
      }

      parent.children[index] = {
        type: 'element',
        tagName: 'figure',
        properties: { className: ['chapter-figure'] },
        children: figureChildren,
      };
    });
  };
}
