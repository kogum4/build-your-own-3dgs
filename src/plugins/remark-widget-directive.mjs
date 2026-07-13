// remark-directive の後段で ::widget{name="..."} をウィジェットのマウント要素に変換する。
// 契約: <div class="widget-mount" data-widget="NAME" [data-props="..."]> — クライアント側
// (src/client/widgets/mount.ts) が data-widget を走査してウィジェットをマウントする。
import { visit } from 'unist-util-visit';

export function remarkWidgetDirective() {
  return (tree, file) => {
    visit(tree, (node) => {
      if (
        node.type !== 'containerDirective' &&
        node.type !== 'leafDirective' &&
        node.type !== 'textDirective'
      ) {
        return;
      }

      if (node.name === 'widget') {
        const attrs = node.attributes ?? {};
        if (!attrs.name) {
          file.message('::widget directive is missing the required name attribute', node);
          return;
        }
        const data = node.data ?? (node.data = {});
        data.hName = 'div';
        data.hProperties = {
          className: ['widget-mount'],
          'data-widget': attrs.name,
          ...(attrs.props ? { 'data-props': attrs.props } : {}),
        };
        return;
      }

      // 教材本文中の「:見出し」のような偶発的マッチが黙って消えないよう、
      // widget 以外のディレクティブは警告してプレーンテキストへ戻す。
      file.message(`Unknown directive "${node.name}" was left as plain text`, node);
      node.type = 'text';
      node.value = `:${node.name}`;
      delete node.children;
    });
  };
}
