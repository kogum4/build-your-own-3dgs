// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import remarkDirective from 'remark-directive';
import rehypeKatex from 'rehype-katex';
import { remarkWidgetDirective } from './src/plugins/remark-widget-directive.mjs';
import { rehypeFigureCaption } from './src/plugins/rehype-figure-caption.mjs';
import { shikiExecTransformer } from './src/plugins/shiki-exec-transformer.mjs';

export default defineConfig({
  site: 'https://kogum4.github.io',
  base: '/build-your-own-3dgs',
  trailingSlash: 'always',
  integrations: [sitemap()],
  i18n: {
    defaultLocale: 'ja',
    locales: ['ja', 'en'],
    routing: {
      prefixDefaultLocale: false,
    },
  },
  vite: {
    worker: {
      format: 'es',
    },
  },
  markdown: {
    remarkPlugins: [remarkMath, remarkDirective, remarkWidgetDirective],
    rehypePlugins: [[rehypeKatex, { strict: false }], rehypeFigureCaption],
    shikiConfig: {
      // 通常のコードブロックはダーク、期待出力ブロックはライトを CSS 側で選択する
      themes: { light: 'github-light', dark: 'nord' },
      defaultColor: 'light',
      transformers: [shikiExecTransformer()],
    },
    smartypants: false,
  },
});
