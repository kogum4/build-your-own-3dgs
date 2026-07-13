// div.widget-mount[data-widget] を走査し、レジストリのウィジェットを遅延マウントする
import { registry } from './registry';
import { pyEnv } from '../runtime/py-env';

export interface WidgetContext {
  chapter: string;
  locale: 'ja' | 'en';
  /** Pyodide 環境 (必要なウィジェットだけが呼ぶ) */
  pyEnv: typeof pyEnv;
}

export type WidgetMountFn = (el: HTMLElement, ctx: WidgetContext) => void | (() => void);

export function mountWidgets(root: ParentNode, chapter: string) {
  const mounts = [...root.querySelectorAll<HTMLElement>('.widget-mount[data-widget]')];
  if (mounts.length === 0) return;

  const locale = document.documentElement.lang === 'en' ? 'en' : 'ja';
  const ctx: WidgetContext = { chapter, locale, pyEnv };

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        observer.unobserve(entry.target);
        void mountOne(entry.target as HTMLElement, ctx);
      }
    },
    { rootMargin: '360px 0px' },
  );
  mounts.forEach((el) => observer.observe(el));
}

async function mountOne(el: HTMLElement, ctx: WidgetContext) {
  const name = el.dataset.widget!;
  const loader = registry[name];
  if (!loader) {
    console.warn(`[widgets] unknown widget: ${name}`);
    el.classList.add('widget-unknown');
    return;
  }
  try {
    const mod = await loader();
    el.innerHTML = '';
    el.classList.add('widget-ready');
    mod.mount(el, ctx);
  } catch (e) {
    console.error(`[widgets] failed to mount ${name}:`, e);
  }
}
