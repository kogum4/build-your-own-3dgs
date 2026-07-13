import { ui, defaultLang, type Lang, type UiKey } from './ui';

/** base を除いた pathname の先頭セグメントから言語を判定する */
export function getLangFromUrl(url: URL): Lang {
  const path = stripBase(url.pathname);
  const [, first] = path.split('/');
  if (first === 'en') return 'en';
  return defaultLang;
}

export function useTranslations(lang: Lang) {
  return function t(key: UiKey, params?: Record<string, string | number>): string {
    let text: string = ui[lang][key] ?? ui[defaultLang][key];
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        text = text.replaceAll(`{${k}}`, String(v));
      }
    }
    return text;
  };
}

/** base prefix を付与する。path は '/' 始まり */
export function withBase(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/+$/, '');
  return `${base}${path}`;
}

/** 言語プレフィックスと base を付けたサイト内パスを作る。path は '/' 始まりの言語非依存パス */
export function localizePath(path: string, lang: Lang): string {
  const localized = lang === defaultLang ? path : `/${lang}${path === '/' ? '/' : path}`;
  return withBase(localized);
}

/** 現在の URL を別言語の対応ページの URL に変換する（言語切替リンク用） */
export function translatePath(url: URL, targetLang: Lang): string {
  let path = stripBase(url.pathname);
  // 現在の言語プレフィックスを除去
  if (path === '/en' || path === '/en/') path = '/';
  else if (path.startsWith('/en/')) path = path.slice('/en'.length);
  return localizePath(path, targetLang);
}

function stripBase(pathname: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/+$/, '');
  const stripped = base && pathname.startsWith(base) ? pathname.slice(base.length) : pathname;
  return stripped === '' ? '/' : stripped;
}
