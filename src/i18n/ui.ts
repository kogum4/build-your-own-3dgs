export const languages = {
  ja: '日本語',
  en: 'English',
} as const;

export type Lang = keyof typeof languages;

export const defaultLang: Lang = 'ja';

export const ui = {
  ja: {
    'site.title': 'Build Your Own 3D Gaussian Splatting',
    'site.tagline': 'NumPyだけで、ゼロから作る3D Gaussian Splatting',
    'site.description':
      'PythonとNumPyだけで3D Gaussian Splattingをスクラッチ実装するインタラクティブ教材。ブラウザ上でコードを実行しながら、数式と実装の対応を一歩ずつ学べます。',
    'nav.home': 'ホーム',
    'nav.chapters': '章一覧',
    'nav.github': 'GitHub',
    'nav.menu': 'メニュー',
    'chapter.label': '第{n}章',
    'chapter.comingSoon': 'Coming Soon',
    'chapter.start': '第1章を読む',
    'chapter.toc': 'この章の目次',
    'chapter.prev': '前の章',
    'chapter.next': '次の章',
    'chapter.backToTop': '章一覧へ戻る',
    'hero.lead':
      'PyTorchもCUDAも使わない。PythonとNumPyだけで、自動微分エンジンからラスタライザまで、3D Gaussian Splattingのすべてをゼロから実装する教材です。',
    'features.scratch.title': '本当にゼロから',
    'features.scratch.body':
      '使うのはNumPyだけ。自動微分もレンダラーもオプティマイザも、すべて自分の手で実装します。ブラックボックスは残しません。',
    'features.interactive.title': 'ブラウザで動かせる',
    'features.interactive.body':
      'コードブロックはその場で編集・実行できます。スライダーで数式のパラメータを動かして、挙動を直感的に確かめられます。',
    'features.stepbystep.title': '数式とコードを一歩ずつ',
    'features.stepbystep.body':
      '2Dの楕円を1つ描くところから始めて、16章かけて実写シーンの再構成まで。各章は前の章の完動コードの上に積み上がります。',
    'landing.chapters.title': '章一覧',
    'landing.chapters.publishedNote': '第1〜9章を公開中。続きの章も順次公開予定です。',
    'footer.textLicense': '本文・図',
    'footer.codeLicense': 'コード',
    'footer.sourceOn': 'ソースコード',
    'translation.banner.title': 'この章はまだ英語に翻訳されていません。',
    'translation.banner.body': '以下は日本語版の本文です。翻訳は順次公開予定です。',
    'translation.banner.link': '英語版の第1章を読む',
    'notfound.title': 'ページが見つかりません',
    'notfound.body': 'お探しのページは存在しないか、移動しました。',
    'notfound.back': 'トップページへ戻る',
  },
  en: {
    'site.title': 'Build Your Own 3D Gaussian Splatting',
    'site.tagline': 'Build 3D Gaussian Splatting from scratch, with nothing but NumPy',
    'site.description':
      'An interactive textbook that implements 3D Gaussian Splatting from scratch with Python and NumPy. Run the code in your browser and follow the math, one step at a time.',
    'nav.home': 'Home',
    'nav.chapters': 'Chapters',
    'nav.github': 'GitHub',
    'nav.menu': 'Menu',
    'chapter.label': 'Chapter {n}',
    'chapter.comingSoon': 'Coming Soon',
    'chapter.start': 'Start Chapter 1',
    'chapter.toc': 'On this page',
    'chapter.prev': 'Previous',
    'chapter.next': 'Next',
    'chapter.backToTop': 'All chapters',
    'hero.lead':
      'No PyTorch. No CUDA. Just Python and NumPy — build everything in 3D Gaussian Splatting from scratch, from an autograd engine to a rasterizer.',
    'features.scratch.title': 'Truly from scratch',
    'features.scratch.body':
      'NumPy is the only dependency. You implement the autograd engine, the renderer, and the optimizer yourself. No black boxes.',
    'features.interactive.title': 'Runs in your browser',
    'features.interactive.body':
      'Every code block can be edited and executed on the page. Drag sliders to see how each parameter shapes the math.',
    'features.stepbystep.title': 'Math and code, step by step',
    'features.stepbystep.body':
      'Start by drawing a single 2D ellipse, and work up to reconstructing a real scene over 16 chapters. Each chapter builds on working code from the last.',
    'landing.chapters.title': 'Chapters',
    'landing.chapters.publishedNote': 'Chapters 1–9 are available now. More chapters are on the way.',
    'footer.textLicense': 'Text & figures',
    'footer.codeLicense': 'Code',
    'footer.sourceOn': 'Source code on',
    'translation.banner.title': 'This chapter has not been translated into English yet.',
    'translation.banner.body': 'The Japanese version is shown below. Translations are coming soon.',
    'translation.banner.link': 'Read Chapter 1 in English',
    'notfound.title': 'Page not found',
    'notfound.body': 'The page you are looking for does not exist or has moved.',
    'notfound.back': 'Back to home',
  },
} as const satisfies Record<Lang, Record<string, string>>;

export type UiKey = keyof (typeof ui)[typeof defaultLang];
