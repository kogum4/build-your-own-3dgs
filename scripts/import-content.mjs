#!/usr/bin/env node
// 教材ソース (F:/Obsidian/.../output/chapters) をリポジトリへ取り込む変換スクリプト。
// 再実行可能・冪等。生成物 (src/content, src/assets, public/code) はコミットする。
//
//   pnpm import                     # 設定ファイルの対象章すべて
//   pnpm import -- --chapter 03     # 単章のみ
//   pnpm import -- --src <path>     # ソースルート上書き
//   pnpm import -- --lang en        # 言語 (既定 ja)。en は chapter-NN.en.md をソースにする
//
// 変換内容:
//   1. H1 からタイトル抽出 → frontmatter 化、本文から H1 除去
//   2. python フェンスに exec / file= / mode= メタを付与
//      - ソースのフェンスメタ (exec / noexec / file= / mode=) が最優先
//      - 先頭行がインデントされた断片コードは exec なし
//      - 直前の散文から「`X.py` を新規作成」「`X.py` の末尾に追加」等を検出して file= を推定
//   3. python ブロック直後の言語タグなしフェンス → ```text output (期待される出力)
//   4. 図参照 (figures/*.png) を src/assets へコピーしパス書換。生 HTML <img> は md 記法へ
//   5. 章直下の *.py を public/code/chapter-NN/ へコピーし manifest.json を再生成
//   6. 旧プロトタイプ記法 (::: file / ::: widget) を正規形へ変換

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');

// ---------- CLI / 設定 ----------
const argv = process.argv.slice(2);
const onlyChapters = [];
let srcOverride;
let lang = 'ja';
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--chapter') onlyChapters.push(String(argv[++i]).padStart(2, '0'));
  else if (argv[i] === '--src') srcOverride = argv[++i];
  else if (argv[i] === '--lang') lang = argv[++i];
}

const config = JSON.parse(fs.readFileSync(path.join(__dirname, 'import.config.json'), 'utf8'));
const srcRoot = srcOverride ?? config.srcRoot;
const chapters = onlyChapters.length > 0 ? onlyChapters : config.chapters;
const pyExclude = (config.pyExclude ?? []).map((p) => new RegExp(p));

if (!fs.existsSync(srcRoot)) {
  console.error(`ソースディレクトリが見つかりません: ${srcRoot}`);
  process.exit(1);
}

// ---------- ユーティリティ ----------
function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function copyIfChanged(src, dest) {
  const srcStat = fs.statSync(src);
  if (fs.existsSync(dest)) {
    const destStat = fs.statSync(dest);
    if (destStat.size === srcStat.size && destStat.mtimeMs >= srcStat.mtimeMs) return false;
  }
  ensureDir(path.dirname(dest));
  fs.copyFileSync(src, dest);
  return true;
}

function yamlString(s) {
  return JSON.stringify(s); // JSON 文字列は妥当な YAML
}

// ---------- 散文からのファイル指示検出 ----------
// 戻り値: { file, mode } または null
function detectFileInstruction(proseLines) {
  const text = proseLines.join('\n');
  const sentences = text.split('。');
  let result = null;
  for (const sentence of sentences) {
    const found = matchFileInstruction(sentence);
    if (found) result = found; // フェンスに近い(後の)文を優先
  }
  return result;
}

function matchFileInstruction(sentence) {
  const NEW_PATTERNS = [
    /`([\w./-]+\.py)`\s*(?:を|として)\s*(?:新規作成|作成)/,
    /以下を\s*`([\w./-]+\.py)`\s*として保存/,
  ];
  const APPEND_PATTERNS = [
    /`([\w./-]+\.py)`\s*(?:の末尾に|の続きに|に続けて|へ)[^。]{0,40}?(?:追加|追記)/,
    /`([\w./-]+\.py)`\s*に[^。]{0,40}?(?:追加|追記)/,
    /`([\w./-]+\.py)`\s*の\s*`?[\w]+`?\s*(?:クラス|関数)に[^。]{0,40}?(?:追加|追記)/,
    /`([\w./-]+\.py)`\s*の最終形/,
  ];
  for (const re of NEW_PATTERNS) {
    const m = sentence.match(re);
    if (m) return { file: m[1], mode: 'new' };
  }
  for (const re of APPEND_PATTERNS) {
    const m = sentence.match(re);
    if (m) return { file: m[1], mode: 'append' };
  }
  return null;
}

// ---------- Markdown 変換 ----------
function transformChapter(md, chapterSlug) {
  const audit = [];
  const referencedImages = new Set();
  const lines = md.replace(/^﻿/, '').split(/\r?\n/);
  const out = [];

  const assetPrefix = `../../../assets/chapters/chapter-${chapterSlug}`;

  // --- H1 抽出 ---
  let title = null;
  let firstContent = 0;
  while (firstContent < lines.length && lines[firstContent].trim() === '') firstContent++;
  const h1 = lines[firstContent]?.match(/^#\s+(?:第\d+章|Chapter\s+\d+)[:：]\s*(.+)$/);
  if (h1) {
    title = h1[1].replace(/`/g, '').trim();
    firstContent++;
  } else {
    console.warn(`  [warn] H1 が想定形式ではありません: ${lines[firstContent]}`);
  }

  // 散文バッファ: 直前のフェンス/見出し以降の本文行 (ファイル指示検出用)
  let proseBuffer = [];
  // 旧記法 ::: file の pending 状態
  let pendingLegacyFile = null;
  let inLegacyFileBlock = false;
  // 直前に出力した python ブロックの終了直後か (出力ブロックのペアリング用)
  let afterPythonBlock = false;

  function rewriteImages(line) {
    // Markdown 画像: ![alt](figures/xxx.png)
    let replaced = line.replace(/!\[([^\]]*)\]\((?:\.\/)?figures\/([^)\s]+)\)/g, (_m, alt, file) => {
      referencedImages.add(`figures/${file}`);
      return `![${alt}](${assetPrefix}/${file})`;
    });
    // 生 HTML: <img src="target.png" ...> → md 記法へ (章ルート直下の画像)
    replaced = replaced.replace(/<img\s+src="([\w./-]+\.(?:png|jpg|jpeg|gif))"[^>]*\/?>/g, (_m, file) => {
      const rel = file.replace(/^\.\//, '');
      referencedImages.add(rel);
      return `![](${assetPrefix}/${path.basename(rel)})`;
    });
    return replaced;
  }

  let i = firstContent;
  while (i < lines.length) {
    const line = lines[i];

    // --- 旧記法: ::: widget NAME ::: / ::: widget ::: ---
    const legacyWidget = line.match(/^:::\s*widget\s*([\w-]*)\s*:::\s*$/);
    if (legacyWidget) {
      const name = legacyWidget[1] || `ch${Number(chapterSlug)}-widget`;
      out.push(`::widget{name="${name}"}`);
      audit.push({ line: i + 1, kind: 'widget(legacy)', detail: name });
      i++;
      continue;
    }

    // --- 旧記法: ::: file NAME ---
    const legacyFile = line.match(/^:::\s*file\s+(.+?)\s*$/);
    if (legacyFile) {
      let name = legacyFile[1];
      let mode = 'new';
      if (/に追記$/.test(name)) {
        name = name.replace(/\s*に追記$/, '');
        mode = 'append';
      }
      pendingLegacyFile = { file: name, mode };
      inLegacyFileBlock = true;
      i++;
      continue;
    }
    if (inLegacyFileBlock && /^:::\s*$/.test(line)) {
      inLegacyFileBlock = false;
      i++;
      continue;
    }

    // --- フェンス開始 ---
    const fence = line.match(/^```(\S*)\s*(.*)$/);
    if (fence) {
      const fenceLang = fence[1];
      const origMeta = fence[2].trim();
      // 閉じフェンスを探す
      let j = i + 1;
      while (j < lines.length && !/^```\s*$/.test(lines[j])) j++;
      const codeLines = lines.slice(i + 1, j);

      if (fenceLang === 'python') {
        // メタ解析 (ソース明示が最優先)
        const metaTokens = origMeta.split(/\s+/).filter(Boolean);
        const explicit = {
          exec: metaTokens.includes('exec'),
          noexec: metaTokens.includes('noexec'),
          file: metaTokens.find((t) => t.startsWith('file='))?.slice(5),
          mode: metaTokens.find((t) => t.startsWith('mode='))?.slice(5),
        };

        const firstCode = codeLines.find((l) => l.trim() !== '') ?? '';
        const isFragment = /^\s/.test(firstCode);

        // Pyodide FS にはフラットに配置するため file= は常にベース名にする
        let file = explicit.file ?? pendingLegacyFile?.file;
        if (file) file = path.posix.basename(file.replace(/\\/g, '/'));
        let mode = explicit.mode ?? pendingLegacyFile?.mode;
        let source = explicit.file ? 'meta' : pendingLegacyFile ? 'legacy' : null;
        if (!file && !isFragment) {
          const detected = detectFileInstruction(proseBuffer);
          if (detected) {
            file = path.posix.basename(detected.file.replace(/\\/g, '/'));
            mode = detected.mode;
            source = 'prose';
          }
        }
        pendingLegacyFile = null;

        let exec;
        if (explicit.noexec) exec = false;
        else if (explicit.exec) exec = true;
        else exec = !isFragment;

        const newTokens = ['python'];
        if (exec) newTokens.push('exec');
        if (exec && file) {
          newTokens.push(`file=${file}`);
          if (mode === 'append') newTokens.push('mode=append');
        }
        out.push('```' + newTokens[0] + (newTokens.length > 1 ? ' ' + newTokens.slice(1).join(' ') : ''));
        out.push(...codeLines);
        out.push('```');
        audit.push({
          line: i + 1,
          kind: exec ? 'exec' : 'static',
          detail: file ? `${file}${mode === 'append' ? ' (追記)' : ''} [${source}]` : isFragment ? '断片' : '',
        });
        proseBuffer = [];
        afterPythonBlock = true;
        i = j + 1;
        continue;
      }

      if (fenceLang === '' && afterPythonBlock) {
        // python ブロック直後の裸フェンス → 期待される出力
        out.push('```text output');
        out.push(...codeLines);
        out.push('```');
        audit.push({ line: i + 1, kind: 'output', detail: `${codeLines.length}行` });
        proseBuffer = [];
        afterPythonBlock = false;
        i = j + 1;
        continue;
      }

      // その他のフェンスはそのまま
      out.push(line);
      out.push(...codeLines);
      out.push('```');
      proseBuffer = [];
      afterPythonBlock = false;
      i = j + 1;
      continue;
    }

    // --- 通常行 ---
    if (line.trim() !== '') {
      // 空行以外が現れたら「python 直後」状態を解除 (出力ペアリングは直後のみ)
      if (afterPythonBlock) afterPythonBlock = false;
      if (/^#{1,6}\s/.test(line)) proseBuffer = [];
      else proseBuffer.push(line);
    }
    // 1行 $$...$$ はインライン数式扱いになり \tag が KaTeX エラーになるため
    // 複数行のディスプレイ形式に正規化する
    const singleLineMath = line.match(/^\$\$(.+)\$\$\s*$/);
    if (singleLineMath) {
      out.push('$$', singleLineMath[1].trim(), '$$');
      i++;
      continue;
    }
    out.push(rewriteImages(line));
    i++;
  }

  // 末尾の連続空行を1つに
  while (out.length > 1 && out[out.length - 1].trim() === '' && out[out.length - 2].trim() === '') {
    out.pop();
  }

  return { title, body: out.join('\n'), audit, referencedImages };
}

// ---------- メイン ----------
let hadError = false;

for (const slug of chapters) {
  const chapterDir = path.join(srcRoot, `chapter-${slug}`);
  const mdName = lang === 'ja' ? `chapter-${slug}.md` : `chapter-${slug}.${lang}.md`;
  const mdPath = path.join(chapterDir, mdName);
  if (!fs.existsSync(mdPath)) {
    console.error(`[error] ソースが見つかりません: ${mdPath}`);
    hadError = true;
    continue;
  }

  console.log(`\n=== chapter-${slug} (${lang}) ===`);
  const md = fs.readFileSync(mdPath, 'utf8');
  const { title, body, audit, referencedImages } = transformChapter(md, slug);

  if (!title) {
    console.error(`[error] タイトルを抽出できませんでした: ${mdPath}`);
    hadError = true;
    continue;
  }

  // --- frontmatter + 本文書き出し ---
  const description = config.descriptions?.[slug] ?? '';
  const frontmatter = [
    '---',
    `title: ${yamlString(title)}`,
    `chapter: ${Number(slug)}`,
    `lang: ${lang}`,
    `description: ${yamlString(description)}`,
    'pyodide: true',
    '---',
    '',
  ].join('\n');

  const outMd = path.join(repoRoot, 'src', 'content', 'chapters', lang, `chapter-${slug}.md`);
  ensureDir(path.dirname(outMd));
  fs.writeFileSync(outMd, frontmatter + body + '\n', 'utf8');
  console.log(`  md: ${path.relative(repoRoot, outMd)} (title: ${title})`);

  // --- 図のコピー ---
  let copied = 0;
  let missing = 0;
  for (const rel of referencedImages) {
    const src = path.join(chapterDir, rel);
    if (!fs.existsSync(src)) {
      console.warn(`  [warn] 参照画像が見つかりません: ${rel}`);
      missing++;
      continue;
    }
    const dest = path.join(repoRoot, 'src', 'assets', 'chapters', `chapter-${slug}`, path.basename(rel));
    if (copyIfChanged(src, dest)) copied++;
  }
  console.log(`  図: ${referencedImages.size}枚参照 (コピー ${copied}, 欠落 ${missing})`);

  // --- .py スナップショットのコピー (ja のときのみ; en は共有) ---
  if (lang === 'ja') {
    const pyDest = path.join(repoRoot, 'public', 'code', `chapter-${slug}`);
    const pyFiles = fs
      .readdirSync(chapterDir)
      .filter((f) => f.endsWith('.py') && !pyExclude.some((re) => re.test(f)));
    // 章のスナップショットを完全に同期 (削除されたファイルも反映)
    if (fs.existsSync(pyDest)) fs.rmSync(pyDest, { recursive: true });
    ensureDir(pyDest);
    for (const f of pyFiles) {
      fs.copyFileSync(path.join(chapterDir, f), path.join(pyDest, f));
    }
    console.log(`  code: ${pyFiles.length}ファイル → public/code/chapter-${slug}/`);
  }

  // --- 変換の監査ログ ---
  for (const a of audit) {
    console.log(`    L${String(a.line).padStart(4)} ${a.kind.padEnd(14)} ${a.detail}`);
  }
}

// 注: chapters/common/ は manim ベースの図生成スクリプトのためランタイム配信しない

// --- manifest.json 再生成 (public/code の実態から) ---
const codeRoot = path.join(repoRoot, 'public', 'code');
const manifest = { chapters: {}, common: [] };
if (fs.existsSync(codeRoot)) {
  for (const entry of fs.readdirSync(codeRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const files = fs.readdirSync(path.join(codeRoot, entry.name)).filter((f) => f.endsWith('.py'));
    if (entry.name === 'common') manifest.common = files;
    else if (entry.name.startsWith('chapter-')) {
      manifest.chapters[entry.name.replace('chapter-', '')] = files;
    }
  }
  fs.writeFileSync(path.join(codeRoot, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  console.log(`manifest: public/code/manifest.json を再生成`);
}

process.exit(hadError ? 1 : 0);
