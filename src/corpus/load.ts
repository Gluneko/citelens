/**
 * 语料装载：读 data/corpus/*.md，解析极简 frontmatter。
 * 不引第三方 md 解析库——语料格式是我们自己定的，越简单越不会出事。
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { basename, join } from "node:path";
import type { Doc } from "../types.js";

const FM = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;

export function parseDoc(id: string, raw: string): Doc {
  const m = raw.match(FM);
  const meta: Record<string, string> = {};
  let text = raw;
  if (m) {
    for (const line of m[1]!.split(/\r?\n/)) {
      const i = line.indexOf(":");
      if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
    }
    text = raw.slice(m[0].length);
  }
  return {
    id,
    title: meta.title ?? id,
    text: text.trim(),
    source: meta.source ?? "unknown",
    license: meta.license ?? "unknown",
  };
}

/**
 * 装载语料。默认读两个目录：
 *   data/corpus       —— 12 篇合成语料，入库，保证 clone 即可跑（离线夹具）
 *   data/corpus-wiki  —— 维基抓取语料，不入库，需自行 pnpm tsx scripts/fetch-wiki.ts
 * 抓取语料只是把"干草堆"变大，金标准答案始终落在合成语料里——
 * 语料越大，同一套金标准题就越难，指标才有区分度。
 */
export function loadCorpus(dirs: string[] = ["data/corpus", "data/corpus-wiki"]): Doc[] {
  const docs: Doc[] = [];
  for (const dir of dirs) {
    if (!existsSync(dir)) continue;
    for (const f of readdirSync(dir).filter((x) => x.endsWith(".md")).sort()) {
      docs.push(parseDoc(basename(f, ".md"), readFileSync(join(dir, f), "utf-8")));
    }
  }
  return docs;
}
