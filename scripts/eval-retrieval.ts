/**
 * 检索层评测：把 20 道金标准题跑一遍，报出 recall@k 与 MRR。
 *
 * 用法：pnpm eval                     默认参数
 *       pnpm eval -- --k1 1.2 --b 0.5 调参对比
 *       pnpm eval -- --verbose        逐题打印，看清楚每道题赢在哪、输在哪
 *
 * 全程零网络零 API：BM25 是纯统计，跑一次不到一秒——所以你可以放心地反复调参。
 */
import { readFileSync } from "node:fs";
import { loadCorpus } from "../src/corpus/load.js";
import { loadGold } from "../src/eval/gold.js";
import { summarize, toDocRanking } from "../src/eval/metrics.js";
import { buildBm25 } from "../src/retrieve/bm25.js";
import type { Chunk } from "../src/types.js";

const argv = process.argv.slice(2).filter((a) => a !== "--");
const flag = (name: string, dflt: number) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 && argv[i + 1] ? Number(argv[i + 1]) : dflt;
};
const k1 = flag("k1", 1.5);
const b = flag("b", 0.75);
const verbose = argv.includes("--verbose");

const chunks: Chunk[] = readFileSync("data/chunks.jsonl", "utf-8")
  .split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l) as Chunk);
const docs = loadCorpus();
const gold = loadGold();
const index = buildBm25(chunks, { k1, b });

console.log(`🔎 BM25 检索评测 | 文档 ${docs.length} 篇 / chunk ${index.size} 个 / 词表 ${index.vocabSize} 词`);
console.log(`   参数 k1=${k1} b=${b}\n`);

const t0 = performance.now();
const results = gold.map((c) => {
  const hits = index.search(c.question, 10);
  const ranked = toDocRanking(hits.map((h) => h.chunk.docId));
  return { case: c, hits, ranked };
});
const ms = performance.now() - t0;

const s = summarize(results.map((r) => ({ ranked: r.ranked, gold: r.case.goldDocIds })));

if (verbose) {
  for (const r of results) {
    const gold0 = r.case.goldDocIds;
    if (gold0.length === 0) {
      const top = r.hits[0];
      console.log(`  🚫 ${r.case.id} 拒答题｜最高分 ${top ? top.score.toFixed(2) : "0"}（${top?.chunk.docId ?? "无命中"}）`);
      continue;
    }
    const rank = r.ranked.findIndex((d) => gold0.includes(d));
    const mark = rank === 0 ? "✅" : rank > 0 && rank < 5 ? "🟡" : "❌";
    console.log(`  ${mark} ${r.case.id} 第 ${rank < 0 ? ">10" : rank + 1} 名｜${r.case.question}`);
    console.log(`      期望 ${gold0.join("/")}｜实得 ${r.ranked.slice(0, 3).join(" > ")}`);
    console.log(`      命中词 ${(r.hits[0]?.matched ?? []).join("、") || "（无）"}`);
  }
  console.log("");
}

console.log("📊 检索层成绩（文档级判定，20 题中 19 题计分，1 题为拒答题）");
console.log(`   recall@1  ${(s.recall[1]! * 100).toFixed(1)}%`);
console.log(`   recall@3  ${(s.recall[3]! * 100).toFixed(1)}%`);
console.log(`   recall@5  ${(s.recall[5]! * 100).toFixed(1)}%   ← 基线数字，记住它`);
console.log(`   recall@10 ${(s.recall[10]! * 100).toFixed(1)}%`);
console.log(`   MRR@10    ${s.mrr.toFixed(3)}`);
console.log(`   耗时      ${ms.toFixed(0)}ms（${(ms / gold.length).toFixed(1)}ms/题）`);

const refuse = results.find((r) => r.case.goldDocIds.length === 0);
if (refuse) {
  const top = refuse.hits[0];
  const best = results.filter((r) => r.case.goldDocIds.length > 0)
    .map((r) => r.hits[0]?.score ?? 0);
  const avg = best.reduce((x, y) => x + y, 0) / (best.length || 1);
  console.log(`\n🚫 拒答题体检：最高分 ${top ? top.score.toFixed(2) : "0"}，正常题平均最高分 ${avg.toFixed(2)}`);
  console.log(`   两者差距就是 Day 4 拒答阈值的立足点——差距越大，"我不知道"越好判。`);
}
