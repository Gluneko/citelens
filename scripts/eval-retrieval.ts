/**
 * 检索层评测：把 20 道金标准题跑一遍，报 recall@k / MRR / 事实层成绩。
 *
 * 用法：
 *   pnpm eval                      默认 BM25
 *   pnpm eval -- --mode vector     向量检索（需先 pnpm vectors）
 *   pnpm eval -- --mode both       两路并排对照 ← Day 2 的重头戏
 *   pnpm eval -- --verbose         逐题详情
 *   pnpm eval -- --k1 1.2 --b 0.5  BM25 调参
 */
import { existsSync, readFileSync } from "node:fs";
import { loadCorpus } from "../src/corpus/load.js";
import { loadGold } from "../src/eval/gold.js";
import { factComplete, factRecall, summarize, toDocRanking } from "../src/eval/metrics.js";
import { buildBm25 } from "../src/retrieve/bm25.js";
import { LocalEmbedder } from "../src/retrieve/embed.js";
import { unpackVectors, VectorIndex, type VectorStoreFile } from "../src/retrieve/vector.js";
import type { Chunk, GoldCase } from "../src/types.js";

const argv = process.argv.slice(2).filter((a) => a !== "--");
const opt = (name: string) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : undefined;
};
const k1 = Number(opt("k1") ?? 1.5);
const b = Number(opt("b") ?? 0.75);
const verbose = argv.includes("--verbose");
const mode = (opt("mode") ?? "bm25") as "bm25" | "vector" | "both";

const VEC = "data/index/vectors.json";
if (mode !== "bm25" && !existsSync(VEC)) {
  console.error(`❌ 找不到 ${VEC}，请先跑：pnpm vectors`);
  process.exit(1);
}

const chunks: Chunk[] = readFileSync("data/chunks.jsonl", "utf-8")
  .split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l) as Chunk);
const docs = loadCorpus();
const gold = loadGold();

interface Retrieved { chunks: Chunk[]; scores: number[] }
type Runner = (q: string) => Promise<Retrieved>;

const runners: Array<{ label: string; run: Runner }> = [];

if (mode === "bm25" || mode === "both") {
  const index = buildBm25(chunks, { k1, b });
  console.log(`🔎 BM25｜chunk ${index.size}｜词表 ${index.vocabSize} 词｜k1=${k1} b=${b}`);
  runners.push({
    label: "BM25",
    run: async (q) => {
      const hits = index.search(q, 10);
      return { chunks: hits.map((h) => h.chunk), scores: hits.map((h) => h.score) };
    },
  });
}

if (mode === "vector" || mode === "both") {
  const file = JSON.parse(readFileSync(VEC, "utf-8")) as VectorStoreFile;
  const vecs = unpackVectors(file);
  const vindex = new VectorIndex(file.model, file.dim);
  for (const c of chunks) {
    const v = vecs.get(c.id);
    if (v) vindex.add(c, v);
  }
  if (vindex.size !== chunks.length) {
    console.error(`⚠️  向量缓存只覆盖 ${vindex.size}/${chunks.length} 个 chunk——切分参数改过？请重跑 pnpm vectors`);
  }
  const embedder = new LocalEmbedder(file.model);
  console.log(`🧠 向量｜${file.model}｜${vindex.size} 条 × ${file.dim} 维`);
  runners.push({
    label: "向量",
    run: async (q) => {
      const hits = await vindex.search(embedder, q, 10);
      return { chunks: hits.map((h) => h.chunk), scores: hits.map((h) => h.score) };
    },
  });
}

console.log(`📚 文档 ${docs.length} 篇\n`);

interface CaseResult { case: GoldCase; rank: number; ranked: string[]; res: Retrieved }
const byRunner = new Map<string, CaseResult[]>();

for (const r of runners) {
  const out: CaseResult[] = [];
  const t0 = performance.now();
  for (const c of gold) {
    const res = await r.run(c.question);
    const ranked = toDocRanking(res.chunks.map((x) => x.docId));
    out.push({ case: c, ranked, res, rank: ranked.findIndex((d) => c.goldDocIds.includes(d)) });
  }
  const ms = performance.now() - t0;
  byRunner.set(r.label, out);
  console.log(`⏱  ${r.label}：${ms.toFixed(0)}ms（${(ms / gold.length).toFixed(1)}ms/题）`);
}

function report(label: string, rs: CaseResult[]) {
  const s = summarize(rs.map((r) => ({ ranked: r.ranked, gold: r.case.goldDocIds })));
  const facts = rs.filter((r) => (r.case.mustContain?.length ?? 0) > 0).map((r) => {
    const texts = r.res.chunks.slice(0, 5).map((c) => c.text);
    return {
      id: r.case.id,
      recall: factRecall(texts, r.case.mustContain!),
      complete: factComplete(texts, r.case.mustContain!),
      missing: r.case.mustContain!.filter((m) => !texts.join("\n").includes(m)),
    };
  });
  const fAvg = facts.reduce((x, y) => x + y.recall, 0) / (facts.length || 1);
  const fComp = facts.reduce((x, y) => x + y.complete, 0) / (facts.length || 1);
  const refuse = rs.find((r) => r.case.goldDocIds.length === 0);
  const normalTop = rs.filter((r) => r.case.goldDocIds.length > 0).map((r) => r.res.scores[0] ?? 0);
  const avgTop = normalTop.reduce((x, y) => x + y, 0) / (normalTop.length || 1);
  return {
    label, s, fAvg, fComp,
    broken: facts.filter((f) => f.complete === 0),
    refuseTop: refuse?.res.scores[0] ?? 0,
    avgTop,
  };
}

const reports = [...byRunner].map(([label, rs]) => report(label, rs));

if (verbose && mode === "both") {
  const [A, B] = [...byRunner.values()];
  console.log("\n📋 逐题对照（名次越小越好，>10 表示前十未命中）");
  console.log("   题号  BM25  向量   胜负   问题");
  for (let i = 0; i < gold.length; i++) {
    const c = gold[i]!;
    const ra = A![i]!.rank, rb = B![i]!.rank;
    if (c.goldDocIds.length === 0) {
      console.log(`   ${c.id}    —     —    拒答题  最高分 BM25 ${A![i]!.res.scores[0]?.toFixed(2)} / 向量 ${B![i]!.res.scores[0]?.toFixed(3)}`);
      continue;
    }
    const fmt = (r: number) => (r < 0 ? ">10" : String(r + 1)).padStart(4);
    const win = ra === rb ? " 平手 " : (ra >= 0 && (rb < 0 || ra < rb)) ? "BM25赢" : "向量赢";
    console.log(`   ${c.id} ${fmt(ra)}  ${fmt(rb)}   ${win}  ${c.question}`);
  }
} else if (verbose) {
  for (const [label, rs] of byRunner) {
    console.log(`\n📋 ${label} 逐题`);
    for (const r of rs) {
      if (r.case.goldDocIds.length === 0) {
        console.log(`  🚫 ${r.case.id} 拒答题｜最高分 ${r.res.scores[0]?.toFixed(3) ?? 0}`);
        continue;
      }
      const mark = r.rank === 0 ? "✅" : r.rank > 0 && r.rank < 5 ? "🟡" : "❌";
      console.log(`  ${mark} ${r.case.id} 第 ${r.rank < 0 ? ">10" : r.rank + 1} 名｜${r.case.question}`);
      console.log(`      期望 ${r.case.goldDocIds.join("/")}｜实得 ${r.ranked.slice(0, 3).join(" > ")}`);
    }
  }
}

console.log("\n📊 成绩对照");
const pad = (s: string, n: number) => s.padEnd(n);
console.log(`   ${pad("指标", 16)}${reports.map((r) => pad(r.label, 12)).join("")}`);
const rows: Array<[string, (r: typeof reports[number]) => string]> = [
  ["recall@1", (r) => `${(r.s.recall[1]! * 100).toFixed(1)}%`],
  ["recall@3", (r) => `${(r.s.recall[3]! * 100).toFixed(1)}%`],
  ["recall@5", (r) => `${(r.s.recall[5]! * 100).toFixed(1)}%`],
  ["recall@10", (r) => `${(r.s.recall[10]! * 100).toFixed(1)}%`],
  ["MRR@10", (r) => r.s.mrr.toFixed(3)],
  ["事实召回率@5", (r) => `${(r.fAvg * 100).toFixed(1)}%`],
  ["一条不缺率@5", (r) => `${(r.fComp * 100).toFixed(1)}%`],
];
for (const [name, fn] of rows) {
  console.log(`   ${pad(name, 16)}${reports.map((r) => pad(fn(r), 12)).join("")}`);
}

for (const r of reports) {
  if (r.broken.length) {
    console.log(`\n   ${r.label} 缺事实的题：${r.broken.map((f) => `${f.id}（缺「${f.missing.join("、")}」）`).join("，")}`);
  }
  console.log(`   ${r.label} 拒答题体检：最高分 ${r.refuseTop.toFixed(3)}，正常题平均最高分 ${r.avgTop.toFixed(3)}，比值 ${(r.refuseTop / (r.avgTop || 1)).toFixed(2)}`);
}
