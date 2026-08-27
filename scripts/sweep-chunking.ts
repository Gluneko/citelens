/**
 * 切分策略 A/B：四种切法 × 两路检索，在同一把尺子上比。
 *
 * 用法：pnpm sweep              四种策略全跑（会给每种策略算一次向量，耗时几分钟）
 *       pnpm sweep -- --no-vec  只比 BM25（秒出，用来先看词法层的差异）
 *
 * 为什么这一步值得花几分钟：切分是整条 RAG 链路里【最便宜也最见效】的一环。
 * 改切法不用换模型、不用加算力，却常常比调参数管用得多。
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { chunkAllBy, type Strategy, type StrategyOptions } from "../src/corpus/chunk.js";
import { loadCorpus } from "../src/corpus/load.js";
import { loadGold } from "../src/eval/gold.js";
import { factComplete, factRecall, summarize, toDocRanking } from "../src/eval/metrics.js";
import { buildBm25 } from "../src/retrieve/bm25.js";
import { LocalEmbedder } from "../src/retrieve/embed.js";
import { packVectors, unpackVectors, VectorIndex, type VectorStoreFile } from "../src/retrieve/vector.js";
import type { Chunk } from "../src/types.js";

const argv = process.argv.slice(2).filter((a) => a !== "--");
const useVec = !argv.includes("--no-vec");

const PLANS: Array<{ strategy: Strategy; opts: StrategyOptions; label: string }> = [
  { strategy: "fixed", opts: { size: 300 }, label: "定长300" },
  { strategy: "overlap", opts: { size: 300, overlap: 60 }, label: "定长300+重叠60" },
  { strategy: "structure", opts: { size: 300 }, label: "按结构" },
  { strategy: "parent-child", opts: { childSize: 120, parentSize: 400 }, label: "父子块120/400" },
];

const docs = loadCorpus();
const gold = loadGold();
const embedder = useVec ? new LocalEmbedder() : null;

/** 喂给模型的文本：父子块策略下是父块，其他策略下就是块本身 */
const ctx = (c: Chunk) => c.context ?? c.text;

async function vectorsFor(tag: string, chunks: Chunk[]): Promise<Map<string, Float32Array>> {
  const path = `data/index/vectors-${tag}.json`;
  let cache = new Map<string, Float32Array>();
  if (existsSync(path)) {
    const old = JSON.parse(readFileSync(path, "utf-8")) as VectorStoreFile;
    if (old.model === embedder!.name) cache = unpackVectors(old);
  }
  const todo = chunks.filter((c) => !cache.has(c.id));
  if (todo.length) {
    process.stdout.write(`   算向量 ${todo.length} 条`);
    for (let i = 0; i < todo.length; i += 32) {
      const batch = todo.slice(i, i + 32);
      const vs = await embedder!.embed(batch.map((c) => `${c.title} ${c.text}`));
      batch.forEach((c, j) => cache.set(c.id, vs[j]!));
      process.stdout.write(".");
    }
    process.stdout.write("\n");
    mkdirSync("data/index", { recursive: true });
    const ids = chunks.map((c) => c.id);
    writeFileSync(path, JSON.stringify(packVectors(embedder!.name, embedder!.dim, ids, ids.map((id) => cache.get(id)!))));
  }
  return cache;
}

interface Row {
  label: string; retriever: string; nChunks: number; avgLen: number;
  r1: number; r5: number; mrr: number; fact: number; complete: number;
  g12: string; refuseRatio: number;
}
const rows: Row[] = [];

for (const plan of PLANS) {
  const chunks = chunkAllBy(docs, plan.strategy, plan.opts);
  const avgLen = chunks.reduce((s, c) => s + c.text.length, 0) / chunks.length;
  console.log(`\n📦 ${plan.label}｜${chunks.length} 块｜平均 ${avgLen.toFixed(0)} 字`);

  const runners: Array<{ name: string; run: (q: string) => Promise<Chunk[]> }> = [];
  const bm = buildBm25(chunks);
  runners.push({ name: "BM25", run: async (q) => bm.search(q, 10).map((h) => h.chunk) });

  if (useVec) {
    const cache = await vectorsFor(`${plan.strategy}`, chunks);
    const vi = new VectorIndex(embedder!.name, embedder!.dim);
    for (const c of chunks) { const v = cache.get(c.id); if (v) vi.add(c, v); }
    runners.push({ name: "向量", run: async (q) => (await vi.search(embedder!, q, 10)).map((h) => h.chunk) });
  }

  for (const r of runners) {
    const results = [];
    for (const c of gold) {
      const hits = await r.run(c.question);
      results.push({ c, hits, ranked: toDocRanking(hits.map((h) => h.docId)) });
    }
    const s = summarize(results.map((x) => ({ ranked: x.ranked, gold: x.c.goldDocIds })));
    const withFacts = results.filter((x) => (x.c.mustContain?.length ?? 0) > 0);
    const fact = withFacts.reduce((acc, x) => acc + factRecall(x.hits.slice(0, 5).map(ctx), x.c.mustContain!), 0) / withFacts.length;
    const comp = withFacts.reduce((acc, x) => acc + factComplete(x.hits.slice(0, 5).map(ctx), x.c.mustContain!), 0) / withFacts.length;
    const g12 = results.find((x) => x.c.id === "g12")!;
    const g12rank = g12.ranked.findIndex((d) => g12.c.goldDocIds.includes(d));
    rows.push({
      label: plan.label, retriever: r.name, nChunks: chunks.length, avgLen,
      r1: s.recall[1]!, r5: s.recall[5]!, mrr: s.mrr, fact, complete: comp,
      g12: g12rank < 0 ? ">10" : String(g12rank + 1),
      refuseRatio: 0,
    });
  }
}

console.log("\n\n📊 切分策略 × 检索方式");
const head = ["切分策略", "检索", "块数", "均长", "R@1", "R@5", "MRR", "事实", "不缺", "g12"];
const w = [18, 6, 6, 5, 7, 7, 7, 7, 7, 5];
const fmt = (cells: string[]) => cells.map((c, i) => c.padEnd(w[i]!)).join("");
console.log("   " + fmt(head));
console.log("   " + "─".repeat(w.reduce((a, b) => a + b, 0)));
for (const r of rows) {
  console.log("   " + fmt([
    r.label, r.retriever, String(r.nChunks), r.avgLen.toFixed(0),
    `${(r.r1 * 100).toFixed(1)}%`, `${(r.r5 * 100).toFixed(1)}%`, r.mrr.toFixed(3),
    `${(r.fact * 100).toFixed(1)}%`, `${(r.complete * 100).toFixed(1)}%`, r.g12,
  ]));
}
console.log("\n   g12 列 = 那道「火山石头上的小洞」在文档级排名（>10 表示前十未命中）");
console.log("   假设：块越小，气孔那句的信号越不被稀释，向量越该找得到它。看这一列验证。");
