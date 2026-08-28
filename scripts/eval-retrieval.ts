/**
 * 检索层评测 CLI。
 *
 * 用法：
 *   pnpm eval                                 BM25 单路（默认）
 *   pnpm eval -- --mode vector                语义单路
 *   pnpm eval -- --mode hybrid                两路 RRF 融合
 *   pnpm eval -- --mode hybrid --rerank       融合召回 + cross-encoder 精排 ← Day 3 全家桶
 *   pnpm eval -- --compare                    四种配置一次跑完并排对照
 *   pnpm eval -- --verbose                    逐题详情（精排模式下会显示"从第几名被顶上来"）
 *   pnpm eval -- --pool 50                    候选池大小（默认 50）
 *   pnpm eval -- --guarantee 15               混合模式下每路保底名额（默认 15，设 0 关闭）
 *   pnpm eval -- --rewrite                    LLM 查询改写（口语→术语，多路召回合并；需 API key）
 *   pnpm eval -- --hyde                       HyDE：先编假答案再拿它检索（需 API key）
 *   pnpm eval -- --graph                      文档图谱多跳扩展（标题提及建边，离线）
 *   pnpm eval -- --rerank-top 20              只精排候选池前 N 条（默认全池，精排很贵）
 */
import { existsSync, readFileSync } from "node:fs";
import { loadCorpus } from "../src/corpus/load.js";
import { loadGold } from "../src/eval/gold.js";
import { runCases, summarizeRun, type EvalReport } from "../src/eval/run.js";
import { buildBm25 } from "../src/retrieve/bm25.js";
import { buildDocGraph, graphStats } from "../src/retrieve/graph.js";
import { HydeExpander, LlmRewriter, type QueryExpander } from "../src/retrieve/rewrite.js";
import { requireApiKey } from "../src/config.js";
import { LocalEmbedder } from "../src/retrieve/embed.js";
import { RetrievalPipeline, type RetrieveMode } from "../src/retrieve/pipeline.js";
import { CrossEncoderReranker } from "../src/retrieve/rerank.js";
import { unpackVectors, VectorIndex, type VectorStoreFile } from "../src/retrieve/vector.js";
import type { Chunk } from "../src/types.js";

const argv = process.argv.slice(2).filter((a) => a !== "--");
const opt = (n: string) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : undefined; };
const verbose = argv.includes("--verbose");
const compare = argv.includes("--compare");
const poolSize = Number(opt("pool") ?? 50);
const topK = Number(opt("top") ?? 5);
const guarantee = Number(opt("guarantee") ?? 15);
const rerankTop = Number(opt("rerank-top") ?? 0);

const VEC = "data/index/vectors.json";
const chunks: Chunk[] = readFileSync("data/chunks.jsonl", "utf-8")
  .split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l) as Chunk);
const docs = loadCorpus();
const gold = loadGold();

const bm25 = buildBm25(chunks);
let vector: VectorIndex | undefined;
let embedder: LocalEmbedder | undefined;
if (existsSync(VEC)) {
  const file = JSON.parse(readFileSync(VEC, "utf-8")) as VectorStoreFile;
  const vecs = unpackVectors(file);
  vector = new VectorIndex(file.model, file.dim);
  for (const c of chunks) { const v = vecs.get(c.id); if (v) vector.add(c, v); }
  embedder = new LocalEmbedder(file.model);
  if (vector.size !== chunks.length) {
    console.error(`⚠️  向量只覆盖 ${vector.size}/${chunks.length} 块——切分改过？重跑 pnpm vectors`);
  }
}

console.log(`📚 文档 ${docs.length} 篇｜chunk ${chunks.length}｜候选池 ${poolSize}｜保底 ${guarantee}｜精排前 ${rerankTop || poolSize}｜返回 ${topK}`);
if (vector) console.log(`🧠 向量 ${vector.model}`);

const plans: Array<{ mode: RetrieveMode; rerank: boolean }> = compare
  ? [
      { mode: "bm25", rerank: false },
      { mode: "vector", rerank: false },
      { mode: "hybrid", rerank: false },
      { mode: "vector", rerank: true },
      { mode: "hybrid", rerank: true },
    ]
  : [{ mode: (opt("mode") ?? "bm25") as RetrieveMode, rerank: argv.includes("--rerank") }];

let expander: QueryExpander | undefined;
if (argv.includes("--rewrite")) expander = new LlmRewriter();
else if (argv.includes("--hyde")) expander = new HydeExpander();
if (expander) {
  requireApiKey();
  console.log(`✍️  查询扩写：${expander.name}（每题一次 LLM 调用）`);
}
// 扩写结果按题缓存：--compare 下四种配置共用同一份扩写，省钱且变量可控
const expandCache = new Map<string, string[]>();
const cachedExpander = expander
  ? {
      name: expander.name,
      expand: async (q: string) => {
        const hit = expandCache.get(q);
        if (hit) return { queries: hit, cost: 0 };
        const r = await expander!.expand(q);
        expandCache.set(q, r.queries);
        return r;
      },
    }
  : undefined;

const useGraph = argv.includes("--graph");
let graph;
if (useGraph) {
  graph = buildDocGraph(docs);
  const gs = graphStats(graph);
  console.log(`🕸  文档图谱：${gs.nodes} 节点 / ${gs.edges} 边｜孤点 ${gs.isolated.length}｜枢纽 ${gs.topHubs.slice(0, 3).map(([d, n]) => `${d}(${n})`).join(" ")}`);
}

const reranker = plans.some((p) => p.rerank) ? new CrossEncoderReranker() : undefined;
const reports: EvalReport[] = [];
const lastOutcomes = new Map<string, Awaited<ReturnType<typeof runCases>>>();

for (const plan of plans) {
  if (plan.mode !== "bm25" && !vector) {
    console.error(`⏭  跳过 ${plan.mode}：未找到 ${VEC}，先跑 pnpm vectors`);
    continue;
  }
  const pipeline = new RetrievalPipeline(
    { bm25, vector, embedder, reranker: plan.rerank ? reranker : undefined, graph, allChunks: useGraph ? chunks : undefined },
    { mode: plan.mode, poolSize, topK, guarantee, rerankTop, graphExpand: useGraph ? 3 : 0 },
  );
  const t0 = performance.now();
  const outcomes = await runCases(pipeline, gold, { expander: cachedExpander });
  const ms = performance.now() - t0;
  reports.push(summarizeRun(pipeline.label, outcomes, ms));
  lastOutcomes.set(pipeline.label, outcomes);
  console.log(`⏱  ${pipeline.label}：${ms.toFixed(0)}ms（${(ms / gold.length).toFixed(1)}ms/题）`);
}

if (verbose) {
  for (const [label, outcomes] of lastOutcomes) {
    console.log(`\n📋 ${label} 逐题`);
    for (const o of outcomes) {
      if (o.case.goldDocIds.length === 0) {
        console.log(`  🚫 ${o.case.id} 拒答题`);
        continue;
      }
      const mark = o.rank === 0 ? "✅" : o.rank > 0 && o.rank < 5 ? "🟡" : "❌";
      const pool = o.poolRank < 0
        ? "候选池也没有"
        : `候选池文档第 ${o.poolRank + 1}/片段第 ${o.poolRankChunk + 1}`;
      const cut = rerankTop > 0 && o.poolRankChunk >= rerankTop
        ? `｜⚠️ 片段第 ${o.poolRankChunk + 1} 位 > 精排前 ${rerankTop}，根本没进精排`
        : "";
      const lift = o.liftedFrom ? `｜精排把前 ${topK} 名从候选 ${o.liftedFrom.join("/")} 位顶上来` : "";
      console.log(`  ${mark} ${o.case.id} 第 ${o.rank < 0 ? ">" + topK : o.rank + 1} 名（${pool}）${cut}${lift}`);
      if (o.rank !== 0) console.log(`      ${o.case.question}`);
    }
  }
}

console.log("\n📊 成绩对照");
const pad = (s: string, n: number) => s.padEnd(n);
const W = 14;
console.log(`   ${pad("指标", 16)}${reports.map((r) => pad(r.label, W)).join("")}`);
const rows: Array<[string, (r: EvalReport) => string]> = [
  ["候选池召回率", (r) => `${(r.poolRecall * 100).toFixed(1)}%`],
  ["recall@1", (r) => `${(r.recall[1]! * 100).toFixed(1)}%`],
  ["recall@3", (r) => `${(r.recall[3]! * 100).toFixed(1)}%`],
  ["recall@5", (r) => `${(r.recall[5]! * 100).toFixed(1)}%`],
  ["MRR@10", (r) => r.mrr.toFixed(3)],
  ["事实召回率@5", (r) => `${(r.factRecall * 100).toFixed(1)}%`],
  ["一条不缺率@5", (r) => `${(r.factComplete * 100).toFixed(1)}%`],
  ["耗时/题", (r) => `${(r.ms / gold.length).toFixed(1)}ms`],
];
for (const [name, fn] of rows) {
  console.log(`   ${pad(name, 16)}${reports.map((r) => pad(fn(r), W)).join("")}`);
}
for (const r of reports) {
  if (r.missingFacts.length) {
    console.log(`\n   ${r.label} 缺事实：${r.missingFacts.map((m) => `${m.id}（缺「${m.missing.join("、")}」）`).join("，")}`);
  }
}
console.log(`\n   候选池召回率 = 召回阶段有没有把正确答案带进来。它是精排的天花板：`);
console.log(`   召回漏了，精排再强也救不回；召回带进来了，剩下的就是排序问题。`);
