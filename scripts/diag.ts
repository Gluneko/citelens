/**
 * 单题诊断：这道题为什么失手？
 *
 * 用法：pnpm diag g12                                  只看两路原始召回
 *       pnpm diag g12 --mode vector --rerank           跑完整流水线，看精排最终挑了什么
 *
 * （名字别叫 why——pnpm 有同名内置命令，会把脚本挡掉且静默无输出。踩过。）
 *
 * 它回答一个评测报告答不了的问题：
 * 「答案所在的那个片段，到底排第几？分数差在哪？」
 * 检索调优最忌讳凭感觉改参数——先看清楚失手的形状，再决定动哪一层。
 */
import { existsSync, readFileSync } from "node:fs";
import { loadGold } from "../src/eval/gold.js";
import { buildBm25 } from "../src/retrieve/bm25.js";
import { LocalEmbedder } from "../src/retrieve/embed.js";
import { tokenize } from "../src/retrieve/tokenize.js";
import { RetrievalPipeline, type RetrieveMode } from "../src/retrieve/pipeline.js";
import { CrossEncoderReranker } from "../src/retrieve/rerank.js";
import { unpackVectors, VectorIndex, type VectorStoreFile } from "../src/retrieve/vector.js";
import type { Chunk } from "../src/types.js";

const rawArgs = process.argv.slice(2).filter((a) => a !== "--");
const id = rawArgs.find((a) => !a.startsWith("--"));
const argOf = (n: string) => { const i = rawArgs.indexOf(`--${n}`); return i >= 0 ? rawArgs[i + 1] : undefined; };
const wantPipeline = rawArgs.includes("--rerank") || rawArgs.includes("--mode");
if (!id) { console.error("用法：pnpm diag g12"); process.exit(1); }

const c = loadGold().find((x) => x.id === id);
if (!c) { console.error(`没有这道题：${id}`); process.exit(1); }

const chunks: Chunk[] = readFileSync("data/chunks.jsonl", "utf-8")
  .split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l) as Chunk);

console.log(`\n❓ ${c.id}｜${c.question}`);
console.log(`   期望文档 ${c.goldDocIds.join("/") || "（拒答题）"}｜关键事实 ${(c.mustContain ?? []).join("、") || "（无）"}`);
console.log(`   查询分词 ${tokenize(c.question).join(" / ")}`);

// 找出"真正含答案"的片段：属于标注文档且含关键事实
const must = c.mustContain ?? [];
const answerChunks = chunks.filter(
  (x) => c.goldDocIds.includes(x.docId) && must.some((m) => x.text.includes(m)),
);
// 全库里还有谁提到了同样的关键事实——语料一大，"正确答案"往往不止一处
const alsoHasFact = chunks.filter(
  (x) => !c.goldDocIds.includes(x.docId) && must.some((m) => x.text.includes(m)),
);
console.log(`\n📌 含答案的片段：${answerChunks.length} 个`);
for (const a of answerChunks) {
  const hitFacts = must.filter((m) => a.text.includes(m));
  console.log(`   ${a.id}（含「${hitFacts.join("、")}」）`);
  console.log(`     ${a.text.slice(0, 90).replace(/\n/g, " ")}…`);
}
if (!answerChunks.length) {
  console.log("   ⚠️ 一个都没有——要么关键事实标错了，要么切分把答案切碎了");
}
if (alsoHasFact.length) {
  console.log(`\n📎 标注文档之外，还有 ${alsoHasFact.length} 个片段也含该关键事实：`);
  for (const a of alsoHasFact.slice(0, 6)) {
    console.log(`   ${a.id}｜${a.text.slice(0, 60).replace(/\n/g, " ")}…`);
  }
  console.log("   （它们同样能支撑正确回答——文档级判定会把命中它们算作失手，事实级不会）");
}

const bm = buildBm25(chunks);
const bmAll = bm.search(c.question, chunks.length);
console.log(`\n🔎 BM25 前 5`);
for (const [i, h] of bmAll.slice(0, 5).entries()) {
  console.log(`   ${i + 1}. ${h.chunk.id}（${h.score.toFixed(2)}）命中词 ${h.matched.join("、") || "（无）"}`);
  console.log(`      ${(h.chunk.text).slice(0, 70).replace(/\n/g, " ")}…`);
}
for (const a of answerChunks) {
  const r = bmAll.findIndex((h) => h.chunk.id === a.id);
  const hit = r >= 0 ? bmAll[r]! : undefined;
  console.log(`   答案片段 ${a.id} 排第 ${r < 0 ? "未命中" : r + 1} 名｜分数 ${hit?.score.toFixed(2) ?? 0}｜命中词 ${hit?.matched.join("、") || "（无）"}`);
}

const VEC = "data/index/vectors.json";
if (existsSync(VEC)) {
  const file = JSON.parse(readFileSync(VEC, "utf-8")) as VectorStoreFile;
  const vecs = unpackVectors(file);
  const vi = new VectorIndex(file.model, file.dim);
  for (const x of chunks) { const v = vecs.get(x.id); if (v) vi.add(x, v); }
  const emb = new LocalEmbedder(file.model);
  const vAll = await vi.search(emb, c.question, chunks.length);
  console.log(`\n🧠 向量前 5（${file.model}）`);
  for (const [i, h] of vAll.slice(0, 5).entries()) {
    console.log(`   ${i + 1}. ${h.chunk.id}（余弦 ${h.score.toFixed(3)}）`);
    console.log(`      ${(h.chunk.text).slice(0, 70).replace(/\n/g, " ")}…`);
  }
  for (const a of answerChunks) {
    const r = vAll.findIndex((h) => h.chunk.id === a.id);
    console.log(`   答案片段 ${a.id} 排第 ${r < 0 ? "未命中" : r + 1} 名｜余弦 ${vAll[r]?.score.toFixed(3) ?? 0}`);
  }
  const top = vAll[0]!;
  console.log(`\n   分数区间：最高 ${top.score.toFixed(3)}，第 10 名 ${vAll[9]?.score.toFixed(3)}，最低 ${vAll.at(-1)!.score.toFixed(3)}`);
  console.log(`   （余弦分数挤在一个很窄的区间里，是向量检索的通病——Day 4 讲拒答阈值时会用到这个观察）`);
} else {
  console.log(`\n（未找到 ${VEC}，跳过向量诊断；先跑 pnpm vectors）`);
}

// ---------- 完整流水线诊断：精排到底挑了什么 ----------
if (wantPipeline) {
  if (!existsSync(VEC)) {
    console.log("\n（跑流水线需要向量，先 pnpm vectors）");
  } else {
    const file = JSON.parse(readFileSync(VEC, "utf-8")) as VectorStoreFile;
    const vecs = unpackVectors(file);
    const vi = new VectorIndex(file.model, file.dim);
    for (const x of chunks) { const v = vecs.get(x.id); if (v) vi.add(x, v); }
    const emb = new LocalEmbedder(file.model);
    const useRerank = rawArgs.includes("--rerank");
    const pipe = new RetrievalPipeline(
      { bm25: bm, vector: vi, embedder: emb, reranker: useRerank ? new CrossEncoderReranker() : undefined },
      {
        mode: (argOf("mode") ?? "hybrid") as RetrieveMode,
        poolSize: Number(argOf("pool") ?? 50),
        topK: Number(argOf("top") ?? 5),
        guarantee: Number(argOf("guarantee") ?? 15),
        rerankTop: Number(argOf("rerank-top") ?? 0),
      },
    );
    const r = await pipe.search(c.question);
    console.log(`\n🎯 ${pipe.label} 最终前 ${r.hits.length}`);
    for (const [i, h] of r.hits.entries()) {
      const from = r.liftedFrom ? `（候选第 ${r.liftedFrom[i]} 位）` : "";
      const hasFact = must.filter((m) => (h.context ?? h.text).includes(m));
      const flag = hasFact.length ? ` ✅含「${hasFact.join("、")}」` : "";
      console.log(`   ${i + 1}. ${h.id}${from}${flag}`);
      console.log(`      ${(h.context ?? h.text).slice(0, 80).replace(/\n/g, " ")}…`);
    }
    const poolIdx = r.pool.findIndex((x) => answerChunks.some((a) => a.id === x.id));
    const rtop = Number(argOf("rerank-top") ?? 0);
    console.log(`\n   答案片段在候选池【片段级】第 ${poolIdx < 0 ? "未进池" : poolIdx + 1} 位｜最终结果里${
      r.hits.some((h) => answerChunks.some((a) => a.id === h.id)) ? "在" : "不在"
    }`);
    if (poolIdx >= 0 && rtop > 0 && poolIdx >= rtop) {
      console.log(`   ⚠️ 它排在精排窗口（前 ${rtop} 条）之外——精排从未见过它。`);
      console.log(`      去掉 --rerank-top 再跑一次，才是对精排能力的公平检验。`);
    }
    if (poolIdx >= 0 && !r.hits.some((h) => answerChunks.some((a) => a.id === h.id))) {
      console.log("   → 召回带进来了、精排没选上：这是【排序问题】，不是召回问题。");
      console.log("     若前 5 名本身就能回答这个问题，那多半是金标准太窄；若不能，就是精排判错。");
    }
  }
}
