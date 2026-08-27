/**
 * 拒答阈值校准：用金标准集实测"有答案"与"无答案"两组的精排分数分布，
 * 再决定阈值——而不是拍脑袋定一个 0。
 *
 * 用法：pnpm calibrate
 *
 * 为什么必须实测：
 * Day 3 已证明余弦不能用（相关与无关都挤在 0.55~0.60）。
 * 但精排分数也不是天然可分——实弹里 g12（有答案）最高分 -1.67，
 * 拒答题最高分 -2.10，只差 0.43。阈值定在 0 会把 g12 一起误拒。
 * 所以这一步的产物不是"某个阈值"，而是**这两组分布到底有没有分开**。
 */
import { existsSync, readFileSync } from "node:fs";
import { loadGold } from "../src/eval/gold.js";
import { buildBm25 } from "../src/retrieve/bm25.js";
import { LocalEmbedder } from "../src/retrieve/embed.js";
import { RetrievalPipeline, type RetrieveMode } from "../src/retrieve/pipeline.js";
import { CrossEncoderReranker } from "../src/retrieve/rerank.js";
import { unpackVectors, VectorIndex, type VectorStoreFile } from "../src/retrieve/vector.js";
import type { Chunk } from "../src/types.js";

const argv = process.argv.slice(2).filter((a) => a !== "--");
const opt = (n: string) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : undefined; };

const chunks: Chunk[] = readFileSync("data/chunks.jsonl", "utf-8")
  .split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l) as Chunk);
const VEC = "data/index/vectors.json";
if (!existsSync(VEC)) { console.error("先跑 pnpm vectors"); process.exit(1); }
const file = JSON.parse(readFileSync(VEC, "utf-8")) as VectorStoreFile;
const vecs = unpackVectors(file);
const vector = new VectorIndex(file.model, file.dim);
for (const c of chunks) { const v = vecs.get(c.id); if (v) vector.add(c, v); }

const pipeline = new RetrievalPipeline(
  { bm25: buildBm25(chunks), vector, embedder: new LocalEmbedder(file.model), reranker: new CrossEncoderReranker() },
  { mode: (opt("mode") ?? "vector") as RetrieveMode, poolSize: Number(opt("pool") ?? 50), topK: 5 },
);

const gold = loadGold();
console.log(`🎚 拒答阈值校准｜${gold.length} 题（${gold.filter((c) => !c.goldDocIds.length).length} 道拒答题）\n`);

interface Row { id: string; answerable: boolean; top1: number; top5: number; gap: number }
const rows: Row[] = [];
for (const c of gold) {
  const r = await pipeline.search(c.question);
  const s = r.scores ?? [];
  const top1 = s[0] ?? Number.NEGATIVE_INFINITY;
  const top5 = s[s.length - 1] ?? top1;
  rows.push({ id: c.id, answerable: c.goldDocIds.length > 0, top1, top5, gap: top1 - top5 });
  process.stdout.write(`\r   ${rows.length}/${gold.length}`);
}
process.stdout.write("\n\n");

const yes = rows.filter((r) => r.answerable).sort((a, b) => a.top1 - b.top1);
const no = rows.filter((r) => !r.answerable).sort((a, b) => b.top1 - a.top1);

const show = (label: string, rs: Row[]) => {
  console.log(`${label}（${rs.length} 题）`);
  for (const r of rs) console.log(`   ${r.id}  top1=${r.top1.toFixed(2)}  top5=${r.top5.toFixed(2)}  落差=${r.gap.toFixed(2)}`);
  const v = rs.map((r) => r.top1);
  console.log(`   最低 ${Math.min(...v).toFixed(2)}｜中位 ${v.sort((a, b) => a - b)[Math.floor(v.length / 2)]!.toFixed(2)}｜最高 ${Math.max(...v).toFixed(2)}\n`);
};
show("✅ 有答案", yes);
show("🚫 拒答题", no);

const minYes = Math.min(...yes.map((r) => r.top1));
const maxNo = Math.max(...no.map((r) => r.top1));
console.log("─".repeat(56));
if (minYes > maxNo) {
  const mid = (minYes + maxNo) / 2;
  console.log(`✅ 两组完全分开：有答案最低 ${minYes.toFixed(2)} > 拒答题最高 ${maxNo.toFixed(2)}`);
  console.log(`   建议阈值 ${mid.toFixed(2)}（取中点，两侧各留 ${((minYes - maxNo) / 2).toFixed(2)} 余量）`);
  console.log(`   用法：pnpm ask "..." -- --refuse-below ${mid.toFixed(2)}`);
} else {
  console.log(`⚠️ 两组重叠：有答案最低 ${minYes.toFixed(2)} ≤ 拒答题最高 ${maxNo.toFixed(2)}`);
  console.log(`   重叠区内无论阈值定在哪，都会同时产生误拒与漏拒——`);
  console.log(`   **单一绝对阈值在这个语料上不可靠**，需要换判据（见下）。`);
  const overlapYes = yes.filter((r) => r.top1 <= maxNo).map((r) => r.id);
  const overlapNo = no.filter((r) => r.top1 >= minYes).map((r) => r.id);
  console.log(`   落入重叠区：有答案 ${overlapYes.join("、") || "无"}｜拒答题 ${overlapNo.join("、") || "无"}`);
  console.log(`\n   可选替代判据：`);
  console.log(`   · 交给模型自己判断（本项目默认）：铁律 4 要求无证据时拒答，校验器只在明确判定不足时强制`);
  console.log(`   · 用 top1 与 top5 的落差：有答案时头部应明显高于尾部，无答案时整体平坦`);
  const gapYes = yes.map((r) => r.gap), gapNo = no.map((r) => r.gap);
  const avg = (a: number[]) => a.reduce((x, y) => x + y, 0) / (a.length || 1);
  console.log(`     实测落差：有答案平均 ${avg(gapYes).toFixed(2)}｜拒答题平均 ${avg(gapNo).toFixed(2)}`);
}
