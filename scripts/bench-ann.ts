/**
 * 暴力搜索 vs 手写 HNSW：把"该不该上向量库"变成一张实测表。
 *
 * 用法：pnpm bench:ann            用真实语料向量（803 条）+ 复制扩增模拟更大规模
 *
 * 方法：以真实 chunk 向量为种子，加微扰复制出 N 倍规模的"仿真库"；
 * 从库里抽 50 条向量做查询（自身排除），暴力 top10 为真值，
 * 报每个规模下两者的查询耗时与 HNSW 的召回率。
 * 全程离线（复用 pnpm vectors 的缓存），无需网络与 API key。
 */
import { existsSync, readFileSync } from "node:fs";
import { cosine, normalize } from "../src/retrieve/embed.js";
import { HnswIndex, mulberry32 } from "../src/retrieve/hnsw.js";
import { unpackVectors, type VectorStoreFile } from "../src/retrieve/vector.js";

const VEC = "data/index/vectors.json";
if (!existsSync(VEC)) { console.error("先跑 pnpm vectors"); process.exit(1); }
const file = JSON.parse(readFileSync(VEC, "utf-8")) as VectorStoreFile;
const base = [...unpackVectors(file).values()];
console.log(`🏁 ANN 基准｜真实向量 ${base.length} 条 × ${file.dim} 维（${file.model}）\n`);

const rand = mulberry32(2026);
function inflate(times: number): Float32Array[] {
  if (times === 1) return base;
  const out: Float32Array[] = [...base];
  for (let t = 1; t < times; t++) {
    for (const v of base) {
      const w = new Float32Array(v.length);
      for (let i = 0; i < v.length; i++) w[i] = v[i]! + (rand() - 0.5) * 0.06; // 微扰≈近邻噪声
      out.push(normalize(w));
    }
  }
  return out;
}

const K = 10, EF = 64, NQ = 50;
console.log(`   规模        暴力/查询   HNSW/查询   加速   HNSW召回@${K}   建索引耗时`);
console.log("   " + "─".repeat(66));

for (const times of [1, 10, 50]) {
  const vecs = inflate(times);
  const n = vecs.length;

  const tb = performance.now();
  const idx = new HnswIndex({ seed: 7 });
  vecs.forEach((v, i) => idx.add(`v${i}`, v));
  const buildMs = performance.now() - tb;

  const qIdx = Array.from({ length: NQ }, () => Math.floor(rand() * n));

  // 暴力：全库点积 + 排序
  const t0 = performance.now();
  const truths: Array<Set<string>> = [];
  for (const qi of qIdx) {
    const q = vecs[qi]!;
    const scored: Array<[number, number]> = [];
    for (let i = 0; i < n; i++) { if (i !== qi) scored.push([i, cosine(q, vecs[i]!)]); }
    scored.sort((a, b) => b[1] - a[1]);
    truths.push(new Set(scored.slice(0, K).map(([i]) => `v${i}`)));
  }
  const bruteMs = (performance.now() - t0) / NQ;

  // HNSW
  const t1 = performance.now();
  let hit = 0;
  qIdx.forEach((qi, j) => {
    const res = idx.search(vecs[qi]!, K + 1, EF).filter((r) => r.id !== `v${qi}`).slice(0, K);
    for (const r of res) if (truths[j]!.has(r.id)) hit++;
  });
  const hnswMs = (performance.now() - t1) / NQ;
  const recall = hit / (NQ * K);

  console.log(
    `   ${String(n).padEnd(10)}` +
    `${bruteMs.toFixed(2).padStart(7)}ms   ` +
    `${hnswMs.toFixed(2).padStart(6)}ms   ` +
    `${(bruteMs / hnswMs).toFixed(1).padStart(4)}x   ` +
    `${(recall * 100).toFixed(1).padStart(8)}%     ` +
    `${(buildMs / 1000).toFixed(1)}s`,
  );
}

console.log(`\n📌 读表指南：`);
console.log(`   · 加速比随规模增长——暴力是 O(N)，HNSW 近似 O(logN)`);
console.log(`   · 召回 <100%：ANN 的 A 是"近似"，速度是拿精度换的（ef=${EF} 可调）`);
console.log(`   · 建索引的耗时是暴力没有的前置成本——写多读少的场景先想清楚`);
console.log(`   · 本项目 803 条的规模上暴力 <1ms：**此规模不需要向量库**，这本身是结论`);
