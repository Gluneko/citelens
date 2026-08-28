/**
 * HNSW 单测：不测内部结构，测三个承诺——
 * 高召回、确定性（同种子同结果）、以及"近似"这个词的诚实含义。
 */
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { FakeEmbedder, cosine, normalize } from "./embed.js";
import { HnswIndex, mulberry32 } from "./hnsw.js";

function randomVecs(n: number, dim: number, seed = 7): Float32Array[] {
  const rand = mulberry32(seed);
  return Array.from({ length: n }, () => {
    const v = new Float32Array(dim);
    for (let i = 0; i < dim; i++) v[i] = rand() * 2 - 1;
    return normalize(v);
  });
}

function bruteTop(vecs: Float32Array[], q: Float32Array, k: number): string[] {
  return vecs
    .map((v, i) => [i, cosine(q, v)] as const)
    .sort((a, b) => b[1] - a[1])
    .slice(0, k)
    .map(([i]) => `v${i}`);
}

test("召回率：500 条随机向量上，HNSW top10 对暴力真值的召回 ≥ 90%", () => {
  const vecs = randomVecs(500, 32);
  const idx = new HnswIndex({ seed: 1 });
  vecs.forEach((v, i) => idx.add(`v${i}`, v));
  const queries = randomVecs(20, 32, 99);
  let hit = 0, total = 0;
  for (const q of queries) {
    const truth = new Set(bruteTop(vecs, q, 10));
    for (const r of idx.search(q, 10, 64)) if (truth.has(r.id)) hit++;
    total += 10;
  }
  const recall = hit / total;
  assert.ok(recall >= 0.9, `召回率仅 ${(recall * 100).toFixed(1)}%`);
});

test("确定性：同种子两次构建，查询结果完全一致", () => {
  const vecs = randomVecs(200, 16);
  const build = () => {
    const idx = new HnswIndex({ seed: 5 });
    vecs.forEach((v, i) => idx.add(`v${i}`, v));
    return idx;
  };
  const q = randomVecs(1, 16, 123)[0]!;
  assert.deepEqual(
    build().search(q, 5).map((r) => r.id),
    build().search(q, 5).map((r) => r.id),
  );
});

test("ef 旋钮：更大的 ef 不应降低召回（准与快的交换方向必须正确）", () => {
  const vecs = randomVecs(400, 24);
  const idx = new HnswIndex({ seed: 3 });
  vecs.forEach((v, i) => idx.add(`v${i}`, v));
  const queries = randomVecs(10, 24, 55);
  const recallAt = (ef: number) => {
    let hit = 0;
    for (const q of queries) {
      const truth = new Set(bruteTop(vecs, q, 10));
      for (const r of idx.search(q, 10, ef)) if (truth.has(r.id)) hit++;
    }
    return hit / (queries.length * 10);
  };
  assert.ok(recallAt(128) >= recallAt(8), "ef 调大召回反而下降，束搜索有 bug");
});

test("精确命中：查询向量与库中某条完全相同时，它必须排第一", async () => {
  const emb = new FakeEmbedder(32);
  const texts = ["玄武岩", "花岗岩", "腕足动物", "气孔构造"];
  const vecs = await emb.embed(texts);
  const idx = new HnswIndex({ seed: 2 });
  texts.forEach((t, i) => idx.add(t, vecs[i]!));
  const hits = idx.search(vecs[3]!, 2);
  assert.equal(hits[0]!.id, "气孔构造");
});

test("空索引与单点索引不炸", () => {
  const idx = new HnswIndex();
  assert.deepEqual(idx.search(new Float32Array(8), 5), []);
  idx.add("only", normalize(new Float32Array([1, 0, 0, 0, 0, 0, 0, 0])));
  assert.equal(idx.search(normalize(new Float32Array([1, 1, 0, 0, 0, 0, 0, 0])), 3)[0]!.id, "only");
});
