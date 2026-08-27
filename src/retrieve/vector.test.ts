/**
 * 向量检索单测：全部用确定性假向量器，离线可跑。
 * 测的是检索逻辑本身，不是模型好不好——模型质量由评测集回答，不由单测回答。
 */
import { strict as assert } from "node:assert";
import { test } from "node:test";
import type { Chunk } from "../types.js";
import { cosine, FakeEmbedder, normalize } from "./embed.js";
import { packVectors, unpackVectors, VectorIndex } from "./vector.js";

const C = (id: string, text: string): Chunk => ({
  id, docId: id, title: "", text, start: 0, end: text.length, source: "t",
});

test("归一化后模长为 1，余弦即点积", () => {
  const v = normalize(new Float32Array([3, 4]));
  assert.ok(Math.abs(Math.hypot(v[0]!, v[1]!) - 1) < 1e-6);
  assert.ok(Math.abs(cosine(v, v) - 1) < 1e-6, "自己与自己的余弦必须是 1");
});

test("余弦：方向相同为 1，垂直为 0，相反为 -1", () => {
  const a = normalize(new Float32Array([1, 0]));
  const b = normalize(new Float32Array([0, 1]));
  const c = normalize(new Float32Array([-1, 0]));
  assert.ok(Math.abs(cosine(a, a) - 1) < 1e-6);
  assert.ok(Math.abs(cosine(a, b) - 0) < 1e-6);
  assert.ok(Math.abs(cosine(a, c) + 1) < 1e-6);
});

test("维度不一致必须报错，而不是算出一个假分数", () => {
  assert.throws(() => cosine(new Float32Array(4), new Float32Array(8)));
  assert.throws(() => new VectorIndex("m", 4).add(C("a", "x"), new Float32Array(8)));
});

test("检索：与查询完全相同的文本必须排第一", async () => {
  const emb = new FakeEmbedder(32);
  const chunks = [C("a", "玄武岩"), C("b", "花岗岩"), C("c", "腕足动物")];
  const idx = new VectorIndex(emb.name, emb.dim);
  const vecs = await emb.embed(chunks.map((c) => c.text));
  chunks.forEach((c, i) => idx.add(c, vecs[i]!));
  const hits = await idx.search(emb, "玄武岩", 3);
  assert.equal(hits[0]!.chunk.id, "a");
  assert.ok(hits[0]!.score >= hits[1]!.score, "结果必须按分数降序");
});

test("结果可复现：同一查询跑两次完全一致", async () => {
  const emb = new FakeEmbedder(16);
  const idx = new VectorIndex(emb.name, emb.dim);
  const cs = [C("a", "同分文本"), C("b", "同分文本"), C("c", "同分文本")];
  const vs = await emb.embed(cs.map((c) => c.text));
  cs.forEach((c, i) => idx.add(c, vs[i]!));
  const one = (await idx.search(emb, "同分文本", 3)).map((h) => h.chunk.id);
  const two = (await idx.search(emb, "同分文本", 3)).map((h) => h.chunk.id);
  assert.deepEqual(one, two);
});

test("向量缓存：打包再解包，必须逐位还原（缓存不能悄悄损失精度）", async () => {
  const emb = new FakeEmbedder(8);
  const vecs = await emb.embed(["甲", "乙", "丙"]);
  const round = unpackVectors(packVectors(emb.name, emb.dim, ["a", "b", "c"], vecs));
  assert.deepEqual([...round.get("b")!], [...vecs[1]!]);
  assert.equal(round.size, 3);
});
