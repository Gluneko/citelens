import { strict as assert } from "node:assert";
import { test } from "node:test";
import type { Chunk } from "../types.js";
import { buildBm25 } from "./bm25.js";
import { FakeEmbedder } from "./embed.js";
import { RetrievalPipeline } from "./pipeline.js";
import { FakeReranker } from "./rerank.js";
import { VectorIndex } from "./vector.js";

const C = (id: string, text: string): Chunk => ({
  id, docId: id, title: "", text, start: 0, end: text.length, source: "t",
});
const CHUNKS = [
  C("a", "玄武岩是基性喷出岩"),
  C("b", "气孔是气体逸出留下的孔洞"),
  C("c", "花岗岩是酸性深成岩"),
  C("d", "腕足动物有两瓣壳"),
];

async function mkVector() {
  const emb = new FakeEmbedder(32);
  const vi = new VectorIndex(emb.name, emb.dim);
  const vs = await emb.embed(CHUNKS.map((c) => c.text));
  CHUNKS.forEach((c, i) => vi.add(c, vs[i]!));
  return { emb, vi };
}

test("三种模式都能跑通并返回候选池", async () => {
  const { emb, vi } = await mkVector();
  const bm = buildBm25(CHUNKS);
  for (const mode of ["bm25", "vector", "hybrid"] as const) {
    const p = new RetrievalPipeline({ bm25: bm, vector: vi, embedder: emb }, { mode, poolSize: 4, topK: 2 });
    // 用两个都能命中的词——BM25 只返回真正命中查询词的块，这是它的正确行为
    const r = await p.search("玄武岩 花岗岩");
    assert.equal(r.hits.length, 2, `${mode} 结果数`);
    assert.ok(r.pool.length >= r.hits.length, `${mode} 候选池不应小于结果`);
  }
});

test("候选池按 poolSize 截断，最终结果按 topK 截断", async () => {
  const { emb, vi } = await mkVector();
  const p = new RetrievalPipeline({ bm25: buildBm25(CHUNKS), vector: vi, embedder: emb },
    { mode: "hybrid", poolSize: 3, topK: 1 });
  const r = await p.search("岩");
  assert.ok(r.pool.length <= 3);
  assert.equal(r.hits.length, 1);
});

test("精排能把候选池里靠后的顶进最终结果", async () => {
  const bm = buildBm25(CHUNKS);
  const p = new RetrievalPipeline({ bm25: bm, reranker: new FakeReranker() },
    { mode: "bm25", poolSize: 4, topK: 1 });
  const r = await p.search("气孔孔洞");
  assert.equal(r.hits[0]!.id, "b");
  assert.ok(r.liftedFrom, "启用精排时必须记录精排前名次");
});

test("缺依赖时明确报错，而不是悄悄降级", async () => {
  const p = new RetrievalPipeline({}, { mode: "vector" });
  await assert.rejects(() => p.search("x"), /需要/);
});

test("label 如实反映当前配置", async () => {
  const bm = buildBm25(CHUNKS);
  assert.equal(new RetrievalPipeline({ bm25: bm }, { mode: "bm25" }).label, "BM25");
  assert.equal(
    new RetrievalPipeline({ bm25: bm, reranker: new FakeReranker() }, { mode: "bm25" }).label,
    "BM25+精排",
  );
});

test("单路保底：只有一路能找到的答案，不会被 RRF 挤出候选池", async () => {
  // 构造 RRF 的典型偏见场景：真答案只有语义路能捞到，且不在最前面
  const many = Array.from({ length: 30 }, (_, i) => C(`noise${i}`, `无关内容${i} 岩石 玄武岩`));
  const gem = C("gem", "气孔是气体逸出留下的孔洞");
  const all = [...many, gem];
  const emb = new FakeEmbedder(32);
  const vi = new VectorIndex(emb.name, emb.dim);
  const vs = await emb.embed(all.map((c) => c.text));
  all.forEach((c, i) => vi.add(c, vs[i]!));
  const bm = buildBm25(all);

  const noGuard = new RetrievalPipeline({ bm25: bm, vector: vi, embedder: emb },
    { mode: "hybrid", poolSize: 5, topK: 5, guarantee: 0 });
  const withGuard = new RetrievalPipeline({ bm25: bm, vector: vi, embedder: emb },
    { mode: "hybrid", poolSize: 5, topK: 5, guarantee: 3 });

  const q = "气孔孔洞";
  const a = await noGuard.search(q);
  const b = await withGuard.search(q);
  // 保底版的候选池必须包含语义路前 3 名，无保底版不保证
  const semTop3 = (await vi.search(emb, q, 3)).map((h) => h.chunk.id);
  for (const id of semTop3) {
    assert.ok(b.pool.some((c) => c.id === id), `保底应把语义路第一梯队 ${id} 纳入候选池`);
  }
  assert.equal(b.pool.length, 5, "保底不应撑大候选池——它替换尾部，不加座位");
  assert.ok(a.pool.length === 5);
});

test("rerankTop 只精排候选池前 N 条（成本旋钮）", async () => {
  const cs = Array.from({ length: 20 }, (_, i) => C(`c${i}`, `内容${i} 玄武岩`));
  cs.push(C("target", "气孔孔洞在此"));
  const bm = buildBm25(cs);
  const p = new RetrievalPipeline({ bm25: bm, reranker: new FakeReranker() },
    { mode: "bm25", poolSize: 21, topK: 3, rerankTop: 2 });
  const r = await p.search("玄武岩");
  assert.ok(r.liftedFrom!.every((x) => x <= 2), "只精排前 2 条时，结果不可能来自更靠后的候选");
});

test("多查询检索：改写查询能把原问题捞不到的答案带进候选池", async () => {
  const chunks = [
    C("noise1", "火山活动与板块边界的关系"),
    C("noise2", "岩石圈的分层结构概述"),
    C("gem", "玄武岩的气孔构造由气体逸出形成"),
  ];
  const bm = buildBm25(chunks);
  const p = new RetrievalPipeline({ bm25: bm }, { mode: "bm25", poolSize: 3, topK: 3 });
  // 原问题全口语，BM25 捞不到 gem；改写后的术语查询能
  const miss = await p.searchMulti(["石头上的小洞哪来的"]);
  assert.ok(!miss.pool.some((c) => c.id === "gem"), "原问题不该命中（前提校验）");
  const hit = await p.searchMulti(["石头上的小洞哪来的", "玄武岩 气孔构造"]);
  assert.ok(hit.pool.some((c) => c.id === "gem"), "改写查询应把答案带进候选池");
});

test("多查询 + 精排：精排用原始问题打分，且答案能进最终结果", async () => {
  const chunks = [C("a", "板块构造概述"), C("gem", "气孔是气体逸出留下的孔洞")];
  const p = new RetrievalPipeline(
    { bm25: buildBm25(chunks), reranker: new FakeReranker() },
    { mode: "bm25", poolSize: 2, topK: 1 },
  );
  const r = await p.searchMulti(["小洞 气孔 孔洞", "气孔 孔洞"]);
  assert.equal(r.hits[0]!.id, "gem");
  assert.ok(r.scores, "多查询+精排也要返回精排分数");
});
