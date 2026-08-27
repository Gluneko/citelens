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
