import { strict as assert } from "node:assert";
import { test } from "node:test";
import { FakeReranker, rerank } from "./rerank.js";

test("精排能把靠后的候选顶上来——这是它存在的全部意义", async () => {
  const candidates = ["完全无关的内容", "也不相关的东西", "气孔是气体逸出留下的孔洞"];
  const out = await rerank(new FakeReranker(), "气孔孔洞", candidates, (x) => x, 3);
  assert.equal(out[0]!.item, "气孔是气体逸出留下的孔洞");
  assert.equal(out[0]!.before, 3, "它进来时排第 3，精排后升到第 1");
});

test("before 字段如实记录精排前的名次", async () => {
  const out = await rerank(new FakeReranker(), "甲", ["乙", "甲"], (x) => x, 2);
  assert.equal(out[0]!.before, 2);
  assert.equal(out[1]!.before, 1);
});

test("同分时保持召回顺序，结果可复现", async () => {
  const cands = ["同样内容", "同样内容", "同样内容"];
  const one = await rerank(new FakeReranker(), "同", cands, (x) => x, 3);
  assert.deepEqual(one.map((o) => o.before), [1, 2, 3]);
});

test("空候选池不炸", async () => {
  assert.deepEqual(await rerank(new FakeReranker(), "q", [], (x: string) => x), []);
});

test("topK 截断生效", async () => {
  const out = await rerank(new FakeReranker(), "甲乙丙", ["甲", "甲乙", "甲乙丙"], (x) => x, 2);
  assert.equal(out.length, 2);
  assert.equal(out[0]!.item, "甲乙丙");
});
