import { strict as assert } from "node:assert";
import { test } from "node:test";
import { factComplete, factRecall, recallAt, reciprocalRank, summarize, toDocRanking } from "./metrics.js";

test("recall@k：正确答案在前 k 条内才算命中", () => {
  const ranked = ["a", "b", "c", "d", "e"];
  assert.equal(recallAt(ranked, ["c"], 3), 1);
  assert.equal(recallAt(ranked, ["c"], 2), 0);
  assert.equal(recallAt(ranked, ["e"], 5), 1);
});

test("MRR：排第几就是几分之一，排不进则计 0", () => {
  assert.equal(reciprocalRank(["a", "b", "c"], ["a"]), 1);
  assert.equal(reciprocalRank(["a", "b", "c"], ["b"]), 0.5);
  assert.equal(reciprocalRank(["a", "b", "c"], ["z"]), 0);
});

test("拒答题不计入检索指标（它考的是诚实，不是召回）", () => {
  const s = summarize([
    { ranked: ["a"], gold: ["a"] },
    { ranked: ["x"], gold: [] },
  ]);
  assert.equal(s.n, 1);
  assert.equal(s.recall[1], 1);
});

test("chunk 级排名压成文档级：同文档多次命中只留最靠前的一次", () => {
  assert.deepEqual(toDocRanking(["a", "a", "b", "a", "c"]), ["a", "b", "c"]);
});

test("事实级召回：只看关键事实齐不齐，不管来自哪篇文档", () => {
  const texts = ["玄武岩二氧化硅含量 45%～52%，属基性岩。"];
  assert.equal(factRecall(texts, ["45", "52"]), 1);
  assert.equal(factRecall(texts, ["45", "99"]), 0.5);
  assert.equal(factComplete(texts, ["45", "99"]), 0);
});

test("事实可以跨片段拼齐——这正是 topK 存在的意义", () => {
  const texts = ["沉积岩仅占地壳体积的约 5%", "却覆盖了大陆表面约 75% 的面积"];
  assert.equal(factComplete(texts, ["5%", "75%"]), 1);
});

test("拒答题没有关键事实，恒计满分（它不由这把尺子衡量）", () => {
  assert.equal(factRecall(["任意内容"], []), 1);
});
