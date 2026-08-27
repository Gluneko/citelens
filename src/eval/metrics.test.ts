import { strict as assert } from "node:assert";
import { test } from "node:test";
import { recallAt, reciprocalRank, summarize, toDocRanking } from "./metrics.js";

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
