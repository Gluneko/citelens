/**
 * 金标准集自体检：出题人也要被判卷。
 */
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { loadCorpus } from "../corpus/load.js";
import { declaresGap, expectedOf, lintGold, loadGold } from "./gold.js";

test("金标准集自洽（docId 存在、关键事实确实在标注文档里）", () => {
  const issues = lintGold(loadGold(), loadCorpus(["data/corpus"]));
  assert.deepEqual(issues, [], JSON.stringify(issues, null, 2));
});

test("题量与题型覆盖", () => {
  const cases = loadGold();
  assert.equal(cases.length, 28);
  assert.equal(cases.filter((c) => expectedOf(c) === "refuse").length, 7, "拒答题");
  assert.equal(cases.filter((c) => expectedOf(c) === "partial").length, 2, "部分作答题");
  assert.ok(cases.filter((c) => c.goldDocIds.length > 1).length >= 1, "至少 1 道跨文档题");
});

test("lint 能抓到出题人的错（反向验证 lint 本身有效）", () => {
  const docs = loadCorpus(["data/corpus"]);
  const bad = lintGold(
    [{ id: "x", question: "问", goldDocIds: ["不存在的文档"], note: "测试" }],
    docs,
  );
  assert.ok(bad.some((i) => i.problem.includes("不存在")));
});

test("缺口声明检测：模型明说'片段未提及'才算合格", () => {
  assert.ok(declaresGap("片段中并未说明超过 3% 后的特殊步骤"));
  assert.ok(declaresGap("检索片段没有给出该数据"));
  assert.ok(declaresGap("资料中没有相关记载"));
  assert.ok(!declaresGap("烧失量超过 3% 时应扣除后归一到 100%。"), "只作答不声明缺口，不算合格");
});

test("期望行为：显式声明优先，否则按 goldDocIds 推断", () => {
  assert.equal(expectedOf({ id: "x", question: "q", goldDocIds: ["a"], note: "n" }), "answer");
  assert.equal(expectedOf({ id: "x", question: "q", goldDocIds: [], note: "n" }), "refuse");
  assert.equal(expectedOf({ id: "x", question: "q", goldDocIds: [], note: "n", expect: "partial" }), "partial");
});
