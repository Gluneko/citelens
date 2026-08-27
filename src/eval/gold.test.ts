/**
 * 金标准集自体检：出题人也要被判卷。
 */
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { loadCorpus } from "../corpus/load.js";
import { lintGold, loadGold } from "./gold.js";

test("金标准集自洽（docId 存在、关键事实确实在标注文档里）", () => {
  const issues = lintGold(loadGold(), loadCorpus(["data/corpus"]));
  assert.deepEqual(issues, [], JSON.stringify(issues, null, 2));
});

test("题量与题型覆盖", () => {
  const cases = loadGold();
  assert.equal(cases.length, 20);
  assert.equal(cases.filter((c) => c.goldDocIds.length === 0).length, 1, "必须有且仅有 1 道拒答题");
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
