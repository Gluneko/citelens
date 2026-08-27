import { strict as assert } from "node:assert";
import { test } from "node:test";
import { termFreq, tokenize } from "./tokenize.js";

test("中文被切成词，而不是整句或单字", () => {
  const t = tokenize("玄武岩是一种基性喷出岩。");
  assert.ok(t.includes("玄武岩"), `实际切分：${t.join("/")}`);
  assert.ok(!t.includes("是"), "停用词应被滤掉");
  assert.ok(!t.some((x) => x.includes("。")), "标点不应进入词表");
});

test("拉丁词归一化为小写（CIPW 与 cipw 视为同一个词）", () => {
  assert.deepEqual(tokenize("CIPW"), tokenize("cipw"));
});

test("数字与百分号：数字要能被检索到", () => {
  const t = tokenize("二氧化硅含量 45%～52%");
  assert.ok(t.includes("45") && t.includes("52"), `实际切分：${t.join("/")}`);
});

test("词频统计正确", () => {
  const tf = termFreq(["金", "铜", "金"]);
  assert.equal(tf.get("金"), 2);
  assert.equal(tf.get("铜"), 1);
});
