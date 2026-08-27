/**
 * 切分器单测。重点不是"切了几块"，而是那条不变量：
 * chunk.text 必须逐字等于 doc.text.slice(start, end)——引文溯源的地基。
 */
import { strict as assert } from "node:assert";
import { test } from "node:test";
import type { Doc } from "../types.js";
import { chunkFixed } from "./chunk.js";
import { parseDoc } from "./load.js";

const doc = (text: string): Doc => ({
  id: "d",
  title: "T",
  text,
  source: "s",
  license: "l",
});

test("定长切分：块数与边界正确", () => {
  const cs = chunkFixed(doc("a".repeat(1000)), { size: 300, overlap: 0 });
  assert.equal(cs.length, 4); // 300+300+300+100
  assert.equal(cs[0]!.start, 0);
  assert.equal(cs[0]!.end, 300);
  assert.equal(cs[3]!.end, 1000);
});

test("重叠生效：相邻块共享 overlap 个字符", () => {
  const cs = chunkFixed(doc("a".repeat(1000)), { size: 300, overlap: 60 });
  assert.equal(cs[1]!.start, 240); // 300 - 60
  assert.ok(cs[0]!.end > cs[1]!.start, "必须有重叠区");
});

test("不变量：chunk.text 逐字等于原文对应区间（引文溯源的地基）", () => {
  const d = doc("玄武岩是一种细粒致密的火成岩。".repeat(40));
  for (const c of chunkFixed(d, { size: 137, overlap: 29 })) {
    assert.equal(c.text, d.text.slice(c.start, c.end), `${c.id} 区间与文本不符`);
  }
});

test("覆盖完整：所有块并起来覆盖全文，不漏字", () => {
  const d = doc("一二三四五六七八九十".repeat(30));
  const cs = chunkFixed(d, { size: 90, overlap: 20 });
  assert.equal(cs[0]!.start, 0);
  assert.equal(cs.at(-1)!.end, d.text.length);
  for (let i = 1; i < cs.length; i++) {
    assert.ok(cs[i]!.start <= cs[i - 1]!.end, "相邻块之间不能有空洞");
  }
});

test("非法参数被拒", () => {
  assert.throws(() => chunkFixed(doc("x"), { size: 0 }));
  assert.throws(() => chunkFixed(doc("x"), { size: 100, overlap: 100 }));
});

test("frontmatter 解析：出处与许可必须被读出来", () => {
  const d = parseDoc("basalt", "---\ntitle: 玄武岩\nsource: 合成语料\nlicense: CC0\n---\n正文开始。");
  assert.equal(d.title, "玄武岩");
  assert.equal(d.source, "合成语料");
  assert.equal(d.license, "CC0");
  assert.equal(d.text, "正文开始。");
});
