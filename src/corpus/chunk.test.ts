/**
 * 切分器单测。重点不是"切了几块"，而是那条不变量：
 * chunk.text 必须逐字等于 doc.text.slice(start, end)——引文溯源的地基。
 */
import { strict as assert } from "node:assert";
import { test } from "node:test";
import type { Doc } from "../types.js";
import { chunkBy, chunkFixed, chunkParentChild, chunkStructure } from "./chunk.js";
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

test("结构切分：不在句子中间下刀", () => {
  const d = doc("# 标题\n\n第一句话结束。第二句话也结束。\n\n第二段开始了。第二段结束了。");
  for (const c of chunkStructure(d, 20)) {
    const t = c.text.trim();
    if (!t) continue;
    assert.ok(/[。！？；#]$|^#/.test(t) || t === d.text.slice(c.start, c.end).trim(),
      `块不该断在句中：「${t}」`);
  }
});

test("结构切分同样守住区间不变量", () => {
  const d = doc("# 玄武岩\n\n第一段内容。很长的一段话在这里。\n\n第二段内容。也有若干句子。");
  for (const c of chunkStructure(d, 30)) {
    assert.equal(c.text, d.text.slice(c.start, c.end));
  }
});

test("父子块：子块小、父块大，且父块必须包含子块", () => {
  const d = doc("# 玄武岩\n\n玄武岩是喷出岩。气体逸出留下孔洞，形成气孔构造。孔洞被充填后称杏仁构造。\n\n玄武岩浆黏度低。常形成熔岩台地。");
  const cs = chunkParentChild(d, 25, 200);
  assert.ok(cs.length > 0);
  for (const c of cs) {
    assert.equal(c.text, d.text.slice(c.start, c.end), "子块区间不变量");
    assert.ok(c.context, "父子块必须带 context");
    assert.ok(c.context!.includes(c.text), "父块必须包含子块全文");
    assert.ok(c.context!.length >= c.text.length);
  }
});

test("父子块确实把长块拆细了（这是它的全部意义）", () => {
  const long = "这是一个句子。".repeat(30);
  const d = doc(`# 标题\n\n${long}`);
  const pc = chunkParentChild(d, 60, 400);
  const fixed = chunkFixed(d, { size: 300, overlap: 60 });
  const avgPc = pc.reduce((s, c) => s + c.text.length, 0) / pc.length;
  const avgFx = fixed.reduce((s, c) => s + c.text.length, 0) / fixed.length;
  assert.ok(avgPc < avgFx / 2, `子块应显著更短：${avgPc.toFixed(0)} vs ${avgFx.toFixed(0)}`);
});

test("四种策略都能跑，且 id 唯一", () => {
  const d = doc("# 标题\n\n第一段。有两句话。\n\n第二段。也有两句。");
  for (const s of ["fixed", "overlap", "structure", "parent-child"] as const) {
    const cs = chunkBy(d, s);
    assert.ok(cs.length > 0, `${s} 切出 0 块`);
    assert.equal(new Set(cs.map((c) => c.id)).size, cs.length, `${s} 有重复 id`);
  }
});
