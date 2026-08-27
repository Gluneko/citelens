/**
 * BM25 单测：不测"分数等于某个魔数"，测的是三条直觉是否真的成立。
 * 这样即使日后换参数、换分词器，测试仍然有意义。
 */
import { strict as assert } from "node:assert";
import { test } from "node:test";
import type { Chunk } from "../types.js";
import { buildBm25 } from "./bm25.js";

const C = (id: string, text: string, title = ""): Chunk => ({
  id, docId: id, title, text, start: 0, end: text.length, source: "t",
});

test("能检索到：查询词命中的文档排在前面", () => {
  const idx = buildBm25([
    C("a", "玄武岩是基性喷出岩，二氧化硅含量 45%～52%。"),
    C("b", "花岗岩是酸性深成岩，石英含量高。"),
    C("c", "腕足动物是海生底栖无脊椎动物。"),
  ]);
  assert.equal(idx.search("玄武岩", 3)[0]!.chunk.id, "a");
  assert.equal(idx.search("石英", 3)[0]!.chunk.id, "b");
});

test("IDF 直觉：罕见词的权重必须高于常见词", () => {
  const idx = buildBm25([
    C("a", "岩石 岩石 霓石"),
    C("b", "岩石 岩石 岩石"),
    C("c", "岩石 石英"),
  ]);
  assert.ok(idx.idf("霓石") > idx.idf("岩石"), "只出现一次的词应比处处都有的词更值钱");
});

test("TF 饱和：词频翻十倍，得分绝不应翻十倍", () => {
  const one = buildBm25([C("x", "金 " + "背景 ".repeat(50))]).search("金", 1)[0]!.score;
  const ten = buildBm25([C("x", "金 ".repeat(10) + "背景 ".repeat(50))]).search("金", 1)[0]!.score;
  assert.ok(ten > one, "词频高应当加分");
  assert.ok(ten < one * 3, `饱和失效：${one.toFixed(3)} → ${ten.toFixed(3)}`);
});

test("长度归一：同样命中一次，短文档应排在长文档前面", () => {
  const idx = buildBm25([
    C("long", "金 " + "无关内容 ".repeat(200)),
    C("short", "金矿 金"),
  ]);
  assert.equal(idx.search("金", 2)[0]!.chunk.id, "short");
});

test("标题参与检索：标题里的词也能命中", () => {
  const idx = buildBm25([C("a", "本文讨论若干问题。", "水系沉积物测量"), C("b", "另一段无关文本。", "花岗岩")]);
  assert.equal(idx.search("水系沉积物", 2)[0]!.chunk.id, "a");
});

test("查询词全库都没有时返回空，而不是抛错或硬凑", () => {
  const idx = buildBm25([C("a", "玄武岩")]);
  assert.deepEqual(idx.search("量子色动力学", 5), []);
});

test("结果可复现：同分时按稳定顺序，跑两次完全一致", () => {
  const cs = [C("a", "金 铜"), C("b", "金 铜"), C("c", "金 铜")];
  const idx = buildBm25(cs);
  assert.deepEqual(idx.search("金", 3).map((h) => h.chunk.id), idx.search("金", 3).map((h) => h.chunk.id));
});

test("matched 字段如实记录命中了哪些查询词", () => {
  const idx = buildBm25([C("a", "玄武岩 二氧化硅 45%")]);
  const hit = idx.search("玄武岩 二氧化硅", 1)[0]!;
  assert.deepEqual([...hit.matched].sort(), ["二氧化硅", "玄武岩"]);
});
