import { strict as assert } from "node:assert";
import { test } from "node:test";
import type { Chunk, Doc } from "../types.js";
import { buildDocGraph, expandPoolByGraph, graphStats } from "./graph.js";

const D = (id: string, title: string, text: string): Doc => ({ id, title, text, source: "t", license: "t" });
const C = (id: string, docId: string, text: string, start = 0): Chunk => ({
  id, docId, title: "", text, start, end: start + text.length, source: "t",
});

const DOCS = [
  D("ridge", "大洋中脊", "大洋中脊喷出的岩浆以玄武岩为主。"),
  D("basalt", "玄武岩", "玄武岩是基性喷出岩。"),
  D("gabbro", "辉长岩", "辉长岩与玄武岩化学成分相当。"),
  D("coal", "煤", "煤是生物成因的沉积岩。"),
];

test("标题提及建边：A 的标题出现在 B 正文 → 连边（双向）", () => {
  const g = buildDocGraph(DOCS);
  assert.ok(g.get("basalt")!.has("ridge"), "玄武岩标题出现在大洋中脊正文里");
  assert.ok(g.get("ridge")!.has("basalt"), "边是双向的");
  assert.ok(g.get("basalt")!.has("gabbro"), "玄武岩标题出现在辉长岩正文里");
  assert.ok(!g.get("ridge")!.has("gabbro"), "大洋中脊与辉长岩无直接提及，不该有边");
});

test("单字标题不建边（到处误连，宁缺毋滥）", () => {
  const g = buildDocGraph([...DOCS, D("x", "岩", "什么岩都提。")]);
  assert.equal(g.get("x")!.size, 0);
});

test("两跳路径存在：ridge → basalt → gabbro", () => {
  const g = buildDocGraph(DOCS);
  const oneHop = g.get("ridge")!;
  const twoHop = new Set([...oneHop].flatMap((d) => [...g.get(d)!]));
  assert.ok(twoHop.has("gabbro"), "第二跳应能到达辉长岩");
});

test("池扩展：邻居文档的首段被补进池尾，池长度不变", () => {
  const g = buildDocGraph(DOCS);
  const chunks = [
    C("ridge#0", "ridge", "大洋中脊喷出的岩浆以玄武岩为主。"),
    C("basalt#0", "basalt", "玄武岩是基性喷出岩。"),
    C("gabbro#0", "gabbro", "辉长岩与玄武岩化学成分相当。"),
    C("coal#0", "coal", "煤是生物成因的沉积岩。"),
  ];
  const pool = [chunks[0]!, chunks[1]!, chunks[3]!]; // 池里没有 gabbro
  const { pool: out, added } = expandPoolByGraph(pool, g, chunks, { seedDocs: 2, maxAdd: 2 });
  assert.ok(added.includes("gabbro"), "basalt 的邻居 gabbro 应被补入");
  assert.equal(out.length, pool.length, "替换池尾，不加座位");
  assert.ok(out.some((c) => c.docId === "gabbro"));
});

test("无可扩展时原样返回", () => {
  const g = buildDocGraph(DOCS.slice(3)); // 只有孤立的煤
  const chunks = [C("coal#0", "coal", "煤。")];
  const { pool, added } = expandPoolByGraph(chunks, g, chunks);
  assert.deepEqual(added, []);
  assert.equal(pool.length, 1);
});

test("graphStats 统计正确", () => {
  const s = graphStats(buildDocGraph(DOCS));
  assert.equal(s.nodes, 4);
  assert.ok(s.edges >= 2);
  assert.ok(s.isolated.includes("coal"));
});
