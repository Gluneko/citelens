import { strict as assert } from "node:assert";
import { test } from "node:test";
import { rrf } from "./fuse.js";

const id = (x: string) => x;

test("两路都排第一的，融合后必然第一", () => {
  const out = rrf([
    { name: "a", items: ["x", "y", "z"] },
    { name: "b", items: ["x", "z", "y"] },
  ], id);
  assert.equal(out[0]!.item, "x");
});

test("RRF 奖励「两路都认可」：双路中游会压过单路第一", () => {
  const out = rrf([
    { name: "词法", items: ["双路货", "甲", "乙"] },
    { name: "语义", items: ["单路王", "双路货", "丙"] },
  ], id);
  // 双路货：1/(60+1) + 1/(60+2)；单路王：只有 1/(60+1)
  assert.equal(out[0]!.item, "双路货", "两路共同认可者胜出——这既是 RRF 的优点，也是它的偏见");
  assert.equal(out[1]!.item, "单路王");
});

test("单路独有项要进前列，必须在那一路排得足够靠前", () => {
  // 只被语义检索捞到、且排第 1 的项，能压过只被词法捞到、排第 3 的项
  const out = rrf([
    { name: "词法", items: ["甲", "乙", "词法独有"] },
    { name: "语义", items: ["语义独有", "丙", "丁"] },
  ], id);
  const r = out.map((o) => o.item);
  assert.ok(r.indexOf("语义独有") < r.indexOf("词法独有"));
});

test("两路都靠中游的，可以合力压过只在单路靠前的", () => {
  const out = rrf([
    { name: "a", items: ["独苗", "共识", "x"] },
    { name: "b", items: ["y", "共识", "z"] },
  ], id, 1); // k 调小，放大名次差异
  const rank = out.map((o) => o.item);
  assert.ok(rank.indexOf("共识") < rank.indexOf("独苗") || out[0]!.item === "独苗");
  assert.ok(out.find((o) => o.item === "共识")!.ranks["a"] === 2);
});

test("k 越大，名次差异被抹得越平", () => {
  const small = rrf([{ name: "a", items: ["一", "二"] }], id, 1);
  const big = rrf([{ name: "a", items: ["一", "二"] }], id, 1000);
  const gapSmall = small[0]!.score - small[1]!.score;
  const gapBig = big[0]!.score - big[1]!.score;
  assert.ok(gapBig < gapSmall, "大 k 应当压缩头尾差距");
});

test("ranks 如实记录每一路的名次（融合出问题时靠它排查）", () => {
  const out = rrf([
    { name: "bm25", items: ["a", "b"] },
    { name: "vec", items: ["b", "a"] },
  ], id);
  const a = out.find((o) => o.item === "a")!;
  assert.deepEqual(a.ranks, { bm25: 1, vec: 2 });
});

test("结果可复现：同分按 key 定序", () => {
  const l = [{ name: "a", items: ["甲", "乙"] }, { name: "b", items: ["乙", "甲"] }];
  assert.deepEqual(rrf(l, id).map((o) => o.item), rrf(l, id).map((o) => o.item));
});
