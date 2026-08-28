/**
 * 归因校验器攻防测试：诚实回答放行，四种造假各抓一次。
 * 全部纯字符串比对，无需模型、无需网络。
 */
import { strict as assert } from "node:assert";
import { test } from "node:test";
import type { Answer } from "./schema.js";
import { extractNumbers, normalizeForQuote, stripChunkIds, verifyAttribution, type EvidenceChunk } from "./verify.js";

const EVIDENCE: EvidenceChunk[] = [
  { id: "basalt#0", text: "玄武岩是分布最广的喷出岩，二氧化硅含量介于45%～52%之间，属基性火山岩。岩浆中溶解的气体逸出后留下孔洞，形成气孔构造。" },
  { id: "granite#0", text: "花岗岩是酸性深成岩，石英体积含量一般大于20%。" },
];

const honest = (): Answer => ({
  refused: false,
  gaps: [],
  claims: [
    {
      text: "玄武岩的二氧化硅含量介于45%～52%之间。",
      chunkId: "basalt#0",
      quote: "二氧化硅含量介于45%～52%之间",
    },
    {
      text: "气孔构造由岩浆中溶解的气体逸出后留下孔洞形成。",
      chunkId: "basalt#0",
      quote: "岩浆中溶解的气体逸出后留下孔洞，形成气孔构造",
    },
  ],
  summary: "玄武岩二氧化硅含量为45%～52%；其气孔构造由气体逸出留下的孔洞形成。",
});

test("诚实回答通过", () => {
  const r = verifyAttribution(honest(), EVIDENCE);
  assert.equal(r.passed, true, JSON.stringify(r.issues, null, 2));
  assert.equal(r.stats.claims, 2);
});

test("铁律1：引用不存在的片段被抓（幻觉引用）", () => {
  const a = honest();
  a.claims[0]!.chunkId = "根本没有这个片段#9";
  const r = verifyAttribution(a, EVIDENCE);
  assert.ok(r.issues.some((i) => i.rule === "cite.unknown-chunk"));
  assert.equal(r.passed, false);
});

test("铁律2：引文被改写一个字就被抓", () => {
  const a = honest();
  a.claims[0]!.quote = "二氧化硅含量介于45%～55%之间"; // 52 改成 55
  const r = verifyAttribution(a, EVIDENCE);
  assert.ok(r.issues.some((i) => i.rule === "quote.not-substring"));
});

test("铁律2：把两处原文拼接起来也被抓（拼接式引文）", () => {
  const a = honest();
  a.claims[0]!.quote = "玄武岩是分布最广的喷出岩，形成气孔构造";
  const r = verifyAttribution(a, EVIDENCE);
  assert.ok(r.issues.some((i) => i.rule === "quote.not-substring"));
});

test("铁律2：仅换行/空格差异应当放行（归一化空白，但不放过改字）", () => {
  const a = honest();
  a.claims[0]!.quote = "二氧化硅含量介于 45%～52% 之间";
  const r = verifyAttribution(a, EVIDENCE);
  assert.equal(r.passed, true, JSON.stringify(r.issues));
});

test("铁律3：数字幻觉被抓（引文没改，但断言里的数字变了）", () => {
  const a = honest();
  a.claims[0]!.text = "玄武岩的二氧化硅含量介于45%～58%之间。";
  const r = verifyAttribution(a, EVIDENCE);
  const hit = r.issues.find((i) => i.rule === "number.unsupported");
  assert.ok(hit, "应抓到数字幻觉");
  assert.ok(hit!.message.includes("58"));
});

test("铁律3：跨片段张冠李戴被抓（引玄武岩却写花岗岩的数字）", () => {
  const a = honest();
  a.claims[0]!.text = "玄武岩的石英含量大于20%。";
  const r = verifyAttribution(a, EVIDENCE);
  assert.ok(r.issues.some((i) => i.rule === "number.unsupported"),
    "20 只出现在 granite#0，引 basalt#0 时不该出现");
});

test("铁律4：证据不足却硬答，被抓", () => {
  const r = verifyAttribution(honest(), EVIDENCE, { evidenceInsufficient: true });
  assert.ok(r.issues.some((i) => i.rule === "refusal.required"));
});

test("铁律4：证据不足时诚实拒答，通过", () => {
  const a: Answer = { refused: true, refusalReason: "检索结果中没有相关信息", claims: [], gaps: [], summary: "无法回答。" };
  const r = verifyAttribution(a, EVIDENCE, { evidenceInsufficient: true });
  assert.equal(r.passed, true, JSON.stringify(r.issues));
});

test("拒答却仍给断言，被抓（自相矛盾）", () => {
  const a: Answer = { ...honest(), refused: true, refusalReason: "无证据" };
  const r = verifyAttribution(a, EVIDENCE);
  assert.ok(r.issues.some((i) => i.rule === "refusal.with-claims"));
});

test("既不拒答也不给断言，被抓", () => {
  const a: Answer = { refused: false, claims: [], gaps: [], summary: "大概是这样吧。" };
  const r = verifyAttribution(a, EVIDENCE);
  assert.ok(r.issues.some((i) => i.rule === "claims.empty"));
});

test("正文出现断言未覆盖的数字 → warn 不拦截", () => {
  const a = honest();
  a.summary = "玄武岩二氧化硅含量为45%～52%，密度约2.9克每立方厘米。";
  const r = verifyAttribution(a, EVIDENCE);
  const hit = r.issues.find((i) => i.rule === "summary.unclaimed-number");
  assert.ok(hit && hit.severity === "warn");
  assert.equal(r.passed, true, "提醒级不应拦截");
});

test("工具函数：空白归一化与数字抽取", () => {
  assert.equal(normalizeForQuote("45% ～ 52%\n之间"), "45%～52%之间");
  assert.deepEqual(extractNumbers("含量45%～52.5%，共2类"), ["45", "52.5", "2"]);
});

test("误报回归（实弹事故锁定）：summary 里的片段 id 不该被当成未溯源数字", () => {
  const a = honest();
  // 模型规范地标注了出处，id 自带数字——这不是事实性数字
  a.summary = "据「basalt#0」与「granite#0」，玄武岩二氧化硅含量为45%～52%。";
  const ev = [...EVIDENCE, { id: "wiki-020-玄武岩#0", text: "玄武岩二氧化硅的含量大约是45-52%" }];
  const r = verifyAttribution(a, ev);
  assert.ok(!r.issues.some((i) => i.rule === "summary.unclaimed-number"),
    `不该误报：${JSON.stringify(r.issues)}`);
});

test("剔除 id 后，真正未溯源的数字仍然要报", () => {
  const a = honest();
  a.summary = "据「basalt#0」，玄武岩密度约2.9克每立方厘米。";
  const r = verifyAttribution(a, EVIDENCE);
  const hit = r.issues.find((i) => i.rule === "summary.unclaimed-number");
  assert.ok(hit && hit.message.includes("2.9"), "2.9 未被任何断言覆盖，必须报");
});

test("stripChunkIds：长 id 优先替换，不留残片", () => {
  const out = stripChunkIds("见 wiki-020-玄武岩#0 与 basalt#0", ["basalt#0", "wiki-020-玄武岩#0"]);
  assert.equal(extractNumbers(out).length, 0, `残留数字：${out}`);
});
