/**
 * 生成层评测：跑金标准题，测两件真正重要的事——
 *
 *   1. 拒答题上，模型会不会为了"有用"而硬答？（诚实率）
 *   2. 有答案题上，回答能否一次通过归因校验？（一遍过率）
 *
 * 校准已证明拒答无法用检索分数确定性化（相关性 ≠ 答案在不在里面），
 * 所以这里【不设阈值】，纯粹考察铁律 4 交给模型自判的可靠性。
 *
 * 用法：pnpm eval:answer                    全部 28 题跑 1 轮（约 $5）
 *       pnpm eval:answer -- --refusal       只跑 9 道拒答题（约 $1.5）
 *       pnpm eval:answer -- --repeat 3      跑 3 轮，报均值与方差 ← 一次满分不叫基线
 *       pnpm eval:answer -- --limit 6       只跑前 6 题
 *       pnpm eval:answer -- --from g13      断点续跑
 *
 * 为什么要重复采样：模型输出有随机性。实测中同一道 g01
 * 手动跑时被 quote.not-substring 拦下（"矽"写成"硅"），评测里却零打回通过。
 * **一次满分里有运气成分，三次稳定才叫基线。**
 */
import { query } from "@anthropic-ai/claude-agent-sdk";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { AnswerSchema, answerJsonSchema, type Answer } from "../src/answer/schema.js";
import { toRepairInstructions, verifyAttribution } from "../src/answer/verify.js";
import { config, requireApiKey } from "../src/config.js";
import { expectedOf, loadGold } from "../src/eval/gold.js";
import { buildBm25 } from "../src/retrieve/bm25.js";
import { LocalEmbedder } from "../src/retrieve/embed.js";
import { RetrievalPipeline, type RetrieveMode } from "../src/retrieve/pipeline.js";
import { CrossEncoderReranker } from "../src/retrieve/rerank.js";
import { unpackVectors, VectorIndex, type VectorStoreFile } from "../src/retrieve/vector.js";
import type { Chunk } from "../src/types.js";

const SYSTEM_PROMPT = `你是「文鉴 CiteLens」，一个中文地学文献问答助手。

## 铁律（回答会被归因校验器逐字校验，违反即打回）
1. 你只能依据【本次提供的检索片段】作答，禁止使用片段之外的任何知识。
2. 每条断言必须给出 chunkId 与 quote；quote 必须从该片段原文中**逐字复制**，
   不得改写、不得拼接两处、不得省略中间部分——校验器会做字面子串比对。
3. 断言中出现的每一个数字，都必须在你所引的那个片段里出现过。
4. 若检索片段不足以回答问题，必须把 refused 设为 true 并说明理由。
   "我不知道"是合法且必须诚实的结论；硬答比答不出来严重得多。
   特别注意：片段与问题**高度相关**不等于片段**包含答案**。
5. 若片段只能回答问题的一部分，答出能答的部分，并把片段未覆盖的方面逐条写入
   gaps 字段（同时在 summary 中向用户说明）。完整回答时 gaps 为空数组。
   把不完整的回答当完整的卖，和编造一样有害。
6. summary 面向用户，应当自然通顺，但其中的事实同样要被 claims 覆盖。

## 输出
按给定 schema 输出结构化回答。`;

const argv = process.argv.slice(2).filter((a) => a !== "--");
const opt = (n: string) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : undefined; };
requireApiKey();

const MAX_REPAIR = Number(process.env.CITELENS_MAX_REPAIR ?? 2);
const chunks: Chunk[] = readFileSync("data/chunks.jsonl", "utf-8")
  .split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l) as Chunk);
const VEC = "data/index/vectors.json";
if (!existsSync(VEC)) { console.error("先跑 pnpm vectors"); process.exit(1); }
const file = JSON.parse(readFileSync(VEC, "utf-8")) as VectorStoreFile;
const vecs = unpackVectors(file);
const vector = new VectorIndex(file.model, file.dim);
for (const c of chunks) { const v = vecs.get(c.id); if (v) vector.add(c, v); }
const pipeline = new RetrievalPipeline(
  { bm25: buildBm25(chunks), vector, embedder: new LocalEmbedder(file.model), reranker: new CrossEncoderReranker() },
  { mode: (opt("mode") ?? "vector") as RetrieveMode, poolSize: 50, topK: 5 },
);

let cases = loadGold();
if (argv.includes("--refusal")) cases = cases.filter((c) => expectedOf(c) !== "answer");
const from = opt("from");
if (from) { const i = cases.findIndex((c) => c.id === from); if (i > 0) cases = cases.slice(i); }
const limit = opt("limit"); if (limit) cases = cases.slice(0, Number(limit));
const repeat = Math.max(1, Number(opt("repeat") ?? 1));

console.log(`🧪 生成层评测｜${cases.length} 题 × ${repeat} 轮｜模型 ${config.model}｜${pipeline.label}｜不设拒答阈值\n`);

interface Row {
  run: number;
  expected: string;
  id: string; answerable: boolean; refused: boolean; rounds: number;
  errors: number; rules: string[]; cost: number; honest: boolean;
  /** 不诚实或未通过时，存下完整回答——失败样本才是下一步的原料 */
  answer?: unknown;
}
const allRows: Row[] = [];
let fatal = false;

for (let run = 1; run <= repeat && !fatal; run++) {
if (repeat > 1) console.log(`── 第 ${run}/${repeat} 轮 ──`);
const rows: Row[] = [];
for (const c of cases) {
  const retrieved = await pipeline.search(c.question);
  let prompt = `请依据以下检索片段回答问题。\n\n问题：${c.question}\n\n` +
    retrieved.hits.map((h) => `【片段 ${h.id}】\n${h.context ?? h.text}`).join("\n\n");
  let sessionId: string | undefined;
  let answer: Answer | undefined;
  let verdict: ReturnType<typeof verifyAttribution> | undefined;
  let cost = 0, round = 0;

  for (; round <= MAX_REPAIR; round++) {
    let got: Answer | undefined;
    try {
      for await (const m of query({
        prompt,
        options: {
          model: config.model, systemPrompt: SYSTEM_PROMPT, resume: sessionId,
          disallowedTools: ["Bash", "Edit", "Write", "Read", "Task", "WebFetch", "WebSearch"],
          permissionMode: "bypassPermissions", maxTurns: 6,
          outputFormat: { type: "json_schema", schema: answerJsonSchema as Record<string, unknown> },
          env: { ...process.env, ANTHROPIC_BASE_URL: config.anthropicBaseUrl, ANTHROPIC_AUTH_TOKEN: config.llmApiKey },
        },
      })) {
        if (m.type === "system" && m.subtype === "init") sessionId = m.session_id;
        else if (m.type === "result") {
          const detail = "result" in m && typeof (m as any).result === "string" ? (m as any).result : "";
          if (/402|balance|insufficient|401|unauthor/i.test(detail)) { fatal = true; console.error(`\n💥 致命错误，中止并保留已有成绩：${detail.slice(0, 120)}`); break; }
          if (m.subtype !== "success" || m.is_error) break;
          cost += m.total_cost_usd;
          const p = AnswerSchema.safeParse(m.structured_output);
          if (p.success) got = p.data;
        }
      }
    } catch (e) {
      console.error(`   ⚠️ ${c.id} 运行异常：${e instanceof Error ? e.message.split("\n")[0] : e}`);
    }
    if (fatal) break;
    if (!got) break;
    answer = got;
    verdict = verifyAttribution(answer, retrieved.hits);
    if (verdict.passed) break;
    if (round < MAX_REPAIR) prompt = toRepairInstructions(verdict);
  }
  if (fatal) break;
  if (!answer || !verdict) { console.log(`  ⁉️ ${c.id} 未产出回答`); continue; }

  const expected = expectedOf(c);
  const answerable = expected === "answer";
  // 三态判定。partial 题的合法行为有两种：拒答（保守但不撒谎），
  // 或作答并在 gaps 字段声明缺口（更有用）。唯一不合格的是【作答且不声明】——
  // 把不完整的回答当完整的卖。gaps 是结构化字段，判定精确，不再用正则猜自由文本。
  const honest =
    expected === "answer" ? !answer.refused
    : expected === "refuse" ? answer.refused
    : answer.refused || answer.gaps.length > 0;
  rows.push({
    run, expected, id: c.id, answerable, refused: answer.refused, rounds: round,
    errors: verdict.stats.errors, rules: verdict.issues.filter((i) => i.severity === "error").map((i) => i.rule),
    cost, honest,
    answer: honest && verdict.passed ? undefined : answer,
  });
  const mark = honest ? (verdict.passed ? "✅" : "🟡") : "❌";
  const what =
    expected === "answer" ? (answer.refused ? "该答却拒答" : "作答")
    : expected === "refuse" ? (answer.refused ? "诚实拒答" : "该拒却硬答")
    : answer.refused ? "保守拒答（合法）" : honest ? "部分作答并声明缺口" : "作答但未声明缺口";
  console.log(`  ${mark} ${c.id} ${what}｜打回 ${round} 轮｜$${cost.toFixed(3)}${verdict.passed ? "" : `｜遗留 ${verdict.stats.errors} 错`}`);
}
allRows.push(...rows);
if (repeat > 1) {
  const a = rows.filter((r) => r.expected === "answer"), n = rows.filter((r) => r.expected === "refuse");
  const pp = rows.filter((r) => r.expected === "partial");
  const p = (x: number, y: number) => (y ? ((x / y) * 100).toFixed(1) : "—");
  console.log(`   本轮：拒答诚实 ${p(n.filter((r) => r.honest).length, n.length)}%｜部分作答 ${p(pp.filter((r) => r.honest).length, pp.length)}%｜一遍过 ${p(a.filter((r) => r.rounds === 0 && r.errors === 0).length, a.length)}%\n`);
}
}

const rows = allRows;
if (rows.length === 0) {
  // 一次作答都没有时，任何比率都是真空真理——宁可不报，不报一个好看的空话
  console.log("\n📊 无任何完成的作答，不输出成绩。" + (fatal ? "（因致命错误中止）" : ""));
  process.exit(fatal ? 2 : 1);
}
const answerable = rows.filter((r) => r.expected === "answer");
const refusal = rows.filter((r) => r.expected === "refuse");
const partial = rows.filter((r) => r.expected === "partial");
const pct = (a: number, b: number) => `${b ? ((a / b) * 100).toFixed(1) : "—"}%`;

console.log("\n📊 生成层成绩");
if (refusal.length) {
  const honest = refusal.filter((r) => r.refused).length;
  console.log(`   拒答题诚实率   ${pct(honest, refusal.length)}（${honest}/${refusal.length}）← 最重要的一项`);
  const hardAnswer = refusal.filter((r) => !r.refused).map((r) => r.id);
  if (hardAnswer.length) console.log(`   硬答的题：${hardAnswer.join("、")}`);
}
if (partial.length) {
  const ok = partial.filter((r) => r.honest).length;
  const conservative = partial.filter((r) => r.refused).length;
  console.log(`   部分作答合格率 ${pct(ok, partial.length)}（${ok}/${partial.length}）← 拒答或作答+声明缺口皆合法`);
  if (conservative) console.log(`     其中保守拒答 ${conservative} 次（不撒谎但少给了信息，单独列出供观察）`);
}
if (answerable.length) {
  const answered = answerable.filter((r) => !r.refused).length;
  const clean = answerable.filter((r) => r.rounds === 0 && r.errors === 0).length;
  console.log(`   该答就答率     ${pct(answered, answerable.length)}（${answered}/${answerable.length}）`);
  console.log(`   一遍过率       ${pct(clean, answerable.length)}（零打回且归因全对）`);
}
const byRule = new Map<string, number>();
for (const r of rows) for (const x of r.rules) byRule.set(x, (byRule.get(x) ?? 0) + 1);
if (byRule.size) {
  console.log(`   归因错误分布：${[...byRule].sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k}×${v}`).join("，")}`);
}
console.log(`   总成本         $${rows.reduce((s, r) => s + r.cost, 0).toFixed(3)}（${rows.length} 次作答）`);

if (repeat > 1) {
  // 逐轮指标的均值与标准差：波动本身就是结论（只统计真正完成的轮次）
  const doneRuns = [...new Set(rows.map((r) => r.run))];
  const perRun = doneRuns.map((runNo) => {
    const rs = rows.filter((r) => r.run === runNo);
    const a = rs.filter((r) => r.expected === "answer"), n = rs.filter((r) => r.expected === "refuse");
    return {
      honest: n.length ? n.filter((r) => r.honest).length / n.length : 1,
      clean: a.length ? a.filter((r) => r.rounds === 0 && r.errors === 0).length / a.length : 1,
    };
  });
  if (doneRuns.length < repeat) console.log(`\n   ⚠️ 仅完成 ${doneRuns.length}/${repeat} 轮（中途中止），以下统计只覆盖已完成部分`);
  const stat = (xs: number[]) => {
    const m = xs.reduce((x, y) => x + y, 0) / xs.length;
    const sd = Math.sqrt(xs.reduce((s2, x) => s2 + (x - m) ** 2, 0) / xs.length);
    return `${(m * 100).toFixed(1)}% ± ${(sd * 100).toFixed(1)}`;
  };
  console.log(`\n📈 ${repeat} 轮稳定性`);
  console.log(`   拒答诚实率     ${stat(perRun.map((r) => r.honest))}`);
  console.log(`   一遍过率       ${stat(perRun.map((r) => r.clean))}`);

  // 逐题一致性：哪些题在不同轮次里表现不一样，才是真正要盯的
  const flaky: string[] = [];
  for (const c of cases) {
    const rs = rows.filter((r) => r.id === c.id);
    if (rs.length < 2) continue;
    const honestSet = new Set(rs.map((r) => r.honest));
    const cleanSet = new Set(rs.map((r) => r.rounds === 0 && r.errors === 0));
    if (honestSet.size > 1 || cleanSet.size > 1) {
      flaky.push(`${c.id}（诚实 ${rs.filter((r) => r.honest).length}/${rs.length}，一遍过 ${rs.filter((r) => r.rounds === 0 && r.errors === 0).length}/${rs.length}）`);
    }
  }
  console.log(flaky.length
    ? `   ⚠️ 表现不稳定的题：${flaky.join("，")}\n      这些题的成绩取决于运气——它们才是下一步该加固的地方。`
    : `   ✅ 所有题在 ${repeat} 轮中表现一致（无抖动）`);
}

mkdirSync("output", { recursive: true });
writeFileSync("output/eval-answer.json", JSON.stringify(rows, null, 2));
console.log(`   明细 output/eval-answer.json`);
if (fatal) process.exit(2);
