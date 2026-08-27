/**
 * 生成层评测：跑金标准题，测两件真正重要的事——
 *
 *   1. 拒答题上，模型会不会为了"有用"而硬答？（诚实率）
 *   2. 有答案题上，回答能否一次通过归因校验？（一遍过率）
 *
 * 校准已证明拒答无法用检索分数确定性化（相关性 ≠ 答案在不在里面），
 * 所以这里【不设阈值】，纯粹考察铁律 4 交给模型自判的可靠性。
 *
 * 用法：pnpm eval:answer                 全部 24 题（约 $5，慎跑）
 *       pnpm eval:answer -- --refusal    只跑 5 道拒答题（约 $1）
 *       pnpm eval:answer -- --limit 6    只跑前 6 题
 *       pnpm eval:answer -- --from g13   断点续跑
 */
import { query } from "@anthropic-ai/claude-agent-sdk";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { AnswerSchema, answerJsonSchema, type Answer } from "../src/answer/schema.js";
import { toRepairInstructions, verifyAttribution } from "../src/answer/verify.js";
import { config, requireApiKey } from "../src/config.js";
import { loadGold } from "../src/eval/gold.js";
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
5. summary 面向用户，应当自然通顺，但其中的事实同样要被 claims 覆盖。

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
if (argv.includes("--refusal")) cases = cases.filter((c) => c.goldDocIds.length === 0);
const from = opt("from");
if (from) { const i = cases.findIndex((c) => c.id === from); if (i > 0) cases = cases.slice(i); }
const limit = opt("limit"); if (limit) cases = cases.slice(0, Number(limit));

console.log(`🧪 生成层评测｜${cases.length} 题｜模型 ${config.model}｜${pipeline.label}｜不设拒答阈值\n`);

interface Row {
  id: string; answerable: boolean; refused: boolean; rounds: number;
  errors: number; rules: string[]; cost: number; honest: boolean;
}
const rows: Row[] = [];
let fatal = false;

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

  const answerable = c.goldDocIds.length > 0;
  const honest = answerable ? !answer.refused : answer.refused;
  rows.push({
    id: c.id, answerable, refused: answer.refused, rounds: round,
    errors: verdict.stats.errors, rules: verdict.issues.filter((i) => i.severity === "error").map((i) => i.rule),
    cost, honest,
  });
  const mark = honest ? (verdict.passed ? "✅" : "🟡") : "❌";
  const what = answerable ? (answer.refused ? "该答却拒答" : "作答") : (answer.refused ? "诚实拒答" : "该拒却硬答");
  console.log(`  ${mark} ${c.id} ${what}｜打回 ${round} 轮｜$${cost.toFixed(3)}${verdict.passed ? "" : `｜遗留 ${verdict.stats.errors} 错`}`);
}

const answerable = rows.filter((r) => r.answerable);
const refusal = rows.filter((r) => !r.answerable);
const pct = (a: number, b: number) => `${b ? ((a / b) * 100).toFixed(1) : "—"}%`;

console.log("\n📊 生成层成绩");
if (refusal.length) {
  const honest = refusal.filter((r) => r.refused).length;
  console.log(`   拒答题诚实率   ${pct(honest, refusal.length)}（${honest}/${refusal.length}）← 最重要的一项`);
  const hardAnswer = refusal.filter((r) => !r.refused).map((r) => r.id);
  if (hardAnswer.length) console.log(`   硬答的题：${hardAnswer.join("、")}`);
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
console.log(`   总成本         $${rows.reduce((s, r) => s + r.cost, 0).toFixed(3)}（${rows.length} 题）`);

mkdirSync("output", { recursive: true });
writeFileSync("output/eval-answer.json", JSON.stringify(rows, null, 2));
console.log(`   明细 output/eval-answer.json`);
if (fatal) process.exit(2);
