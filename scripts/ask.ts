/**
 * 文鉴 CiteLens 问答 CLI：检索 → 生成带出处的回答 → 归因校验 → 不过则定向打回。
 *
 * 用法：pnpm ask "玄武岩的二氧化硅含量是多少？"
 *       pnpm ask "那种黑色的石头..." -- --rewrite   先把口语翻译成术语再检索
 *       pnpm ask "华南某矿区2024年金平均品位" --refuse-below 0
 *       pnpm ask "..." --top 5 --pool 50 --no-rerank
 *
 * 只有这一步需要 API key；检索层全本地。
 */
import { query } from "@anthropic-ai/claude-agent-sdk";
import { existsSync, readFileSync } from "node:fs";
import { config, requireApiKey } from "../src/config.js";
import { AnswerSchema, answerJsonSchema, type Answer } from "../src/answer/schema.js";
import { formatAttribution, toRepairInstructions, verifyAttribution } from "../src/answer/verify.js";
import { buildBm25 } from "../src/retrieve/bm25.js";
import { LocalEmbedder } from "../src/retrieve/embed.js";
import { RetrievalPipeline, type RetrieveMode } from "../src/retrieve/pipeline.js";
import { CrossEncoderReranker } from "../src/retrieve/rerank.js";
import { LlmRewriter } from "../src/retrieve/rewrite.js";
import { unpackVectors, VectorIndex, type VectorStoreFile } from "../src/retrieve/vector.js";
import type { Chunk } from "../src/types.js";

const SYSTEM_PROMPT = `你是「文鉴 CiteLens」，一个中文地学文献问答助手。

## 铁律（回答会被归因校验器逐字校验，违反即打回）
1. 你只能依据【本次提供的检索片段】作答，禁止使用片段之外的任何知识。
2. 每条断言必须给出 chunkId 与 quote；quote 必须从该片段原文中**逐字复制**，
   不得改写、不得拼接两处、不得省略中间部分——校验器会做字面子串比对。
3. 断言中出现的每一个数字，都必须在你所引的那个片段里出现过。
   宁可不写数字，也不要写一个"差不多"的数字。
4. 若检索片段不足以回答问题，必须把 refused 设为 true 并说明理由。
   "我不知道"是合法且必须诚实的结论；硬答比答不出来严重得多。
5. 若片段只能回答问题的一部分，答出能答的部分，并把片段未覆盖的方面逐条写入
   gaps 字段（同时在 summary 中向用户说明）。完整回答时 gaps 为空数组。
6. summary 面向用户，应当自然通顺，但其中的事实同样要被 claims 覆盖。

## 输出
按给定 schema 输出结构化回答。`;

const argv = process.argv.slice(2).filter((a) => a !== "--");
const opt = (n: string) => { const i = argv.indexOf(`--${n}`); return i >= 0 ? argv[i + 1] : undefined; };
const question = argv.filter((a) => !a.startsWith("--") && argv[argv.indexOf(a) - 1]?.startsWith("--") !== true)[0]
  ?? argv.find((a) => !a.startsWith("--"));
if (!question) { console.error('用法：pnpm ask "你的问题"'); process.exit(1); }

requireApiKey();

const topK = Number(opt("top") ?? 5);
const poolSize = Number(opt("pool") ?? 50);
const useRerank = !argv.includes("--no-rerank");
const refuseBelow = opt("refuse-below") !== undefined ? Number(opt("refuse-below")) : undefined;
const MAX_REPAIR = Number(process.env.CITELENS_MAX_REPAIR ?? 2);

const chunks: Chunk[] = readFileSync("data/chunks.jsonl", "utf-8")
  .split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l) as Chunk);
const VEC = "data/index/vectors.json";
if (!existsSync(VEC)) { console.error("❌ 缺少向量索引，先跑 pnpm vectors"); process.exit(1); }

const file = JSON.parse(readFileSync(VEC, "utf-8")) as VectorStoreFile;
const vecs = unpackVectors(file);
const vector = new VectorIndex(file.model, file.dim);
for (const c of chunks) { const v = vecs.get(c.id); if (v) vector.add(c, v); }

const pipeline = new RetrievalPipeline(
  {
    bm25: buildBm25(chunks),
    vector,
    embedder: new LocalEmbedder(file.model),
    reranker: useRerank ? new CrossEncoderReranker() : undefined,
  },
  { mode: (opt("mode") ?? "vector") as RetrieveMode, poolSize, topK },
);

console.log(`🔎 文鉴 CiteLens | 模型 ${config.model} | ${pipeline.label} | chunk ${chunks.length}`);
console.log(`❓ ${question}\n`);

const t0 = performance.now();
let retrieved;
if (argv.includes("--rewrite")) {
  const exp = await new LlmRewriter().expand(question);
  console.log(`✍️  查询改写：${exp.queries.slice(1).map((q) => `「${q}」`).join(" ")}`);
  retrieved = await pipeline.searchMulti(exp.queries);
} else {
  retrieved = await pipeline.search(question);
}
const retrieveMs = performance.now() - t0;

console.log(`📚 检索到 ${retrieved.hits.length} 个片段（${retrieveMs.toFixed(0)}ms）`);
retrieved.hits.forEach((h, i) => {
  const sc = retrieved.scores ? `｜相关性 ${retrieved.scores[i]!.toFixed(2)}` : "";
  console.log(`   ${i + 1}. ${h.id}${sc}｜${(h.context ?? h.text).slice(0, 46).replace(/\n/g, " ")}…`);
});

// 拒答判据：用精排分数而非余弦——余弦把相关与无关都挤在 0.5～0.6，几乎不可分
const topScore = retrieved.scores?.[0];
const insufficient = refuseBelow !== undefined && topScore !== undefined && topScore < refuseBelow;
if (refuseBelow !== undefined) {
  console.log(`\n🚧 拒答阈值 ${refuseBelow}｜本次最高相关性 ${topScore?.toFixed(2) ?? "—"} → ${insufficient ? "判定证据不足，要求拒答" : "证据充分"}`);
}

const evidenceBlock = retrieved.hits
  .map((h) => `【片段 ${h.id}】\n${h.context ?? h.text}`)
  .join("\n\n");

let prompt = `请依据以下检索片段回答问题。

问题：${question}

${evidenceBlock}`;

let sessionId: string | undefined;
let answer: Answer | undefined;
let verdict: ReturnType<typeof verifyAttribution> | undefined;
let cost = 0;
const rounds: string[] = [];

for (let round = 0; round <= MAX_REPAIR; round++) {
  if (round > 0) console.log(`\n🔁 定向打回（第 ${round}/${MAX_REPAIR} 轮）`);
  let roundAnswer: Answer | undefined;
  let roundCost = 0;
  const run = query({
    prompt,
    options: {
      model: config.model,
      systemPrompt: SYSTEM_PROMPT,
      resume: sessionId,
      disallowedTools: ["Bash", "Edit", "Write", "Read", "Task", "WebFetch", "WebSearch"],
      permissionMode: "bypassPermissions",
      maxTurns: 6,
      outputFormat: { type: "json_schema", schema: answerJsonSchema as Record<string, unknown> },
      env: { ...process.env, ANTHROPIC_BASE_URL: config.anthropicBaseUrl, ANTHROPIC_AUTH_TOKEN: config.llmApiKey },
    },
  });
  try {
    for await (const m of run) {
      if (m.type === "system" && m.subtype === "init") sessionId = m.session_id;
      else if (m.type === "result") {
        if (m.subtype !== "success" || m.is_error) {
          // 把真实错误打出来——只印 subtype 等于什么都没说（踩过）
          const detail = "result" in m && typeof (m as any).result === "string"
            ? (m as any).result
            : JSON.stringify(m).slice(0, 400);
          console.error(`\n❌ 生成失败（subtype=${m.subtype}, is_error=${m.is_error}）`);
          console.error(`   ${detail}`);
          if (/401|unauthor|invalid.*key|authentication/i.test(detail)) {
            console.error(`\n   看起来是鉴权失败。检查 .env 里的 LLM_API_KEY 是否已填成真实密钥。`);
            console.error(`   当前端点：${config.anthropicBaseUrl}`);
          }
          if (/402|balance|insufficient/i.test(detail)) {
            console.error(`\n   看起来是余额不足——账户充值后重跑即可。`);
          }
          break;
        }
        roundCost = m.total_cost_usd;
        const parsed = AnswerSchema.safeParse(m.structured_output);
        if (parsed.success) roundAnswer = parsed.data;
        else console.error("❌ 输出未过 schema：", parsed.error.issues.slice(0, 3));
      }
    }
  } catch (e) {
    console.error(`\n❌ 运行中断：${e instanceof Error ? e.message.split("\n")[0] : e}`);
  }
  cost += roundCost;
  if (!roundAnswer) { if (answer) break; process.exit(1); }
  answer = roundAnswer;
  verdict = verifyAttribution(answer, retrieved.hits, { evidenceInsufficient: insufficient });
  rounds.push(`R${round}=${verdict.stats.errors}错误`);
  console.log(`\n🔍 归因校验（第 ${round} 轮）\n${formatAttribution(verdict)}`);
  if (verdict.passed) break;
  if (round < MAX_REPAIR) prompt = toRepairInstructions(verdict);
}

if (!answer || !verdict) process.exit(1);

console.log("\n" + "─".repeat(60));
console.log(`📊 ${rounds.join(" → ")}｜$${cost.toFixed(4)}（牌价折算仅供对比）`);
if (answer.refused) {
  console.log(`\n🚫 拒答：${answer.refusalReason ?? "（未说明理由）"}`);
  console.log(`\n${answer.summary}`);
} else {
  console.log(`\n【回答】\n${answer.summary}`);
  if (answer.gaps.length) console.log(`\n【未覆盖】${answer.gaps.map((g) => `「${g}」`).join(" ")}——检索片段未涉及这些方面`);
  console.log(`\n【逐条溯源】`);
  answer.claims.forEach((c, i) => {
    console.log(`  ${i + 1}. ${c.text}`);
    console.log(`     ← ${c.chunkId}：「${c.quote}」`);
  });
}
console.log(verdict.passed ? "\n✅ 每条断言均已逐字溯源" : "\n⚠️ 达到最大重排轮数，仍有归因错误");
