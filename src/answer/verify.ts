/**
 * 归因校验器 —— 本项目的灵魂，也是"确定性第六种形态"的落点。
 *
 * RAG 公认的难点是"模型会不会编"。这个问题有一半可以确定性化：
 * 不去判断答案"对不对"（那要靠人或 judge），而是判断答案
 * **有没有资格这么说**——它引的片段真的检索到了吗？它引的原文一字不差吗？
 * 它写的数字在原文里出现过吗？证据不足时它认账了吗？
 *
 * 四条铁律全是纯字符串比对，没有一处依赖模型判断：
 *   1. cite.unknown-chunk   引用的片段必须真在本次检索结果里
 *   2. quote.not-substring  引文必须是该片段的字面子串（仅归一化空白）
 *   3. number.unsupported   断言里的每个数字都必须在被引片段中出现过
 *   4. refusal.required     无可用证据时必须拒答，而不是硬答
 */
import type { Answer } from "./schema.js";

export interface AttributionIssue {
  rule: string;
  severity: "error" | "warn";
  path: string;
  message: string;
  detail?: string;
}

export interface AttributionResult {
  passed: boolean;
  issues: AttributionIssue[];
  stats: { errors: number; warns: number; claims: number };
}

/** 供检索片段用的最小形状 */
export interface EvidenceChunk {
  id: string;
  text: string;
  context?: string;
}

/** 只归一化空白与全角空格：引文可以有换行差异，但一个字都不能变 */
export function normalizeForQuote(s: string): string {
  return s.replace(/[\s　]+/g, "");
}

/**
 * 抽取断言中的数字。
 * 只认阿拉伯数字（含小数），因为它们最容易被模型悄悄改写——
 * "45%～52%" 写成 "45%～55%" 读起来毫无破绽，但已经是幻觉。
 */
export function extractNumbers(text: string): string[] {
  return [...text.matchAll(/\d+(?:\.\d+)?/g)].map((m) => m[0]!);
}

export interface VerifyAnswerOptions {
  /** 本次检索是否没捞到可用证据。为 true 时模型【必须】拒答 */
  evidenceInsufficient?: boolean;
}

export function verifyAttribution(
  answer: Answer,
  evidence: EvidenceChunk[],
  opts: VerifyAnswerOptions = {},
): AttributionResult {
  const issues: AttributionIssue[] = [];
  const byId = new Map(evidence.map((c) => [c.id, c.context ?? c.text]));

  // 铁律 4：证据不足必须拒答
  if (opts.evidenceInsufficient && !answer.refused) {
    issues.push({
      rule: "refusal.required",
      severity: "error",
      path: "refused",
      message: "检索未提供足够证据，必须如实拒答而不是硬答",
    });
  }
  if (answer.refused) {
    if (answer.claims.length > 0) {
      issues.push({
        rule: "refusal.with-claims",
        severity: "error",
        path: "claims",
        message: "既然拒答，就不应同时给出断言",
      });
    }
    if (!answer.refusalReason?.trim()) {
      issues.push({
        rule: "refusal.no-reason",
        severity: "warn",
        path: "refusalReason",
        message: "拒答未说明理由",
      });
    }
  } else if (answer.claims.length === 0) {
    issues.push({
      rule: "claims.empty",
      severity: "error",
      path: "claims",
      message: "既未拒答，也没有任何带出处的断言",
    });
  }

  answer.claims.forEach((claim, i) => {
    const path = `claims[${i}]`;

    // 铁律 1：引用的片段必须真实存在于本次检索结果
    const source = byId.get(claim.chunkId);
    if (source === undefined) {
      issues.push({
        rule: "cite.unknown-chunk",
        severity: "error",
        path,
        message: `引用了本次检索结果中不存在的片段「${claim.chunkId}」——幻觉引用`,
      });
      return; // 片段本身是假的，后两条无从校验
    }

    // 铁律 2：引文必须是该片段的字面子串
    if (!normalizeForQuote(source).includes(normalizeForQuote(claim.quote))) {
      issues.push({
        rule: "quote.not-substring",
        severity: "error",
        path,
        message: "引文不是被引片段的原文子串——改写、拼接或臆造",
        detail: claim.quote.slice(0, 60),
      });
    }

    // 铁律 3：断言里的数字必须在被引片段中出现过
    const bad = extractNumbers(claim.text).filter((n) => !source.includes(n));
    if (bad.length) {
      issues.push({
        rule: "number.unsupported",
        severity: "error",
        path,
        message: `断言中的数字「${bad.join("、")}」未出现在被引片段中——数字幻觉`,
        detail: claim.text.slice(0, 60),
      });
    }
  });

  // 提醒级：summary 里出现了任何断言都没提过的数字
  const claimed = new Set(answer.claims.flatMap((c) => extractNumbers(c.text)));
  const orphan = extractNumbers(answer.summary).filter((n) => !claimed.has(n));
  if (orphan.length && !answer.refused) {
    issues.push({
      rule: "summary.unclaimed-number",
      severity: "warn",
      path: "summary",
      message: `回答正文出现了未被任何断言覆盖的数字「${orphan.join("、")}」`,
    });
  }

  const errors = issues.filter((x) => x.severity === "error").length;
  return {
    passed: errors === 0,
    issues,
    stats: { errors, warns: issues.length - errors, claims: answer.claims.length },
  };
}

export function formatAttribution(r: AttributionResult): string {
  const head = r.passed
    ? `✅ 归因校验通过（${r.stats.claims} 条断言逐条溯源，${r.stats.warns} 条提醒）`
    : `❌ 归因校验未通过：${r.stats.errors} 个错误 / ${r.stats.warns} 条提醒`;
  return [head, ...r.issues.map((i) =>
    `  ${i.severity === "error" ? "✗" : "⚠"} [${i.rule}] ${i.path}: ${i.message}` +
    (i.detail ? `\n      「${i.detail}…」` : ""),
  )].join("\n");
}

/** 不通过时，把问题变成给模型的定向修复指令 */
export function toRepairInstructions(r: AttributionResult): string {
  const errors = r.issues.filter((x) => x.severity === "error");
  if (!errors.length) return "";
  return [
    "上一版回答未通过归因校验，请只针对以下问题定向修复，其余不变：",
    ...errors.map((e, i) => `${i + 1}. [${e.rule}] ${e.path}: ${e.message}`),
    "提醒：quote 必须从片段原文逐字复制；断言中的数字必须来自被引片段；无证据时请如实拒答。",
  ].join("\n");
}
