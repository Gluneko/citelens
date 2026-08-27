/**
 * 回答的结构化契约 —— 归因校验的前提。
 *
 * 核心设计：**答案不是一段文字，而是一组带出处的断言**。
 * 每条断言必须自带 chunkId（来自哪个片段）与 quote（原文里的哪一句）。
 * 只有把"出处"变成结构化字段，校验器才有东西可查；
 * 若让模型自由写一段带 [1][2] 角标的散文，校验就退化成正则猜谜。
 */
import { z } from "zod";

export const ClaimSchema = z.object({
  text: z.string().min(1).describe("一条断言，用中文陈述句，必须完全由所引片段支撑"),
  chunkId: z.string().min(1).describe("支撑这条断言的片段 id，必须来自本次提供的检索结果"),
  quote: z
    .string()
    .min(2)
    .describe("从该片段中【逐字摘录】的原文，必须是片段文本的子串，不得改写、不得拼接、不得省略中间部分"),
});

export const AnswerSchema = z.object({
  refused: z.boolean().describe("检索到的片段不足以回答时置为 true"),
  refusalReason: z.string().optional().describe("拒答理由；refused 为 true 时必填"),
  claims: z.array(ClaimSchema).describe("带出处的断言列表；拒答时为空数组"),
  summary: z.string().describe("面向用户的完整回答；拒答时说明无法作答"),
});

export type Claim = z.infer<typeof ClaimSchema>;
export type Answer = z.infer<typeof AnswerSchema>;

/** Agent SDK 的 outputFormat 需要 draft-7 JSON Schema */
export const answerJsonSchema = z.toJSONSchema(AnswerSchema, { target: "draft-7" });
