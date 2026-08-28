/**
 * 查询侧改写 —— 治的是检索链路治不了的那类病。
 *
 * g12 的完整病历证明：当问题本身没有一个词是"文档的语言"时，
 * 换切分、换检索、加精排都只能勉强补救。病在查询，就该治查询：
 * 在检索之前，先把"用户的话"翻译成"资料的话"。
 *
 * 两种药：
 *   rewrite —— 直译：把口语问题改写成 2 个术语化检索查询
 *              （"黑色火山石头上的小洞" → "玄武岩 气孔构造 成因"）
 *   HyDE    —— 迂回：让模型先【编】一段假想的答案，拿假答案去检索。
 *              反直觉但有效：假答案和真文档"长得像"，而问题和文档长得不像。
 *              注意：假答案只用于检索，绝不进入最终回答——它是钓饵，不是证据。
 *
 * 多查询的合并复用 RRF：原始问题 + 各改写各跑一路召回，按名次融合。
 * 原始问题永远保留为第一路——改写可能改错，原问题是保底。
 */
import { z } from "zod";
import { runStructured } from "../llm.js";

export interface QueryExpansion {
  /** 第一个永远是原始问题 */
  queries: string[];
  cost: number;
}

export interface QueryExpander {
  readonly name: string;
  expand(question: string): Promise<QueryExpansion>;
}

const RewriteSchema = z.object({
  rewrites: z.array(z.string().min(2)).min(1).max(3)
    .describe("2 个术语化检索查询，每个是若干专业术语的组合，不是完整句子"),
});
const rewriteJson = z.toJSONSchema(RewriteSchema, { target: "draft-7" });

export class LlmRewriter implements QueryExpander {
  readonly name = "rewrite";
  async expand(question: string): Promise<QueryExpansion> {
    const { value, cost } = await runStructured({
      system:
        "你是地学文献检索助手。把用户的口语问题改写成适合全文检索的术语化查询：" +
        "用领域专业术语替换口语说法（如'黑色火山石头上的小洞'→'玄武岩 气孔构造'），" +
        "每个查询是术语的组合而非完整句子。只输出查询，不回答问题。",
      prompt: `口语问题：${question}`,
      schema: rewriteJson as Record<string, unknown>,
      parse: (x) => RewriteSchema.safeParse(x).data,
    });
    return { queries: [question, ...value.rewrites], cost };
  }
}

const HydeSchema = z.object({
  passage: z.string().min(20).max(400)
    .describe("一段可能出现在地学教科书里的、直接回答该问题的短文（60~120字）。允许凭常识撰写，不要求真实准确"),
});
const hydeJson = z.toJSONSchema(HydeSchema, { target: "draft-7" });

export class HydeExpander implements QueryExpander {
  readonly name = "hyde";
  async expand(question: string): Promise<QueryExpansion> {
    const { value, cost } = await runStructured({
      system:
        "你是地学教科书作者。针对问题写一段 60~120 字的假想答案段落，" +
        "使用教科书式的专业措辞。这段文字只用于检索匹配，允许不完全准确。",
      prompt: `问题：${question}`,
      schema: hydeJson as Record<string, unknown>,
      parse: (x) => HydeSchema.safeParse(x).data,
    });
    return { queries: [question, value.passage], cost };
  }
}

/** 离线测试用：固定映射表，行为完全可预测 */
export class FakeExpander implements QueryExpander {
  readonly name = "fake";
  constructor(private map: Record<string, string[]>) {}
  async expand(question: string): Promise<QueryExpansion> {
    return { queries: [question, ...(this.map[question] ?? [])], cost: 0 };
  }
}
