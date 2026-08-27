/**
 * 评测执行与汇总：把"跑一遍金标准集"这件事抽出来，
 * CLI 与 sweep 共用同一套判定，免得两处口径悄悄漂移。
 * （OreScout 的实弹教训：同一内核、不同配置，就不是同一计算。）
 */
import type { Chunk, GoldCase } from "../types.js";
import type { RetrievalPipeline } from "../retrieve/pipeline.js";
import { factComplete, factRecall, summarize, toDocRanking } from "./metrics.js";

/** 喂给模型的文本：父子块下是父块，其余等于块本身 */
export const ctxOf = (c: Chunk) => c.context ?? c.text;

export interface CaseOutcome {
  case: GoldCase;
  hits: Chunk[];
  pool: Chunk[];
  ranked: string[];
  /** 正确文档在最终结果里的名次，-1 表示未命中 */
  rank: number;
  /**
   * 正确文档在候选池里的【文档级】名次，-1 表示召回阶段就漏了。
   * 注意：这是把多个 chunk 压成一篇之后的名次。
   */
  poolRank: number;
  /**
   * 正确片段在候选池里的【chunk 级】名次。
   * 必须与 poolRank 分开报——rerankTop 是按 chunk 数截断的，
   * 用文档级名次去判断"有没有被精排看到"会得出完全错误的结论。
   * （实弹教训：文档级第 17、chunk 级第 43，rerankTop=20 时其实压根没进精排。）
   */
  poolRankChunk: number;
  liftedFrom?: number[];
}

export async function runCases(pipeline: RetrievalPipeline, gold: GoldCase[]): Promise<CaseOutcome[]> {
  const out: CaseOutcome[] = [];
  for (const c of gold) {
    const r = await pipeline.search(c.question);
    const ranked = toDocRanking(r.hits.map((h) => h.docId));
    out.push({
      case: c,
      hits: r.hits,
      pool: r.pool,
      ranked,
      rank: ranked.findIndex((d) => c.goldDocIds.includes(d)),
      poolRank: toDocRanking(r.pool.map((h) => h.docId)).findIndex((d) => c.goldDocIds.includes(d)),
      poolRankChunk: r.pool.findIndex((h) => c.goldDocIds.includes(h.docId)),
      liftedFrom: r.liftedFrom,
    });
  }
  return out;
}

export interface EvalReport {
  label: string;
  n: number;
  recall: Record<number, number>;
  mrr: number;
  /** 候选池召回率：召回阶段有没有把正确答案带进来。精排的上限就是它 */
  poolRecall: number;
  factRecall: number;
  factComplete: number;
  missingFacts: Array<{ id: string; missing: string[] }>;
  /** 拒答题最高分与正常题平均最高分的比值——越低越好判 */
  refuseRatio: number;
  ms: number;
}

export function summarizeRun(label: string, outcomes: CaseOutcome[], ms: number): EvalReport {
  const scored = outcomes.filter((o) => o.case.goldDocIds.length > 0);
  const s = summarize(outcomes.map((o) => ({ ranked: o.ranked, gold: o.case.goldDocIds })));
  const withFacts = outcomes.filter((o) => (o.case.mustContain?.length ?? 0) > 0);
  const texts = (o: CaseOutcome) => o.hits.slice(0, 5).map(ctxOf);

  return {
    label,
    n: scored.length,
    recall: s.recall,
    mrr: s.mrr,
    poolRecall: scored.filter((o) => o.poolRank >= 0).length / (scored.length || 1),
    factRecall: withFacts.reduce((a, o) => a + factRecall(texts(o), o.case.mustContain!), 0) / (withFacts.length || 1),
    factComplete: withFacts.reduce((a, o) => a + factComplete(texts(o), o.case.mustContain!), 0) / (withFacts.length || 1),
    missingFacts: withFacts
      .map((o) => ({
        id: o.case.id,
        missing: o.case.mustContain!.filter((m) => !texts(o).join("\n").includes(m)),
      }))
      .filter((x) => x.missing.length > 0),
    refuseRatio: 0,
    ms,
  };
}
