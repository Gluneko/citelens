/**
 * 检索层评测指标。
 *
 * 这三个指标回答三个不同的问题，缺一不可：
 *
 * - recall@k（命中率）：前 k 条里有没有正确答案？——"捞没捞到"
 * - MRR（平均倒数排名）：正确答案排在第几？——"排得靠不靠前"
 *   第 1 名计 1 分，第 2 名 1/2 分，第 5 名 1/5 分。它对头部排名极敏感。
 * - 无关分数（拒答题）：语料里根本没有答案时，最高分有多低？
 *   这是 Day 4 拒答阈值的依据——它不是检索指标，而是【诚实】的度量。
 */

/** 一次检索的结果：按相关性从高到低排列的文档 id（已去重） */
export type RankedDocIds = string[];

export function recallAt(ranked: RankedDocIds, gold: string[], k: number): 0 | 1 {
  return ranked.slice(0, k).some((d) => gold.includes(d)) ? 1 : 0;
}

/** 第一个正确答案的倒数排名；一个都没排进 topK 则计 0 */
export function reciprocalRank(ranked: RankedDocIds, gold: string[], k = 10): number {
  const i = ranked.slice(0, k).findIndex((d) => gold.includes(d));
  return i < 0 ? 0 : 1 / (i + 1);
}

export interface RetrievalSummary {
  n: number;
  recall: Record<number, number>;
  mrr: number;
}

export function summarize(
  results: Array<{ ranked: RankedDocIds; gold: string[] }>,
  ks: number[] = [1, 3, 5, 10],
): RetrievalSummary {
  const scored = results.filter((r) => r.gold.length > 0); // 拒答题不进检索指标
  const recall: Record<number, number> = {};
  for (const k of ks) {
    recall[k] = scored.reduce((s, r) => s + recallAt(r.ranked, r.gold, k), 0) / (scored.length || 1);
  }
  return {
    n: scored.length,
    recall,
    mrr: scored.reduce((s, r) => s + reciprocalRank(r.ranked, r.gold), 0) / (scored.length || 1),
  };
}

/** 把 chunk 级检索结果压成文档级排名（同一篇文档多次命中只保留最靠前的一次） */
export function toDocRanking(chunkDocIds: string[]): RankedDocIds {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const d of chunkDocIds) {
    if (seen.has(d)) continue;
    seen.add(d);
    out.push(d);
  }
  return out;
}
