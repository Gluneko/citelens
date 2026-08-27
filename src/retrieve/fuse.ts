/**
 * RRF —— 倒数排名融合（Reciprocal Rank Fusion）。
 *
 * 要把 BM25 和向量两路结果合成一路，最直觉的做法是"把分数加起来"。
 * 但这行不通：BM25 的分是 0～30 的无上界统计量，余弦是 -1～1 的几何量，
 * 两者不同量纲、不同分布，相加等于拿体重加身高。
 *
 * RRF 的聪明之处是【只看名次，不看分数】：
 *
 *     score(d) = Σ  1 / (k + rank_i(d))
 *
 * 每一路里排第 1 的贡献 1/(k+1)，第 2 名 1/(k+2)……名次越靠后贡献越小。
 * k 是阻尼常数（论文经验值 60）：k 越大，头部与尾部的差距越平缓，
 * 越不容易被某一路的"第一名"独断。
 *
 * 为什么混合几乎总赢：两路犯的错不相关。
 * 词法在"通篇没有高信息量词"的问题上失灵，语义在"专名与数字"上失灵——
 * 一路漏掉的，另一路常常还留着。
 */

export interface FusedItem<T> {
  item: T;
  score: number;
  /** 它在各路里的名次（1 起算），未进入该路则缺席——排查融合行为时极有用 */
  ranks: Record<string, number>;
}

export function rrf<T>(
  lists: Array<{ name: string; items: T[] }>,
  keyOf: (item: T) => string,
  k = 60,
): Array<FusedItem<T>> {
  const acc = new Map<string, FusedItem<T>>();
  for (const { name, items } of lists) {
    items.forEach((item, i) => {
      const key = keyOf(item);
      let e = acc.get(key);
      if (!e) acc.set(key, (e = { item, score: 0, ranks: {} }));
      e.score += 1 / (k + i + 1);
      e.ranks[name] = i + 1;
    });
  }
  return [...acc.values()].sort(
    (a, b) => b.score - a.score || keyOf(a.item).localeCompare(keyOf(b.item)),
  );
}
