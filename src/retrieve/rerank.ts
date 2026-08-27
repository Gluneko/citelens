/**
 * 重排序（rerank）—— 理解整套 RAG 架构的钥匙。
 *
 * 召回阶段用的是 bi-encoder：问题与文档【分开】编码。
 *   优点：文档可以离线算好、入库，查询时只做一次点积 → 能扛百万级语料
 *   代价：模型从未同时看到"问题"和"这篇文档"，它只能比较两个各自压缩过的摘要
 *
 * 精排阶段用的是 cross-encoder：把问题和文档【拼在一起】送进模型。
 *   优点：模型能逐词对照，直接判断"小洞"是否就是"气孔构造"
 *   代价：每个候选都要跑一次完整前向，无法预计算 → 只能用在少量候选上
 *
 * 于是必然是漏斗：
 *   便宜的召回捞 50 条（任务是【别漏掉】）→ 昂贵的精排选 5 条（任务是【排第一】）
 *
 * 实弹依据：g12 的答案片段在向量里排第 43、在 BM25 里排第 137。
 * 召回 top50 能把它纳入候选池，但要把它顶到第一，只能靠精排。
 */

export interface Reranker {
  readonly name: string;
  /** 返回与 docs 等长的相关性分数（越大越相关） */
  score(query: string, docs: string[]): Promise<number[]>;
}

/** bge-reranker：中文 cross-encoder，输出未归一化的相关性 logit */
export class CrossEncoderReranker implements Reranker {
  private tokenizer: any = null;
  private model: any = null;

  constructor(readonly name = "Xenova/bge-reranker-base", private batch = 16) {}

  private async ensure() {
    if (this.model) return;
    const { AutoTokenizer, AutoModelForSequenceClassification } = await import("@huggingface/transformers");
    this.tokenizer = await AutoTokenizer.from_pretrained(this.name);
    this.model = await AutoModelForSequenceClassification.from_pretrained(this.name, { dtype: "q8" });
  }

  async score(query: string, docs: string[]): Promise<number[]> {
    if (!docs.length) return [];
    await this.ensure();
    const out: number[] = [];
    for (let i = 0; i < docs.length; i += this.batch) {
      const batch = docs.slice(i, i + this.batch);
      // 关键：query 与 doc 成对送入（text_pair），这正是 cross-encoder 与 bi-encoder 的分野
      const inputs = this.tokenizer(new Array(batch.length).fill(query), {
        text_pair: batch,
        padding: true,
        truncation: true,
      });
      const { logits } = await this.model(inputs);
      const data = logits.data as Float32Array;
      for (let j = 0; j < batch.length; j++) out.push(data[j]!);
    }
    return out;
  }
}

/** 离线测试用：按"查询词在文档里出现几个"打分，行为可预测且不需要模型 */
export class FakeReranker implements Reranker {
  readonly name = "fake-overlap";
  async score(query: string, docs: string[]): Promise<number[]> {
    const qs = [...new Set(Array.from(query))].filter((ch) => /\S/.test(ch));
    return docs.map((d) => qs.filter((ch) => d.includes(ch)).length);
  }
}

export interface RerankedItem<T> {
  item: T;
  score: number;
  /** 精排前的名次（1 起算）——用来观察"精排把谁从多少名捞了上来" */
  before: number;
}

/** 对候选池重新打分排序，返回前 topK */
export async function rerank<T>(
  reranker: Reranker,
  query: string,
  candidates: T[],
  textOf: (item: T) => string,
  topK = 5,
): Promise<Array<RerankedItem<T>>> {
  const scores = await reranker.score(query, candidates.map(textOf));
  return candidates
    .map((item, i) => ({ item, score: scores[i]!, before: i + 1 }))
    .sort((a, b) => b.score - a.score || a.before - b.before)
    .slice(0, topK);
}
