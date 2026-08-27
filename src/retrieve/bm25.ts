/**
 * BM25 —— 词法检索的工业标准，1990 年代的产物，至今仍是所有 RAG 系统的基石之一。
 *
 * 它回答一个问题：给定查询词，哪篇文档最"相关"？
 * 三个直觉叠在一起：
 *
 * 1) TF（词频）——一个词在本文档出现得越多，本文档越可能在讲它。
 *    但不能线性叠加："玄武岩"出现 100 次不该是出现 10 次的 10 倍相关。
 *    所以 BM25 让 TF 饱和：tf / (tf + k1)，k1 控制饱和速度。
 *
 * 2) IDF（逆文档频率）——一个词在全库越罕见，命中它越说明问题。
 *    "岩石"人人都有，"霓石"千金难求。
 *
 * 3) 文档长度归一——长文档天然更容易撞上任何词，要把这份便宜压回去。
 *    b 控制压多狠：b=0 完全不管长度，b=1 完全按长度归一。
 *
 * 合起来就是每个查询词的得分，再对所有查询词求和：
 *
 *   score(D, Q) = Σ  IDF(q) · ( tf · (k1+1) ) / ( tf + k1·(1 - b + b·|D|/avgdl) )
 *
 * 没有任何机器学习，纯统计。这也是它的可贵之处：结果完全可解释、可复现。
 */
import type { Chunk } from "../types.js";
import { termFreq, tokenize } from "./tokenize.js";

export interface Bm25Options {
  /** 词频饱和速度。经验值 1.2~2.0，越小越快饱和 */
  k1?: number;
  /** 文档长度归一强度，0~1。经验值 0.75 */
  b?: number;
}

export interface Hit {
  chunk: Chunk;
  score: number;
  /** 命中了哪些查询词——调试与讲解时极有用：分数从哪来一目了然 */
  matched: string[];
}

export class Bm25Index {
  private readonly k1: number;
  private readonly b: number;
  private readonly chunks: Chunk[] = [];
  /** 倒排索引：词 → [(文档下标, 该词在此文档的词频)]。检索的核心数据结构 */
  private readonly postings = new Map<string, Array<[number, number]>>();
  private readonly docLen: number[] = [];
  private avgdl = 0;

  constructor(opts: Bm25Options = {}) {
    this.k1 = opts.k1 ?? 1.5;
    this.b = opts.b ?? 0.75;
  }

  /**
   * 建索引。注意这里做的事：不是"把文档存起来"，而是把它们【翻过来】——
   * 从"文档里有哪些词"翻成"每个词出现在哪些文档里"。
   * 这就是倒排（inverted）二字的由来，也是检索能在毫秒级完成的原因：
   * 查询时只需取出查询词那几条 postings，而不用遍历全库。
   */
  add(chunks: Chunk[]): this {
    for (const c of chunks) {
      const idx = this.chunks.length;
      this.chunks.push(c);
      // 标题也参与检索：标题词往往是最强的相关性信号
      const tokens = tokenize(`${c.title} ${c.text}`);
      this.docLen.push(tokens.length);
      for (const [term, tf] of termFreq(tokens)) {
        let p = this.postings.get(term);
        if (!p) this.postings.set(term, (p = []));
        p.push([idx, tf]);
      }
    }
    this.avgdl = this.docLen.reduce((s, n) => s + n, 0) / (this.docLen.length || 1);
    return this;
  }

  get size(): number {
    return this.chunks.length;
  }

  get vocabSize(): number {
    return this.postings.size;
  }

  /** IDF：用的是 BM25 的概率式变体，罕见词得分高，且对超高频词给出接近 0 的权重 */
  idf(term: string): number {
    const df = this.postings.get(term)?.length ?? 0;
    const N = this.chunks.length;
    return Math.log(1 + (N - df + 0.5) / (df + 0.5));
  }

  search(query: string, topK = 5): Hit[] {
    const qTerms = [...new Set(tokenize(query))];
    const scores = new Map<number, number>();
    const matched = new Map<number, string[]>();

    for (const term of qTerms) {
      const postings = this.postings.get(term);
      if (!postings) continue; // 查询词全库都没有，直接跳过
      const idf = this.idf(term);
      for (const [idx, tf] of postings) {
        const norm = 1 - this.b + this.b * (this.docLen[idx]! / this.avgdl);
        const gain = idf * ((tf * (this.k1 + 1)) / (tf + this.k1 * norm));
        scores.set(idx, (scores.get(idx) ?? 0) + gain);
        const m = matched.get(idx);
        if (m) m.push(term);
        else matched.set(idx, [term]);
      }
    }

    return [...scores.entries()]
      .sort((a, b) => b[1] - a[1] || a[0] - b[0]) // 分数相同按下标定序，保证结果可复现
      .slice(0, topK)
      .map(([idx, score]) => ({ chunk: this.chunks[idx]!, score, matched: matched.get(idx) ?? [] }));
  }
}

export function buildBm25(chunks: Chunk[], opts?: Bm25Options): Bm25Index {
  return new Bm25Index(opts).add(chunks);
}
