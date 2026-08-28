/**
 * 检索流水线：把四种配置收敛成同一个接口，评测才能公平对比。
 *
 *   bm25    词法单路
 *   vector  语义单路
 *   hybrid  两路 RRF 融合
 *   + rerank  在上述任一召回之后接 cross-encoder 精排
 *
 * 关键设计：search() 同时返回【候选池 pool】与【最终结果 hits】。
 * 因为这两段的职责根本不同——
 *   召回阶段的 KPI 是"别漏掉"（看候选池召回率）
 *   精排阶段的 KPI 是"排第一"（看 recall@1 / MRR）
 * 把它们混在一个数字里看，就永远说不清是哪一段拖了后腿。
 */
import type { Chunk } from "../types.js";
import type { Bm25Index } from "./bm25.js";
import type { Embedder } from "./embed.js";
import { rrf } from "./fuse.js";
import { expandPoolByGraph, type DocGraph } from "./graph.js";
import { rerank, type Reranker } from "./rerank.js";
import type { VectorIndex } from "./vector.js";

export type RetrieveMode = "bm25" | "vector" | "hybrid";

export interface PipelineDeps {
  bm25?: Bm25Index;
  vector?: VectorIndex;
  embedder?: Embedder;
  reranker?: Reranker;
  /** 文档图谱 + 全量 chunk：启用多跳扩展时必须一起给 */
  graph?: DocGraph;
  allChunks?: Chunk[];
}

export interface PipelineOptions {
  mode: RetrieveMode;
  /** 召回候选数。漏斗的宽口——太窄会漏，太宽精排会慢 */
  poolSize?: number;
  topK?: number;
  rrfK?: number;
  /**
   * 单路保底名额：混合模式下，强制把每一路各自的前 N 名放进候选池。
   *
   * 为什么需要它（实弹教训）：RRF 只奖励"两路共识"，
   * 于是【只有一路能找到的真答案】会被一堆两路都中游的项挤出候选池。
   * 实测 g12 的答案在向量单路排第 17，融合后却掉出前 50——
   * 融合反而让候选池召回率从 100% 掉到 94.7%。
   * 保底名额就是给"另辟蹊径的那一路"留座位。设 0 关闭。
   */
  guarantee?: number;
  /** 只精排候选池的前 N 条（精排很贵，这是成本旋钮）。0/未设表示全池精排 */
  rerankTop?: number;
  /** 图谱扩展：取池中前 N 个文档的邻居首段补进池尾（多跳）。0/未设关闭 */
  graphExpand?: number;
}

export interface PipelineResult {
  /** 精排后的最终结果（无精排时即召回前 topK） */
  hits: Chunk[];
  /** 召回候选池（精排前），按召回顺序 */
  pool: Chunk[];
  /** 若启用精排，记录每个最终结果精排前的名次 */
  liftedFrom?: number[];
  /**
   * 若启用精排，记录每个最终结果的精排分数。
   * cross-encoder 的分数分布远比余弦分散，因此它是【比余弦好得多的拒答判据】——
   * 余弦把相关与无关都挤在 0.5～0.6，精排分数则能拉开好几个数量级。
   */
  scores?: number[];
}

export class RetrievalPipeline {
  constructor(private deps: PipelineDeps, private opts: PipelineOptions) {}

  get label(): string {
    const base = { bm25: "BM25", vector: "向量", hybrid: "混合RRF" }[this.opts.mode];
    return this.deps.reranker ? `${base}+精排` : base;
  }

  async search(query: string): Promise<PipelineResult> {
    const pool = this.maybeExpand(await this.recall(query));
    const topK = this.opts.topK ?? 5;
    if (!this.deps.reranker) return { hits: pool.slice(0, topK), pool };

    const n = this.opts.rerankTop && this.opts.rerankTop > 0 ? this.opts.rerankTop : pool.length;
    const toRank = pool.slice(0, n);
    const ranked = await rerank(this.deps.reranker, query, toRank, (c) => c.context ?? c.text, topK);
    return {
      hits: ranked.map((r) => r.item),
      pool,
      liftedFrom: ranked.map((r) => r.before),
      scores: ranked.map((r) => r.score),
    };
  }

  /**
   * 多查询检索：原始问题 + 若干改写各跑一路召回，RRF 按名次融合成一个候选池。
   * 精排仍用【原始问题】打分——改写只负责把答案捞进池子，
   * 最终"哪段最相关"必须以用户真正问的话为准。
   */
  async searchMulti(queries: string[]): Promise<PipelineResult> {
    if (queries.length === 0) throw new Error("searchMulti 需要至少一个查询");
    if (queries.length === 1) return this.search(queries[0]!);
    const n = this.opts.poolSize ?? 50;
    const pools: Array<{ name: string; items: Chunk[] }> = [];
    for (const [i, q] of queries.entries()) {
      pools.push({ name: `q${i}`, items: await this.recall(q) });
    }
    const pool = this.maybeExpand(rrf(pools, (c) => c.id, this.opts.rrfK ?? 60).slice(0, n).map((f) => f.item));
    const topK = this.opts.topK ?? 5;
    if (!this.deps.reranker) return { hits: pool.slice(0, topK), pool };
    const m = this.opts.rerankTop && this.opts.rerankTop > 0 ? this.opts.rerankTop : pool.length;
    const ranked = await rerank(this.deps.reranker, queries[0]!, pool.slice(0, m), (c) => c.context ?? c.text, topK);
    return { hits: ranked.map((r) => r.item), pool, liftedFrom: ranked.map((r) => r.before), scores: ranked.map((r) => r.score) };
  }

  /** 图谱多跳扩展（可选）：邻居首段补池尾，只扩召回、不改排序权 */
  private maybeExpand(pool: Chunk[]): Chunk[] {
    const n = this.opts.graphExpand ?? 0;
    if (n <= 0 || !this.deps.graph || !this.deps.allChunks) return pool;
    return expandPoolByGraph(pool, this.deps.graph, this.deps.allChunks, { seedDocs: n, maxAdd: 5 }).pool;
  }

  private async recall(query: string): Promise<Chunk[]> {
    const n = this.opts.poolSize ?? 50;
    const { bm25, vector, embedder } = this.deps;

    if (this.opts.mode === "bm25") {
      if (!bm25) throw new Error("bm25 模式需要 Bm25Index");
      return bm25.search(query, n).map((h) => h.chunk);
    }
    if (this.opts.mode === "vector") {
      if (!vector || !embedder) throw new Error("vector 模式需要 VectorIndex 与 Embedder");
      return (await vector.search(embedder, query, n)).map((h) => h.chunk);
    }
    if (!bm25 || !vector || !embedder) throw new Error("hybrid 模式需要两路索引");
    const lex = bm25.search(query, n).map((h) => h.chunk);
    const sem = (await vector.search(embedder, query, n)).map((h) => h.chunk);
    const fused = rrf(
      [{ name: "bm25", items: lex }, { name: "vector", items: sem }],
      (c) => c.id,
      this.opts.rrfK ?? 60,
    ).map((f) => f.item);

    const pool = fused.slice(0, n);
    const g = this.opts.guarantee ?? 0;
    if (g <= 0) return pool;

    // 单路保底：把每路各自前 g 名中被融合挤掉的，塞回候选池尾部
    const have = new Set(pool.map((c) => c.id));
    const rescued: Chunk[] = [];
    for (const c of [...lex.slice(0, g), ...sem.slice(0, g)]) {
      if (!have.has(c.id)) { have.add(c.id); rescued.push(c); }
    }
    if (!rescued.length) return pool;
    return [...pool.slice(0, Math.max(0, n - rescued.length)), ...rescued];
  }
}
