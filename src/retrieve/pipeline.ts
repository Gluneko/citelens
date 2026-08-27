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
import { rerank, type Reranker } from "./rerank.js";
import type { VectorIndex } from "./vector.js";

export type RetrieveMode = "bm25" | "vector" | "hybrid";

export interface PipelineDeps {
  bm25?: Bm25Index;
  vector?: VectorIndex;
  embedder?: Embedder;
  reranker?: Reranker;
}

export interface PipelineOptions {
  mode: RetrieveMode;
  /** 召回候选数。漏斗的宽口——太窄会漏，太宽精排会慢 */
  poolSize?: number;
  topK?: number;
  rrfK?: number;
}

export interface PipelineResult {
  /** 精排后的最终结果（无精排时即召回前 topK） */
  hits: Chunk[];
  /** 召回候选池（精排前），按召回顺序 */
  pool: Chunk[];
  /** 若启用精排，记录每个最终结果精排前的名次 */
  liftedFrom?: number[];
}

export class RetrievalPipeline {
  constructor(private deps: PipelineDeps, private opts: PipelineOptions) {}

  get label(): string {
    const base = { bm25: "BM25", vector: "向量", hybrid: "混合RRF" }[this.opts.mode];
    return this.deps.reranker ? `${base}+精排` : base;
  }

  async search(query: string): Promise<PipelineResult> {
    const pool = await this.recall(query);
    const topK = this.opts.topK ?? 5;
    if (!this.deps.reranker) return { hits: pool.slice(0, topK), pool };

    const ranked = await rerank(this.deps.reranker, query, pool, (c) => c.context ?? c.text, topK);
    return { hits: ranked.map((r) => r.item), pool, liftedFrom: ranked.map((r) => r.before) };
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
    return rrf(
      [{ name: "bm25", items: lex }, { name: "vector", items: sem }],
      (c) => c.id,
      this.opts.rrfK ?? 60,
    ).slice(0, n).map((f) => f.item);
  }
}
