/**
 * 向量检索：暴力余弦搜索。
 *
 * 是的，暴力——把查询向量和全库每一条向量都点积一遍，取最高的 K 条。
 * 复杂度 O(N·d)，803 个 chunk × 512 维 = 41 万次乘加，现代 CPU 上不到 1 毫秒。
 *
 * 为什么不一上来就用向量库？因为 Day 4 我们要亲手量一次：
 * 暴力搜索在多大规模上开始撑不住，HNSW 又是拿什么换来的速度。
 * 没有这个对照，"该不该上向量库"就只能靠听说。
 *
 * 顺带一提：暴力搜索的召回率恒为 100%（它检查了每一条），
 * 而所有 ANN 索引的召回率都 < 100%。速度是拿精度换的。
 */
import type { Chunk } from "../types.js";
import { cosine, type Embedder } from "./embed.js";

export interface VectorHit {
  chunk: Chunk;
  score: number;
}

export class VectorIndex {
  private readonly chunks: Chunk[] = [];
  private readonly vectors: Float32Array[] = [];

  constructor(readonly model: string, readonly dim: number) {}

  add(chunk: Chunk, vector: Float32Array): this {
    if (vector.length !== this.dim) throw new Error(`维度不一致：期望 ${this.dim}，实得 ${vector.length}`);
    this.chunks.push(chunk);
    this.vectors.push(vector);
    return this;
  }

  get size(): number {
    return this.chunks.length;
  }

  /** 用已有的查询向量检索 */
  searchByVector(q: Float32Array, topK = 5): VectorHit[] {
    const scored: Array<[number, number]> = [];
    for (let i = 0; i < this.vectors.length; i++) {
      scored.push([i, cosine(q, this.vectors[i]!)]);
    }
    return scored
      .sort((a, b) => b[1] - a[1] || a[0] - b[0]) // 同分按下标，保证可复现
      .slice(0, topK)
      .map(([i, score]) => ({ chunk: this.chunks[i]!, score }));
  }

  async search(embedder: Embedder, query: string, topK = 5): Promise<VectorHit[]> {
    return this.searchByVector(await embedder.embedQuery(query), topK);
  }
}

/** 向量缓存的磁盘格式：向量存 base64（比 JSON 数组小 4 倍且不丢精度） */
export interface VectorStoreFile {
  model: string;
  dim: number;
  ids: string[];
  /** Float32Array 的原始字节，base64 编码 */
  data: string;
}

export function packVectors(model: string, dim: number, ids: string[], vecs: Float32Array[]): VectorStoreFile {
  const flat = new Float32Array(ids.length * dim);
  vecs.forEach((v, i) => flat.set(v, i * dim));
  return { model, dim, ids, data: Buffer.from(flat.buffer).toString("base64") };
}

export function unpackVectors(file: VectorStoreFile): Map<string, Float32Array> {
  const buf = Buffer.from(file.data, "base64");
  const flat = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const map = new Map<string, Float32Array>();
  file.ids.forEach((id, i) => {
    map.set(id, new Float32Array(flat.slice(i * file.dim, (i + 1) * file.dim)));
  });
  return map;
}
