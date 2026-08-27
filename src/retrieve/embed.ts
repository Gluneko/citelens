/**
 * 向量化（embedding）：把一段文字映射成一串数字，即"语义坐标"。
 *
 * 核心直觉：意思接近的两段话，坐标方向也接近。
 * 于是"小洞"和"气孔"、"火山石头"和"喷出岩"可以互相找到——
 * 这正是 BM25 在 g12 上做不到的事。
 *
 * 这里刻意把 Embedder 抽成接口：
 *  - 单测用确定性的假向量器，离线、毫秒级、结果可复现；
 *  - 真跑时才换成本地模型。
 * 检索逻辑（余弦、topK、归一化）与"谁来产生向量"解耦，是这层唯一重要的设计。
 */

export interface Embedder {
  readonly name: string;
  readonly dim: number;
  /** 文档侧向量化 */
  embed(texts: string[]): Promise<Float32Array[]>;
  /** 查询侧向量化。某些模型要求查询加指令前缀，见下 */
  embedQuery(text: string): Promise<Float32Array>;
}

/**
 * BGE 系列中文模型要求：**查询要加指令前缀，文档不加**。
 *
 * 为什么不对称？因为检索是"短问题 → 长文档"的匹配，两者形态天然不同。
 * 加上这句前缀，等于告诉模型"我在检索，请按检索的方式表示这句话"，
 * 官方实测能提升几个点。忘了加是使用 BGE 最常见的错误。
 */
export const BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章：";

/** L2 归一化：归一化之后，余弦相似度就等于点积——省一次开方，也避免长度干扰 */
export function normalize(v: Float32Array): Float32Array {
  let sum = 0;
  for (const x of v) sum += x * x;
  const norm = Math.sqrt(sum) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i]! / norm;
  return out;
}

/** 两个【已归一化】向量的余弦相似度 = 点积，取值 [-1, 1]，越大越像 */
export function cosine(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error(`维度不一致：${a.length} vs ${b.length}`);
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
  return dot;
}

/**
 * 本地 embedding：transformers.js 在 Node 里跑 ONNX 模型，不联网、不花钱。
 * 首次运行会从 HuggingFace 下载模型（约 100MB），之后走本地缓存。
 */
export class LocalEmbedder implements Embedder {
  readonly name: string;
  dim = 512; // bge-small-zh-v1.5 的向量维度，首次 embed 后按实际值校正
  private pipe: any = null;

  constructor(name = "Xenova/bge-small-zh-v1.5") {
    this.name = name;
  }

  private async ensure() {
    if (this.pipe) return this.pipe;
    const { pipeline } = await import("@huggingface/transformers");
    this.pipe = await pipeline("feature-extraction", this.name, { dtype: "q8" });
    return this.pipe;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    const pipe = await this.ensure();
    // mean pooling + L2 归一化，是 BGE 官方推荐的取向量方式
    const out = await pipe(texts, { pooling: "mean", normalize: true });
    const dim = out.dims.at(-1) as number;
    this.dim = dim;
    const flat = out.data as Float32Array;
    return texts.map((_, i) => new Float32Array(flat.slice(i * dim, (i + 1) * dim)));
  }

  async embedQuery(text: string): Promise<Float32Array> {
    const [v] = await this.embed([BGE_ZH_QUERY_PREFIX + text]);
    return v!;
  }
}

/**
 * 确定性假向量器：把字符哈希撒进固定维度，同样的文本永远得到同样的向量。
 * 它当然不懂语义——它的用途是让"余弦 / topK / 归一化 / 缓存"这些逻辑
 * 能在无网络、无模型的环境里被单测覆盖。
 */
export class FakeEmbedder implements Embedder {
  readonly name = "fake-hash";
  constructor(readonly dim = 32) {}

  private one(text: string): Float32Array {
    const v = new Float32Array(this.dim);
    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      v[code % this.dim] = v[code % this.dim]! + 1;
    }
    return normalize(v);
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    return texts.map((t) => this.one(t));
  }

  async embedQuery(text: string): Promise<Float32Array> {
    return this.one(text);
  }
}
