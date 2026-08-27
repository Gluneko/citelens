/**
 * 给所有 chunk 算向量，缓存到 data/index/vectors.json。
 *
 * ⚠️ 首次运行会从 HuggingFace 下载模型（约 100MB），需要网络。
 *    国内可设镜像：HF_ENDPOINT=https://hf-mirror.com pnpm vectors
 *    模型缓存后离线可用；向量算完后 pnpm eval 也完全离线。
 *
 * 增量：已算过的 chunk 直接复用（按 chunk.id + 模型名匹配），改切分参数后只补新块。
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { LocalEmbedder } from "../src/retrieve/embed.js";
import { packVectors, unpackVectors, type VectorStoreFile } from "../src/retrieve/vector.js";
import type { Chunk } from "../src/types.js";

const OUT = "data/index/vectors.json";
const BATCH = 32;

const chunks: Chunk[] = readFileSync("data/chunks.jsonl", "utf-8")
  .split(/\r?\n/).filter(Boolean).map((l) => JSON.parse(l) as Chunk);

const embedder = new LocalEmbedder();
console.log(`🧠 模型 ${embedder.name}｜chunk ${chunks.length} 个`);

let cache = new Map<string, Float32Array>();
if (existsSync(OUT)) {
  const old = JSON.parse(readFileSync(OUT, "utf-8")) as VectorStoreFile;
  if (old.model === embedder.name) {
    cache = unpackVectors(old);
    console.log(`♻️  复用缓存 ${cache.size} 条`);
  } else {
    console.log(`⚠️  缓存来自另一个模型（${old.model}），全部重算`);
  }
}

const todo = chunks.filter((c) => !cache.has(c.id));
console.log(`📐 需要新算 ${todo.length} 条`);

const t0 = performance.now();
for (let i = 0; i < todo.length; i += BATCH) {
  const batch = todo.slice(i, i + BATCH);
  // 文档侧不加查询前缀——BGE 的查询/文档是不对称的，加错了反而掉点
  const vecs = await embedder.embed(batch.map((c) => `${c.title} ${c.text}`));
  batch.forEach((c, j) => cache.set(c.id, vecs[j]!));
  const done = Math.min(i + BATCH, todo.length);
  process.stdout.write(`\r   ${done}/${todo.length}（${((done / todo.length) * 100).toFixed(0)}%）`);
}
if (todo.length) process.stdout.write("\n");
const ms = performance.now() - t0;

// 只保留当前 chunk 集合里的向量，避免缓存无限膨胀
const ids = chunks.map((c) => c.id);
const vecs = ids.map((id) => cache.get(id)!);
mkdirSync("data/index", { recursive: true });
writeFileSync(OUT, JSON.stringify(packVectors(embedder.name, embedder.dim, ids, vecs)));

const mb = (ids.length * embedder.dim * 4) / 1024 / 1024;
console.log(`✅ ${ids.length} 条 × ${embedder.dim} 维 = ${mb.toFixed(2)}MB → ${OUT}`);
if (todo.length) console.log(`   耗时 ${(ms / 1000).toFixed(1)}s（${(ms / todo.length).toFixed(0)}ms/条）`);
