/**
 * 语料 → chunks.jsonl
 * 用法：pnpm chunks            （默认 size=300 overlap=60）
 *       pnpm chunks -- 500 100
 *
 * 顺带打印分布统计——切分参数改一次就看一次，是最便宜的直觉训练。
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { chunkAll } from "../src/corpus/chunk.js";
import { loadCorpus } from "../src/corpus/load.js";

const argv = process.argv.slice(2).filter((a) => a !== "--");
const size = Number(argv[0] ?? 300);
const overlap = Number(argv[1] ?? 60);

const docs = loadCorpus();
const chunks = chunkAll(docs, { size, overlap });

mkdirSync("data", { recursive: true });
writeFileSync("data/chunks.jsonl", chunks.map((c) => JSON.stringify(c)).join("\n") + "\n");

const lens = chunks.map((c) => c.text.length).sort((a, b) => a - b);
const pct = (p: number) => lens[Math.min(lens.length - 1, Math.floor(lens.length * p))]!;
const totalChars = docs.reduce((s, d) => s + d.text.length, 0);

console.log(`📚 文档 ${docs.length} 篇 / 正文 ${totalChars} 字`);
console.log(`✂️  切分 size=${size} overlap=${overlap} → ${chunks.length} 个 chunk`);
console.log(`   长度分布 min=${lens[0]} p50=${pct(0.5)} p90=${pct(0.9)} max=${lens.at(-1)}`);
console.log(`   平均每篇 ${(chunks.length / docs.length).toFixed(1)} 块`);
console.log(`💾 data/chunks.jsonl`);
