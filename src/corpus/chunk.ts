/**
 * 切分（Day 0 只做最笨的一种：定长 + 重叠）。
 *
 * 为什么从最笨的开始：Day 2 下午我们会把四种切法放在同一把尺子（金标准集）上比，
 * 没有这个"笨基线"，后面的提升就没有对照。
 *
 * 一个必须守住的不变量：chunk.text === doc.text.slice(chunk.start, chunk.end)。
 * 它是 Day 4 引文溯源的地基——引文能不能"指回原文第几个字"，全靠这个等式成立。
 */
import type { Chunk, Doc } from "../types.js";

export interface FixedOptions {
  /** 每个 chunk 的目标字符数（中文按字符算，不是 token） */
  size?: number;
  /** 相邻 chunk 的重叠字符数：防止一句话正好被切成两半，两边都答不全 */
  overlap?: number;
}

export function chunkFixed(doc: Doc, opts: FixedOptions = {}): Chunk[] {
  const size = opts.size ?? 300;
  const overlap = opts.overlap ?? 60;
  if (size <= 0) throw new Error("size 必须为正");
  if (overlap < 0 || overlap >= size) throw new Error("overlap 必须在 [0, size) 内");

  const text = doc.text;
  const chunks: Chunk[] = [];
  const stride = size - overlap;

  for (let start = 0, n = 0; start < text.length; start += stride, n++) {
    const end = Math.min(start + size, text.length);
    const slice = text.slice(start, end);
    // 末尾若只剩零星空白，不值得单独成块
    if (slice.trim().length === 0) break;
    chunks.push({
      id: `${doc.id}#${n}`,
      docId: doc.id,
      title: doc.title,
      text: slice,
      start,
      end,
      source: doc.source,
    });
    if (end === text.length) break;
  }
  return chunks;
}

export function chunkAll(docs: Doc[], opts?: FixedOptions): Chunk[] {
  return docs.flatMap((d) => chunkFixed(d, opts));
}
