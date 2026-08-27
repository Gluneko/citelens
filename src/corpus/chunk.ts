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
import { sentenceSpans, structureSpans } from "./split.js";

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

/**
 * 策略三：按结构切。标题与空行是天然语义边界，且绝不在句子中间下刀。
 * 直觉：一段话讲一件事，把它整段留住，比机械地数 300 个字更可能保住完整答案。
 */
export function chunkStructure(doc: Doc, maxSize = 300): Chunk[] {
  return structureSpans(doc.text, maxSize).map(([start, end], n) => ({
    id: `${doc.id}#${n}`,
    docId: doc.id,
    title: doc.title,
    text: doc.text.slice(start, end),
    start,
    end,
    source: doc.source,
  }));
}

/**
 * 策略四：父子块（small-to-big）。
 *
 * 这是本课最重要的一个想法：**检索和回答，要的是两种不同的东西**。
 *  - 检索要「信号纯」：块越小，主题越单一，向量越不会被无关内容稀释
 *  - 回答要「上下文全」：块太小，模型看不到前因后果，照样答不好
 * 于是：用小块（child）去命中，命中后把它所属的大块（parent）喂给模型。
 *
 * 实弹动机：g12 问"火山石头上的小洞"，答案「气孔构造」躺在一个 300 字块的第 182 字，
 * 那个块的主体是玄武岩的定义、含硅量、颜色、密度——**平均池化把气孔的信号冲淡了**。
 */
export function chunkParentChild(doc: Doc, childSize = 120, parentSize = 400): Chunk[] {
  const out: Chunk[] = [];
  let n = 0;
  for (const [pa, pb] of structureSpans(doc.text, parentSize)) {
    const parentText = doc.text.slice(pa, pb);
    let cur: [number, number] | null = null;
    const flush = () => {
      if (!cur) return;
      out.push({
        id: `${doc.id}#${n++}`,
        docId: doc.id,
        title: doc.title,
        text: doc.text.slice(cur[0], cur[1]),
        start: cur[0],
        end: cur[1],
        source: doc.source,
        context: parentText, // ← 检索命中子块，喂给模型的却是父块
      });
      cur = null;
    };
    for (const [sa, sb] of sentenceSpans(parentText)) {
      const [A, B] = [pa + sa, pa + sb];
      if (cur && B - cur[0] <= childSize) cur[1] = B;
      else { flush(); cur = [A, B]; }
    }
    flush();
  }
  return out;
}

export type Strategy = "fixed" | "overlap" | "structure" | "parent-child";

export interface StrategyOptions {
  size?: number;
  overlap?: number;
  childSize?: number;
  parentSize?: number;
}

export function chunkBy(doc: Doc, strategy: Strategy, o: StrategyOptions = {}): Chunk[] {
  switch (strategy) {
    case "fixed": return chunkFixed(doc, { size: o.size ?? 300, overlap: 0 });
    case "overlap": return chunkFixed(doc, { size: o.size ?? 300, overlap: o.overlap ?? 60 });
    case "structure": return chunkStructure(doc, o.size ?? 300);
    case "parent-child": return chunkParentChild(doc, o.childSize ?? 120, o.parentSize ?? 400);
  }
}

export function chunkAll(docs: Doc[], opts?: FixedOptions): Chunk[] {
  return docs.flatMap((d) => chunkFixed(d, opts));
}

export function chunkAllBy(docs: Doc[], strategy: Strategy, o?: StrategyOptions): Chunk[] {
  return docs.flatMap((d) => chunkBy(d, strategy, o));
}
