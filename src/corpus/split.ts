/**
 * 文本切分的两个基础动作：按句子断、按结构断。
 * 独立成文件是因为它们与"切分策略"正交——策略是怎么攒，这里是怎么断。
 */

/** 中文句末标点；保留标点在句尾，切完拼回去要与原文逐字一致 */
const SENT_END = /[。！？；\n]/;

/** 按句子切，返回 [起, 止) 区间，保证无缝覆盖全文 */
export function sentenceSpans(text: string): Array<[number, number]> {
  const spans: Array<[number, number]> = [];
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    if (SENT_END.test(text[i]!)) {
      // 把连续的句末标点/换行都并进本句
      let j = i;
      while (j + 1 < text.length && SENT_END.test(text[j + 1]!)) j++;
      spans.push([start, j + 1]);
      start = j + 1;
      i = j;
    }
  }
  if (start < text.length) spans.push([start, text.length]);
  return spans.filter(([a, b]) => text.slice(a, b).trim().length > 0);
}

/**
 * 按结构切：markdown 标题与空行是天然的语义边界。
 * 返回块区间，每块尽量不超过 maxSize，但**绝不在句子中间下刀**。
 */
export function structureSpans(text: string, maxSize: number): Array<[number, number]> {
  // 先按"标题行 / 空行"划出自然段
  const blocks: Array<[number, number]> = [];
  let start = 0;
  const lines = text.split("\n");
  let pos = 0;
  for (const line of lines) {
    const isBoundary = /^#{1,6}\s/.test(line) || line.trim() === "";
    if (isBoundary && pos > start) {
      blocks.push([start, pos]);
      start = pos;
    }
    pos += line.length + 1;
  }
  if (start < text.length) blocks.push([start, text.length]);

  // 自然段太长就按句子再分，太短就与相邻块合并
  const out: Array<[number, number]> = [];
  for (const [a, b] of blocks) {
    if (text.slice(a, b).trim().length === 0) continue;
    if (b - a <= maxSize) {
      const last = out.at(-1);
      if (last && last[1] - last[0] + (b - a) <= maxSize) last[1] = b; // 合并短块
      else out.push([a, b]);
      continue;
    }
    // 长块按句子攒到 maxSize
    let cur: [number, number] | null = null;
    for (const [sa, sb] of sentenceSpans(text.slice(a, b))) {
      const [A, B] = [a + sa, a + sb];
      if (cur && B - cur[0] <= maxSize) cur[1] = B;
      else {
        if (cur) out.push(cur);
        cur = [A, B];
      }
    }
    if (cur) out.push(cur);
  }
  return out;
}
