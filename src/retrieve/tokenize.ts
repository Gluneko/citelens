/**
 * 中文分词。
 *
 * 英文检索天然有空格分词，中文没有——"大洋中脊玄武岩"要切成
 * 「大洋 / 中脊 / 玄武岩」还是「大洋中脊 / 玄武岩」，直接决定了能不能被检索到。
 *
 * 这里用 Node 内置的 Intl.Segmenter（ICU 词典分词），零依赖。
 * 它不如专业分词器聪明（专业术语常被切碎），但正因为如此，
 * Day 3 你会亲眼看到"词法检索的天花板"从哪来。
 */

/** 高频虚词：出现在几乎每篇文档里，IDF 极低，留着只是徒增计算与噪音 */
const STOP = new Set([
  "的", "了", "是", "在", "和", "与", "或", "而", "则", "也", "都", "被", "把", "对", "为",
  "其", "此", "该", "这", "那", "有", "个", "上", "中", "下", "内", "外", "等", "及", "以",
  "并", "但", "由", "于", "从", "到", "至", "者", "所", "之", "它", "他", "她", "我", "你",
  "可以", "可能", "通常", "一般", "因此", "所以", "由于", "如果", "虽然", "但是", "以及",
  "一", "二", "三", "不", "很", "会", "能", "要", "就", "还", "又", "使", "让", "多", "少",
]);

const segmenter = new Intl.Segmenter("zh-Hans-CN", { granularity: "word" });

/**
 * 切词并归一化：
 * - 只保留"像词"的片段（isWordLike 会滤掉标点与空白）
 * - 拉丁字母统一转小写（CIPW 与 cipw 应当命中同一个词）
 * - 去掉停用词与单字符的拉丁/数字碎片
 */
export function tokenize(text: string): string[] {
  const out: string[] = [];
  for (const seg of segmenter.segment(text)) {
    if (!seg.isWordLike) continue;
    const w = seg.segment.trim().toLowerCase();
    if (!w || STOP.has(w)) continue;
    // 单个拉丁字母或单个数字几乎不携带检索信息；单个汉字则保留（"金""铜"都是有效检索词）
    if (w.length === 1 && /[a-z0-9]/.test(w)) continue;
    out.push(w);
  }
  return out;
}

/** 词频统计：BM25 里的 TF 就来自这里 */
export function termFreq(tokens: string[]): Map<string, number> {
  const tf = new Map<string, number>();
  for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
  return tf;
}
