/**
 * 全项目的数据契约（单一事实来源）。
 * 一条 RAG 流水线只有两种核心数据：文档 Doc（切分前）与片段 Chunk（切分后、检索的最小单位）。
 */

/** 语料里的一篇文档 */
export interface Doc {
  /** 文档 id，用文件名（稳定、可读，出问题时一眼看出是哪篇） */
  id: string;
  title: string;
  /** 正文（已去掉 frontmatter） */
  text: string;
  /** 出处与许可——RAG 项目里这不是可选项：答案要能指回来源 */
  source: string;
  license: string;
}

/**
 * 检索的最小单位。
 * 关键设计：chunk 必须保留 `start`/`end`——即它在原文中的字符区间。
 * 没有这两个数字，就无法把答案里的引文"指回原文第几个字"，Day 4 的归因校验器就无从谈起。
 */
export interface Chunk {
  /** 形如 basalt#3：文档 id + 序号，人眼可读 */
  id: string;
  docId: string;
  title: string;
  text: string;
  /** 在原文 text 中的起止字符下标（含 start，不含 end） */
  start: number;
  end: number;
  source: string;
  /**
   * 喂给模型的上下文。多数策略下等于 text；
   * 父子块策略下是「父块全文」——**用小块去检索，用大块去回答**。
   * 检索要的是信号纯，回答要的是上下文全，这两件事可以分开优化。
   */
  context?: string;
}

/**
 * 期望行为。二元的「作答 / 拒答」不够用——实弹逼出了第三态。
 *
 * answer  语料能完整回答 → 应当作答
 * refuse  语料完全没有相关事实 → 应当拒答
 * partial 语料只有一部分 → **应当作答已有的部分，并明确声明缺口**
 *
 * 第三态的由来（g26 实弹）：问「烧失量超过 3% 应如何校正」，
 * 语料写了通用做法（扣除烧失量归一到 100%），但没有针对 >3% 的特殊步骤。
 * 模型的回答是：讲清通用做法 + 明说"片段未提及额外的特殊校正步骤"。
 * 这既诚实又有用，**比拒答更好**——而当时的二元判定把它记成了失败。
 *
 * 与 BrachioKey 的三态输出同构：那里确立了"无法判定是合法结论"，
 * 这里补上对称的一条：**"只能部分回答"也是合法结论**。
 */
export type ExpectedBehavior = "answer" | "refuse" | "partial";

/** 金标准问答：一个问题 + 它应该命中的 chunk 所属文档/关键片段 */
export interface GoldCase {
  id: string;
  question: string;
  /** 期望行为；缺省时按 goldDocIds 是否为空推断 answer / refuse */
  expect?: ExpectedBehavior;
  /** 正确答案应当来自这些文档（文档级判定，比逐 chunk 标注省力且够用） */
  goldDocIds: string[];
  /** 答案必须包含的关键事实（Day 4 生成层评测用；检索层只用 goldDocIds） */
  mustContain?: string[];
  /** 这条题考的是什么——写给未来的自己看 */
  note: string;
}
