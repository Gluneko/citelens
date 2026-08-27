/**
 * 金标准集装载 + 自洽性校验。
 *
 * 这里立一条规矩（PetroLens 的 case lint 制度照搬）：
 * 金标准里引用的 docId 必须真实存在于语料中。出题人写错，比考生答错更隐蔽也更致命——
 * 所以它进单测，每次跑测试都自动体检。
 */
import { readFileSync } from "node:fs";
import type { Doc, GoldCase } from "../types.js";

export function loadGold(path = "data/gold/questions.jsonl"): GoldCase[] {
  return readFileSync(path, "utf-8")
    .split(/\r?\n/)
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as GoldCase);
}

export interface GoldLintIssue {
  caseId: string;
  problem: string;
}

export function lintGold(cases: GoldCase[], docs: Doc[]): GoldLintIssue[] {
  const issues: GoldLintIssue[] = [];
  const docIds = new Set(docs.map((d) => d.id));
  const seen = new Set<string>();

  for (const c of cases) {
    if (seen.has(c.id)) issues.push({ caseId: c.id, problem: "id 重复" });
    seen.add(c.id);
    if (!c.question.trim()) issues.push({ caseId: c.id, problem: "问题为空" });
    if (!c.note?.trim()) issues.push({ caseId: c.id, problem: "缺 note：未来的你会想知道这题考的是什么" });

    for (const g of c.goldDocIds) {
      if (!docIds.has(g)) issues.push({ caseId: c.id, problem: `goldDocId「${g}」在语料中不存在` });
    }

    // 关键事实必须真的能在标注文档里找到，否则这题永远不可能答对
    for (const must of c.mustContain ?? []) {
      const hit = docs.some((d) => c.goldDocIds.includes(d.id) && d.text.includes(must));
      if (!hit) issues.push({ caseId: c.id, problem: `mustContain「${must}」在标注文档中找不到` });
    }

    if (c.goldDocIds.length === 0 && (c.mustContain?.length ?? 0) > 0) {
      issues.push({ caseId: c.id, problem: "拒答题不应有 mustContain" });
    }
  }
  return issues;
}
