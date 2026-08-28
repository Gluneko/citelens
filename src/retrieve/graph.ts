/**
 * 文档图谱增强 —— 治向量检索的另一个天生短板：多跳问题。
 *
 * "形成大洋中脊新洋壳的岩石，对应的深成岩是什么？"
 * 答案要走两步：大洋中脊 →（玄武岩）→ 辉长岩。
 * 检索只看"这段文字和问题像不像"，而第二跳的文档和问题可能一个词都不重合。
 *
 * 图从哪来：不做实体抽取（那需要模型），用一条完全确定性的规则——
 * **文档 A 的标题出现在文档 B 的正文里，就连一条边**。
 * 标题提及是作者亲手写下的关联，比任何统计共现都可靠。
 *
 * 用在哪：召回之后，沿着 top 文档的边把邻居文档的首段补进候选池尾部。
 * 纪律与单路保底一致：**只许扩大召回，不许有否决权**——
 * 替换池尾、不加座位，最终排序仍由精排按原始问题决定。
 */
import type { Chunk, Doc } from "../types.js";

export type DocGraph = Map<string, Set<string>>;

export function buildDocGraph(docs: Doc[]): DocGraph {
  const g: DocGraph = new Map(docs.map((d) => [d.id, new Set<string>()]));
  for (const a of docs) {
    const title = a.title.trim();
    if (title.length < 2) continue; // 单字标题（如"煤"）到处误连，宁缺毋滥
    for (const b of docs) {
      if (a.id === b.id) continue;
      if (b.text.includes(title)) {
        g.get(a.id)!.add(b.id);
        g.get(b.id)!.add(a.id); // 无向：提及是双向线索
      }
    }
  }
  return g;
}

export interface GraphStats {
  nodes: number;
  edges: number;
  isolated: string[];
  topHubs: Array<[string, number]>;
}

export function graphStats(g: DocGraph): GraphStats {
  let edges = 0;
  const isolated: string[] = [];
  const degree: Array<[string, number]> = [];
  for (const [id, nb] of g) {
    edges += nb.size;
    degree.push([id, nb.size]);
    if (nb.size === 0) isolated.push(id);
  }
  return {
    nodes: g.size,
    edges: edges / 2,
    isolated,
    topHubs: degree.sort((a, b) => b[1] - a[1]).slice(0, 5),
  };
}

/**
 * 候选池图谱扩展：取池中前 seedDocs 个文档，把它们的邻居文档
 * （尚未出现在池中的）各取首段补进池尾。返回新池（长度不变）。
 */
export function expandPoolByGraph(
  pool: Chunk[],
  graph: DocGraph,
  allChunks: Chunk[],
  opts: { seedDocs?: number; maxAdd?: number } = {},
): { pool: Chunk[]; added: string[] } {
  const seedDocs = opts.seedDocs ?? 3;
  const maxAdd = opts.maxAdd ?? 5;
  const inPool = new Set(pool.map((c) => c.docId));
  const firstChunk = new Map<string, Chunk>();
  for (const c of allChunks) {
    const cur = firstChunk.get(c.docId);
    if (!cur || c.start < cur.start) firstChunk.set(c.docId, c);
  }

  const seeds: string[] = [];
  for (const c of pool) {
    if (!seeds.includes(c.docId)) seeds.push(c.docId);
    if (seeds.length >= seedDocs) break;
  }

  const added: string[] = [];
  const extra: Chunk[] = [];
  for (const seed of seeds) {
    for (const nb of graph.get(seed) ?? []) {
      if (inPool.has(nb) || added.length >= maxAdd) continue;
      const fc = firstChunk.get(nb);
      if (!fc) continue;
      inPool.add(nb);
      added.push(nb);
      extra.push(fc);
    }
  }
  if (!extra.length) return { pool, added };
  return { pool: [...pool.slice(0, Math.max(0, pool.length - extra.length)), ...extra], added };
}
