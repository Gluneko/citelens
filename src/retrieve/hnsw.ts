/**
 * 手写 HNSW —— 近似最近邻（ANN）索引。
 *
 * 暴力搜索每次查询要和全库逐条点积，O(N)。803 条无所谓，百万条就是灾难。
 * HNSW 的直觉是【跳表的高维版】：
 *   - 每个点随机抽签决定"住几层"（几何分布：大多数点只住底层，越高越稀疏）
 *   - 高层稀疏，用来快速跳到目标的大致区域；底层稠密，用来精细逼近
 *   - 查询：从顶层入口贪心下潜，每层走到局部最近，落到底层后做一次宽度为 ef 的束搜索
 *
 * 必须诚实的三件事：
 *   1. ANN 的 A 是"近似"——召回率 < 100%，速度是拿精度换的
 *   2. ef 是运行时旋钮：调大→更准更慢，调小→更快更漏
 *   3. 构建有成本：插入比暴力"存进数组"贵得多，写多读少的场景要三思
 *
 * 实现为教学级：单线程、无删除、无持久化；随机源可注入种子，保证测试可复现。
 */
import { cosine } from "./embed.js";

/** 可复现的伪随机数（mulberry32）——ANN 的层数抽签必须可复现，否则测试就是掷骰子 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

interface Node {
  id: string;
  vec: Float32Array;
  /** neighbors[l] = 第 l 层的邻居下标 */
  neighbors: number[][];
}

export interface HnswOptions {
  /** 每层最大邻居数。大→图更稠密更准，索引更大 */
  M?: number;
  /** 构建时的束宽。大→建得慢但图质量高 */
  efConstruction?: number;
  seed?: number;
}

export class HnswIndex {
  private nodes: Node[] = [];
  private entry = -1;
  private topLayer = -1;
  private readonly M: number;
  private readonly M0: number;
  private readonly efC: number;
  private readonly mL: number;
  private readonly rand: () => number;

  constructor(opts: HnswOptions = {}) {
    this.M = opts.M ?? 12;
    this.M0 = this.M * 2;          // 底层允许双倍邻居（经验规则）
    this.efC = opts.efConstruction ?? 100;
    this.mL = 1 / Math.log(this.M);
    this.rand = mulberry32(opts.seed ?? 42);
  }

  get size(): number { return this.nodes.length; }

  add(id: string, vec: Float32Array): void {
    const level = Math.floor(-Math.log(Math.max(this.rand(), 1e-12)) * this.mL);
    const node: Node = { id, vec, neighbors: Array.from({ length: level + 1 }, () => []) };
    const idx = this.nodes.length;
    this.nodes.push(node);

    if (this.entry < 0) { this.entry = idx; this.topLayer = level; return; }

    // 1) 从顶层贪心下潜到 level+1（每层只找一个局部最近点作为下一层入口）
    let ep = this.entry;
    for (let l = this.topLayer; l > level; l--) ep = this.greedy(vec, ep, l);

    // 2) level..0 每层：束搜索找候选，连边并互连（超限时按距离裁剪）
    for (let l = Math.min(level, this.topLayer); l >= 0; l--) {
      const cand = this.searchLayer(vec, ep, l, this.efC);
      const cap = l === 0 ? this.M0 : this.M;
      const chosen = cand.slice(0, this.M);
      node.neighbors[l] = chosen.map(([i]) => i);
      for (const [ni] of chosen) {
        const nb = this.nodes[ni]!.neighbors[l];
        if (!nb) continue;
        nb.push(idx);
        if (nb.length > cap) {
          // 裁剪：只保留离该节点最近的 cap 个
          nb.sort((a, b) => cosine(this.nodes[ni]!.vec, this.nodes[b]!.vec) - cosine(this.nodes[ni]!.vec, this.nodes[a]!.vec));
          nb.length = cap;
        }
      }
      ep = cand[0]?.[0] ?? ep;
    }
    if (level > this.topLayer) { this.entry = idx; this.topLayer = level; }
  }

  /** 单层贪心：一步步挪向更近的邻居，直到局部最优 */
  private greedy(q: Float32Array, start: number, layer: number): number {
    let cur = start;
    let curSim = cosine(q, this.nodes[cur]!.vec);
    for (;;) {
      let best = cur, bestSim = curSim;
      for (const ni of this.nodes[cur]!.neighbors[layer] ?? []) {
        const s = cosine(q, this.nodes[ni]!.vec);
        if (s > bestSim) { best = ni; bestSim = s; }
      }
      if (best === cur) return cur;
      cur = best; curSim = bestSim;
    }
  }

  /** 单层束搜索：维护宽度 ef 的候选集，返回按相似度降序的 [下标, 相似度] */
  private searchLayer(q: Float32Array, start: number, layer: number, ef: number): Array<[number, number]> {
    const visited = new Set<number>([start]);
    const startSim = cosine(q, this.nodes[start]!.vec);
    // 教学级用排序数组代替堆：规模内性能足够，代码好读
    let frontier: Array<[number, number]> = [[start, startSim]];
    const found: Array<[number, number]> = [[start, startSim]];
    while (frontier.length) {
      const [cur, curSim] = frontier.shift()!;
      const worst = found.length >= ef ? found[found.length - 1]![1] : -Infinity;
      if (curSim < worst && found.length >= ef) break;
      for (const ni of this.nodes[cur]!.neighbors[layer] ?? []) {
        if (visited.has(ni)) continue;
        visited.add(ni);
        const s = cosine(q, this.nodes[ni]!.vec);
        if (found.length < ef || s > found[found.length - 1]![1]) {
          insertSorted(found, [ni, s]);
          if (found.length > ef) found.length = ef;
          insertSorted(frontier, [ni, s]);
        }
      }
    }
    return found;
  }

  /** 查询：ef 越大越准越慢（运行时旋钮，与构建参数无关） */
  search(q: Float32Array, topK = 10, ef = 64): Array<{ id: string; score: number }> {
    if (this.entry < 0) return [];
    let ep = this.entry;
    for (let l = this.topLayer; l > 0; l--) ep = this.greedy(q, ep, l);
    return this.searchLayer(q, ep, 0, Math.max(ef, topK))
      .slice(0, topK)
      .map(([i, s]) => ({ id: this.nodes[i]!.id, score: s }));
  }
}

function insertSorted(arr: Array<[number, number]>, item: [number, number]): void {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid]![1] >= item[1]) lo = mid + 1;
    else hi = mid;
  }
  arr.splice(lo, 0, item);
}
