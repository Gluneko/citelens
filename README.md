# 文鉴 CiteLens 🔎🪨

> 中文地学文献 RAG —— 确定性归因校验 × 混合检索 × 评测先行
>
> 姊妹项目（一套方法论，六种确定性形态）：
> [食途 TripBite](https://github.com/Gluneko/tripbite)（规则约束）·
> [岩鉴 PetroLens](https://github.com/Gluneko/petrolens)（数值复算）·
> [腕鉴 BrachioKey](https://github.com/Gluneko/brachiokey)（决策树复走）·
> [鸟探 BirdScout](https://github.com/Gluneko/birdscout)（数据血缘）·
> [矿探 OreScout](https://github.com/Gluneko/orescout)（统计推断确定性化）·
> 本项目（**归因的确定性化**）

**输入**：中文地学语料 + 一个问题
**输出**：带引文的回答——每个断言标注来源片段，每个数字都能回溯到原文

## 核心设计：引文可以被逐字校验

RAG 公认的难点是"模型会不会编"。这个问题有一半可以确定性化：

```
语料 → 切分（保留 start/end 字符区间）
        ▼
   混合召回（BM25 词法 + 向量语义 → RRF 融合）
        ▼
   重排序（cross-encoder 精排，召回 50 → 精排 5）
        ▼
   生成（结构化输出：每个断言带 chunkId + 原文引文）
        ▼
   归因校验器（确定性，四条铁律）：
     1. 引用的 chunkId 必须真的在本次检索结果里 —— 禁止幻觉引用
     2. 引文必须是该 chunk 的【字面子串】 —— 一个字符都不许飘
     3. 答案里的每个数字必须在被引片段中出现过 —— 数字幻觉专项
     4. 证据不足必须拒答 —— "不知道"是合法且必须诚实的结论
        ▼
   不通过 → 结构化原因定向打回 │ 通过 → 输出
```

第 2 条能成立，靠的是切分时保留了 `chunk.start` / `chunk.end`，且单测锁死了不变量
`chunk.text === doc.text.slice(start, end)`——引文溯源的地基在切分那一步就已经打好。

## 快速开始

```bash
pnpm install
pnpm test      # 无需网络与 API key
pnpm chunks    # 语料 → data/chunks.jsonl，并打印长度分布
pnpm eval      # 检索层评测：recall@k / MRR / 事实层（零网络零 API）

pnpm vectors                  # 给所有 chunk 算向量（首次需下模型 ~100MB）
# 后端自动选择：原生 onnxruntime-node 优先，缺二进制时回落 wasm
# 强制指定：CITELENS_DEVICE=wasm pnpm vectors
# Apple Silicon 上若装的是 x64 版 Node（Rosetta），原生后端无 darwin/x64 二进制，
# 换 arm64 版 Node 可显著提速：nvm install 22 --arch=arm64
pnpm eval -- --mode hybrid           # 两路 RRF 融合召回
pnpm eval -- --mode hybrid --rerank  # 融合召回 + cross-encoder 精排
pnpm eval -- --compare --verbose     # 四种配置一次跑完并排对照
pnpm eval -- --k1 1.2 --b 0.5        # BM25 调参

pnpm diag g12                 # 单题诊断：答案片段排第几、两路各捞回了什么
# （别叫 why——pnpm 有同名内置命令会静默挡掉脚本）
pnpm sweep                    # 四种切分策略 × 两路检索的 A/B（几分钟）
pnpm sweep -- --no-vec        # 只比 BM25，秒出

cp .env.example .env          # 只有生成层需要 API key
pnpm ask "玄武岩的二氧化硅含量是多少？"          # 检索→生成带出处的回答→归因校验→定向打回
pnpm calibrate                # 拒答阈值校准（结论：相关性≠答案在不在里面，阈值不可靠）
pnpm eval:answer -- --refusal # 生成层评测：只跑拒答题（约 $1）
pnpm eval:answer              # 生成层评测：全部 24 题（约 $4）

pnpm fetch:wiki  # 可选：抓维基地学条目扩充语料（需要能访问维基的网络）
```

## 语料与许可

- `data/corpus/`：12 篇**合成教学语料**（依据公开教科书知识撰写，CC0），入库，
  保证 clone 下来就能跑通全流程与 CI——不依赖任何外部服务
- `data/corpus-wiki/`：可选的中文维基百科地学条目（CC BY-SA 4.0），**不入库**，
  用 `pnpm fetch:wiki` 自行抓取（118 个条目清单，限频 1.1 秒，标明 UA）
- 金标准答案始终落在合成语料里；抓取语料只负责把"干草堆"变大，让同一套题更难、指标更有区分度
- 教科书、行业标准、付费文献一律不入库

## 评测先行

`data/gold/questions.jsonl` —— 20 道金标准题，在写任何检索代码**之前**就已建好：

| 题型 | 用意 |
|---|---|
| 专名型（Middlemost、CIPW） | 检验词法检索的主场 |
| 口语改写型（"黑色火山石上的小洞"） | 检验语义检索的主场 |
| 数字型（45%～52%、20%、5% 与 75%） | Day 4 数字幻觉校验的靶子 |
| 清单型 / 过程型 | 检验切分有没有把完整答案切碎 |
| 跨文档型 | 为图谱增强与多跳做铺垫 |
| 拒答题（语料里根本没有的信息） | 最重要的一题：诚实比正确更难 |

出题人同样被判卷：`lintGold` 校验金标准自洽（goldDocId 必须存在、mustContain 必须真的能在标注文档里找到），并进单测。

## 路线图

- [x] Day 0：语料 + 定长切分（保留字符区间）+ 20 题金标准 + case lint
- [x] Day 1：手写 BM25（倒排索引 / TF-IDF / 中文分词）+ recall@k / MRR 评测 → **recall@5 基线**
- [x] Day 2：向量检索（本地 bge-small-zh，暴力余弦）+ 四种切分策略 A/B + 单题诊断工具
- [x] Day 3：RRF 混合召回 + cross-encoder 重排 + 候选池召回率（漏斗两段分开量）
- [x] Day 4：归因校验器（四条铁律）+ 结构化回答 + 定向打回
- [x] Day 4 余项：5 道对抗拒答题 + 拒答阈值校准 + 生成层评测（诚实率/一遍过率）
- [ ] Day 4 遗留：向量库选型（暴力 vs HNSW）+ 重复采样看方差
- [ ] Day 5：查询改写 / HyDE / 会话式指代消解 + 图谱增强 + 评测收口

## 边界声明

个人学习与作品集项目。合成语料为教学演示用，不构成任何地学专业依据；
真实研究请使用规范文献与专业数据库。
