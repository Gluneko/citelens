/**
 * 语料扩充：从中文维基百科抓地学条目正文。
 *
 * ⚠️ 需要网络。云端沙箱与本地 VM 都不通维基，这个脚本请在你自己的 Mac 终端跑。
 *    仓库自带的 12 篇合成语料是离线夹具，保证 clone 下来就能跑通全流程与 CI。
 *
 * 合规：只读公开的 MediaWiki API，限频 ≥1 秒，标明 UA；正文按 CC BY-SA 4.0 使用，
 *      每篇 frontmatter 写明来源 URL 与许可。抓下来的语料不入库（见 .gitignore）。
 *
 * 用法：pnpm tsx scripts/fetch-wiki.ts            抓默认清单
 *       pnpm tsx scripts/fetch-wiki.ts 花岗岩 片麻岩   抓指定条目
 */
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const OUT = "data/corpus-wiki";
const UA = "CiteLens/0.1 (personal RAG study project; contact via github.com/Gluneko)";
const API = "https://zh.wikipedia.org/w/api.php";
const DELAY_MS = 1100;

/** 地学条目清单：覆盖三大岩类、矿物、构造、古生物、化探，且彼此有交叉引用（利于造多跳问题） */
const TITLES = [
  "岩石", "火成岩", "沉积岩", "变质岩", "矿物", "造岩矿物", "石英", "长石", "云母", "辉石",
  "角闪石", "橄榄石", "方解石", "黄铁矿", "黄铜矿", "方铅矿", "闪锌矿", "白钨矿", "锡石", "辉钼矿",
  "玄武岩", "安山岩", "流纹岩", "英安岩", "粗面岩", "响岩", "辉长岩", "闪长岩", "花岗岩", "花岗闪长岩",
  "橄榄岩", "科马提岩", "凝灰岩", "黑曜岩", "浮岩", "火山碎屑岩", "岩浆", "岩浆房", "火山", "火山喷发",
  "砂岩", "泥岩", "页岩", "砾岩", "石灰岩", "白云岩", "燧石", "蒸发岩", "煤", "石油",
  "板岩", "千枚岩", "片岩", "片麻岩", "大理岩", "石英岩", "混合岩", "榴辉岩", "角岩", "变质作用",
  "板块构造论", "海底扩张", "大洋中脊", "俯冲带", "海沟", "岛弧", "造山运动", "地震", "断层", "褶皱",
  "地层学", "岩石地层学", "生物地层学", "地质年代", "寒武纪", "奥陶纪", "志留纪", "泥盆纪", "石炭纪", "二叠纪",
  "三叠纪", "侏罗纪", "白垩纪", "古近纪", "新近纪", "第四纪", "化石", "三叶虫", "笔石", "菊石",
  "腕足动物门", "双壳纲", "珊瑚", "有孔虫", "放射虫", "牙形石", "古生物学", "地层对比", "沉积相", "层理",
  "地球化学", "勘查地球化学", "土壤地球化学测量", "元素丰度", "克拉克值", "微量元素", "稀土元素", "同位素地质年代学",
  "矿床", "斑岩铜矿", "矽卡岩", "热液矿床", "伟晶岩", "成矿作用", "找矿", "地球物理勘探", "钻探", "地质图",
];

async function fetchExtract(title: string): Promise<string | null> {
  const url = `${API}?action=query&format=json&formatversion=2&prop=extracts&explaintext=1&exsectionformat=plain&redirects=1&titles=${encodeURIComponent(title)}`;
  const res = await fetch(url, { headers: { "User-Agent": UA, "Accept-Encoding": "gzip" } });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const json: any = await res.json();
  const page = json?.query?.pages?.[0];
  if (!page || page.missing) return null;
  return typeof page.extract === "string" ? page.extract : null;
}

/** 去掉参考文献/外部链接等尾部小节，它们对问答无用却污染检索 */
function stripTail(text: string): string {
  const cut = text.search(/\n\s*(参见|参考文献|外部連結|外部链接|延伸閱讀|延伸阅读|注釋|注释)\s*\n/);
  return (cut > 0 ? text.slice(0, cut) : text).trim();
}

function slug(title: string, i: number): string {
  return `wiki-${String(i).padStart(3, "0")}-${title.replace(/[\/\\:*?"<>|\s]/g, "_")}`;
}

const titles = process.argv.slice(2).filter((a) => a !== "--");
const list = titles.length ? titles : TITLES;
mkdirSync(OUT, { recursive: true });

let ok = 0, miss = 0, fail = 0, chars = 0;
for (const [i, title] of list.entries()) {
  const file = join(OUT, `${slug(title, i)}.md`);
  if (existsSync(file)) { console.log(`⏭  ${title}（已存在）`); continue; }
  try {
    const raw = await fetchExtract(title);
    if (!raw) { miss++; console.log(`∅  ${title}（无此条目）`); }
    else {
      const text = stripTail(raw);
      if (text.length < 200) { miss++; console.log(`∅  ${title}（正文过短 ${text.length} 字）`); }
      else {
        const fm = [
          "---",
          `title: ${title}`,
          `source: https://zh.wikipedia.org/wiki/${encodeURIComponent(title)}`,
          "license: CC BY-SA 4.0",
          "---",
          "",
        ].join("\n");
        writeFileSync(file, fm + text + "\n");
        ok++; chars += text.length;
        console.log(`✓  ${title}（${text.length} 字）`);
      }
    }
  } catch (e) {
    fail++;
    console.log(`✗  ${title}：${e instanceof Error ? e.message : e}`);
  }
  await new Promise((r) => setTimeout(r, DELAY_MS)); // 限频，做个有礼貌的抓取者
}
console.log(`\n📦 成功 ${ok} / 缺失 ${miss} / 失败 ${fail}，共 ${chars} 字 → ${OUT}/`);
if (fail > 0) console.log("提示：若全部失败，多半是网络不通——请在能访问维基的终端里跑。");
