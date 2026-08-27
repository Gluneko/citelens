/**
 * 生成层配置。检索层全本地零成本，只有这一层需要 API key。
 */
export const config = {
  llmApiKey: process.env.LLM_API_KEY ?? "",
  anthropicBaseUrl: process.env.ANTHROPIC_BASE_URL ?? "https://api.deepseek.com/anthropic",
  model: process.env.CITELENS_MODEL ?? "deepseek-chat",
};

/** .env.example 里的占位值——照抄过去而没改，是最常见的第一次失败 */
const PLACEHOLDERS = new Set(["your-api-key", "sk-xxx", "changeme", ""]);

export function requireApiKey(): void {
  const key = config.llmApiKey.trim();
  if (PLACEHOLDERS.has(key)) {
    console.error("❌ LLM_API_KEY 还是占位值。请编辑 .env 填入真实密钥：");
    console.error("     LLM_API_KEY=sk-你的真实密钥");
    console.error(`   当前端点 ${config.anthropicBaseUrl}｜模型 ${config.model}`);
    console.error("   （只有生成层需要密钥，检索层 pnpm eval / sweep / diag 全部离线可跑）");
    process.exit(1);
  }
  if (key.length < 16) {
    console.error(`❌ LLM_API_KEY 看起来不像有效密钥（长度 ${key.length}）。请检查 .env`);
    process.exit(1);
  }
}
