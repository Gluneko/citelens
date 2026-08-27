/**
 * 生成层配置。检索层全本地零成本，只有这一层需要 API key。
 */
export const config = {
  llmApiKey: process.env.LLM_API_KEY ?? "",
  anthropicBaseUrl: process.env.ANTHROPIC_BASE_URL ?? "https://api.deepseek.com/anthropic",
  model: process.env.CITELENS_MODEL ?? "deepseek-chat",
};

export function requireApiKey(): void {
  if (!config.llmApiKey) {
    console.error("❌ 缺少 LLM_API_KEY。请 cp .env.example .env 并填入（检索层不需要，只有生成层需要）");
    process.exit(1);
  }
}
