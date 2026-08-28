/**
 * 生成层小工具：跑一次结构化 LLM 调用。
 * ask / eval-answer / 查询改写共用，免得三处各写一遍消息循环。
 */
import { query } from "@anthropic-ai/claude-agent-sdk";
import { config } from "./config.js";

export async function runStructured<T>(opts: {
  system: string;
  prompt: string;
  schema: Record<string, unknown>;
  parse: (x: unknown) => T | undefined;
  maxTurns?: number;
}): Promise<{ value: T; cost: number }> {
  let value: T | undefined;
  let cost = 0;
  for await (const m of query({
    prompt: opts.prompt,
    options: {
      model: config.model,
      systemPrompt: opts.system,
      disallowedTools: ["Bash", "Edit", "Write", "Read", "Task", "WebFetch", "WebSearch"],
      permissionMode: "bypassPermissions",
      maxTurns: opts.maxTurns ?? 4,
      outputFormat: { type: "json_schema", schema: opts.schema },
      env: { ...process.env, ANTHROPIC_BASE_URL: config.anthropicBaseUrl, ANTHROPIC_AUTH_TOKEN: config.llmApiKey },
    },
  })) {
    if (m.type === "result") {
      if (m.subtype === "success" && !m.is_error) {
        cost = m.total_cost_usd;
        value = opts.parse(m.structured_output);
      } else {
        const detail = "result" in m && typeof (m as any).result === "string" ? (m as any).result : m.subtype;
        throw new Error(`LLM 调用失败：${detail}`.slice(0, 200));
      }
    }
  }
  if (value === undefined) throw new Error("LLM 未返回合法结构化输出");
  return { value, cost };
}
