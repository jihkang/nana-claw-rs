use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::types::*;
use super::{Provider, ProviderError};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const MAX_TOKENS: u32 = 4096;

// ─── Anthropic Provider ─────────────────────────────────────────

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String, base_url: Option<String>, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            model,
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        system_prompt: Option<&str>,
    ) -> Result<LlmResponse, ProviderError> {
        let url = format!("{}/v1/messages", self.base_url);
        debug!(provider = "anthropic", model = %self.model, url = %url, "sending chat request");

        let api_messages = convert_messages(messages);
        let api_tools = convert_tools(tools);

        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "messages": api_messages,
        });

        if let Some(system) = system_prompt {
            body["system"] = serde_json::json!(system);
        }

        if !api_tools.is_empty() {
            body["tools"] = serde_json::json!(api_tools);
        }

        debug!(body = %serde_json::to_string(&body).unwrap_or_default(), "request body");

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = response.text().await.unwrap_or_default();
            error!(
                provider = "anthropic",
                status = status_code,
                body = %body_text,
                "API request failed"
            );
            return match status_code {
                429 => Err(ProviderError::RateLimited {
                    message: body_text,
                }),
                401 | 403 => Err(ProviderError::AuthError {
                    message: body_text,
                }),
                _ => Err(ProviderError::ApiError {
                    status: status_code,
                    body: body_text,
                }),
            };
        }

        let api_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::ParseError(format!("failed to parse response JSON: {e}")))?;

        debug!(
            provider = "anthropic",
            stop_reason = ?api_response.stop_reason,
            input_tokens = api_response.usage.input_tokens,
            output_tokens = api_response.usage.output_tokens,
            "received response"
        );

        let content = api_response
            .content
            .into_iter()
            .map(|block| match block {
                AnthropicContentBlock::Text { text } => ContentBlock::Text { text },
                AnthropicContentBlock::ToolUse { id, name, input } => {
                    ContentBlock::ToolUse { id, name, input }
                }
            })
            .collect();

        let stop_reason = match api_response.stop_reason.as_deref() {
            Some("end_turn") => StopReason::EndTurn,
            Some("tool_use") => StopReason::ToolUse,
            Some("max_tokens") => StopReason::MaxTokens,
            Some("stop_sequence") => StopReason::StopSequence,
            _ => StopReason::EndTurn,
        };

        Ok(LlmResponse {
            content,
            stop_reason,
            usage: Usage {
                input_tokens: api_response.usage.input_tokens,
                output_tokens: api_response.usage.output_tokens,
            },
        })
    }

    fn name(&self) -> &str {
        "anthropic"
    }
}

// ─── Anthropic API Types ────────────────────────────────────────

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ─── Conversion Helpers ─────────────────────────────────────────

fn convert_messages(messages: &[Message]) -> Vec<AnthropicMessage> {
    // Anthropic expects messages with "user" and "assistant" roles only;
    // system messages are sent via the top-level "system" parameter.
    // We skip System messages here (the caller passes system_prompt separately).
    messages
        .iter()
        .filter_map(|msg| match msg {
            Message::System { .. } => None,
            Message::User { content } => Some(AnthropicMessage {
                role: "user".to_string(),
                content: serde_json::json!(content),
            }),
            Message::Assistant { content } => {
                let blocks: Vec<serde_json::Value> = content
                    .iter()
                    .map(|block| match block {
                        ContentBlock::Text { text } => serde_json::json!({
                            "type": "text",
                            "text": text,
                        }),
                        ContentBlock::ToolUse { id, name, input } => serde_json::json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": input,
                        }),
                    })
                    .collect();
                Some(AnthropicMessage {
                    role: "assistant".to_string(),
                    content: serde_json::json!(blocks),
                })
            }
            Message::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => Some(AnthropicMessage {
                role: "user".to_string(),
                content: serde_json::json!([{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                    "is_error": is_error,
                }]),
            }),
        })
        .collect()
}

fn convert_tools(tools: &[ToolDefinition]) -> Vec<AnthropicTool> {
    tools
        .iter()
        .map(|tool| AnthropicTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
        })
        .collect()
}
