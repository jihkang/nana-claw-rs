use async_trait::async_trait;
use reqwest::Client;
use tracing::{debug, error};

use crate::types::*;
use super::openai;
use super::{Provider, ProviderError};

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api";
const APP_TITLE: &str = "NanoClaw";
const APP_REFERER: &str = "https://github.com/neocode24/nano_claw";

// ─── OpenRouter Provider ────────────────────────────────────────

pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl OpenRouterProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
        }
    }
}

#[async_trait]
impl Provider for OpenRouterProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        system_prompt: Option<&str>,
    ) -> Result<LlmResponse, ProviderError> {
        let url = format!("{}/v1/chat/completions", OPENROUTER_BASE_URL);
        debug!(provider = "openrouter", model = %self.model, url = %url, "sending chat request");

        // Reuse OpenAI conversion logic for messages and tools.
        let api_messages = openai::convert_messages(messages, system_prompt);
        let api_tools = openai::convert_tools(tools);

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": api_messages,
        });

        if !api_tools.is_empty() {
            body["tools"] = serde_json::json!(api_tools);
        }

        debug!(body = %serde_json::to_string(&body).unwrap_or_default(), "request body");

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", APP_REFERER)
            .header("X-Title", APP_TITLE)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = response.text().await.unwrap_or_default();
            error!(
                provider = "openrouter",
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

        // OpenRouter returns the same response format as OpenAI.
        let api_response: OpenRouterResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::ParseError(format!("failed to parse response JSON: {e}")))?;

        let choice = api_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::ParseError("no choices in response".to_string()))?;

        debug!(
            provider = "openrouter",
            finish_reason = ?choice.finish_reason,
            usage_prompt = api_response.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
            usage_completion = api_response.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
            "received response"
        );

        let content = parse_response_message(choice.message)?;

        let stop_reason = match choice.finish_reason.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("tool_calls") => StopReason::ToolUse,
            Some("length") => StopReason::MaxTokens,
            Some("content_filter") => StopReason::StopSequence,
            _ => StopReason::EndTurn,
        };

        let usage = api_response
            .usage
            .map(|u| Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
            })
            .unwrap_or_default();

        Ok(LlmResponse {
            content,
            stop_reason,
            usage,
        })
    }

    fn name(&self) -> &str {
        "openrouter"
    }
}

// ─── OpenRouter Response Types ──────────────────────────────────
// These mirror OpenAI's format. We define them locally to avoid
// leaking private deserialization types from the openai module.

#[derive(Debug, serde::Deserialize)]
struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
    usage: Option<OpenRouterUsage>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenRouterChoice {
    message: OpenRouterMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenRouterMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenRouterToolCall>>,
}

#[derive(Debug, serde::Deserialize)]
struct OpenRouterToolCall {
    id: String,
    function: OpenRouterFunctionCall,
}

#[derive(Debug, serde::Deserialize)]
struct OpenRouterFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, serde::Deserialize)]
struct OpenRouterUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

/// Parse an OpenRouter/OpenAI-format assistant message into our ContentBlock vec.
fn parse_response_message(msg: OpenRouterMessage) -> Result<Vec<ContentBlock>, ProviderError> {
    let mut content = Vec::new();

    if let Some(text) = msg.content {
        if !text.is_empty() {
            content.push(ContentBlock::Text { text });
        }
    }

    if let Some(tool_calls) = msg.tool_calls {
        for tc in tool_calls {
            let input: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or_else(|_| serde_json::json!({}));
            content.push(ContentBlock::ToolUse {
                id: tc.id,
                name: tc.function.name,
                input,
            });
        }
    }

    Ok(content)
}
