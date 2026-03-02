use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};

use crate::types::*;
use super::{Provider, ProviderError};

const DEFAULT_BASE_URL: &str = "https://api.openai.com";

// ─── OpenAI Provider ────────────────────────────────────────────

pub struct OpenAiProvider {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl OpenAiProvider {
    pub fn new(api_key: String, base_url: Option<String>, model: String) -> Self {
        Self {
            client: Client::new(),
            api_key,
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            model,
        }
    }

    /// Build a provider with a pre-configured client, base URL, and extra
    /// headers already baked into the client. Used by OpenRouter.
    #[allow(dead_code)]
    pub(crate) fn with_client(
        client: Client,
        api_key: String,
        base_url: String,
        model: String,
    ) -> Self {
        Self {
            client,
            api_key,
            base_url,
            model,
        }
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        system_prompt: Option<&str>,
    ) -> Result<LlmResponse, ProviderError> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        debug!(provider = "openai", model = %self.model, url = %url, "sending chat request");

        let api_messages = convert_messages(messages, system_prompt);
        let api_tools = convert_tools(tools);

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
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let body_text = response.text().await.unwrap_or_default();
            error!(
                provider = "openai",
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

        let api_response: OpenAiResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::ParseError(format!("failed to parse response JSON: {e}")))?;

        let choice = api_response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::ParseError("no choices in response".to_string()))?;

        debug!(
            provider = "openai",
            finish_reason = ?choice.finish_reason,
            usage_prompt = api_response.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
            usage_completion = api_response.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
            "received response"
        );

        let content = parse_assistant_message(choice.message)?;

        let stop_reason = match choice.finish_reason.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("tool_calls") => StopReason::ToolUse,
            Some("length") => StopReason::MaxTokens,
            Some("content_filter") => StopReason::StopSequence,
            _ => StopReason::EndTurn,
        };

        let usage = api_response.usage.map(|u| Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
        }).unwrap_or_default();

        Ok(LlmResponse {
            content,
            stop_reason,
            usage,
        })
    }

    fn name(&self) -> &str {
        "openai"
    }
}

// ─── OpenAI API Types ───────────────────────────────────────────

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAiFunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct OpenAiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAiFunctionDef,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiFunctionDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// ─── Conversion Helpers (pub(crate) so openrouter can reuse) ────

pub(crate) fn convert_messages(messages: &[Message], system_prompt: Option<&str>) -> Vec<OpenAiMessage> {
    let mut result = Vec::new();

    // OpenAI uses a "system" role message at the start.
    if let Some(system) = system_prompt {
        result.push(OpenAiMessage {
            role: "system".to_string(),
            content: Some(serde_json::json!(system)),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    for msg in messages {
        match msg {
            Message::System { content } => {
                // Include any explicit system messages from the conversation too.
                result.push(OpenAiMessage {
                    role: "system".to_string(),
                    content: Some(serde_json::json!(content)),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            Message::User { content } => {
                result.push(OpenAiMessage {
                    role: "user".to_string(),
                    content: Some(serde_json::json!(content)),
                    tool_calls: None,
                    tool_call_id: None,
                });
            }
            Message::Assistant { content } => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<OpenAiToolCall> = Vec::new();

                for block in content {
                    match block {
                        ContentBlock::Text { text } => {
                            text_parts.push(text.clone());
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            tool_calls.push(OpenAiToolCall {
                                id: id.clone(),
                                call_type: "function".to_string(),
                                function: OpenAiFunctionCall {
                                    name: name.clone(),
                                    arguments: serde_json::to_string(input)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                },
                            });
                        }
                    }
                }

                let text_content = if text_parts.is_empty() {
                    None
                } else {
                    Some(serde_json::json!(text_parts.join("\n")))
                };

                let tool_calls_opt = if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                };

                result.push(OpenAiMessage {
                    role: "assistant".to_string(),
                    content: text_content,
                    tool_calls: tool_calls_opt,
                    tool_call_id: None,
                });
            }
            Message::ToolResult {
                tool_use_id,
                content,
                is_error: _,
            } => {
                result.push(OpenAiMessage {
                    role: "tool".to_string(),
                    content: Some(serde_json::json!(content)),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                });
            }
        }
    }

    result
}

pub(crate) fn convert_tools(tools: &[ToolDefinition]) -> Vec<OpenAiTool> {
    tools
        .iter()
        .map(|tool| OpenAiTool {
            tool_type: "function".to_string(),
            function: OpenAiFunctionDef {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.input_schema.clone(),
            },
        })
        .collect()
}

/// Parse an OpenAI assistant response message into our ContentBlock vec.
fn parse_assistant_message(msg: OpenAiResponseMessage) -> Result<Vec<ContentBlock>, ProviderError> {
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
