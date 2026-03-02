use futures::future::join_all;
use tracing::info;

use crate::provider::{Provider, ProviderError};
use crate::session::Session;
use crate::tools::ToolRegistry;
use crate::types::*;

// ─── Errors ─────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),

    #[error("Max turns exceeded ({0})")]
    MaxTurnsExceeded(usize),
}

// ─── Result ─────────────────────────────────────────────────────

pub struct AgentResult {
    pub final_text: Option<String>,
    #[allow(dead_code)]
    pub messages: Vec<Message>,
    pub total_usage: Usage,
}

// ─── Agent ──────────────────────────────────────────────────────

pub struct Agent {
    provider: Box<dyn Provider>,
    tools: ToolRegistry,
    system_prompt: Option<String>,
    max_turns: usize,
}

impl Agent {
    pub fn new(
        provider: Box<dyn Provider>,
        tools: ToolRegistry,
        system_prompt: Option<String>,
    ) -> Self {
        Self {
            provider,
            tools,
            system_prompt,
            max_turns: 100,
        }
    }

    /// Override the default max-turns safety limit.
    #[allow(dead_code)]
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Run the agent loop for a single query.
    ///
    /// 1. Add user message to history
    /// 2. Call provider.chat(messages, tools)
    /// 3. If response has tool_use blocks:
    ///    - Execute each tool call via ToolRegistry (in parallel)
    ///    - Add assistant message (with tool_use) and tool results to history
    ///    - Go to step 2
    /// 4. If response is end_turn (text only):
    ///    - Return the final text
    /// 5. If max_turns exceeded, return error
    pub async fn run(
        &self,
        prompt: &str,
        session: &mut Session,
    ) -> Result<AgentResult, AgentError> {
        // Step 1: Add user message to history
        session.push(Message::User {
            content: prompt.to_string(),
        });

        let tool_defs = self.tools.definitions();
        let mut total_usage = Usage::default();

        for turn in 1..=self.max_turns {
            // Step 2: Call the provider
            info!(
                turn = turn,
                provider = self.provider.name(),
                history_len = session.messages.len(),
                "starting turn"
            );

            let response = self.provider.chat(
                &session.messages,
                &tool_defs,
                self.system_prompt.as_deref(),
            ).await?;

            // Accumulate usage
            total_usage.input_tokens += response.usage.input_tokens;
            total_usage.output_tokens += response.usage.output_tokens;

            info!(
                turn = turn,
                stop_reason = ?response.stop_reason,
                input_tokens = response.usage.input_tokens,
                output_tokens = response.usage.output_tokens,
                content_blocks = response.content.len(),
                "received response"
            );

            // Extract tool calls from the response
            let tool_calls: Vec<ToolCall> = response
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::ToolUse { id, name, input } => Some(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    }),
                    _ => None,
                })
                .collect();

            // Step 3: If there are tool calls, execute them
            if !tool_calls.is_empty() {
                info!(
                    turn = turn,
                    tool_count = tool_calls.len(),
                    tools = ?tool_calls.iter().map(|tc| tc.name.as_str()).collect::<Vec<_>>(),
                    "executing tool calls"
                );

                // Add the assistant message (contains both text and tool_use blocks)
                session.push(Message::Assistant {
                    content: response.content,
                });

                // Execute all tool calls in parallel
                let futures: Vec<_> = tool_calls
                    .iter()
                    .map(|tc| {
                        let name = tc.name.clone();
                        let input = tc.input.clone();
                        async move {
                            let result = self.tools.execute(&name, input).await;
                            (tc.id.clone(), result)
                        }
                    })
                    .collect();

                let results = join_all(futures).await;

                // Add each tool result as a separate message
                for (tool_use_id, result) in results {
                    info!(
                        tool_use_id = %tool_use_id,
                        is_error = result.is_error,
                        output_len = result.content.len(),
                        "tool result"
                    );

                    session.push(Message::ToolResult {
                        tool_use_id,
                        content: result.content,
                        is_error: result.is_error,
                    });
                }

                // Continue the loop (go back to step 2)
                continue;
            }

            // Step 4: Handle stop reasons without tool calls

            // If max_tokens, auto-continue by injecting a "Please continue" message
            if response.stop_reason == StopReason::MaxTokens {
                info!(turn = turn, "max_tokens reached, auto-continuing");

                session.push(Message::Assistant {
                    content: response.content,
                });
                session.push(Message::User {
                    content: "Please continue.".to_string(),
                });

                continue;
            }

            // EndTurn or StopSequence — extract final text and return
            let final_text = extract_text(&response.content);

            // Add the final assistant message to history
            session.push(Message::Assistant {
                content: response.content,
            });

            info!(
                turn = turn,
                total_input_tokens = total_usage.input_tokens,
                total_output_tokens = total_usage.output_tokens,
                has_final_text = final_text.is_some(),
                "agent loop complete"
            );

            return Ok(AgentResult {
                final_text,
                messages: session.messages.clone(),
                total_usage,
            });
        }

        // Step 5: Max turns exceeded
        Err(AgentError::MaxTurnsExceeded(self.max_turns))
    }
}

/// Extract concatenated text from content blocks.
fn extract_text(content: &[ContentBlock]) -> Option<String> {
    let texts: Vec<&str> = content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();

    if texts.is_empty() {
        None
    } else {
        Some(texts.join(""))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_from_mixed_content() {
        let content = vec![
            ContentBlock::Text {
                text: "Hello ".to_string(),
            },
            ContentBlock::ToolUse {
                id: "tool_1".to_string(),
                name: "bash".to_string(),
                input: serde_json::json!({"command": "ls"}),
            },
            ContentBlock::Text {
                text: "world".to_string(),
            },
        ];

        assert_eq!(extract_text(&content), Some("Hello world".to_string()));
    }

    #[test]
    fn test_extract_text_empty() {
        let content: Vec<ContentBlock> = vec![];
        assert_eq!(extract_text(&content), None);
    }

    #[test]
    fn test_extract_text_tool_use_only() {
        let content = vec![ContentBlock::ToolUse {
            id: "tool_1".to_string(),
            name: "bash".to_string(),
            input: serde_json::json!({"command": "ls"}),
        }];
        assert_eq!(extract_text(&content), None);
    }
}
