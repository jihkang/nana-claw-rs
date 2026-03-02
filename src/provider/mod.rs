pub mod anthropic;
pub mod openai;
pub mod openrouter;

use async_trait::async_trait;
use crate::types::*;

// ─── Provider Trait ─────────────────────────────────────────────

#[async_trait]
pub trait Provider: Send + Sync {
    /// Send a chat request with messages, tools, and an optional system prompt.
    /// Returns the LLM response or a provider error.
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        system_prompt: Option<&str>,
    ) -> Result<LlmResponse, ProviderError>;

    /// Human-readable name of this provider (e.g. "anthropic", "openai").
    fn name(&self) -> &str;
}

// ─── Provider Errors ────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API returned error status {status}: {body}")]
    ApiError { status: u16, body: String },

    #[error("Rate limited (HTTP 429): {message}")]
    RateLimited { message: String },

    #[error("Authentication failed (HTTP 401/403): {message}")]
    AuthError { message: String },

    #[error("Failed to parse response: {0}")]
    ParseError(String),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),
}

// ─── Factory ────────────────────────────────────────────────────

/// Create a boxed Provider from a ProviderConfig.
pub fn create_provider(config: ProviderConfig) -> Box<dyn Provider> {
    match config {
        ProviderConfig::Anthropic {
            api_key,
            base_url,
            model,
        } => Box::new(anthropic::AnthropicProvider::new(api_key, base_url, model)),

        ProviderConfig::OpenAi {
            api_key,
            base_url,
            model,
        } => Box::new(openai::OpenAiProvider::new(api_key, base_url, model)),

        ProviderConfig::OpenRouter { api_key, model } => {
            Box::new(openrouter::OpenRouterProvider::new(api_key, model))
        }
    }
}
