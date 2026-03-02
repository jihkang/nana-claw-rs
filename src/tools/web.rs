use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;
use tracing::debug;

use super::Tool;
use crate::types::ToolResult;

/// Maximum response body size (bytes) before we truncate.
const MAX_BODY_BYTES: usize = 512 * 1024; // 512 KB

/// Maximum characters of processed text we return.
const MAX_TEXT_CHARS: usize = 100_000;

/// HTTP request timeout.
const FETCH_TIMEOUT: Duration = Duration::from_secs(30);

// ═════════════════════════════════════════════════════════════════
//  WebFetchTool
// ═════════════════════════════════════════════════════════════════

pub struct WebFetchTool;

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch the content of a URL and return it as plain text. \
         HTML tags are stripped. Very long responses are truncated."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["url"],
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional prompt describing what to extract (informational only)"
                }
            }
        })
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let url = match input.get("url").and_then(|v| v.as_str()) {
            Some(u) => u,
            None => {
                return ToolResult {
                    content: "Missing required field: url".to_string(),
                    is_error: true,
                };
            }
        };

        debug!(url, "web_fetch");

        let client = match reqwest::Client::builder()
            .timeout(FETCH_TIMEOUT)
            .user_agent("NanoClaw-Agent/0.1")
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                return ToolResult {
                    content: format!("Failed to create HTTP client: {e}"),
                    is_error: true,
                };
            }
        };

        let response = match client.get(url).send().await {
            Ok(r) => r,
            Err(e) => {
                return ToolResult {
                    content: format!("HTTP request failed: {e}"),
                    is_error: true,
                };
            }
        };

        let status = response.status();
        if !status.is_success() {
            return ToolResult {
                content: format!("HTTP {status} for {url}"),
                is_error: true,
            };
        }

        // Read the body, limiting size.
        let bytes = match response.bytes().await {
            Ok(b) => b,
            Err(e) => {
                return ToolResult {
                    content: format!("Failed to read response body: {e}"),
                    is_error: true,
                };
            }
        };

        let body_bytes = if bytes.len() > MAX_BODY_BYTES {
            &bytes[..MAX_BODY_BYTES]
        } else {
            &bytes[..]
        };

        let body_text = String::from_utf8_lossy(body_bytes);

        // Simple HTML-to-text conversion: strip tags.
        let text = strip_html_tags(&body_text);

        // Collapse whitespace runs and trim.
        let cleaned = collapse_whitespace(&text);

        // Truncate if still very long.
        let output = if cleaned.len() > MAX_TEXT_CHARS {
            let truncated = &cleaned[..MAX_TEXT_CHARS];
            format!("{truncated}\n\n... (truncated, {MAX_TEXT_CHARS} char limit)")
        } else {
            cleaned
        };

        ToolResult {
            content: output,
            is_error: false,
        }
    }
}

/// Minimal HTML tag stripper. Removes `<script>`, `<style>` blocks entirely,
/// then strips remaining tags. Good enough for extracting readable text from
/// most web pages.
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut skip_until_close: Option<&str> = None;

    // First pass: remove <script> and <style> blocks entirely.
    let lower = html.to_lowercase();
    let mut out = String::with_capacity(html.len());
    let mut i = 0;
    let bytes = html.as_bytes();

    while i < bytes.len() {
        if let Some(close_tag) = skip_until_close {
            if let Some(pos) = lower[i..].find(close_tag) {
                i += pos + close_tag.len();
                skip_until_close = None;
            } else {
                break; // Unterminated block; skip the rest.
            }
        } else if i + 7 <= lower.len() && &lower[i..i + 7] == "<script" {
            skip_until_close = Some("</script>");
        } else if i + 6 <= lower.len() && &lower[i..i + 6] == "<style" {
            skip_until_close = Some("</style>");
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }

    // Second pass: strip remaining HTML tags.
    let mut in_tag = false;
    for ch in out.chars() {
        match ch {
            '<' => in_tag = true,
            '>' if in_tag => {
                in_tag = false;
                result.push(' '); // Replace tag with space for readability.
            }
            _ if !in_tag => result.push(ch),
            _ => {} // Inside a tag, skip.
        }
    }

    // Decode common HTML entities.
    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

/// Collapse runs of whitespace into single spaces and trim blank lines.
fn collapse_whitespace(text: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut prev_blank = false;

    for line in text.lines() {
        let trimmed: String = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if trimmed.is_empty() {
            if !prev_blank {
                lines.push(String::new());
                prev_blank = true;
            }
        } else {
            lines.push(trimmed);
            prev_blank = false;
        }
    }

    lines.join("\n").trim().to_string()
}

// ═════════════════════════════════════════════════════════════════
//  WebSearchTool
// ═════════════════════════════════════════════════════════════════

pub struct WebSearchTool;

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web for information. \
         Note: this is a placeholder; web search requires an external API."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            }
        })
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("<no query>");

        debug!(query, "web_search (stub)");

        ToolResult {
            content: "Web search is not available in standalone mode. \
                      Use the bash tool to run `curl` for specific URLs, \
                      or use the web_fetch tool to retrieve a known page."
                .to_string(),
            is_error: false,
        }
    }
}
