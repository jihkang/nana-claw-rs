use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;
use tokio::process::Command;
use tracing::debug;

use super::Tool;
use crate::types::ToolResult;

/// Hard-coded environment variables that are always stripped before running
/// a user command, regardless of the caller-provided list.
const ALWAYS_STRIP: &[&str] = &["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"];

/// Default command timeout in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 120;

// ─── BashTool ────────────────────────────────────────────────────

pub struct BashTool {
    /// Union of hard-coded + caller-provided secret env-var names.
    secret_vars: Vec<String>,
}

impl BashTool {
    pub fn new(extra_secret_vars: &[String]) -> Self {
        let mut secret_vars: Vec<String> = ALWAYS_STRIP.iter().map(|s| s.to_string()).collect();
        for v in extra_secret_vars {
            if !secret_vars.contains(v) {
                secret_vars.push(v.clone());
            }
        }
        Self { secret_vars }
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command and return its output. \
         The command runs in /bin/bash -c. Stdout and stderr are combined."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (default 120)"
                },
                "description": {
                    "type": "string",
                    "description": "Human-readable description of the command (ignored by execution)"
                }
            }
        })
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let command = match input.get("command").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => {
                return ToolResult {
                    content: "Missing required field: command".to_string(),
                    is_error: true,
                };
            }
        };

        let timeout_secs = input
            .get("timeout")
            .and_then(|v| v.as_f64())
            .map(|t| t.max(1.0) as u64)
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        debug!(command, timeout_secs, "bash: executing");

        // Build the Command, stripping secret env vars.
        let mut cmd = Command::new("/bin/bash");
        cmd.arg("-c").arg(command);

        for var in &self.secret_vars {
            cmd.env_remove(var);
        }

        // Merge stdout + stderr into a single stream by redirecting stderr to
        // stdout at the shell level is tempting, but it's easier to just
        // capture both separately and concatenate.
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let result = tokio::time::timeout(Duration::from_secs(timeout_secs), cmd.output()).await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                let mut combined = String::new();
                if !stdout.is_empty() {
                    combined.push_str(&stdout);
                }
                if !stderr.is_empty() {
                    if !combined.is_empty() && !combined.ends_with('\n') {
                        combined.push('\n');
                    }
                    combined.push_str(&stderr);
                }

                // Trim trailing newline for cleaner output.
                let content = combined.trim_end_matches('\n').to_string();

                let is_error = !output.status.success();

                ToolResult { content, is_error }
            }
            Ok(Err(e)) => ToolResult {
                content: format!("Failed to spawn bash: {e}"),
                is_error: true,
            },
            Err(_) => ToolResult {
                content: format!(
                    "Command timed out after {timeout_secs} seconds. \
                     Consider increasing the timeout or simplifying the command."
                ),
                is_error: true,
            },
        }
    }
}
