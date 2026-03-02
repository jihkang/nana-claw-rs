use async_trait::async_trait;
use serde_json::json;
use std::path::Path;
use tracing::debug;

use super::Tool;
use crate::types::ToolResult;

/// Default maximum number of lines returned by ReadTool.
const DEFAULT_LINE_LIMIT: usize = 2000;

// ═════════════════════════════════════════════════════════════════
//  ReadTool
// ═════════════════════════════════════════════════════════════════

pub struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a file's contents. Returns numbered lines like `cat -n`. \
         Supports optional offset (1-based line number) and limit."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["file_path"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                },
                "offset": {
                    "type": "number",
                    "description": "1-based line number to start reading from"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to return (default 2000)"
                }
            }
        })
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let file_path = match input.get("file_path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return ToolResult {
                    content: "Missing required field: file_path".to_string(),
                    is_error: true,
                };
            }
        };

        let offset = input
            .get("offset")
            .and_then(|v| v.as_u64())
            .map(|n| n.max(1) as usize)
            .unwrap_or(1);

        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_LINE_LIMIT);

        debug!(file_path, offset, limit, "read_file");

        let content = match tokio::fs::read_to_string(file_path).await {
            Ok(c) => c,
            Err(e) => {
                return ToolResult {
                    content: format!("Error reading {file_path}: {e}"),
                    is_error: true,
                };
            }
        };

        let lines: Vec<&str> = content.lines().collect();
        let start = (offset - 1).min(lines.len());
        let end = (start + limit).min(lines.len());

        let mut output = String::new();
        for (idx, line) in lines[start..end].iter().enumerate() {
            let line_num = start + idx + 1;
            // Truncate very long lines to keep output manageable.
            let truncated = if line.len() > 2000 {
                &line[..2000]
            } else {
                line
            };
            output.push_str(&format!("{line_num}\t{truncated}\n"));
        }

        ToolResult {
            content: output,
            is_error: false,
        }
    }
}

// ═════════════════════════════════════════════════════════════════
//  WriteTool
// ═════════════════════════════════════════════════════════════════

pub struct WriteTool;

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file, creating parent directories if needed. \
         Overwrites the file if it already exists."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["file_path", "content"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            }
        })
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let file_path = match input.get("file_path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return ToolResult {
                    content: "Missing required field: file_path".to_string(),
                    is_error: true,
                };
            }
        };

        let content = match input.get("content").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => {
                return ToolResult {
                    content: "Missing required field: content".to_string(),
                    is_error: true,
                };
            }
        };

        debug!(file_path, "write_file");

        // Ensure parent directories exist.
        if let Some(parent) = Path::new(file_path).parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return ToolResult {
                    content: format!("Failed to create parent directories: {e}"),
                    is_error: true,
                };
            }
        }

        match tokio::fs::write(file_path, content).await {
            Ok(()) => {
                let bytes = content.len();
                ToolResult {
                    content: format!("Successfully wrote {bytes} bytes to {file_path}"),
                    is_error: false,
                }
            }
            Err(e) => ToolResult {
                content: format!("Error writing {file_path}: {e}"),
                is_error: true,
            },
        }
    }
}

// ═════════════════════════════════════════════════════════════════
//  EditTool
// ═════════════════════════════════════════════════════════════════

pub struct EditTool;

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Find an exact string in a file and replace it. The old_string must \
         appear exactly once unless replace_all is true."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["file_path", "old_string", "new_string"],
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to edit"
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find in the file"
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text"
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default false)"
                }
            }
        })
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let file_path = match input.get("file_path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => {
                return ToolResult {
                    content: "Missing required field: file_path".to_string(),
                    is_error: true,
                };
            }
        };

        let old_string = match input.get("old_string").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return ToolResult {
                    content: "Missing required field: old_string".to_string(),
                    is_error: true,
                };
            }
        };

        let new_string = match input.get("new_string").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => {
                return ToolResult {
                    content: "Missing required field: new_string".to_string(),
                    is_error: true,
                };
            }
        };

        let replace_all = input
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        debug!(file_path, replace_all, "edit_file");

        // Read the existing content.
        let content = match tokio::fs::read_to_string(file_path).await {
            Ok(c) => c,
            Err(e) => {
                return ToolResult {
                    content: format!("Error reading {file_path}: {e}"),
                    is_error: true,
                };
            }
        };

        // Count occurrences.
        let count = content.matches(old_string).count();

        if count == 0 {
            return ToolResult {
                content: format!(
                    "old_string not found in {file_path}. \
                     Make sure the string matches exactly (including whitespace)."
                ),
                is_error: true,
            };
        }

        if count > 1 && !replace_all {
            return ToolResult {
                content: format!(
                    "old_string found {count} times in {file_path}. \
                     Provide more context to make it unique, or set replace_all to true."
                ),
                is_error: true,
            };
        }

        let new_content = if replace_all {
            content.replace(old_string, new_string)
        } else {
            // Replace only the first occurrence.
            content.replacen(old_string, new_string, 1)
        };

        match tokio::fs::write(file_path, &new_content).await {
            Ok(()) => {
                let msg = if replace_all {
                    format!("Replaced {count} occurrence(s) in {file_path}")
                } else {
                    format!("Successfully edited {file_path}")
                };
                ToolResult {
                    content: msg,
                    is_error: false,
                }
            }
            Err(e) => ToolResult {
                content: format!("Error writing {file_path}: {e}"),
                is_error: true,
            },
        }
    }
}

// ═════════════════════════════════════════════════════════════════
//  GlobTool
// ═════════════════════════════════════════════════════════════════

pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern (e.g. \"**/*.rs\"). \
         Returns matching file paths, one per line."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. \"**/*.ts\", \"src/**/*.rs\")"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (default: cwd)"
                }
            }
        })
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let pattern = match input.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => {
                return ToolResult {
                    content: "Missing required field: pattern".to_string(),
                    is_error: true,
                };
            }
        };

        let base_dir = input
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        debug!(pattern, base_dir, "glob");

        // globwalk is blocking — run on the blocking threadpool.
        let base_dir_owned = base_dir.to_string();
        let pattern_owned = pattern.clone();

        let result = tokio::task::spawn_blocking(move || {
            let walker = match globwalk::GlobWalkerBuilder::from_patterns(&base_dir_owned, &[&pattern_owned])
                .build()
            {
                Ok(w) => w,
                Err(e) => return Err(format!("Invalid glob pattern: {e}")),
            };

            let mut paths: Vec<String> = Vec::new();
            for entry in walker {
                match entry {
                    Ok(e) => {
                        paths.push(e.path().display().to_string());
                    }
                    Err(e) => {
                        // Non-fatal: skip entries we can't read.
                        debug!("glob entry error: {e}");
                    }
                }
            }
            paths.sort();
            Ok(paths)
        })
        .await;

        match result {
            Ok(Ok(paths)) => {
                if paths.is_empty() {
                    ToolResult {
                        content: format!("No files matched pattern: {pattern}"),
                        is_error: false,
                    }
                } else {
                    let count = paths.len();
                    let listing = paths.join("\n");
                    ToolResult {
                        content: format!("{listing}\n\n({count} files)"),
                        is_error: false,
                    }
                }
            }
            Ok(Err(e)) => ToolResult {
                content: e,
                is_error: true,
            },
            Err(e) => ToolResult {
                content: format!("Glob task panicked: {e}"),
                is_error: true,
            },
        }
    }
}
