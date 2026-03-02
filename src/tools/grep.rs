use async_trait::async_trait;
use regex::Regex;
use serde_json::json;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::debug;

use super::Tool;
use crate::types::ToolResult;

/// Maximum number of result lines before we truncate output.
const MAX_RESULT_LINES: usize = 5000;

// ─── GrepTool ────────────────────────────────────────────────────

pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents using regex patterns. Supports glob filtering, \
         context lines, and multiple output modes."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (default: cwd)"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g. \"*.rs\", \"*.{ts,tsx}\")"
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output mode (default: files_with_matches)"
                },
                "context": {
                    "type": "number",
                    "description": "Number of context lines before and after each match"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)"
                }
            }
        })
    }

    async fn execute(&self, input: serde_json::Value) -> ToolResult {
        let pattern_str = match input.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => {
                return ToolResult {
                    content: "Missing required field: pattern".to_string(),
                    is_error: true,
                };
            }
        };

        let search_path = input
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".")
            .to_string();

        let glob_filter = input
            .get("glob")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let output_mode = input
            .get("output_mode")
            .and_then(|v| v.as_str())
            .unwrap_or("files_with_matches")
            .to_string();

        let context_lines = input
            .get("context")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let case_insensitive = input
            .get("case_insensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        debug!(
            pattern = %pattern_str,
            path = %search_path,
            output_mode = %output_mode,
            "grep"
        );

        // Build the regex.
        let re_pattern = if case_insensitive {
            format!("(?i){pattern_str}")
        } else {
            pattern_str.clone()
        };

        let re = match Regex::new(&re_pattern) {
            Ok(r) => r,
            Err(e) => {
                return ToolResult {
                    content: format!("Invalid regex pattern: {e}"),
                    is_error: true,
                };
            }
        };

        // Collect files to search.
        let search_path_buf = PathBuf::from(&search_path);

        // Run the blocking search on the blocking threadpool.
        let result = tokio::task::spawn_blocking(move || {
            let files = collect_files(&search_path_buf, glob_filter.as_deref())?;

            match output_mode.as_str() {
                "files_with_matches" => search_files_with_matches(&re, &files),
                "count" => search_count(&re, &files),
                "content" | _ => search_content(&re, &files, context_lines),
            }
        })
        .await;

        match result {
            Ok(Ok(output)) => {
                if output.is_empty() {
                    ToolResult {
                        content: "No matches found.".to_string(),
                        is_error: false,
                    }
                } else {
                    ToolResult {
                        content: output,
                        is_error: false,
                    }
                }
            }
            Ok(Err(e)) => ToolResult {
                content: e,
                is_error: true,
            },
            Err(e) => ToolResult {
                content: format!("Grep task panicked: {e}"),
                is_error: true,
            },
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────

/// Collect all files under `path`, optionally filtered by a glob pattern.
fn collect_files(path: &Path, glob_filter: Option<&str>) -> Result<Vec<PathBuf>, String> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }

    if !path.is_dir() {
        return Err(format!("Path does not exist: {}", path.display()));
    }

    // If we have a glob filter, use globwalk.
    if let Some(pattern) = glob_filter {
        let walker = globwalk::GlobWalkerBuilder::from_patterns(path, &[pattern])
            .build()
            .map_err(|e| format!("Invalid glob filter: {e}"))?;

        let mut files = Vec::new();
        for entry in walker.into_iter().filter_map(Result::ok) {
            if entry.path().is_file() {
                files.push(entry.path().to_path_buf());
            }
        }
        files.sort();
        return Ok(files);
    }

    // No glob filter: walk the entire directory, skipping hidden dirs and common
    // noise directories.
    let mut files = Vec::new();
    walk_dir(path, &mut files);
    files.sort();
    Ok(files)
}

/// Simple recursive directory walk, skipping hidden and known-noisy dirs.
fn walk_dir(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip hidden entries and common noise directories.
        if name_str.starts_with('.') {
            continue;
        }
        if matches!(
            name_str.as_ref(),
            "node_modules" | "target" | "dist" | "build" | "__pycache__" | ".git"
        ) {
            continue;
        }

        if path.is_dir() {
            walk_dir(&path, out);
        } else if path.is_file() {
            // Skip binary-looking files by extension.
            if !is_likely_binary(&path) {
                out.push(path);
            }
        }
    }
}

/// Heuristic: skip files that are likely binary based on extension.
fn is_likely_binary(path: &Path) -> bool {
    let ext = match path.extension().and_then(|e| e.to_str()) {
        Some(e) => e.to_lowercase(),
        None => return false,
    };

    matches!(
        ext.as_str(),
        "png" | "jpg" | "jpeg" | "gif" | "bmp" | "ico" | "svg"
            | "woff" | "woff2" | "ttf" | "eot"
            | "zip" | "tar" | "gz" | "bz2" | "xz" | "7z"
            | "exe" | "dll" | "so" | "dylib"
            | "o" | "a" | "lib"
            | "pdf" | "doc" | "docx"
            | "mp3" | "mp4" | "avi" | "mov" | "mkv"
            | "wasm" | "pyc" | "class"
    )
}

/// Mode: files_with_matches — just list file paths that contain a match.
fn search_files_with_matches(re: &Regex, files: &[PathBuf]) -> Result<String, String> {
    let mut matched_files = Vec::new();

    for file in files {
        let content = match fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue, // Skip unreadable files.
        };

        if re.is_match(&content) {
            matched_files.push(file.display().to_string());
        }

        if matched_files.len() >= MAX_RESULT_LINES {
            matched_files.push("... (truncated)".to_string());
            break;
        }
    }

    Ok(matched_files.join("\n"))
}

/// Mode: count — show match counts per file.
fn search_count(re: &Regex, files: &[PathBuf]) -> Result<String, String> {
    let mut counts: Vec<String> = Vec::new();

    for file in files {
        let content = match fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let count = re.find_iter(&content).count();
        if count > 0 {
            counts.push(format!("{}:{count}", file.display()));
        }

        if counts.len() >= MAX_RESULT_LINES {
            counts.push("... (truncated)".to_string());
            break;
        }
    }

    Ok(counts.join("\n"))
}

/// Mode: content — show matching lines with optional context.
fn search_content(
    re: &Regex,
    files: &[PathBuf],
    context: usize,
) -> Result<String, String> {
    let mut output_lines: Vec<String> = Vec::new();

    for file in files {
        let content = match fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let lines: Vec<&str> = content.lines().collect();
        let file_display = file.display().to_string();

        // Find all matching line indices.
        let mut match_indices: Vec<usize> = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if re.is_match(line) {
                match_indices.push(i);
            }
        }

        if match_indices.is_empty() {
            continue;
        }

        // Build a set of line indices to display (matches + context).
        let mut display_set = Vec::new();
        for &idx in &match_indices {
            let start = idx.saturating_sub(context);
            let end = (idx + context + 1).min(lines.len());
            for i in start..end {
                display_set.push(i);
            }
        }
        display_set.sort();
        display_set.dedup();

        // Build the match-set for coloring context vs match lines.
        let match_set: HashSet<usize> =
            match_indices.iter().copied().collect();

        let mut prev_idx: Option<usize> = None;
        for &idx in &display_set {
            // Insert a separator if there's a gap.
            if let Some(prev) = prev_idx {
                if idx > prev + 1 {
                    output_lines.push("--".to_string());
                }
            }

            let line_num = idx + 1;
            let separator = if match_set.contains(&idx) { ":" } else { "-" };
            output_lines.push(format!(
                "{file_display}{separator}{line_num}{separator}{}",
                lines[idx]
            ));
            prev_idx = Some(idx);
        }

        if output_lines.len() >= MAX_RESULT_LINES {
            output_lines.push("... (truncated)".to_string());
            break;
        }
    }

    Ok(output_lines.join("\n"))
}
