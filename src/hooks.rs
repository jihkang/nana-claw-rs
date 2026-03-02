use std::fs;
use std::io::Write;
use std::path::Path;

use chrono::Local;
use tracing::{debug, warn};

use crate::types::*;

// ─── Transcript Archiving ───────────────────────────────────────

/// Archive conversation transcript to `/workspace/group/conversations/`.
/// Called when session is compacted (equivalent to the TypeScript PreCompact hook).
///
/// Reads the session summary from `sessions-index.json` to generate a meaningful
/// filename. Falls back to a timestamp-based name if unavailable.
pub fn archive_transcript(
    messages: &[Message],
    session_id: &str,
    assistant_name: Option<&str>,
) -> Result<(), std::io::Error> {
    if messages.is_empty() {
        debug!("no messages to archive");
        return Ok(());
    }

    // Parse messages into a simplified user/assistant form
    let parsed = parse_messages(messages);
    if parsed.is_empty() {
        debug!("no user/assistant messages to archive");
        return Ok(());
    }

    // Create conversations directory
    let conversations_dir = Path::new("/workspace/group/conversations");
    fs::create_dir_all(conversations_dir)?;

    // Try to get a summary from sessions-index.json
    let summary = get_session_summary(session_id);
    let name = match &summary {
        Some(s) => sanitize_filename(s),
        None => generate_fallback_name(),
    };

    let date = Local::now().format("%Y-%m-%d").to_string();
    let filename = format!("{}-{}.md", date, name);
    let file_path = conversations_dir.join(&filename);

    debug!(
        session_id = %session_id,
        path = %file_path.display(),
        message_count = parsed.len(),
        "archiving transcript"
    );

    // Format and write
    let markdown = format_transcript_markdown(&parsed, summary.as_deref(), assistant_name);
    let mut file = fs::File::create(&file_path)?;
    file.write_all(markdown.as_bytes())?;
    file.flush()?;

    Ok(())
}

// ─── Session Summary ────────────────────────────────────────────

/// Read the summary for a session from `/workspace/group/sessions-index.json`.
/// Returns `None` if the file doesn't exist, can't be parsed, or the session
/// isn't found.
pub fn get_session_summary(session_id: &str) -> Option<String> {
    let index_path = Path::new("/workspace/group/sessions-index.json");

    if !index_path.exists() {
        debug!(path = %index_path.display(), "sessions index not found");
        return None;
    }

    let contents = match fs::read_to_string(index_path) {
        Ok(c) => c,
        Err(e) => {
            warn!(error = %e, "failed to read sessions index");
            return None;
        }
    };

    let index: SessionsIndex = match serde_json::from_str(&contents) {
        Ok(i) => i,
        Err(e) => {
            warn!(error = %e, "failed to parse sessions index");
            return None;
        }
    };

    index
        .entries
        .iter()
        .find(|entry| entry.session_id == session_id)
        .map(|entry| entry.summary.clone())
        .filter(|s| !s.is_empty())
}

// ─── Filename Helpers ───────────────────────────────────────────

/// Sanitize a summary string into a safe filename component.
/// - Lowercased
/// - Non-alphanumeric characters replaced with hyphens
/// - Leading/trailing hyphens trimmed
/// - Maximum 50 characters
pub fn sanitize_filename(summary: &str) -> String {
    let sanitized: String = summary
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();

    // Collapse consecutive hyphens
    let mut result = String::with_capacity(sanitized.len());
    let mut prev_hyphen = false;
    for c in sanitized.chars() {
        if c == '-' {
            if !prev_hyphen {
                result.push(c);
            }
            prev_hyphen = true;
        } else {
            result.push(c);
            prev_hyphen = false;
        }
    }

    // Trim leading and trailing hyphens
    let trimmed = result.trim_matches('-');

    // Truncate to 50 chars (on a char boundary, which is safe for ASCII)
    if trimmed.len() > 50 {
        // Find the last hyphen within the first 50 chars for a cleaner cut
        let slice = &trimmed[..50];
        match slice.rfind('-') {
            Some(pos) if pos > 30 => slice[..pos].to_string(),
            _ => slice.to_string(),
        }
    } else {
        trimmed.to_string()
    }
}

/// Generate a fallback filename when no summary is available.
/// Format: `conversation-HHMM`
pub fn generate_fallback_name() -> String {
    let now = Local::now();
    format!("conversation-{}", now.format("%H%M"))
}

// ─── Message Parsing ────────────────────────────────────────────

/// A simplified message with just role and text content,
/// used for rendering the markdown transcript.
struct ParsedMessage {
    role: ParsedRole,
    content: String,
}

enum ParsedRole {
    User,
    Assistant,
}

/// Extract user and assistant text content from the raw Message enum.
/// Tool messages and system messages are skipped.
fn parse_messages(messages: &[Message]) -> Vec<ParsedMessage> {
    let mut parsed = Vec::new();

    for msg in messages {
        match msg {
            Message::User { content } => {
                if !content.is_empty() {
                    parsed.push(ParsedMessage {
                        role: ParsedRole::User,
                        content: content.clone(),
                    });
                }
            }
            Message::Assistant { content } => {
                // Extract only text blocks, skip tool_use blocks
                let text_parts: Vec<&str> = content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        ContentBlock::ToolUse { .. } => None,
                    })
                    .collect();

                let text = text_parts.join("");
                if !text.is_empty() {
                    parsed.push(ParsedMessage {
                        role: ParsedRole::Assistant,
                        content: text,
                    });
                }
            }
            // Skip system and tool result messages
            _ => {}
        }
    }

    parsed
}

// ─── Markdown Formatting ────────────────────────────────────────

const MAX_MESSAGE_LENGTH: usize = 2000;

/// Format a conversation transcript as a markdown document.
fn format_transcript_markdown(
    messages: &[ParsedMessage],
    title: Option<&str>,
    assistant_name: Option<&str>,
) -> String {
    let now = Local::now();
    let formatted_date = now.format("%b %-d, %-I:%M %p").to_string();

    let mut lines: Vec<String> = Vec::new();

    // Header
    lines.push(format!("# {}", title.unwrap_or("Conversation")));
    lines.push(String::new());
    lines.push(format!("Archived: {}", formatted_date));
    lines.push(String::new());
    lines.push("---".to_string());
    lines.push(String::new());

    // Messages
    for msg in messages {
        let sender = match msg.role {
            ParsedRole::User => "User",
            ParsedRole::Assistant => assistant_name.unwrap_or("Assistant"),
        };

        let content = if msg.content.len() > MAX_MESSAGE_LENGTH {
            let mut truncated = msg.content[..MAX_MESSAGE_LENGTH].to_string();
            truncated.push_str("...");
            truncated
        } else {
            msg.content.clone()
        };

        lines.push(format!("**{}**: {}", sender, content));
        lines.push(String::new());
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_filename_basic() {
        assert_eq!(
            sanitize_filename("Hello World!"),
            "hello-world"
        );
    }

    #[test]
    fn test_sanitize_filename_special_chars() {
        assert_eq!(
            sanitize_filename("Fix bug #123: crash on startup"),
            "fix-bug-123-crash-on-startup"
        );
    }

    #[test]
    fn test_sanitize_filename_leading_trailing() {
        assert_eq!(
            sanitize_filename("--hello--world--"),
            "hello-world"
        );
    }

    #[test]
    fn test_sanitize_filename_truncation() {
        let long = "a".repeat(100);
        let result = sanitize_filename(&long);
        assert!(result.len() <= 50);
    }

    #[test]
    fn test_sanitize_filename_consecutive_hyphens() {
        assert_eq!(
            sanitize_filename("hello   world"),
            "hello-world"
        );
    }

    #[test]
    fn test_generate_fallback_name() {
        let name = generate_fallback_name();
        assert!(name.starts_with("conversation-"));
        // Should be conversation-HHMM (4 digits)
        assert_eq!(name.len(), "conversation-".len() + 4);
    }

    #[test]
    fn test_parse_messages_filters_correctly() {
        let messages = vec![
            Message::System {
                content: "You are helpful.".to_string(),
            },
            Message::User {
                content: "Hello".to_string(),
            },
            Message::Assistant {
                content: vec![
                    ContentBlock::Text {
                        text: "Let me help.".to_string(),
                    },
                    ContentBlock::ToolUse {
                        id: "tool_1".to_string(),
                        name: "bash".to_string(),
                        input: serde_json::json!({"command": "ls"}),
                    },
                ],
            },
            Message::ToolResult {
                tool_use_id: "tool_1".to_string(),
                content: "file.txt".to_string(),
                is_error: false,
            },
            Message::Assistant {
                content: vec![ContentBlock::Text {
                    text: "Done!".to_string(),
                }],
            },
        ];

        let parsed = parse_messages(&messages);
        assert_eq!(parsed.len(), 3); // user, assistant text, assistant text
        assert!(matches!(parsed[0].role, ParsedRole::User));
        assert_eq!(parsed[0].content, "Hello");
        assert!(matches!(parsed[1].role, ParsedRole::Assistant));
        assert_eq!(parsed[1].content, "Let me help.");
        assert!(matches!(parsed[2].role, ParsedRole::Assistant));
        assert_eq!(parsed[2].content, "Done!");
    }

    #[test]
    fn test_format_transcript_markdown_structure() {
        let messages = vec![
            ParsedMessage {
                role: ParsedRole::User,
                content: "Hello".to_string(),
            },
            ParsedMessage {
                role: ParsedRole::Assistant,
                content: "Hi there!".to_string(),
            },
        ];

        let md = format_transcript_markdown(&messages, Some("Test Chat"), Some("Bot"));
        assert!(md.starts_with("# Test Chat\n"));
        assert!(md.contains("Archived:"));
        assert!(md.contains("---"));
        assert!(md.contains("**User**: Hello"));
        assert!(md.contains("**Bot**: Hi there!"));
    }

    #[test]
    fn test_format_transcript_markdown_truncates_long_messages() {
        let long_content = "x".repeat(3000);
        let messages = vec![ParsedMessage {
            role: ParsedRole::User,
            content: long_content,
        }];

        let md = format_transcript_markdown(&messages, None, None);
        // Should contain truncated content + "..."
        assert!(md.contains("..."));
        // The message line should not contain 3000 x's
        for line in md.lines() {
            if line.starts_with("**User**:") {
                assert!(line.len() < 2200); // 2000 chars + prefix + "..."
            }
        }
    }

    #[test]
    fn test_format_transcript_markdown_defaults() {
        let messages = vec![
            ParsedMessage {
                role: ParsedRole::User,
                content: "Hello".to_string(),
            },
            ParsedMessage {
                role: ParsedRole::Assistant,
                content: "Hi".to_string(),
            },
        ];

        let md = format_transcript_markdown(&messages, None, None);
        assert!(md.starts_with("# Conversation\n"));
        assert!(md.contains("**Assistant**: Hi"));
    }
}
