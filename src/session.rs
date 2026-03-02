use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use tracing::{debug, warn};
use uuid::Uuid;

use crate::types::*;

// ─── Errors ─────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Session not found: {0}")]
    NotFound(String),
}

// ─── Session ────────────────────────────────────────────────────

pub struct Session {
    pub id: String,
    pub messages: Vec<Message>,
    #[allow(dead_code)]
    pub group_folder: String,
    sessions_dir: PathBuf,
}

impl Session {
    /// Create a new session with a generated UUID.
    /// The sessions directory is at `/workspace/group/.sessions/`.
    pub fn new(group_folder: &str) -> Self {
        let id = Uuid::new_v4().to_string();
        let sessions_dir = PathBuf::from("/workspace/group/.sessions");

        debug!(
            session_id = %id,
            group_folder = %group_folder,
            sessions_dir = %sessions_dir.display(),
            "created new session"
        );

        Self {
            id,
            messages: Vec::new(),
            group_folder: group_folder.to_string(),
            sessions_dir,
        }
    }

    /// Load an existing session from disk.
    /// Session files are stored in JSONL format (one JSON-serialized Message per line).
    pub fn load(session_id: &str, group_folder: &str) -> Result<Self, SessionError> {
        let sessions_dir = PathBuf::from("/workspace/group/.sessions");
        let file_path = sessions_dir.join(format!("{}.jsonl", session_id));

        if !file_path.exists() {
            return Err(SessionError::NotFound(session_id.to_string()));
        }

        debug!(
            session_id = %session_id,
            path = %file_path.display(),
            "loading session from disk"
        );

        let file = fs::File::open(&file_path)?;
        let reader = BufReader::new(file);
        let mut messages = Vec::new();

        for (line_number, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            let trimmed = line.trim();

            // Skip empty lines
            if trimmed.is_empty() {
                continue;
            }

            match serde_json::from_str::<Message>(trimmed) {
                Ok(msg) => messages.push(msg),
                Err(e) => {
                    warn!(
                        session_id = %session_id,
                        line = line_number + 1,
                        error = %e,
                        "skipping corrupt session entry"
                    );
                    // Be lenient: skip corrupt lines rather than failing entirely
                }
            }
        }

        debug!(
            session_id = %session_id,
            message_count = messages.len(),
            "session loaded"
        );

        Ok(Self {
            id: session_id.to_string(),
            messages,
            group_folder: group_folder.to_string(),
            sessions_dir,
        })
    }

    /// Save current session to disk in JSONL format (one message per line).
    pub fn save(&self) -> Result<(), SessionError> {
        // Ensure the sessions directory exists
        fs::create_dir_all(&self.sessions_dir)?;

        let file_path = self.sessions_dir.join(format!("{}.jsonl", self.id));

        debug!(
            session_id = %self.id,
            path = %file_path.display(),
            message_count = self.messages.len(),
            "saving session to disk"
        );

        let mut file = fs::File::create(&file_path)?;

        for message in &self.messages {
            let json = serde_json::to_string(message)?;
            writeln!(file, "{}", json)?;
        }

        file.flush()?;

        Ok(())
    }

    /// Add a message to the session.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Get total approximate token count.
    /// Uses a rough heuristic: 4 characters per token.
    pub fn estimated_tokens(&self) -> usize {
        let total_chars: usize = self
            .messages
            .iter()
            .map(|msg| message_char_count(msg))
            .sum();

        total_chars / 4
    }

    /// Compact the session: keep system prompt + last N messages.
    /// Returns the removed messages for archiving.
    pub fn compact(&mut self, keep_last: usize) -> Vec<Message> {
        // Separate system messages from the rest
        let (system_msgs, non_system): (Vec<Message>, Vec<Message>) = self
            .messages
            .drain(..)
            .partition(|msg| matches!(msg, Message::System { .. }));

        let total_non_system = non_system.len();

        if total_non_system <= keep_last {
            // Nothing to compact — restore everything
            self.messages = system_msgs;
            self.messages.extend(non_system);
            return Vec::new();
        }

        let split_at = total_non_system - keep_last;
        let (removed, kept) = non_system.into_iter().enumerate().fold(
            (Vec::new(), Vec::new()),
            |(mut removed, mut kept), (i, msg)| {
                if i < split_at {
                    removed.push(msg);
                } else {
                    kept.push(msg);
                }
                (removed, kept)
            },
        );

        debug!(
            session_id = %self.id,
            removed_count = removed.len(),
            kept_count = kept.len(),
            system_count = system_msgs.len(),
            "session compacted"
        );

        // Rebuild: system messages first, then kept messages
        self.messages = system_msgs;
        self.messages.extend(kept);

        removed
    }

    /// Get the path to this session's file on disk.
    #[allow(dead_code)]
    pub fn file_path(&self) -> PathBuf {
        self.sessions_dir.join(format!("{}.jsonl", self.id))
    }
}

/// Estimate the character count of a message for token estimation.
fn message_char_count(message: &Message) -> usize {
    match message {
        Message::System { content } => content.len(),
        Message::User { content } => content.len(),
        Message::Assistant { content } => content
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text } => text.len(),
                ContentBlock::ToolUse { name, input, .. } => {
                    name.len() + input.to_string().len()
                }
            })
            .sum(),
        Message::ToolResult { content, .. } => content.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_session_has_uuid() {
        let session = Session::new("test-group");
        assert!(!session.id.is_empty());
        assert_eq!(session.group_folder, "test-group");
        assert!(session.messages.is_empty());
        // Verify it's a valid UUID format (8-4-4-4-12)
        assert_eq!(session.id.len(), 36);
        assert_eq!(session.id.chars().filter(|c| *c == '-').count(), 4);
    }

    #[test]
    fn test_push_messages() {
        let mut session = Session::new("test-group");
        session.push(Message::User {
            content: "Hello".to_string(),
        });
        session.push(Message::Assistant {
            content: vec![ContentBlock::Text {
                text: "Hi there!".to_string(),
            }],
        });
        assert_eq!(session.messages.len(), 2);
    }

    #[test]
    fn test_estimated_tokens() {
        let mut session = Session::new("test-group");
        // 20 chars => ~5 tokens
        session.push(Message::User {
            content: "12345678901234567890".to_string(),
        });
        assert_eq!(session.estimated_tokens(), 5);
    }

    #[test]
    fn test_compact_keeps_last_n() {
        let mut session = Session::new("test-group");

        // Add a system message
        session.push(Message::System {
            content: "You are helpful.".to_string(),
        });

        // Add 5 user/assistant message pairs
        for i in 0..5 {
            session.push(Message::User {
                content: format!("Message {}", i),
            });
            session.push(Message::Assistant {
                content: vec![ContentBlock::Text {
                    text: format!("Reply {}", i),
                }],
            });
        }

        // Total: 1 system + 10 non-system = 11 messages
        assert_eq!(session.messages.len(), 11);

        // Compact to keep last 4 non-system messages
        let removed = session.compact(4);

        // Should have removed 6 non-system messages
        assert_eq!(removed.len(), 6);
        // Should have kept 1 system + 4 non-system = 5 messages
        assert_eq!(session.messages.len(), 5);
        // First message should be the system prompt
        assert!(matches!(session.messages[0], Message::System { .. }));
    }

    #[test]
    fn test_compact_nothing_to_remove() {
        let mut session = Session::new("test-group");

        session.push(Message::User {
            content: "Hello".to_string(),
        });
        session.push(Message::Assistant {
            content: vec![ContentBlock::Text {
                text: "Hi".to_string(),
            }],
        });

        // Ask to keep more than we have
        let removed = session.compact(10);
        assert!(removed.is_empty());
        assert_eq!(session.messages.len(), 2);
    }

    #[test]
    fn test_message_char_count() {
        assert_eq!(
            message_char_count(&Message::User {
                content: "Hello".to_string()
            }),
            5
        );

        assert_eq!(
            message_char_count(&Message::System {
                content: "System".to_string()
            }),
            6
        );

        assert_eq!(
            message_char_count(&Message::ToolResult {
                tool_use_id: "id".to_string(),
                content: "result text".to_string(),
                is_error: false,
            }),
            11
        );
    }
}
