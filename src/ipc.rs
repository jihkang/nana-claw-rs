use std::fs;
use std::path::{Path, PathBuf};

use serde_json;
use tokio::sync::mpsc;
use tracing::{debug, error, warn};

use crate::types::IpcMessage;

const IPC_INPUT_DIR: &str = "/workspace/ipc/input";
const IPC_INPUT_CLOSE_SENTINEL: &str = "/workspace/ipc/input/_close";
const IPC_POLL_MS: u64 = 500;

// ─── Sentinel ────────────────────────────────────────────────────

/// Check for the `_close` sentinel file.
/// If present, removes it and returns `true`.
pub fn should_close() -> bool {
    let sentinel = Path::new(IPC_INPUT_CLOSE_SENTINEL);
    if sentinel.exists() {
        if let Err(e) = fs::remove_file(sentinel) {
            warn!("Failed to remove _close sentinel: {e}");
        }
        true
    } else {
        false
    }
}

// ─── Drain ───────────────────────────────────────────────────────

/// Drain all pending IPC input messages from `/workspace/ipc/input/*.json`.
///
/// Reads each file, parses `{ "type": "message", "text": "..." }`, deletes the
/// file after reading, and returns the collected message texts in
/// lexicographic (filename) order.
pub fn drain_ipc_input() -> Vec<String> {
    let dir = Path::new(IPC_INPUT_DIR);

    // Ensure the directory exists (first call may race with container setup).
    if let Err(e) = fs::create_dir_all(dir) {
        error!("Failed to create IPC input dir: {e}");
        return Vec::new();
    }

    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            error!("IPC drain error reading dir: {e}");
            return Vec::new();
        }
    };

    // Collect and sort filenames so messages arrive in order.
    let mut json_files: Vec<PathBuf> = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    json_files.sort();

    let mut messages = Vec::new();

    for file_path in json_files {
        match fs::read_to_string(&file_path) {
            Ok(contents) => {
                // Always remove the file after reading, even if parsing fails.
                if let Err(e) = fs::remove_file(&file_path) {
                    warn!("Failed to remove IPC input file {}: {e}", file_path.display());
                }

                match serde_json::from_str::<IpcMessage>(&contents) {
                    Ok(IpcMessage::Message { text }) => {
                        messages.push(text);
                    }
                    Err(e) => {
                        warn!(
                            "Failed to parse IPC input file {}: {e}",
                            file_path.display()
                        );
                    }
                }
            }
            Err(e) => {
                error!(
                    "Failed to read IPC input file {}: {e}",
                    file_path.display()
                );
                // Try to clean up the bad file.
                let _ = fs::remove_file(&file_path);
            }
        }
    }

    messages
}

// ─── Blocking wait ───────────────────────────────────────────────

/// Wait for a new IPC message or the `_close` sentinel.
///
/// Polls every 500 ms.  Returns `None` if the close sentinel is detected,
/// or `Some(text)` when one or more messages are found (joined with `\n`).
pub async fn wait_for_ipc_message() -> Option<String> {
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(IPC_POLL_MS));

    loop {
        interval.tick().await;

        if should_close() {
            return None;
        }

        let messages = drain_ipc_input();
        if !messages.is_empty() {
            return Some(messages.join("\n"));
        }
    }
}

// ─── Atomic write ────────────────────────────────────────────────

/// Write an IPC output file atomically (temp file + rename).
///
/// Used by the MCP server tools to write to `/workspace/ipc/messages/` and
/// `/workspace/ipc/tasks/`.  Creates the target directory if it does not
/// exist.  Returns the generated filename on success.
#[allow(dead_code)]
pub fn write_ipc_file(dir: &str, data: &serde_json::Value) -> std::io::Result<String> {
    let dir_path = Path::new(dir);
    fs::create_dir_all(dir_path)?;

    let filename = format!(
        "{}-{}.json",
        chrono::Utc::now().timestamp_millis(),
        &uuid::Uuid::new_v4().to_string()[..8],
    );

    let file_path = dir_path.join(&filename);
    let temp_path = dir_path.join(format!("{filename}.tmp"));

    let json = serde_json::to_string_pretty(data)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    fs::write(&temp_path, json)?;
    fs::rename(&temp_path, &file_path)?;

    Ok(filename)
}

// ─── Init ────────────────────────────────────────────────────────

/// Initialize IPC directories: create the input dir if it doesn't exist and
/// remove any stale `_close` sentinel left over from a previous container run.
pub fn init_ipc_dirs() {
    if let Err(e) = fs::create_dir_all(IPC_INPUT_DIR) {
        error!("Failed to create IPC input dir: {e}");
    }

    // Clean up stale sentinel from a previous run.
    let sentinel = Path::new(IPC_INPUT_CLOSE_SENTINEL);
    if sentinel.exists() {
        debug!("Removing stale _close sentinel");
        let _ = fs::remove_file(sentinel);
    }
}

// ─── Background poller ──────────────────────────────────────────

/// Start a background IPC poller that feeds messages into an unbounded
/// channel while a query is active.
///
/// Returns `(receiver, join_handle)`.  The `JoinHandle` resolves to `true`
/// if the `_close` sentinel was detected during polling (`closedDuringQuery`),
/// or `false` if the receiver was dropped (the query finished first).
pub fn start_ipc_poller() -> (mpsc::UnboundedReceiver<String>, tokio::task::JoinHandle<bool>) {
    let (tx, rx) = mpsc::unbounded_channel::<String>();

    let handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(IPC_POLL_MS));

        loop {
            interval.tick().await;

            // Check for close sentinel first.
            if should_close() {
                debug!("Close sentinel detected during query");
                return true; // closedDuringQuery = true
            }

            // Drain any pending input messages.
            let messages = drain_ipc_input();
            for text in messages {
                debug!("Piping IPC message into active query ({} chars)", text.len());
                // If the receiver has been dropped the query is done; stop polling.
                if tx.send(text).is_err() {
                    debug!("IPC poller: receiver dropped, stopping");
                    return false;
                }
            }
        }
    });

    (rx, handle)
}

// ─── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Helper: write an IPC message file into a given directory.
    fn write_msg(dir: &Path, name: &str, text: &str) {
        let msg = serde_json::json!({ "type": "message", "text": text });
        fs::write(dir.join(name), serde_json::to_string(&msg).unwrap()).unwrap();
    }

    #[test]
    fn test_write_ipc_file_creates_dir_and_file() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("nested").join("dir");
        let data = serde_json::json!({ "type": "message", "text": "hello" });

        let filename = write_ipc_file(dir.to_str().unwrap(), &data).unwrap();
        assert!(filename.ends_with(".json"));

        let written = fs::read_to_string(dir.join(&filename)).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&written).unwrap();
        assert_eq!(parsed["text"], "hello");
    }
}
