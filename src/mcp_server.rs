//! MCP stdio server for NanoClaw (standalone binary).
//!
//! This is a **separate process** spawned by the agent runner.  It reads
//! JSON-RPC 2.0 requests from stdin (one per line) and writes JSON-RPC 2.0
//! responses to stdout.  The protocol subset implemented here is the
//! minimum required by the Model Context Protocol:
//!
//! - `initialize`  -> server info + capabilities
//! - `tools/list`  -> list of available tools
//! - `tools/call`  -> execute a tool
//! - `notifications/initialized` (notification, no response)
//!
//! Environment variables (set by the agent runner before spawning):
//! - `NANOCLAW_CHAT_JID`
//! - `NANOCLAW_GROUP_FOLDER`
//! - `NANOCLAW_IS_MAIN` ("1" or "0")

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;

use chrono::Utc;
use cron::Schedule as CronSchedule;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::str::FromStr;

// ─── IPC paths (inside the container) ───────────────────────────

const IPC_DIR: &str = "/workspace/ipc";
const MESSAGES_DIR: &str = "/workspace/ipc/messages";
const TASKS_DIR: &str = "/workspace/ipc/tasks";
const CURRENT_TASKS_FILE: &str = "/workspace/ipc/current_tasks.json";

// ─── Environment context ────────────────────────────────────────

struct Context {
    chat_jid: String,
    group_folder: String,
    is_main: bool,
}

impl Context {
    fn from_env() -> Self {
        Self {
            chat_jid: env::var("NANOCLAW_CHAT_JID").unwrap_or_default(),
            group_folder: env::var("NANOCLAW_GROUP_FOLDER").unwrap_or_default(),
            is_main: env::var("NANOCLAW_IS_MAIN").unwrap_or_default() == "1",
        }
    }
}

// ─── JSON-RPC types ─────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i64,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

// ─── Atomic IPC file write ──────────────────────────────────────

fn write_ipc_file(dir: &str, data: &Value) -> io::Result<String> {
    let dir_path = Path::new(dir);
    fs::create_dir_all(dir_path)?;

    let filename = format!(
        "{}-{}.json",
        Utc::now().timestamp_millis(),
        &uuid::Uuid::new_v4().to_string()[..8],
    );

    let file_path = dir_path.join(&filename);
    let temp_path = dir_path.join(format!("{filename}.tmp"));

    let json = serde_json::to_string_pretty(data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    fs::write(&temp_path, json)?;
    fs::rename(&temp_path, &file_path)?;

    Ok(filename)
}

// ─── Tool definitions ───────────────────────────────────────────

fn tool_definitions(is_main: bool) -> Value {
    let mut tools = vec![
        json!({
            "name": "send_message",
            "description": "Send a message to the user or group immediately while you're still running. Use this for progress updates or to send multiple messages. You can call this multiple times. Note: when running as a scheduled task, your final output is NOT sent to the user \u{2014} use this tool if you need to communicate with the user or group.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The message text to send"
                    },
                    "sender": {
                        "type": "string",
                        "description": "Your role/identity name (e.g. \"Researcher\"). When set, messages appear from a dedicated bot in Telegram."
                    }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "schedule_task",
            "description": "Schedule a recurring or one-time task. The task will run as a full agent with access to all tools.\n\nCONTEXT MODE - Choose based on task type:\n\u{2022} \"group\": Task runs in the group's conversation context, with access to chat history.\n\u{2022} \"isolated\": Task runs in a fresh session with no conversation history.\n\nSCHEDULE VALUE FORMAT (all times are LOCAL timezone):\n\u{2022} cron: Standard cron expression (e.g., \"*/5 * * * *\" for every 5 minutes, \"0 9 * * *\" for daily at 9am LOCAL time)\n\u{2022} interval: Milliseconds between runs (e.g., \"300000\" for 5 minutes, \"3600000\" for 1 hour)\n\u{2022} once: Local time WITHOUT \"Z\" suffix (e.g., \"2026-02-01T15:30:00\"). Do NOT use UTC/Z suffix.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "What the agent should do when the task runs. For isolated mode, include all necessary context here."
                    },
                    "schedule_type": {
                        "type": "string",
                        "enum": ["cron", "interval", "once"],
                        "description": "cron=recurring at specific times, interval=recurring every N ms, once=run once at specific time"
                    },
                    "schedule_value": {
                        "type": "string",
                        "description": "cron: \"*/5 * * * *\" | interval: milliseconds like \"300000\" | once: local timestamp like \"2026-02-01T15:30:00\" (no Z suffix!)"
                    },
                    "context_mode": {
                        "type": "string",
                        "enum": ["group", "isolated"],
                        "default": "group",
                        "description": "group=runs with chat history and memory, isolated=fresh session (include context in prompt)"
                    },
                    "target_group_jid": {
                        "type": "string",
                        "description": "(Main group only) JID of the group to schedule the task for. Defaults to the current group."
                    }
                },
                "required": ["prompt", "schedule_type", "schedule_value"]
            }
        }),
        json!({
            "name": "list_tasks",
            "description": "List all scheduled tasks. From main: shows all tasks. From other groups: shows only that group's tasks.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }),
        json!({
            "name": "pause_task",
            "description": "Pause a scheduled task. It will not run until resumed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to pause"
                    }
                },
                "required": ["task_id"]
            }
        }),
        json!({
            "name": "resume_task",
            "description": "Resume a paused task.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to resume"
                    }
                },
                "required": ["task_id"]
            }
        }),
        json!({
            "name": "cancel_task",
            "description": "Cancel and delete a scheduled task.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task ID to cancel"
                    }
                },
                "required": ["task_id"]
            }
        }),
    ];

    if is_main {
        tools.push(json!({
            "name": "register_group",
            "description": "Register a new WhatsApp group so the agent can respond to messages there. Main group only.\n\nUse available_groups.json to find the JID for a group. The folder name should be lowercase with hyphens (e.g., \"family-chat\").",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "jid": {
                        "type": "string",
                        "description": "The WhatsApp JID (e.g., \"120363336345536173@g.us\")"
                    },
                    "name": {
                        "type": "string",
                        "description": "Display name for the group"
                    },
                    "folder": {
                        "type": "string",
                        "description": "Folder name for group files (lowercase, hyphens, e.g., \"family-chat\")"
                    },
                    "trigger": {
                        "type": "string",
                        "description": "Trigger word (e.g., \"@Andy\")"
                    }
                },
                "required": ["jid", "name", "folder", "trigger"]
            }
        }));
    }

    json!(tools)
}

// ─── Tool execution ─────────────────────────────────────────────

/// Result from a tool call, matching MCP's `CallToolResult`.
struct ToolCallResult {
    text: String,
    is_error: bool,
}

fn execute_tool(name: &str, args: &Value, ctx: &Context) -> ToolCallResult {
    match name {
        "send_message" => tool_send_message(args, ctx),
        "schedule_task" => tool_schedule_task(args, ctx),
        "list_tasks" => tool_list_tasks(ctx),
        "pause_task" => tool_pause_task(args, ctx),
        "resume_task" => tool_resume_task(args, ctx),
        "cancel_task" => tool_cancel_task(args, ctx),
        "register_group" => tool_register_group(args, ctx),
        _ => ToolCallResult {
            text: format!("Unknown tool: {name}"),
            is_error: true,
        },
    }
}

// ─── send_message ───────────────────────────────────────────────

fn tool_send_message(args: &Value, ctx: &Context) -> ToolCallResult {
    let text = match args.get("text").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => {
            return ToolCallResult {
                text: "Missing required parameter: text".into(),
                is_error: true,
            }
        }
    };

    let sender = args.get("sender").and_then(|v| v.as_str());

    let mut data = json!({
        "type": "message",
        "chatJid": ctx.chat_jid,
        "text": text,
        "groupFolder": ctx.group_folder,
        "timestamp": Utc::now().to_rfc3339(),
    });

    if let Some(s) = sender {
        data["sender"] = json!(s);
    }

    match write_ipc_file(MESSAGES_DIR, &data) {
        Ok(_) => ToolCallResult {
            text: "Message sent.".into(),
            is_error: false,
        },
        Err(e) => ToolCallResult {
            text: format!("Failed to write IPC message: {e}"),
            is_error: true,
        },
    }
}

// ─── schedule_task ──────────────────────────────────────────────

fn tool_schedule_task(args: &Value, ctx: &Context) -> ToolCallResult {
    let prompt = match args.get("prompt").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => {
            return ToolCallResult {
                text: "Missing required parameter: prompt".into(),
                is_error: true,
            }
        }
    };

    let schedule_type = match args.get("schedule_type").and_then(|v| v.as_str()) {
        Some(st) => st,
        None => {
            return ToolCallResult {
                text: "Missing required parameter: schedule_type".into(),
                is_error: true,
            }
        }
    };

    let schedule_value = match args.get("schedule_value").and_then(|v| v.as_str()) {
        Some(sv) => sv,
        None => {
            return ToolCallResult {
                text: "Missing required parameter: schedule_value".into(),
                is_error: true,
            }
        }
    };

    // Validate schedule_value based on type.
    match schedule_type {
        "cron" => {
            // The `cron` crate expects 6-field (with seconds) or 7-field expressions.
            // Standard 5-field cron needs a seconds prefix: "0 <expr>".
            let cron_expr = if schedule_value.split_whitespace().count() == 5 {
                format!("0 {schedule_value}")
            } else {
                schedule_value.to_string()
            };

            match CronSchedule::from_str(&cron_expr) {
                Ok(_) => {}
                Err(_) => {
                    return ToolCallResult {
                        text: format!(
                            "Invalid cron: \"{schedule_value}\". Use format like \"0 9 * * *\" (daily 9am) or \"*/5 * * * *\" (every 5 min)."
                        ),
                        is_error: true,
                    };
                }
            }
        }
        "interval" => {
            match schedule_value.parse::<u64>() {
                Ok(ms) if ms > 0 => {}
                _ => {
                    return ToolCallResult {
                        text: format!(
                            "Invalid interval: \"{schedule_value}\". Must be positive milliseconds (e.g., \"300000\" for 5 min)."
                        ),
                        is_error: true,
                    };
                }
            }
        }
        "once" => {
            // Reject timezone suffixes (Z, +HH:MM, -HH:MM).
            if schedule_value.ends_with('Z')
                || schedule_value.ends_with('z')
                || has_tz_offset_suffix(schedule_value)
            {
                return ToolCallResult {
                    text: format!(
                        "Timestamp must be local time without timezone suffix. Got \"{schedule_value}\" \u{2014} use format like \"2026-02-01T15:30:00\"."
                    ),
                    is_error: true,
                };
            }
            // Attempt to parse as NaiveDateTime.
            if chrono::NaiveDateTime::parse_from_str(schedule_value, "%Y-%m-%dT%H:%M:%S").is_err()
                && chrono::NaiveDateTime::parse_from_str(schedule_value, "%Y-%m-%dT%H:%M").is_err()
            {
                return ToolCallResult {
                    text: format!(
                        "Invalid timestamp: \"{schedule_value}\". Use local time format like \"2026-02-01T15:30:00\"."
                    ),
                    is_error: true,
                };
            }
        }
        _ => {
            return ToolCallResult {
                text: format!("Invalid schedule_type: \"{schedule_type}\". Must be cron, interval, or once."),
                is_error: true,
            };
        }
    }

    // Non-main groups can only schedule for themselves.
    let target_jid = if ctx.is_main {
        args.get("target_group_jid")
            .and_then(|v| v.as_str())
            .unwrap_or(&ctx.chat_jid)
    } else {
        &ctx.chat_jid
    };

    let context_mode = args
        .get("context_mode")
        .and_then(|v| v.as_str())
        .unwrap_or("group");

    let data = json!({
        "type": "schedule_task",
        "prompt": prompt,
        "schedule_type": schedule_type,
        "schedule_value": schedule_value,
        "context_mode": context_mode,
        "targetJid": target_jid,
        "createdBy": ctx.group_folder,
        "timestamp": Utc::now().to_rfc3339(),
    });

    match write_ipc_file(TASKS_DIR, &data) {
        Ok(filename) => ToolCallResult {
            text: format!("Task scheduled ({filename}): {schedule_type} - {schedule_value}"),
            is_error: false,
        },
        Err(e) => ToolCallResult {
            text: format!("Failed to write IPC task: {e}"),
            is_error: true,
        },
    }
}

/// Check whether a string ends with a UTC offset like `+00:00` or `-05:30`.
fn has_tz_offset_suffix(s: &str) -> bool {
    let bytes = s.as_bytes();
    // Minimum pattern: +HH:MM (6 chars)
    if bytes.len() < 6 {
        return false;
    }
    let tail = &s[s.len() - 6..];
    // Match [+-]DD:DD
    let tb = tail.as_bytes();
    (tb[0] == b'+' || tb[0] == b'-')
        && tb[1].is_ascii_digit()
        && tb[2].is_ascii_digit()
        && tb[3] == b':'
        && tb[4].is_ascii_digit()
        && tb[5].is_ascii_digit()
}

// ─── list_tasks ─────────────────────────────────────────────────

fn tool_list_tasks(ctx: &Context) -> ToolCallResult {
    let tasks_path = Path::new(CURRENT_TASKS_FILE);

    if !tasks_path.exists() {
        return ToolCallResult {
            text: "No scheduled tasks found.".into(),
            is_error: false,
        };
    }

    let contents = match fs::read_to_string(tasks_path) {
        Ok(c) => c,
        Err(e) => {
            return ToolCallResult {
                text: format!("Error reading tasks: {e}"),
                is_error: false,
            }
        }
    };

    let all_tasks: Vec<Value> = match serde_json::from_str(&contents) {
        Ok(t) => t,
        Err(e) => {
            return ToolCallResult {
                text: format!("Error parsing tasks: {e}"),
                is_error: false,
            }
        }
    };

    // Filter: main sees all, others see only their own group.
    let tasks: Vec<&Value> = if ctx.is_main {
        all_tasks.iter().collect()
    } else {
        all_tasks
            .iter()
            .filter(|t| {
                t.get("groupFolder")
                    .or_else(|| t.get("group_folder"))
                    .and_then(|v| v.as_str())
                    == Some(&ctx.group_folder)
            })
            .collect()
    };

    if tasks.is_empty() {
        return ToolCallResult {
            text: "No scheduled tasks found.".into(),
            is_error: false,
        };
    }

    let formatted: Vec<String> = tasks
        .iter()
        .map(|t| {
            let id = t.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let prompt = t.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
            let prompt_preview = if prompt.len() > 50 {
                format!("{}...", &prompt[..50])
            } else {
                prompt.to_string()
            };
            let stype = t
                .get("schedule_type")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let svalue = t
                .get("schedule_value")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let status = t.get("status").and_then(|v| v.as_str()).unwrap_or("?");
            let next_run = t
                .get("next_run")
                .and_then(|v| v.as_str())
                .unwrap_or("N/A");
            format!("- [{id}] {prompt_preview} ({stype}: {svalue}) - {status}, next: {next_run}")
        })
        .collect();

    ToolCallResult {
        text: format!("Scheduled tasks:\n{}", formatted.join("\n")),
        is_error: false,
    }
}

// ─── pause_task ─────────────────────────────────────────────────

fn tool_pause_task(args: &Value, ctx: &Context) -> ToolCallResult {
    let task_id = match args.get("task_id").and_then(|v| v.as_str()) {
        Some(id) => id,
        None => {
            return ToolCallResult {
                text: "Missing required parameter: task_id".into(),
                is_error: true,
            }
        }
    };

    let data = json!({
        "type": "pause_task",
        "taskId": task_id,
        "groupFolder": ctx.group_folder,
        "isMain": ctx.is_main,
        "timestamp": Utc::now().to_rfc3339(),
    });

    match write_ipc_file(TASKS_DIR, &data) {
        Ok(_) => ToolCallResult {
            text: format!("Task {task_id} pause requested."),
            is_error: false,
        },
        Err(e) => ToolCallResult {
            text: format!("Failed to write IPC task: {e}"),
            is_error: true,
        },
    }
}

// ─── resume_task ────────────────────────────────────────────────

fn tool_resume_task(args: &Value, ctx: &Context) -> ToolCallResult {
    let task_id = match args.get("task_id").and_then(|v| v.as_str()) {
        Some(id) => id,
        None => {
            return ToolCallResult {
                text: "Missing required parameter: task_id".into(),
                is_error: true,
            }
        }
    };

    let data = json!({
        "type": "resume_task",
        "taskId": task_id,
        "groupFolder": ctx.group_folder,
        "isMain": ctx.is_main,
        "timestamp": Utc::now().to_rfc3339(),
    });

    match write_ipc_file(TASKS_DIR, &data) {
        Ok(_) => ToolCallResult {
            text: format!("Task {task_id} resume requested."),
            is_error: false,
        },
        Err(e) => ToolCallResult {
            text: format!("Failed to write IPC task: {e}"),
            is_error: true,
        },
    }
}

// ─── cancel_task ────────────────────────────────────────────────

fn tool_cancel_task(args: &Value, ctx: &Context) -> ToolCallResult {
    let task_id = match args.get("task_id").and_then(|v| v.as_str()) {
        Some(id) => id,
        None => {
            return ToolCallResult {
                text: "Missing required parameter: task_id".into(),
                is_error: true,
            }
        }
    };

    let data = json!({
        "type": "cancel_task",
        "taskId": task_id,
        "groupFolder": ctx.group_folder,
        "isMain": ctx.is_main,
        "timestamp": Utc::now().to_rfc3339(),
    });

    match write_ipc_file(TASKS_DIR, &data) {
        Ok(_) => ToolCallResult {
            text: format!("Task {task_id} cancellation requested."),
            is_error: false,
        },
        Err(e) => ToolCallResult {
            text: format!("Failed to write IPC task: {e}"),
            is_error: true,
        },
    }
}

// ─── register_group ─────────────────────────────────────────────

fn tool_register_group(args: &Value, ctx: &Context) -> ToolCallResult {
    if !ctx.is_main {
        return ToolCallResult {
            text: "Only the main group can register new groups.".into(),
            is_error: true,
        };
    }

    let jid = match args.get("jid").and_then(|v| v.as_str()) {
        Some(j) => j,
        None => {
            return ToolCallResult {
                text: "Missing required parameter: jid".into(),
                is_error: true,
            }
        }
    };

    let name = match args.get("name").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => {
            return ToolCallResult {
                text: "Missing required parameter: name".into(),
                is_error: true,
            }
        }
    };

    let folder = match args.get("folder").and_then(|v| v.as_str()) {
        Some(f) => f,
        None => {
            return ToolCallResult {
                text: "Missing required parameter: folder".into(),
                is_error: true,
            }
        }
    };

    let trigger = match args.get("trigger").and_then(|v| v.as_str()) {
        Some(t) => t,
        None => {
            return ToolCallResult {
                text: "Missing required parameter: trigger".into(),
                is_error: true,
            }
        }
    };

    let data = json!({
        "type": "register_group",
        "jid": jid,
        "name": name,
        "folder": folder,
        "trigger": trigger,
        "timestamp": Utc::now().to_rfc3339(),
    });

    match write_ipc_file(TASKS_DIR, &data) {
        Ok(_) => ToolCallResult {
            text: format!("Group \"{name}\" registered. It will start receiving messages immediately."),
            is_error: false,
        },
        Err(e) => ToolCallResult {
            text: format!("Failed to write IPC task: {e}"),
            is_error: true,
        },
    }
}

// ─── JSON-RPC response helpers ──────────────────────────────────

fn success_response(id: Value, result: Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(result),
        error: None,
    }
}

fn error_response(id: Value, code: i64, message: &str) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: None,
        error: Some(JsonRpcError {
            code,
            message: message.into(),
            data: None,
        }),
    }
}

// ─── Request handling ───────────────────────────────────────────

fn handle_request(req: &JsonRpcRequest, ctx: &Context) -> Option<JsonRpcResponse> {
    let id = req.id.clone().unwrap_or(Value::Null);

    match req.method.as_str() {
        // ── initialize ──────────────────────────────────────────
        "initialize" => Some(success_response(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "nanoclaw",
                    "version": "1.0.0"
                }
            }),
        )),

        // ── notifications (no response) ─────────────────────────
        "notifications/initialized" | "notifications/cancelled" => {
            // Notifications have no id and require no response.
            None
        }

        // ── tools/list ──────────────────────────────────────────
        "tools/list" => Some(success_response(
            id,
            json!({
                "tools": tool_definitions(ctx.is_main)
            }),
        )),

        // ── tools/call ──────────────────────────────────────────
        "tools/call" => {
            let tool_name = req
                .params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let arguments = req
                .params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));

            let result = execute_tool(tool_name, &arguments, ctx);

            Some(success_response(
                id,
                json!({
                    "content": [{
                        "type": "text",
                        "text": result.text
                    }],
                    "isError": result.is_error
                }),
            ))
        }

        // ── unknown method ──────────────────────────────────────
        _ => {
            // If the request has no id it is a notification; don't respond.
            if req.id.is_none() {
                None
            } else {
                Some(error_response(
                    id,
                    -32601,
                    &format!("Method not found: {}", req.method),
                ))
            }
        }
    }
}

// ─── Main loop ──────────────────────────────────────────────────

/// Entry point for the MCP stdio server binary.
///
/// Reads JSON-RPC requests line-by-line from stdin and writes responses to
/// stdout.  The process exits when stdin is closed (EOF).
pub fn main() {
    let ctx = Context::from_env();

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break, // EOF or read error
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                // Write a parse error response.
                let resp = error_response(Value::Null, -32700, &format!("Parse error: {e}"));
                if let Ok(json) = serde_json::to_string(&resp) {
                    let _ = writeln!(stdout, "{json}");
                    let _ = stdout.flush();
                }
                continue;
            }
        };

        if let Some(resp) = handle_request(&req, &ctx) {
            if let Ok(json) = serde_json::to_string(&resp) {
                let _ = writeln!(stdout, "{json}");
                let _ = stdout.flush();
            }
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ctx() -> Context {
        Context {
            chat_jid: "123@g.us".into(),
            group_folder: "test-group".into(),
            is_main: false,
        }
    }

    fn main_ctx() -> Context {
        Context {
            chat_jid: "123@g.us".into(),
            group_folder: "main".into(),
            is_main: true,
        }
    }

    #[test]
    fn test_initialize() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(json!(1)),
            method: "initialize".into(),
            params: json!({}),
        };
        let resp = handle_request(&req, &test_ctx()).unwrap();
        let result = resp.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "nanoclaw");
        assert!(result["capabilities"]["tools"].is_object());
    }

    #[test]
    fn test_tools_list_non_main() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(json!(2)),
            method: "tools/list".into(),
            params: json!({}),
        };
        let resp = handle_request(&req, &test_ctx()).unwrap();
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"send_message"));
        assert!(names.contains(&"schedule_task"));
        assert!(names.contains(&"list_tasks"));
        assert!(names.contains(&"pause_task"));
        assert!(names.contains(&"resume_task"));
        assert!(names.contains(&"cancel_task"));
        // register_group should NOT be present for non-main
        assert!(!names.contains(&"register_group"));
    }

    #[test]
    fn test_tools_list_main_includes_register_group() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(json!(3)),
            method: "tools/list".into(),
            params: json!({}),
        };
        let resp = handle_request(&req, &main_ctx()).unwrap();
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        let names: Vec<&str> = tools
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"register_group"));
    }

    #[test]
    fn test_register_group_blocked_for_non_main() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(json!(4)),
            method: "tools/call".into(),
            params: json!({
                "name": "register_group",
                "arguments": {
                    "jid": "456@g.us",
                    "name": "Test",
                    "folder": "test",
                    "trigger": "@test"
                }
            }),
        };
        let resp = handle_request(&req, &test_ctx()).unwrap();
        let result = resp.result.unwrap();
        assert_eq!(result["isError"], true);
        assert!(result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Only the main group"));
    }

    #[test]
    fn test_schedule_task_invalid_cron() {
        let result = execute_tool(
            "schedule_task",
            &json!({
                "prompt": "test",
                "schedule_type": "cron",
                "schedule_value": "not a cron"
            }),
            &test_ctx(),
        );
        assert!(result.is_error);
        assert!(result.text.contains("Invalid cron"));
    }

    #[test]
    fn test_schedule_task_invalid_interval() {
        let result = execute_tool(
            "schedule_task",
            &json!({
                "prompt": "test",
                "schedule_type": "interval",
                "schedule_value": "-100"
            }),
            &test_ctx(),
        );
        assert!(result.is_error);
        assert!(result.text.contains("Invalid interval"));
    }

    #[test]
    fn test_schedule_task_once_rejects_z_suffix() {
        let result = execute_tool(
            "schedule_task",
            &json!({
                "prompt": "test",
                "schedule_type": "once",
                "schedule_value": "2026-02-01T15:30:00Z"
            }),
            &test_ctx(),
        );
        assert!(result.is_error);
        assert!(result.text.contains("without timezone suffix"));
    }

    #[test]
    fn test_schedule_task_once_rejects_offset_suffix() {
        let result = execute_tool(
            "schedule_task",
            &json!({
                "prompt": "test",
                "schedule_type": "once",
                "schedule_value": "2026-02-01T15:30:00+05:30"
            }),
            &test_ctx(),
        );
        assert!(result.is_error);
        assert!(result.text.contains("without timezone suffix"));
    }

    #[test]
    fn test_schedule_task_once_rejects_invalid_timestamp() {
        let result = execute_tool(
            "schedule_task",
            &json!({
                "prompt": "test",
                "schedule_type": "once",
                "schedule_value": "not-a-date"
            }),
            &test_ctx(),
        );
        assert!(result.is_error);
        assert!(result.text.contains("Invalid timestamp"));
    }

    #[test]
    fn test_has_tz_offset_suffix() {
        assert!(has_tz_offset_suffix("2026-02-01T15:30:00+00:00"));
        assert!(has_tz_offset_suffix("2026-02-01T15:30:00-05:30"));
        assert!(!has_tz_offset_suffix("2026-02-01T15:30:00"));
        assert!(!has_tz_offset_suffix("short"));
    }

    #[test]
    fn test_unknown_method_returns_error() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(json!(99)),
            method: "unknown/method".into(),
            params: json!({}),
        };
        let resp = handle_request(&req, &test_ctx()).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
    }

    #[test]
    fn test_notification_returns_none() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: None,
            method: "notifications/initialized".into(),
            params: json!({}),
        };
        assert!(handle_request(&req, &test_ctx()).is_none());
    }

    #[test]
    fn test_send_message_missing_text() {
        let result = execute_tool("send_message", &json!({}), &test_ctx());
        assert!(result.is_error);
        assert!(result.text.contains("Missing required parameter: text"));
    }

    #[test]
    fn test_unknown_tool() {
        let result = execute_tool("nonexistent_tool", &json!({}), &test_ctx());
        assert!(result.is_error);
        assert!(result.text.contains("Unknown tool"));
    }
}
