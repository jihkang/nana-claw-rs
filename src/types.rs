use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Container Protocol (stdin/stdout) ────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContainerInput {
    pub prompt: String,
    pub session_id: Option<String>,
    pub group_folder: String,
    #[allow(dead_code)]
    pub chat_jid: String,
    pub is_main: bool,
    #[serde(default)]
    pub is_scheduled_task: bool,
    pub assistant_name: Option<String>,
    #[serde(default)]
    pub secrets: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContainerOutput {
    pub status: OutputStatus,
    pub result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputStatus {
    Success,
    Error,
}

pub const OUTPUT_START_MARKER: &str = "---NANOCLAW_OUTPUT_START---";
pub const OUTPUT_END_MARKER: &str = "---NANOCLAW_OUTPUT_END---";

// ─── LLM Messages ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: String },
    #[serde(rename = "assistant")]
    Assistant { content: Vec<ContentBlock> },
    #[serde(rename = "tool")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

// ─── Tool Definitions ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

// ─── Provider Response ───────────────────────────────────────────

#[derive(Debug)]
pub struct LlmResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ─── Session ─────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionEntry {
    pub session_id: String,
    pub full_path: String,
    pub summary: String,
    pub first_prompt: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionsIndex {
    pub entries: Vec<SessionEntry>,
}

// ─── IPC ─────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IpcMessage {
    #[serde(rename = "message")]
    Message {
        text: String,
    },
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IpcOutbound {
    #[serde(rename = "message")]
    Message {
        #[serde(rename = "chatJid")]
        chat_jid: String,
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        sender: Option<String>,
        #[serde(rename = "groupFolder")]
        group_folder: String,
        timestamp: String,
    },
    #[serde(rename = "schedule_task")]
    ScheduleTask {
        prompt: String,
        schedule_type: String,
        schedule_value: String,
        context_mode: String,
        #[serde(rename = "targetJid")]
        target_jid: String,
        #[serde(rename = "createdBy")]
        created_by: String,
        timestamp: String,
    },
    #[serde(rename = "pause_task")]
    PauseTask {
        #[serde(rename = "taskId")]
        task_id: String,
        #[serde(rename = "groupFolder")]
        group_folder: String,
        #[serde(rename = "isMain")]
        is_main: bool,
        timestamp: String,
    },
    #[serde(rename = "resume_task")]
    ResumeTask {
        #[serde(rename = "taskId")]
        task_id: String,
        #[serde(rename = "groupFolder")]
        group_folder: String,
        #[serde(rename = "isMain")]
        is_main: bool,
        timestamp: String,
    },
    #[serde(rename = "cancel_task")]
    CancelTask {
        #[serde(rename = "taskId")]
        task_id: String,
        #[serde(rename = "groupFolder")]
        group_folder: String,
        #[serde(rename = "isMain")]
        is_main: bool,
        timestamp: String,
    },
    #[serde(rename = "register_group")]
    RegisterGroup {
        jid: String,
        name: String,
        folder: String,
        trigger: String,
        timestamp: String,
    },
}

// ─── Provider Config ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ProviderConfig {
    Anthropic {
        api_key: String,
        base_url: Option<String>,
        model: String,
    },
    OpenAi {
        api_key: String,
        base_url: Option<String>,
        model: String,
    },
    OpenRouter {
        api_key: String,
        model: String,
    },
}
