pub mod bash;
pub mod filesystem;
pub mod grep;
pub mod web;

use async_trait::async_trait;
use crate::types::{ToolDefinition, ToolResult};

// ─── Tool Trait ──────────────────────────────────────────────────

/// Every built-in tool implements this trait.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Machine-readable name sent to the LLM (e.g. "bash", "read_file").
    fn name(&self) -> &str;

    /// One-sentence description shown to the LLM.
    fn description(&self) -> &str;

    /// JSON Schema describing the expected `input` object.
    fn input_schema(&self) -> serde_json::Value;

    /// Run the tool and return textual output (possibly with is_error).
    async fn execute(&self, input: serde_json::Value) -> ToolResult;
}

// ─── Registry ────────────────────────────────────────────────────

/// Owns every registered tool and dispatches calls by name.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Add a tool to the registry.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    /// Produce the `tools` array sent to the LLM in the API request.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .iter()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                input_schema: t.input_schema(),
            })
            .collect()
    }

    /// Look up a tool by name and run it.
    /// Returns an error result if the tool is not found.
    pub async fn execute(&self, name: &str, input: serde_json::Value) -> ToolResult {
        match self.get(name) {
            Some(tool) => tool.execute(input).await,
            None => ToolResult {
                content: format!("Unknown tool: {name}"),
                is_error: true,
            },
        }
    }

    /// Look up a tool by name (borrowed).
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| &**t)
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Factory ─────────────────────────────────────────────────────

/// Create a registry pre-loaded with every built-in tool.
///
/// `secret_env_vars` – additional env-var names that the bash tool must
/// strip before executing a command (on top of the hard-coded list).
pub fn create_default_registry(secret_env_vars: &[String]) -> ToolRegistry {
    let mut reg = ToolRegistry::new();

    // Shell
    reg.register(Box::new(bash::BashTool::new(secret_env_vars)));

    // Filesystem
    reg.register(Box::new(filesystem::ReadTool));
    reg.register(Box::new(filesystem::WriteTool));
    reg.register(Box::new(filesystem::EditTool));
    reg.register(Box::new(filesystem::GlobTool));

    // Search
    reg.register(Box::new(grep::GrepTool));

    // Web
    reg.register(Box::new(web::WebFetchTool));
    reg.register(Box::new(web::WebSearchTool));

    reg
}
