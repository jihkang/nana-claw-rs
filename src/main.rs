mod agent;
mod hooks;
mod ipc;
#[allow(dead_code)]
mod mcp_server;
#[allow(dead_code)]
mod orchestrator;
mod provider;
mod session;
mod tools;
mod types;

use std::collections::HashMap;
use tokio::io::AsyncReadExt;
use tracing_subscriber::EnvFilter;

use crate::agent::Agent;
use crate::hooks::archive_transcript;
use crate::ipc::{drain_ipc_input, init_ipc_dirs, start_ipc_poller, wait_for_ipc_message};
use crate::orchestrator::Orchestrator;
use crate::provider::create_provider;
use crate::session::Session;
use crate::tools::create_default_registry;
use crate::types::*;

fn write_output(output: &ContainerOutput) {
    println!("{}", OUTPUT_START_MARKER);
    println!("{}", serde_json::to_string(output).unwrap());
    println!("{}", OUTPUT_END_MARKER);
}

fn log(message: &str) {
    eprintln!("[agent-runner] {message}");
}

/// Read all of stdin into a string.
async fn read_stdin() -> Result<String, std::io::Error> {
    let mut buf = String::new();
    tokio::io::stdin().read_to_string(&mut buf).await?;
    Ok(buf)
}

/// Determine provider config from secrets and environment.
fn resolve_provider(secrets: &HashMap<String, String>) -> ProviderConfig {
    // Check for explicit provider override
    let provider_type = secrets
        .get("NANOCLAW_PROVIDER")
        .cloned()
        .or_else(|| std::env::var("NANOCLAW_PROVIDER").ok())
        .unwrap_or_default();

    let model = secrets
        .get("NANOCLAW_MODEL")
        .cloned()
        .or_else(|| std::env::var("NANOCLAW_MODEL").ok());

    match provider_type.as_str() {
        "openai" => ProviderConfig::OpenAi {
            api_key: secrets
                .get("OPENAI_API_KEY")
                .cloned()
                .unwrap_or_default(),
            base_url: secrets.get("OPENAI_BASE_URL").cloned(),
            model: model.unwrap_or_else(|| "gpt-4o".to_string()),
        },
        "openrouter" => ProviderConfig::OpenRouter {
            api_key: secrets
                .get("OPENROUTER_API_KEY")
                .cloned()
                .unwrap_or_default(),
            model: model.unwrap_or_else(|| "anthropic/claude-sonnet-4".to_string()),
        },
        _ => {
            // Default: Anthropic
            ProviderConfig::Anthropic {
                api_key: secrets
                    .get("ANTHROPIC_API_KEY")
                    .cloned()
                    .unwrap_or_default(),
                base_url: secrets.get("ANTHROPIC_BASE_URL").cloned(),
                model: model.unwrap_or_else(|| "claude-sonnet-4-20250514".to_string()),
            }
        }
    }
}

/// Build system prompt from global CLAUDE.md and group context.
fn build_system_prompt(container_input: &ContainerInput) -> Option<String> {
    let global_path = "/workspace/global/CLAUDE.md";
    let group_path = "/workspace/group/CLAUDE.md";

    let mut parts = Vec::new();

    // Group-level CLAUDE.md
    if let Ok(content) = std::fs::read_to_string(group_path) {
        if !content.trim().is_empty() {
            parts.push(content);
        }
    }

    // Global CLAUDE.md (for non-main groups)
    if !container_input.is_main {
        if let Ok(content) = std::fs::read_to_string(global_path) {
            if !content.trim().is_empty() {
                parts.push(format!("\n---\n# Global Instructions\n\n{content}"));
            }
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n"))
    }
}

/// Check if orchestrator mode is enabled via secrets or env.
fn is_orchestrator_enabled(secrets: &std::collections::HashMap<String, String>) -> bool {
    if let Some(v) = secrets.get("NANOCLAW_ORCHESTRATE") {
        return v == "1" || v == "true";
    }
    if let Ok(v) = std::env::var("NANOCLAW_ORCHESTRATE") {
        return v == "1" || v == "true";
    }
    false
}

/// Run a single query: create agent (or orchestrator), feed prompt, handle IPC.
async fn run_query(
    prompt: &str,
    session: &mut Session,
    container_input: &ContainerInput,
    provider_config: &ProviderConfig,
    secret_env_vars: &[String],
) -> Result<QueryResult, Box<dyn std::error::Error>> {
    // Start IPC poller for follow-up messages during query
    let (ipc_rx, poller_handle) = start_ipc_poller();

    let system_prompt = build_system_prompt(container_input);

    let query_result = if is_orchestrator_enabled(&container_input.secrets) {
        // Orchestrator mode: plan → split → parallel execute → merge
        log("Orchestrator mode enabled");
        let orch = Orchestrator::new(
            provider_config.clone(),
            secret_env_vars.to_vec(),
            system_prompt,
        );

        match orch.run(prompt, session).await {
            Ok(result) => {
                if let Some(ref plan) = result.plan {
                    log(&format!(
                        "Orchestrator: {:?} complexity, {} subtasks, {:?} merge",
                        plan.complexity,
                        plan.subtasks.len(),
                        plan.merge_strategy,
                    ));
                }
                log(&format!(
                    "Query done. Usage: {}in/{}out tokens",
                    result.total_usage.input_tokens, result.total_usage.output_tokens
                ));
                Ok(result.final_text)
            }
            Err(e) => Err(Box::new(e) as Box<dyn std::error::Error>),
        }
    } else {
        // Direct mode: single agent
        let provider = create_provider(provider_config.clone());
        let tools = create_default_registry(secret_env_vars);
        let agent = Agent::new(provider, tools, system_prompt);

        match agent.run(prompt, session).await {
            Ok(result) => {
                log(&format!(
                    "Query done. Usage: {}in/{}out tokens",
                    result.total_usage.input_tokens, result.total_usage.output_tokens
                ));
                Ok(result.final_text)
            }
            Err(e) => Err(Box::new(e) as Box<dyn std::error::Error>),
        }
    };

    // Stop poller and check if close was detected
    drop(ipc_rx);
    let closed_during_query = poller_handle.await.unwrap_or(false);

    match query_result {
        Ok(final_text) => Ok(QueryResult {
            final_text,
            closed_during_query,
        }),
        Err(e) => Err(e),
    }
}

struct QueryResult {
    final_text: Option<String>,
    closed_during_query: bool,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    // Read and parse container input from stdin
    let container_input: ContainerInput = match read_stdin().await {
        Ok(data) => match serde_json::from_str(&data) {
            Ok(input) => input,
            Err(e) => {
                write_output(&ContainerOutput {
                    status: OutputStatus::Error,
                    result: None,
                    new_session_id: None,
                    error: Some(format!("Failed to parse input: {e}")),
                });
                std::process::exit(1);
            }
        },
        Err(e) => {
            write_output(&ContainerOutput {
                status: OutputStatus::Error,
                result: None,
                new_session_id: None,
                error: Some(format!("Failed to read stdin: {e}")),
            });
            std::process::exit(1);
        }
    };

    // Delete temp input file (may contain secrets)
    let _ = std::fs::remove_file("/tmp/input.json");
    log(&format!(
        "Received input for group: {}",
        container_input.group_folder
    ));

    // Resolve provider configuration
    let provider_config = resolve_provider(&container_input.secrets);
    log(&format!("Using provider: {:?}", provider_config));

    // Secret env vars to strip from bash subprocesses
    let secret_env_vars: Vec<String> = vec![
        "ANTHROPIC_API_KEY".to_string(),
        "CLAUDE_CODE_OAUTH_TOKEN".to_string(),
        "OPENAI_API_KEY".to_string(),
        "OPENROUTER_API_KEY".to_string(),
    ];

    // Initialize IPC directories
    init_ipc_dirs();

    // Load or create session
    let mut session = if let Some(ref sid) = container_input.session_id {
        match Session::load(sid, &container_input.group_folder) {
            Ok(s) => {
                log(&format!("Resumed session: {}", s.id));
                s
            }
            Err(e) => {
                log(&format!("Failed to load session {sid}: {e}, creating new"));
                Session::new(&container_input.group_folder)
            }
        }
    } else {
        Session::new(&container_input.group_folder)
    };

    let session_id = session.id.clone();

    // Build initial prompt
    let mut prompt = container_input.prompt.clone();
    if container_input.is_scheduled_task {
        prompt = format!(
            "[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n{prompt}"
        );
    }

    // Drain any pending IPC messages into initial prompt
    let pending = drain_ipc_input();
    if !pending.is_empty() {
        log(&format!(
            "Draining {} pending IPC messages into initial prompt",
            pending.len()
        ));
        prompt.push('\n');
        prompt.push_str(&pending.join("\n"));
    }

    // Main query loop
    loop {
        log(&format!("Starting query (session: {})...", session.id));

        match run_query(
            &prompt,
            &mut session,
            &container_input,
            &provider_config,
            &secret_env_vars,
        )
        .await
        {
            Ok(result) => {
                // Write output
                write_output(&ContainerOutput {
                    status: OutputStatus::Success,
                    result: result.final_text,
                    new_session_id: Some(session_id.clone()),
                    error: None,
                });

                // Save session
                if let Err(e) = session.save() {
                    log(&format!("Failed to save session: {e}"));
                }

                // Check if session needs compaction
                let estimated = session.estimated_tokens();
                if estimated > 80_000 {
                    log(&format!(
                        "Session at ~{estimated} tokens, compacting..."
                    ));
                    let removed = session.compact(20);
                    if let Err(e) = archive_transcript(
                        &removed,
                        &session.id,
                        container_input.assistant_name.as_deref(),
                    ) {
                        log(&format!("Failed to archive transcript: {e}"));
                    }
                }

                // Exit if close was detected during query
                if result.closed_during_query {
                    log("Close sentinel consumed during query, exiting");
                    break;
                }

                // Emit session update
                write_output(&ContainerOutput {
                    status: OutputStatus::Success,
                    result: None,
                    new_session_id: Some(session_id.clone()),
                    error: None,
                });

                log("Query ended, waiting for next IPC message...");

                // Wait for next message or close sentinel
                match wait_for_ipc_message().await {
                    Some(next_message) => {
                        log(&format!(
                            "Got new message ({} chars), starting new query",
                            next_message.len()
                        ));
                        prompt = next_message;
                    }
                    None => {
                        log("Close sentinel received, exiting");
                        break;
                    }
                }
            }
            Err(e) => {
                let error_message = e.to_string();
                log(&format!("Agent error: {error_message}"));
                write_output(&ContainerOutput {
                    status: OutputStatus::Error,
                    result: None,
                    new_session_id: Some(session_id.clone()),
                    error: Some(error_message),
                });
                std::process::exit(1);
            }
        }
    }
}
