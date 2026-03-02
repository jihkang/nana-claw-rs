//! Task orchestrator: analyzes task complexity, splits into subtasks,
//! runs them in parallel, and merges results.

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::session::Session;
use crate::agent::{Agent, AgentError};
use crate::types::*;

// ─── Plan Types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPlan {
    pub complexity: Complexity,
    pub subtasks: Vec<SubTask>,
    pub merge_strategy: MergeStrategy,
    /// Optional reasoning from the planner about why it split this way
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Complexity {
    Small,
    Medium,
    Large,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubTask {
    pub id: String,
    pub description: String,
    /// Full prompt/context to give to the sub-agent
    pub prompt: String,
    /// IDs of subtasks this one depends on (must complete first)
    #[serde(default)]
    pub depends_on: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    /// Simply concatenate all results
    Concatenate,
    /// Ask LLM to synthesize/summarize all results
    Summarize,
    /// Ask LLM to merge code changes coherently
    CodeMerge,
}

// ─── Subtask Result ──────────────────────────────────────────────

#[derive(Debug)]
pub struct SubTaskResult {
    pub id: String,
    pub description: String,
    pub output: Option<String>,
    pub usage: Usage,
    pub success: bool,
    pub error: Option<String>,
}

// ─── Orchestrator ────────────────────────────────────────────────

/// The planner prompt instructs the LLM to analyze the task and return
/// a structured JSON plan.
const PLANNER_SYSTEM_PROMPT: &str = r#"You are a task planner. Analyze the user's request and decide how to execute it.

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):

{
  "complexity": "small" | "medium" | "large",
  "subtasks": [
    {
      "id": "unique-id",
      "description": "short description",
      "prompt": "full instructions for this subtask",
      "depends_on": ["id-of-dependency"]  // empty array if independent
    }
  ],
  "merge_strategy": "concatenate" | "summarize" | "code_merge",
  "reasoning": "why you split it this way"
}

Rules:
- "small": 1 subtask, simple enough for a single agent pass. Just wrap the original prompt.
- "medium": 2-3 subtasks, can mostly run in parallel.
- "large": 4+ subtasks, complex with potential dependencies.
- Each subtask prompt must be self-contained with all necessary context.
- Use depends_on for subtasks that need another's output (e.g., "implement" depends on "design").
- Independent subtasks will run in parallel.
- merge_strategy: use "summarize" for research/analysis, "code_merge" for code tasks, "concatenate" for simple aggregation."#;

pub struct Orchestrator {
    /// Provider used for both planning and sub-agent execution
    provider_config: ProviderConfig,
    /// Secret env vars to strip from bash
    secret_env_vars: Vec<String>,
    /// System prompt for sub-agents
    system_prompt: Option<String>,
    /// Threshold: skip planning for short prompts (char count)
    min_prompt_length_for_planning: usize,
}

impl Orchestrator {
    pub fn new(
        provider_config: ProviderConfig,
        secret_env_vars: Vec<String>,
        system_prompt: Option<String>,
    ) -> Self {
        Self {
            provider_config,
            secret_env_vars,
            system_prompt,
            min_prompt_length_for_planning: 200,
        }
    }

    /// Main entry: analyze, split, execute, merge.
    pub async fn run(
        &self,
        prompt: &str,
        session: &mut Session,
    ) -> Result<OrchestratorResult, AgentError> {
        // Short prompts → skip planning, run directly
        if prompt.len() < self.min_prompt_length_for_planning {
            info!(prompt_len = prompt.len(), "Short prompt, skipping planning");
            return self.run_single(prompt, session).await;
        }

        // Step 1: Plan
        let plan = match self.plan(prompt).await {
            Ok(plan) => plan,
            Err(e) => {
                warn!("Planning failed ({e}), falling back to single agent");
                return self.run_single(prompt, session).await;
            }
        };

        info!(
            complexity = ?plan.complexity,
            subtask_count = plan.subtasks.len(),
            merge_strategy = ?plan.merge_strategy,
            reasoning = ?plan.reasoning,
            "Task plan created"
        );

        // Small tasks → just run directly (no overhead)
        if plan.complexity == Complexity::Small || plan.subtasks.len() <= 1 {
            return self.run_single(prompt, session).await;
        }

        // Step 2: Execute subtasks (respecting dependencies)
        let results = self.execute_plan(&plan).await;

        // Step 3: Merge results
        let merged = self.merge_results(&plan, &results).await?;

        // Push the final result into the session
        session.push(Message::User {
            content: prompt.to_string(),
        });
        session.push(Message::Assistant {
            content: vec![ContentBlock::Text {
                text: merged.clone(),
            }],
        });

        let total_usage = results.iter().fold(Usage::default(), |acc, r| Usage {
            input_tokens: acc.input_tokens + r.usage.input_tokens,
            output_tokens: acc.output_tokens + r.usage.output_tokens,
        });

        Ok(OrchestratorResult {
            final_text: Some(merged),
            plan: Some(plan),
            subtask_results: results,
            total_usage,
        })
    }

    /// Fallback: run as a single agent (no planning overhead).
    async fn run_single(
        &self,
        prompt: &str,
        session: &mut Session,
    ) -> Result<OrchestratorResult, AgentError> {
        let provider = crate::provider::create_provider(self.provider_config.clone());
        let tools = crate::tools::create_default_registry(&self.secret_env_vars);
        let agent = Agent::new(provider, tools, self.system_prompt.clone());

        let result = agent.run(prompt, session).await?;

        Ok(OrchestratorResult {
            final_text: result.final_text,
            plan: None,
            subtask_results: vec![],
            total_usage: result.total_usage,
        })
    }

    /// Step 1: Ask LLM to create a plan.
    async fn plan(&self, prompt: &str) -> Result<TaskPlan, PlanError> {
        let provider = crate::provider::create_provider(self.provider_config.clone());

        let messages = vec![Message::User {
            content: format!("Analyze this task and create an execution plan:\n\n{prompt}"),
        }];

        let response = provider
            .chat(&messages, &[], Some(PLANNER_SYSTEM_PROMPT))
            .await
            .map_err(|e| PlanError::ProviderError(e.to_string()))?;

        // Extract text from response
        let text = response
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        // Parse JSON (handle markdown code blocks)
        let json_str = extract_json(&text);
        let plan: TaskPlan =
            serde_json::from_str(json_str).map_err(|e| PlanError::ParseError(e.to_string()))?;

        // Validate
        if plan.subtasks.is_empty() {
            return Err(PlanError::EmptyPlan);
        }

        // Validate dependency references
        let ids: Vec<&str> = plan.subtasks.iter().map(|s| s.id.as_str()).collect();
        for task in &plan.subtasks {
            for dep in &task.depends_on {
                if !ids.contains(&dep.as_str()) {
                    return Err(PlanError::InvalidDependency(dep.clone()));
                }
            }
        }

        Ok(plan)
    }

    /// Step 2: Execute subtasks with dependency-aware parallelism.
    async fn execute_plan(&self, plan: &TaskPlan) -> Vec<SubTaskResult> {
        use std::collections::{HashMap, HashSet};

        let mut results: HashMap<String, SubTaskResult> = HashMap::new();
        let mut completed: HashSet<String> = HashSet::new();
        let total = plan.subtasks.len();

        while completed.len() < total {
            // Find ready tasks (all dependencies satisfied)
            let ready: Vec<&SubTask> = plan
                .subtasks
                .iter()
                .filter(|t| {
                    !completed.contains(&t.id)
                        && t.depends_on.iter().all(|d| completed.contains(d))
                })
                .collect();

            if ready.is_empty() {
                warn!("Deadlock detected: no tasks ready but {} remaining", total - completed.len());
                break;
            }

            info!(
                ready_count = ready.len(),
                completed = completed.len(),
                total,
                "Executing batch of subtasks"
            );

            // Execute ready tasks in parallel
            let handles: Vec<_> = ready
                .into_iter()
                .map(|task| {
                    let task_id = task.id.clone();
                    let task_desc = task.description.clone();
                    let task_prompt = task.prompt.clone();
                    let provider_config = self.provider_config.clone();
                    let secret_env_vars = self.secret_env_vars.clone();
                    let system_prompt = self.system_prompt.clone();

                    // Inject dependency results into prompt if needed
                    let full_prompt = if task.depends_on.is_empty() {
                        task_prompt
                    } else {
                        let mut parts = vec![task_prompt.clone()];
                        parts.push("\n\n--- Results from prerequisite tasks ---\n".to_string());
                        for dep_id in &task.depends_on {
                            if let Some(dep_result) = results.get(dep_id) {
                                parts.push(format!(
                                    "\n[{}] {}:\n{}\n",
                                    dep_id,
                                    dep_result.description,
                                    dep_result.output.as_deref().unwrap_or("(no output)")
                                ));
                            }
                        }
                        parts.join("")
                    };

                    tokio::spawn(async move {
                        info!(task_id = %task_id, desc = %task_desc, "Starting subtask");

                        let provider = crate::provider::create_provider(provider_config);
                        let tools = crate::tools::create_default_registry(&secret_env_vars);
                        let agent = Agent::new(provider, tools, system_prompt);

                        // Each subtask gets its own session (isolated)
                        let mut sub_session = Session::new("_orchestrator");

                        match agent.run(&full_prompt, &mut sub_session).await {
                            Ok(result) => {
                                info!(task_id = %task_id, "Subtask completed");
                                SubTaskResult {
                                    id: task_id,
                                    description: task_desc,
                                    output: result.final_text,
                                    usage: result.total_usage,
                                    success: true,
                                    error: None,
                                }
                            }
                            Err(e) => {
                                warn!(task_id = %task_id, error = %e, "Subtask failed");
                                SubTaskResult {
                                    id: task_id,
                                    description: task_desc,
                                    output: None,
                                    usage: Usage::default(),
                                    success: false,
                                    error: Some(e.to_string()),
                                }
                            }
                        }
                    })
                })
                .collect();

            // Wait for all parallel tasks in this batch
            for handle in handles {
                match handle.await {
                    Ok(result) => {
                        completed.insert(result.id.clone());
                        results.insert(result.id.clone(), result);
                    }
                    Err(e) => {
                        warn!("Task join error: {e}");
                    }
                }
            }
        }

        // Return results in original subtask order
        plan.subtasks
            .iter()
            .filter_map(|t| results.remove(&t.id))
            .collect()
    }

    /// Step 3: Merge subtask results according to the plan's strategy.
    async fn merge_results(
        &self,
        plan: &TaskPlan,
        results: &[SubTaskResult],
    ) -> Result<String, AgentError> {
        match plan.merge_strategy {
            MergeStrategy::Concatenate => {
                let merged = results
                    .iter()
                    .map(|r| {
                        format!(
                            "## {}\n\n{}",
                            r.description,
                            r.output.as_deref().unwrap_or("(no output)")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n---\n\n");
                Ok(merged)
            }
            MergeStrategy::Summarize | MergeStrategy::CodeMerge => {
                // Use LLM to synthesize results
                let provider = crate::provider::create_provider(self.provider_config.clone());

                let context = results
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        format!(
                            "### Subtask {} — {}\n\n{}",
                            i + 1,
                            r.description,
                            r.output.as_deref().unwrap_or("(failed)")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n---\n\n");

                let merge_instruction = match plan.merge_strategy {
                    MergeStrategy::Summarize => {
                        "Synthesize the following subtask results into a coherent, unified response. Remove redundancy and present a clear summary."
                    }
                    MergeStrategy::CodeMerge => {
                        "Merge the following code changes from subtasks into a coherent final result. Resolve any conflicts and ensure consistency."
                    }
                    _ => unreachable!(),
                };

                let messages = vec![Message::User {
                    content: format!("{merge_instruction}\n\n{context}"),
                }];

                let response = provider
                    .chat(&messages, &[], None)
                    .await
                    .map_err(|e| AgentError::Provider(e))?;

                let text = response
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");

                Ok(text)
            }
        }
    }
}

// ─── Result Types ────────────────────────────────────────────────

#[derive(Debug)]
pub struct OrchestratorResult {
    pub final_text: Option<String>,
    pub plan: Option<TaskPlan>,
    pub subtask_results: Vec<SubTaskResult>,
    pub total_usage: Usage,
}

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("provider error: {0}")]
    ProviderError(String),
    #[error("failed to parse plan: {0}")]
    ParseError(String),
    #[error("plan has no subtasks")]
    EmptyPlan,
    #[error("invalid dependency reference: {0}")]
    InvalidDependency(String),
}

// ─── Helpers ─────────────────────────────────────────────────────

/// Extract JSON from a string that might be wrapped in ```json blocks.
fn extract_json(text: &str) -> &str {
    let trimmed = text.trim();

    // Try to find ```json ... ``` block
    if let Some(start) = trimmed.find("```json") {
        let after = &trimmed[start + 7..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }

    // Try to find ``` ... ``` block
    if let Some(start) = trimmed.find("```") {
        let after = &trimmed[start + 3..];
        if let Some(end) = after.find("```") {
            return after[..end].trim();
        }
    }

    // Try to find raw JSON object
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return &trimmed[start..=end];
        }
    }

    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_raw() {
        let input = r#"{"complexity":"small","subtasks":[]}"#;
        assert_eq!(extract_json(input), input);
    }

    #[test]
    fn test_extract_json_code_block() {
        let input = "Here's the plan:\n```json\n{\"complexity\":\"large\"}\n```\nDone.";
        assert_eq!(extract_json(input), r#"{"complexity":"large"}"#);
    }

    #[test]
    fn test_extract_json_with_surrounding_text() {
        let input = "Sure! {\"complexity\":\"small\"} That's it.";
        assert_eq!(extract_json(input), r#"{"complexity":"small"}"#);
    }

    #[test]
    fn test_plan_deserialization() {
        let json = r#"{
            "complexity": "medium",
            "subtasks": [
                {
                    "id": "research",
                    "description": "Research the topic",
                    "prompt": "Find information about X",
                    "depends_on": []
                },
                {
                    "id": "write",
                    "description": "Write the report",
                    "prompt": "Write a report based on the research",
                    "depends_on": ["research"]
                }
            ],
            "merge_strategy": "summarize",
            "reasoning": "Split into research and writing phases"
        }"#;

        let plan: TaskPlan = serde_json::from_str(json).unwrap();
        assert_eq!(plan.complexity, Complexity::Medium);
        assert_eq!(plan.subtasks.len(), 2);
        assert_eq!(plan.subtasks[1].depends_on, vec!["research"]);
    }
}
