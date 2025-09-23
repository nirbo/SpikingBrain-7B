# P1 Critical Rules (Must Follow)

## Glossary:

- statefile: a new file you'll create and manage named "llm.state"

## IMPORTANT:
- In case of a Python codebase, you MUST check whether a venv exists in the cwd and make sure to activate it before attempting to run python code.
- If no venv exists, you MUST approach the user and let them know; Under no circumstances are you to try and install packages outside of a venv.

## Code Structure Requirements

1. **File Headers**: Every .go file MUST have a detailed header comment explaining:
   - All functions in the file Related files and dependencies
   - Context and purpose
   - Integration points
2. **Header Line Limit**: Header comments MUST end at or before line 200, code starts at line 201 or before if the header is shorter.
3. **Code Comments**: Every line of code MUST have comments explaining what it does
4. **File Size Limit**: Files can NEVER exceed 700 lines of code
5. **Reading Strategy**: When reading files, ALWAYS read the header first, then locate specific sections without reading the entire file
6. **Git management** When git commit/push are in order, overwrite or create a file named ".git-commit.txt", add an extensive decription of the changes done, git add -F .git-commit.txt and push.
7. **HTTP API ENDPOINTS DOCUMENTATION** IF YOU UPDATE AN HTTP API CODE, MAKE SURE THAT SWAGGER ANNOTATION ARE CORRECT.
8. **EXCEPTION MANAGEMENT** WE HAVE TO PROVIDE MEANINGFUL EXCEPTION MANAGEMNT TO ALL ACTIONS THAT CAN ISOLATE AND IDENTIFY AN ISSUE WITH AS MUCH DETAIL AS POSSIBLE.
9. **Always revalidate the structure of the database before making a code change to a query!**

#### 1. Codebase Structure and Modularity

1.1 **Function and File Size Limits**: No function shall exceed 150 lines of code, and no file shall exceed 700 lines. If a function or file approaches these limits, refactor it into smaller, modular components before proceeding.
1.2 **Feature Isolation**: All code must be modular, with each feature encapsulated to prevent dependencies. Adding or modifying a feature must not require changes to unrelated features. Use clear interfaces and separation of concerns to enforce this.  
1.3 **Documentation Standards**: Every function must include Swagger annotations with detailed descriptions of inputs, outputs, and purpose. Additionally, provide in-code comments explaining the logic, intent, and context of each major block. Comments must be concise yet comprehensive, covering at least 20% of the code lines for clarity.  
1.4 **Refactoring for Modularity**: Before implementing changes, verify that the affected code adheres to modularity and size limits. If not, refactor first and log the refactoring steps in the statefile.

#### 2. Minimal and Surgical Changes

2.1 **Minimal Change Principle**: Always make the smallest possible changes required to resolve a specific issue or implement a task. Avoid altering unrelated code or introducing unnecessary modifications.  
2.2 **Surgical Precision**: If minimal changes are not feasible due to codebase constraints, immediately inform the user, explain the issue, and request guidance on how to proceed. Do not proceed without explicit user approval.  
2.3 **Change Validation**: After making changes, validate that only the intended code was modified by comparing the before and after states. Log this validation in the statefile.

#### 3. Tool Usage and Parameter Validation

3.1 **Tool Call Integrity**: Before executing any tool (e.g., edit_file), verify that all required parameters are present and valid. If parameters are missing or invalid, halt execution, notify the user, and request clarification.  
3.2 **Single Approval Request**: When seeking user approval for tool usage, ask only once. Accept any positive response (e.g., “yes,” “go ahead”, “approved”) as final authorization. Do not repeatedly prompt the user for approval.  
3.3 **Tool Context Preservation**: Log all tool calls, their parameters, and outcomes in the statefile to maintain context across interactions.

#### 4. User Interaction and Assumption Avoidance

4.1 **No Assumptions**: Never make assumptions about the codebase, task requirements, or user intent. If any detail is unclear, ask the user for clarification before proceeding.  
4.2 **User as Authority**: Treat the user as the definitive source of truth for all task-related information. Log user clarifications in the statefile to prevent repeated queries.  
4.3 **Proactive Clarification**: If the task description or codebase context is ambiguous, proactively ask the user targeted questions to resolve ambiguity before taking any actions.

#### 5. Code Quality and Best Practices

5.1 **Language-Specific Standards**: You MUST safely and strictly follow the best-practices of the programming language in-use, and utilize techniques such as zero-copy / zero-allocation to have the code as efficient and performant as it possibly can. Prioritize clarity, correctness, and maintainability.
5.2 **Code Commenting**: Summarize the purpose and functionality of every major code block in comments. When modifying existing code, read adjacent comments to understand context and ensure changes align with the original intent.  
5.3 **Error Handling**: Include robust error handling in all code to prevent crashes or undefined behavior. Log error conditions in the statefile for debugging purposes.  
5.4 **Code Review Simulation**: Before finalizing changes, simulate a code review by checking for readability, adherence to standards, and potential bugs. Log the results in the statefile.

#### 6. Context Window Management

6.1 **Context Awareness**: Continuously monitor the context window usage. At 85% capacity, complete the current action, update the statefile, and perform context compaction before proceeding.  
6.2 **Context Compaction**: During compaction, summarize the current task state, retain only essential information (e.g., task goal, recent actions, and pending steps), and archive non-critical details in the statefile.  
6.3 **Context Revalidation**: After each user interaction or context compaction, re-read the statefile to confirm alignment with the original task and current progress. If misalignment is detected, notify the user and request clarification.  
6.4 **Context Window Safeguard**: If the context window is at risk of overflow (e.g., >90% capacity), pause all actions, save the current state, and compact the context immediately. Log this event in the statefile.

#### 7. Statefile Management

7.1 **Statefile Creation**: At the start of every task, create a statefile named llm.state to track the task’s progress, context, and metadata. If the file exists, use it while leaving all pre-existing content unchanged.
7.2 **Statefile Structure**: The statefile must include:

- The original task description provided by the user.
- A detailed plan breaking the task into the smallest logical steps.
- A log of completed actions, including tool calls, code changes, and user interactions.
- A list of pending steps with estimated completion criteria.
- A summary of the task’s current state and any unresolved issues.  
  7.3 **Statefile Updates**: Update the statefile after every action, ensuring it reflects the latest task state, code changes, and user inputs.  
  7.4 **Statefile Revalidation**: Before each user interaction, read the statefile to confirm the task’s context, current step, and goal. If the statefile is outdated or inconsistent, notify the user and request guidance.  
  7.5 **Task Completion Summary**: Upon completing a task, append a summary to the statefile detailing what was done, why it was done, and any follow-up recommendations. This ensures quick resumption if the task is revisited.  
  7.6 **Statefile Backup**: Periodically save a backup of the statefile (e.g., llm.state.bak) to prevent data loss in case of errors or context overflow.

#### 8. Task Planning and Execution

8.1 **Task Decomposition**: Break every task into the smallest possible logical steps. Each step must be atomic, with a clear input, output, and success criterion.  
8.2 **Planning Phase**: Before starting a task, create a detailed plan in the statefile outlining each step, its purpose, and dependencies. Update the plan as new information arises.  
8.3 **Step-by-Step Execution**: Execute one step at a time, validating its success before proceeding. Log each step’s outcome in the statefile.  
8.4 **Progress Tracking**: Maintain a progress tracker in the statefile, indicating completed steps, current step, and remaining steps. Update this tracker after each action.  
8.5 **Deviation Detection**: If execution deviates from the planned steps, halt immediately, log the deviation in the statefile, and consult the user for guidance.

#### 9. Continuous Self-Monitoring

9.1 **Context Drift Prevention**: Periodically check that all actions align with the original task goal by referencing the statefile’s task description. If drift is detected, pause and notify the user.  
9.2 **Statefile Integrity Check**: Before and after each statefile update, verify its integrity (e.g., no missing sections, no corruption). If issues are found, revert to the backup and inform the user.  
9.3 **Task Relevance Check**: Before executing any step, confirm its relevance to the current task by cross-referencing the statefile’s plan and task description.  
9.4 **Performance Monitoring**: Track the model’s performance (e.g., context window usage, execution time) and log anomalies in the statefile for future optimization.

#### 10. Error Recovery and Fallback

10.1 **Error Logging**: Log all errors, including their context, cause, and impact, in the statefile. Include a timestamp and the affected task step.  
10.2 **Fallback Mechanism**: If an error prevents progress (e.g., invalid tool call, context overflow), revert to the last known good state using the statefile backup and notify the user.  
10.3 **User Notification**: For any unrecoverable error, provide the user with a clear explanation, the current statefile contents, and a recommended course of action.  
10.4 **Retry Logic**: For transient errors (e.g., temporary tool failure), implement a retry mechanism with a maximum of three attempts. Log each attempt in the statefile.

## Agent-Specific Instructions

- Obey this file’s scope for the whole repo; keep patches targeted.
- Prefer existing Make targets; if missing, add them with concise recipes.
- Do not introduce breaking changes without discussion; format and lint before proposing patches.
- Always re-read files before making changes in them to avoid inconsistencies.

---

## Persistent Agent Context (maintained by the agent)

- Project Goal: Build a self-hosted clone of selected ByteRover capabilities with high performance and strong modularity.
- Language/Env: Go 1.25.0 on macOS (cross-arch builds required).
- Vector DB Decision: Qdrant (Apache-2.0), containerized via Docker/Compose; programmatic startup via testcontainers-go.
- Research Artifacts: `research/byterover_capabilities.md`, `research/vector_db_selection.md`.
- Implementation Priorities: Knowledge/Reflection memories, embeddings + vector search, REST API with Swagger, event bus, schema validation before query changes, comprehensive error handling.
- Statefile: Track all actions in `llm.state` (plan, logs, tool calls, deviations, backups).
