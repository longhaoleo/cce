---
name: experiment-phase-summary
description: Use when the user wants to summarize a completed, paused, active, or checkpointed experiment phase by reconstructing goals, workflow, code paths, outputs, findings, limitations, and next steps from repository evidence, then writing a README, retrospective, log entry, or handoff note.
---

# Experiment Phase Summary

Use this skill when the user wants to capture the logic and outcomes of an experiment phase in a reusable document.

This skill is broader than archive-only cleanup. It applies to:

- active experiment branches that need a checkpoint summary
- recently completed phases that need a retrospective
- paused side tracks that need a handoff note
- historical experiments that should be archived honestly

## Workflow

1. Orient to the current project mainline.
   Read the root `README.md` and nearby index docs such as `log/README.md` so you understand the current recommended workflow and where this experiment phase sits relative to it.

2. Classify the phase before writing.
   Decide whether the target is:
   - active and continuing,
   - completed for now,
   - paused,
   - exploratory side path,
   - or archival.

   Do not assume archive status by default.

3. Reconstruct the phase from evidence.
   Inspect the most relevant:
   - entry scripts,
   - configs,
   - outputs and artifacts,
   - metrics files and plots,
   - logs and retrospectives,
   - target directory README,
   - and, when useful, local git status or recent changes.

   Base claims on what the repository actually shows.

4. Explain the logic in execution order.
   Prefer documenting:
   - phase goal or hypothesis,
   - setup and inputs,
   - main execution flow,
   - important transformations or intermediate representations,
   - metrics and how to interpret them,
   - outputs and how to read them,
   - results, caveats, and unresolved questions.

5. Preserve the real process.
   Avoid writing a fake clean-room narrative that hides iteration. Summaries should retain:
   - what was tried,
   - what changed during the phase,
   - what worked,
   - what failed or stayed inconclusive,
   - and what the next sensible move is.

6. Write the right document for the situation.
   Choose the best target:
   - directory `README.md` for explaining a subproject or experiment folder,
   - log entry for chronological retrospective,
   - handoff note for active continuation,
   - index pointer when discoverability matters.

## Default Summary Shape

When writing or expanding a phase summary, prefer this structure:

- current status and relation to mainline
- phase goal and scope
- key files and outputs
- detailed logic in execution order
- findings, failures, and confidence level
- what this phase means for the project
- next steps or restart guidance

## Documentation Standards

- Distinguish fact from inference.
- Explain metrics carefully, especially when normalization, aggregation, heuristics, or third-party tooling can distort interpretation.
- If the phase is diagnostic rather than intervention-ready, say so explicitly.
- If the phase is still active, make continuation state explicit rather than writing it like a closed retrospective.
- If the phase should be archived, explain why it does not belong in the current mainline.

## Editing Guidance

- Default to adding or improving the most local useful document first.
- Add a short index pointer only when it improves navigation.
- Do not move or delete bulky outputs unless explicitly requested.
- Prefer preserving historical evidence over over-cleaning.

## Prompt Template

If the user wants a reusable prompt version of this workflow, read:

- `references/prompt-template.md`
