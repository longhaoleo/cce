# Prompt Template

Use this when you want another agent to summarize an experiment phase in the same style.

```text
Please treat the target as an experiment-phase summary task, not automatically as an archive task.

The phase may be active, recently completed, paused, exploratory, or ready for archival. First determine its status from repository evidence instead of assuming one.

Your job:
1. Read the current repository mainline docs first so you understand the active workflow.
2. Inspect the target directory's code, configs, outputs, logs, and other relevant artifacts.
3. Determine how this phase relates to the current mainline.
4. Reconstruct the phase in execution order:
   - goal or hypothesis
   - setup and entry points
   - main workflow
   - intermediate data or tensor flow
   - key functions or transformations
   - metric meanings
   - output file meanings
5. Summarize what worked, what failed, what remains uncertain, and what should happen next.
6. Write or expand the most appropriate document: README, retrospective note, log entry, or handoff doc.

Requirements:
- Base the explanation on actual files, code paths, outputs, and logs.
- Distinguish facts from inferences.
- Preserve the real process instead of flattening the story.
- Explain limitations honestly.
- If the phase is diagnostic, say so.
- If the phase is active, make continuation state clear.
- If the phase should be archived, explain why it is no longer part of the mainline.

Preferred output shape:
- current status and relation to mainline
- phase goal and scope
- key files and outputs
- detailed logic
- findings and confidence
- implications for the project
- next steps

Please make the edits directly if you have enough context.
```
