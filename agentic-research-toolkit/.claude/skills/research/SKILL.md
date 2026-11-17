---
name: research
description: Run the full research pipeline on an idea PDF — idea, literature, gap, questions, critique, course-based solutions, implementation, results, critique, report, review — calling the agents in .claude/agents/ in order and stopping at checkpoints A, B and C. Use with an idea PDF, or a projects/<slug> folder to resume.
argument-hint: <idea.pdf | projects/<slug>> [--auto]
---

# Research coordinator

You run the pipeline in `AGENTS.md`. You do not do the stages yourself: you call
the agent for each stage, check its output, keep the project README up to date,
and stop at the checkpoints. Arguments: $ARGUMENTS

## Start or resume
- **A PDF** → create `projects/<slug>/` (short kebab-case name from the idea),
  copy the PDF in as `idea.pdf`, create `README.md` from the template below.
- **A `projects/<slug>` folder** → read its `README.md`; continue at the first
  unticked stage.
- **"Redo stage NN"** → re-run that agent only; list which later stages are now out of date.

## Run each stage
Call the stage's agent (Agent tool, `subagent_type` = agent name) with:
the project path, the input files, the output file, and anything the agent
must know (e.g. the chosen sections of 06 for the critic at 09).
Then:
1. Check the output file exists and does what the agent file asks.
2. For critic stages, add a **Response** section under the critique: what you
   will change and why.
3. Tick the stage in `README.md` and update **Next step**.
4. Tell the student in ≤ 5 lines: what was done, the file, what to look at.

Without subagents (e.g. Codex): read the agent's file and do it yourself as a
clearly labelled pass ("critic pass (same model)" for the critic).

## Checkpoints
- **A** after 05: the revised questions are at the end of `05_rq_critique.md`.
  Ask which to pursue and why. Copy the chosen questions into README Decision A —
  they are authoritative from then on.
- **B** after 06: ask which solution(s) to implement (always with the baseline) and why.
- **C** after 09: ask the student to explain the main result and its biggest limitation in their own words.

Summarise the options with evidence, recommend, ask **one** question, and **stop**.
Record the answer and reason in the student's words.

## Going back
If a critic finds a major flaw, go back to the stage that needs fixing (05 → 04,
09 → 06/07/08), redo it, and note why in README *Notes*. Stage 11 fixes text only;
anything needing a new run goes back to 08.

## --auto (instructor demos)
Do not stop at checkpoints: take your recommendation and record it as
*auto-selected*. At C write "Main result (agent summary — no student answer)".
Act on each critique's major issues at most once (one loop back per critique);
list anything left in README *Notes*.

## README template
```markdown
# <Project title>
Idea: idea.pdf · Started: <date>

## Progress
- [ ] 01 Idea  - [ ] 02 Literature  - [ ] 03 Gap  - [ ] 04 Questions
- [ ] 05 Question critique (A)  - [ ] 06 Solutions (B)  - [ ] 07 Implementation
- [ ] 08 Results  - [ ] 09 Results critique (C)  - [ ] 10 Report  - [ ] 11 Review

## Decisions
- A — research question(s): …  Why: …
- B — solution(s): …  Why: …
- C — main result in my words: …

## Notes
## Next step
```
