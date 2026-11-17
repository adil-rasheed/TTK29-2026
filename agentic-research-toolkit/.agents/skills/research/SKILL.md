---
name: research
description: Run the full research pipeline on an idea PDF (or resume a projects/<slug> folder), stopping at checkpoints A, B and C.
---

Read `AGENTS.md`, then follow `.claude/skills/research/SKILL.md` exactly.
Codex cannot launch the agents in `.claude/agents/` as separate subagents: for each
stage, read that agent's file and carry it out as a clearly labelled pass.
The idea PDF or project folder (and an optional `--auto`) is in the user's message.
