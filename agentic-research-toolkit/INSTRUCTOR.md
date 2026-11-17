# Instructor notes

## What students learn
The research loop: refine → literature → gap → questions → critique →
solutions → implement → results → critique → report → review. Each step is done
by a named agent, so students see what each step is *for*, and spend their own
effort on the decisions (A, B, C) and on reacting to the critiques.

## How it is built
- `AGENTS.md` — the map. `CLAUDE.md` just imports it.
- `.claude/agents/*.md` — the nine agents, each self-contained. **Edit an agent
  here to change its behaviour**; nothing is duplicated elsewhere.
- `.claude/skills/research/SKILL.md` — `/research`, the coordinator: calls the
  agents in order, keeps the project README, stops at A/B/C, handles `--auto`.
- `.agents/skills/research/SKILL.md` — `$research` for Codex; a pointer to the
  coordinator. Codex has no subagents, so it runs each agent file as a labelled pass.
- `.claude/settings.json` — `additionalDirectories: [".."]` lets the agents read
  `../slides` and `../python` without permission prompts.
- `COURSE_CATALOG.md` — grounds the solutions. After adding a module, ask:
  "Update COURSE_CATALOG.md with the new notebooks and slides."

## In class
1. **Demo (20 min).** Run `/research examples/sensor-placement/ideafile.pdf --auto`
   beforehand (≈ 1.5 h, mostly Monte Carlo) and walk through `projects/`.
   `--auto` takes the agent's recommendation at each checkpoint and marks it.
2. **Agents one by one.** Have students open two agent files, then call one by
   hand, e.g. "Use the critic on projects/<slug>/04_research_questions.md".
3. **Students write a 1–2 page idea PDF** and run the pipeline on it, ideally in
   pairs, discussing at checkpoints A, B and C.

Students only need `index.html`: section 1 sets up Claude (desktop app, Code tab,
paid plan; Windows also needs Git) or ChatGPT (desktop app, Codex mode), section 3
has copy-paste prompts that fill in their project name and PDF path. Let them do
the setup and the "check that everything works" prompt before class.

## Slides
`research-workflow-slide.pptx` (three slides for the class) is built by
`tools/build-slides.js`. To rebuild after editing it:

```bash
npm install pptxgenjs
node tools/build-slides.js
```

## What to assess
- **Decisions A, B, C** in the project `README.md`: own reasoning, and sound?
- **Reaction to critiques 05 and 09**: engaged, or ignored?
- **Traceability**: check one number in the report against `runs/`, one citation.
- **Oral check**: ask the critic's "question for the student" from stage 09.

Do not grade the polish of `10_report.md`; the agents write well regardless.

## Known limits
- Literature access: many publishers block automated reading, so sources are
  often read at abstract level. The agents record the read level; check a few.
- Code runs locally in the `ham4digitaltwin` environment; heavy training must be scaled down.
- The critic is independent of the author's reasoning but is the same model;
  it helps, it does not replace you.
