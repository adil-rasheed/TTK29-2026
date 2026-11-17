// Builds agentic-research-toolkit/research-workflow-slide.pptx
// Rebuild after editing:   npm install pptxgenjs   then   node tools/build-slides.js
// Colours are written as hex so no theme post-processing is needed.
const path = require("path");
const pptxgen = require("pptxgenjs");

const OUT = path.join(__dirname, "..", "research-workflow-slide.pptx");

// Palette (hex, no "#")
const COLORS = {
  dark: "1B2B30", white: "FFFFFF", teal: "0E4A52", mist: "EAF2F3",
  orange: "D9692B", grey: "5E7A80", pale: "B8CDD0",
};
const HEAD_FONT = "Cambria", BODY_FONT = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "From an idea to a reviewed report";
pres.theme = { headFontFace: HEAD_FONT, bodyFontFace: BODY_FONT };
// Map the old scheme-colour names onto the hex palette
const C = {
  text1: COLORS.dark, text2: COLORS.teal, background1: COLORS.white, background2: COLORS.mist,
  accent1: COLORS.teal, accent2: COLORS.orange, accent3: COLORS.grey, accent6: COLORS.pale,
};

pres.defineSlideMaster({
  title: "TITLE_ONLY",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.8,
        fontSize: 36, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
});

pres.addSection({ title: "Workflow" });
const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Workflow" });
s.addText("From an idea to a reviewed report in 11 stages", { placeholder: "title" });

// subtitle with the command
s.addText([
  { text: "Paste one prompt into Claude or ChatGPT (Codex). Nine agents do the legwork; ", options: {} },
  { text: "you decide.", options: { bold: true, color: C.accent2 } },
], { x: 0.6, y: 1.15, w: 12.1, h: 0.45, fontSize: 18, color: C.text1, margin: 0, isTextBox: true, objectName: "subtitle" });

const phases = [
  { name: "Understand it", stages: [["01", "Refine the idea"], ["02", "Literature review"], ["03", "Knowledge gap"]] },
  { name: "Ask the question", stages: [["04", "Research questions"], ["05", "Critique the questions", "A"]] },
  { name: "Solve it", stages: [["06", "Candidate solutions", "B"], ["07", "Implement in Python"], ["08", "Run experiments"], ["09", "Critique the results", "C"]] },
  { name: "Communicate it", stages: [["10", "Write the report"], ["11", "Review the report"]] },
];
const critic = new Set(["05", "09", "11"]);
const agent = {
  "01": "idea-refiner", "02": "literature-reviewer", "03": "gap-finder", "04": "question-framer",
  "05": "critic", "06": "solution-architect", "07": "implementer", "08": "experiment-runner",
  "09": "critic", "10": "report-writer", "11": "critic",
};
const cpText = { A: "Checkpoint A: pick question", B: "Checkpoint B: pick solution", C: "Checkpoint C: explain result" };

const X0 = 0.6, GAP = 0.3, W = (12.13 - 3 * GAP) / 4, Y = 1.95, H = 3.95, ROW = 0.78;
phases.forEach((p, i) => {
  const x = X0 + i * (W + GAP);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: Y, w: W, h: H, rectRadius: 0.12,
    fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `phase-${i + 1}` });
  s.addText(`${i + 1}`, { x: x + 0.2, y: Y + 0.18, w: 0.3, h: 0.36, fontSize: 14, bold: true,
    color: C.accent3, margin: 0, isTextBox: true });
  s.addText(p.name, { x: x + 0.45, y: Y + 0.18, w: W - 0.6, h: 0.36, fontSize: 16, bold: true,
    color: C.text2, margin: 0, isTextBox: true, objectName: `phase-${i + 1}-name` });
  p.stages.forEach(([n, label, cp], j) => {
    const ry = Y + 0.75 + j * ROW;
    const cx = x + 0.2;
    s.addShape(pres.shapes.OVAL, { x: cx, y: ry, w: 0.5, h: 0.5,
      fill: { color: cp ? C.accent2 : C.accent1 }, line: { color: cp ? C.accent2 : C.accent1 }, objectName: `stage-${n}-dot` });
    s.addText(n, { x: cx, y: ry, w: 0.5, h: 0.5, fontSize: 13, bold: true, color: C.background1,
      align: "center", valign: "middle", margin: 0, isTextBox: true });
    const runs = [
      { text: label, options: { bold: true, fontSize: 14, color: C.text1, breakLine: true } },
      { text: agent[n], options: { fontFace: "Courier New", fontSize: 11, color: C.accent3, breakLine: !!cp || n === "11" } },
    ];
    if (cp) runs.push({ text: cpText[cp], options: { fontSize: 10, bold: true, color: C.accent2 } });
    if (n === "11") runs.push({ text: "then report-writer fixes", options: { fontSize: 10, color: C.accent3 } });
    s.addText(runs, { x: cx + 0.62, y: ry - 0.06, w: W - 1.0, h: 0.62, valign: "middle", margin: 0, isTextBox: true, objectName: `stage-${n}-text` });
  });
});

// bottom: three principles
const pr = [
  ["Grounded", "Solutions are built on the course notebooks and slides"],
  ["Honest", "Real citations; every number comes from code that ran"],
  ["You decide", "The pipeline stops at A, B and C and asks you why"],
];
const PW = (12.13 - 2 * GAP) / 3;
pr.forEach(([h, t], i) => {
  const x = X0 + i * (PW + GAP);
  s.addShape(pres.shapes.OVAL, { x, y: 6.3, w: 0.22, h: 0.22, fill: { color: i === 2 ? C.accent2 : C.accent1 },
    line: { color: i === 2 ? C.accent2 : C.accent1 } });
  s.addText([
    { text: h + "  ", options: { bold: true, color: C.text2 } },
    { text: t, options: { color: C.text1 } },
  ], { x: x + 0.35, y: 6.12, w: PW - 0.4, h: 0.6, fontSize: 14, valign: "middle", margin: 0, isTextBox: true, objectName: `principle-${i + 1}` });
});

s.addNotes(
  "One command starts the workflow: /research followed by the idea PDF. " +
  "Four phases, 11 stages; each stage writes one numbered file in the project folder. " +
  "A critic reviews the questions (05), the results (09) and the report (11). " +
  "At checkpoints A, B and C the agent stops and the student must choose and explain: the research question, the solution, and the meaning of the result. " +
  "Assessment focuses on those decisions, not on the polish of the report. Open agentic-research-toolkit/index.html for the student guide."
);

// ---------- Slide 2: get started ----------
pres.addSection({ title: "Get started" });
const g = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Get started" });
g.addText("Get started in three steps", { placeholder: "title" });
g.addText([
  { text: "Everything is in ", options: {} },
  { text: "agentic-research-toolkit/index.html", options: { fontFace: "Courier New", bold: true } },
  { text: ". Open it in any browser and follow along.", options: {} },
], { x: 0.6, y: 1.15, w: 12.1, h: 0.45, fontSize: 18, color: C.text1, margin: 0, isTextBox: true, objectName: "subtitle-g" });

const GY = 1.75, GH = 4.45, GW = (12.13 - 2 * 0.3) / 3;
const gx = i => 0.6 + i * (GW + 0.3);
function gcard(i, title, sub, accent) {
  const x = gx(i);
  g.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: GY, w: GW, h: GH, rectRadius: 0.12,
    fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `g-card-${i + 1}` });
  g.addShape(pres.shapes.OVAL, { x: x + 0.2, y: GY + 0.2, w: 0.42, h: 0.42, fill: { color: accent }, line: { color: accent } });
  g.addText(String(i + 1), { x: x + 0.2, y: GY + 0.2, w: 0.42, h: 0.42, fontSize: 14, bold: true,
    color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  g.addText([
    { text: title, options: { bold: true, fontSize: 16, color: C.text2, breakLine: true } },
    { text: sub, options: { fontSize: 12, color: C.accent3 } },
  ], { x: x + 0.75, y: GY + 0.14, w: GW - 0.9, h: 0.6, valign: "middle", margin: 0, isTextBox: true });
  return x;
}
function bullets(x, y, w, h, items, name) {
  g.addText(items.map((t, k) => ({ text: t, options: { bullet: { indent: 14 }, breakLine: k < items.length - 1 } })),
    { x, y, w, h, fontSize: 13, color: C.text1, paraSpaceAfter: 4, valign: "top", margin: 0, isTextBox: true, objectName: name });
}

// 1: set up
let x1 = gcard(0, "Set up (once)", "about 20 minutes", C.accent1);
g.addText("Claude", { x: x1 + 0.22, y: GY + 0.88, w: GW - 0.4, h: 0.28, fontSize: 13, bold: true, color: C.text2, margin: 0, isTextBox: true });
bullets(x1 + 0.22, GY + 1.16, GW - 0.4, 1.05, ["Paid plan (Pro or higher)", "Claude desktop app → Code tab", "Local → Open folder… → the toolkit", "Windows: install Git first"], "g-claude");
g.addText("ChatGPT · Codex", { x: x1 + 0.22, y: GY + 2.3, w: GW - 0.4, h: 0.28, fontSize: 13, bold: true, color: C.text2, margin: 0, isTextBox: true });
bullets(x1 + 0.22, GY + 2.58, GW - 0.4, 0.8, ["Plan with Codex (Plus or higher)", "ChatGPT desktop app → Codex", "New chat → open the toolkit folder"], "g-codex");
g.addText([
  { text: "Both: ", options: { bold: true } },
  { text: "the course Python environment (step 1a in index.html)", options: {} },
], { x: x1 + 0.22, y: GY + 3.55, w: GW - 0.4, h: 0.6, fontSize: 12, color: C.text1, valign: "top", margin: 0, isTextBox: true });

// 2: write the idea
let x2 = gcard(1, "Write your idea", "1–2 page PDF", C.accent1);
bullets(x2 + 0.22, GY + 0.92, GW - 0.4, 2.6, [
  "The problem: what you want to find out",
  "The system and what is known",
  "Course connections",
  "A possible comparison and baseline",
  "What you don't know",
  "Constraints: time, laptop, data",
], "g-idea");
g.addText("Save it in the agentic-research-toolkit folder.", { x: x2 + 0.22, y: GY + 3.55, w: GW - 0.4, h: 0.6,
  fontSize: 12, italic: true, color: C.accent3, valign: "top", margin: 0, isTextBox: true });

// 3: run it
let x3 = gcard(2, "Run it", "copy and paste", C.accent2);
bullets(x3 + 0.22, GY + 0.92, GW - 0.4, 1.0, [
  "In index.html, type your project name and PDF path",
  "Copy the prompt, paste it into the chat",
], "g-run");
g.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x3 + 0.22, y: GY + 1.95, w: GW - 0.44, h: 1.45, rectRadius: 0.08,
  fill: { color: C.background1 }, line: { color: C.accent6, width: 1 }, objectName: "g-prompt-box" });
g.addText("Read AGENTS.md. Run the full research pipeline on my idea ideafile.pdf … Stop at checkpoints A, B and C and wait for my answer.",
  { x: x3 + 0.35, y: GY + 2.02, w: GW - 0.7, h: 1.3, fontFace: "Courier New", fontSize: 11, color: C.text1,
    valign: "middle", margin: 0, isTextBox: true, objectName: "g-prompt" });
g.addText([
  { text: "At A, B and C: ", options: { bold: true, color: C.accent2 } },
  { text: "answer in your own words, with your reason.", options: {} },
], { x: x3 + 0.22, y: GY + 3.55, w: GW - 0.4, h: 0.6, fontSize: 12, color: C.text1, valign: "top", margin: 0, isTextBox: true });

g.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 6.4, w: 12.13, h: 0.6, rectRadius: 0.12,
  fill: { color: C.background1 }, line: { color: C.accent6, width: 1 }, objectName: "g-strip" });
g.addText([
  { text: "Prefer one step at a time? ", options: { bold: true, color: C.text2 } },
  { text: "Open any stage in index.html and copy its prompt — each one calls a single agent.", options: {} },
], { x: 0.85, y: 6.4, w: 11.7, h: 0.6, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true });

g.addNotes(
  "Step 1, once: get a paid Claude plan and the Claude desktop app (Code tab, Local, open the agentic-research-toolkit folder; on Windows install Git first), " +
  "or a ChatGPT plan with Codex and the ChatGPT desktop app (Codex mode, new chat, open the same folder). Both need the course conda environment. " +
  "Section 1c of index.html has a prompt that checks the setup. Step 2: a one- or two-page idea PDF with six headings, saved in the toolkit folder. " +
  "Step 3: in index.html, type the project name and PDF path; every prompt fills in. Paste the Option 1 prompt to run the whole pipeline, " +
  "and answer the checkpoints in your own words. Or open each stage and paste its prompt to run one agent at a time."
);

// ---------- Slide 3: where everything lives ----------
pres.addSection({ title: "Files" });
const f = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Files" });
f.addText("Where the agents, skill and your work live", { placeholder: "title" });
f.addText([
  { text: "Everything is in ", options: {} },
  { text: "agentic-research-toolkit/", options: { fontFace: "Courier New", bold: true } },
  { text: ". To change an agent, edit its one file.", options: {} },
], { x: 0.6, y: 1.15, w: 12.1, h: 0.45, fontSize: 18, color: C.text1, margin: 0, isTextBox: true, objectName: "subtitle-2" });

function cardFrame(slide, { x, y, w, h, num, title, where, accent }) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12,
    fill: { color: C.background2 }, line: { color: C.background2 }, objectName: `card-${num}` });
  slide.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.2, w: 0.42, h: 0.42, fill: { color: accent }, line: { color: accent } });
  slide.addText(String(num), { x: x + 0.2, y: y + 0.2, w: 0.42, h: 0.42, fontSize: 14, bold: true,
    color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  slide.addText([
    { text: title, options: { bold: true, fontSize: 16, color: C.text2, breakLine: true } },
    { text: where, options: { fontFace: "Courier New", fontSize: 11, color: C.accent3 } },
  ], { x: x + 0.75, y: y + 0.14, w: w - 0.9, h: 0.6, valign: "middle", margin: 0, isTextBox: true });
}
function fileRow(slide, x, y, w, path, desc, name) {
  slide.addText([
    { text: path, options: { fontFace: "Courier New", bold: true, fontSize: 12, color: C.text1, breakLine: true } },
    { text: desc, options: { fontSize: 12, color: C.accent3 } },
  ], { x, y, w, h: 0.46, valign: "top", margin: 0, isTextBox: true, objectName: name });
}

const FY = 1.75, FH = 4.4;
// card 1: start here
const c1 = { x: 0.6, w: 3.7 };
cardFrame(f, { ...c1, y: FY, h: FH, num: 1, title: "Start here", where: "agentic-research-toolkit/", accent: C.accent1 });
[["AGENTS.md", "the map: pipeline, rules"], ["index.html", "setup + copy-paste prompts"],
 ["COURSE_CATALOG.md", "what the course covers"], [".claude/settings.json", "lets agents read ../slides, ../python"],
 ["examples/", "example idea PDF"]].forEach(([p, d], j) => fileRow(f, c1.x + 0.22, FY + 0.92 + j * 0.62, c1.w - 0.4, p, d, `c1-${j}`));

// card 2: agents
const c2 = { x: 4.6, w: 4.85 };
cardFrame(f, { ...c2, y: FY, h: FH, num: 2, title: "The agents and the skill", where: ".claude/  ·  .agents/", accent: C.accent1 });
f.addText(".claude/agents/  — one file per agent", { x: c2.x + 0.22, y: FY + 0.92, w: c2.w - 0.4, h: 0.3,
  fontFace: "Courier New", bold: true, fontSize: 12, color: C.text1, margin: 0, isTextBox: true });
const names = ["idea-refiner", "literature-reviewer", "gap-finder", "question-framer", "critic",
  "solution-architect", "implementer", "experiment-runner", "report-writer"];
names.forEach((nm, k) => {
  const col = k < 5 ? 0 : 1, row = k < 5 ? k : k - 5;
  f.addText(nm + ".md", { x: c2.x + 0.4 + col * 2.2, y: FY + 1.28 + row * 0.3, w: 2.15, h: 0.28,
    fontFace: "Courier New", fontSize: 11, color: nm === "critic" ? C.accent2 : C.accent3, margin: 0, isTextBox: true });
});
fileRow(f, c2.x + 0.22, FY + 2.95, c2.w - 0.4, ".claude/skills/research/SKILL.md", "/research — coordinator: runs the agents, stops at A/B/C", "c2-skill");
fileRow(f, c2.x + 0.22, FY + 3.6, c2.w - 0.4, ".agents/skills/research/SKILL.md", "$research for Codex — points to the same coordinator", "c2-codex");

// card 3: project
const c3 = { x: 9.75, w: 2.98 };
cardFrame(f, { ...c3, y: FY, h: FH, num: 3, title: "Your project", where: "projects/<topic>/", accent: C.accent2 });
[["README.md", "progress + decisions A, B, C"], ["01_idea.md … 11_*.md", "one file per stage"],
 ["references.bib", "literature"], ["code/", "Python scripts"], ["runs/ · figures/", "raw results and plots"]]
  .forEach(([p, d], j) => fileRow(f, c3.x + 0.22, FY + 0.92 + j * 0.62, c3.w - 0.4, p, d, `c3-${j}`));

f.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 6.35, w: 12.13, h: 0.65, rectRadius: 0.12,
  fill: { color: C.background1 }, line: { color: C.accent6, width: 1 }, objectName: "course-strip" });
f.addText([
  { text: "One folder up, read but never changed:  ", options: { bold: true, color: C.text2 } },
  { text: "../slides/", options: { fontFace: "Courier New", bold: true } },
  { text: " lectures   ·   ", options: {} },
  { text: "../python/", options: { fontFace: "Courier New", bold: true } },
  { text: " notebooks, pyutils helpers, environment.yml", options: {} },
], { x: 0.85, y: 6.35, w: 11.7, h: 0.65, fontSize: 14, color: C.text1, valign: "middle", margin: 0, isTextBox: true });

f.addNotes(
  "Everything lives in agentic-research-toolkit/, and students open that folder. " +
  "(1) AGENTS.md is the map: the pipeline table and rules; index.html is the guide; COURSE_CATALOG.md lists what the course covers; " +
  ".claude/settings.json lets the agents read the course one folder up. " +
  "(2) .claude/agents/ holds the nine agents, one self-contained file each — open one to see its job, inputs, outputs and rules. " +
  "The critic is reused for questions, results and report. .claude/skills/research is the /research coordinator; .agents/skills/research is the Codex entry. " +
  "(3) Each idea gets projects/<topic>/ with one numbered file per stage, code, runs and figures."
);

pres.writeFile({ fileName: OUT }).then(() => console.log("wrote", OUT));
