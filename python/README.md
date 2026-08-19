# Course — Python side

Explainer notebooks, programming assignments, and the tooling that keeps the
LaTeX side honest. **Every figure in the slides and the book is generated here** —
the notebooks are the single source of truth for figures and for every number
quoted in the teaching material.

## Layout

```
environment.yml     reproducible conda environment for every module
pyutils/            shared helpers imported by notebooks (plot style, save_dual)
tools/
  sync_figures.py   copies figures to the LaTeX side and audits the result
  lint_course.py    checks notation, colours, and light/dark figure usage
NN-name/
  notebooks/        the teaching notebooks (jupytext-paired .py + .ipynb)
  assignment/       programming assignment (nbgrader source = solution version)
  data/             datasets for this module
  figures/          plots written by the notebooks (git-ignored, regenerable)
```

Module folders mirror the LaTeX side (`00-...` … `30-...`) so the two stay aligned.

## Modules, in teaching order

Folder numbers record the order modules were **written**. The course is taught in
six parts; this is the order to read them in.

**Introduction**

`00-introduction` · `01-physics-based-modeling` · `02-data-driven-modeling`

**Part I — Multivariate methods**

`03-least-squares` → `04-multiple-linear-regression` → `05-pca` →
`06-principal-component-regression` → `07-partial-least-squares` →
`08-independent-component-analysis`

**Part II — Nonlinear methods** (unsupervised first, then supervised)

`09-clustering` → `10-tsne` → `11-support-vector-machines` → `12-decision-trees`

**Part III — Time series**

`13-time-series-forecasting` → `14-time-series-classification`

**Part IV — Deep learning based methods**

`15-deep-learning-practice` → `16-convolutional-networks` → `17-sequence-models`
→ `18-diffusion-models`

**Part V — Hybrid models**, ordered by how the physics enters — from the
equations themselves down to physics present only in the training data:

| Module | Physics enters as… |
|---|---|
| `19-reduced-order-modeling` | the equations, projected onto a data basis |
| `20-corrective-source-term` | a residual the data must correct |
| `21-equation-discovery` | the object being recovered |
| `22-physics-informed-nn` | a penalty in the loss |
| `23-dynamic-mode-decomposition` | structure imposed on an operator |
| `24-compressed-sensing` | a basis for reconstructing a field |
| `25-deep-operator-learning` | the solver that generated the data |
| `26-fourier-neural-operator` | the solver that generated the data |

**Part VI — Control algorithms**

`27-model-predictive-control` → `28-koopman-control` → `29-reinforcement-learning`
→ `30-safe-learning-control`

## Setup

```bash
conda env create -f environment.yml
```

```bash
conda activate scientific-ml-course
```

```bash
jupyter lab
```

That installs Python 3.11 with numpy, scipy, matplotlib, pandas, scikit-learn,
statsmodels, jupyterlab, jupytext, nbgrader, plus torch and pysindy via pip.

If the environment already exists and you only need a missing package:

```bash
conda env update -f environment.yml --prune
```

**Anything a notebook imports must be listed in `environment.yml`.** Installing a
package ad hoc makes the repo unreproducible on another machine — that is the one
rule this side of the repo really has.

## Running the notebooks

Notebooks are stored as jupytext-paired `.py` (percent format) **and** `.ipynb`.
The `.py` file is the one that produces readable diffs; the `.ipynb` carries the
executed outputs.

Open either in JupyterLab and run it. To execute one from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace 05-pca/notebooks/pca_monitoring.ipynb
```

To regenerate an `.ipynb` after editing its `.py`:

```bash
jupytext --to notebook 05-pca/notebooks/pca_monitoring.py
```

Every notebook is seeded and ends with asserts that enforce the numbers quoted on
the slides. If a notebook fails, a claim in the teaching material is wrong — that
is the intended alarm.

### Regenerating every figure

```bash
for f in */notebooks/*.ipynb; do jupyter nbconvert --to notebook --execute --inplace "$f" || echo "FAILED $f"; done
```

Then sync them to the LaTeX side (below). Total runtime is a few minutes; each
individual notebook is under ~2 minutes by design.

## Figure bridge to the LaTeX side

Notebooks write figures via `pyutils.plotstyle.save_dual()`, which renders each
figure **twice**:

| file | for | style |
|------|-----|-------|
| `name.png`      | book chapters, theory sheets | dark ink, white background |
| `name_dark.png` | slides                        | light ink, transparent background |

`python/*/figures/` is git-ignored (regenerable); the copies committed under
`latex/modules/*/figures/` are what Overleaf and a fresh clone use. Move them
across with the bridge:

```bash
python tools/sync_figures.py
```

```bash
python tools/sync_figures.py --check --strict
```

The first copies and reports; the second is the CI-style audit — it writes
nothing and exits non-zero on:

* **STALE** — the LaTeX copy differs from the notebook output (run the sync)
* **MISSING** — an `\includegraphics` target that exists on neither side
* **ORPHAN** — a LaTeX figure no notebook produces (`--strict` makes it an error)

Run the plain sync after regenerating figures, and the `--check --strict` form
before committing.

## Linting

```bash
python tools/lint_course.py
```

Checks the whole repo for `\bm`, shortcut macros, `\definecolor` outside
`common/`, and slides that use a light figure variant where they should use
`_dark`. Should print `lint clean`.

## Conventions

* **`jupytext`** — pair every notebook: `jupytext --set-formats ipynb,py:percent nb.ipynb`.
* **`nbgrader`** — author each assignment with solution cells tagged
  `### BEGIN SOLUTION` / `### END SOLUTION`; generate the student copy with
  `nbgrader generate_assignment`. This mirrors the `\ifshowsolutions` toggle on
  the LaTeX side.
* **Plot style** — always `from pyutils import set_plot_style, save_dual, fg, muted`,
  and build each figure inside a zero-argument function passed to `save_dual`
  (matplotlib binds rcParams when artists are created, so the figure must be
  re-drawn for each theme).
* **mathtext** — figure labels follow the same convention as the text:
  `\boldsymbol{x}` for a vector, `\mathbf{A}` for a matrix. matplotlib's
  mathtext renders both correctly; it does not support `\bm`.
