# TTK29 Course Catalog (what this checkout actually covers)

This file is a reference for an AI research agent proposing project solutions. Ground proposals ONLY in the methods, systems, notebooks and datasets listed here.
It lists only what exists in this checkout: 7 slide decks (`slides/`), 7 notebook modules (`python/00, 01, 02, 20, 21, 22, 24`), `python/pyutils/`. `python/README.md` mentions modules 03-19 and 23, 25-30 that are NOT present; do not cite them (some slides do, see Caveats).
To regenerate when modules are added: re-read `slides/*.pdf` (`pdftotext -layout`; use physical 1-based PDF pages, not the slide-footer numbers, which differ) and `python/*/notebooks/*.py` (or the `.ipynb` when no paired `.py`; ignore `*_executed.ipynb`), then add/update rows in the Lectures table, the per-module sections, the Recurring-systems table, pyutils, Method chooser and Datasets.
All page numbers below are physical PDF pages. Paths are repo-relative to `python/` unless they start with `slides/`.

## 1. Lectures

| Deck (`slides/`) | Pages | Topic | Key pages |
|---|---|---|---|
| `L00_Introduction.pdf` | 37 | Digital twins; PBM vs DDM vs hybrid; course map | Capability ladder 0-5: 6-12; industry challenges: 15-20; why no single paradigm wins: 18, 22-24; hybridization idea (squeeze the residual): 25-27; course map: 29; running examples (vessel, heat/flow, turbine): 31-33; setup: 35-37 |
| `L01_PBM.pdf` | 48 | Physics-based modeling: derive, discretize, solve, analyze | Recipe: 4-7 (four benchmarks table p7); heat PDE: 8-17 (FTCS p13, von Neumann stability p14, PCB circuit p15, cable p16-17); Navier-Stokes: 18-25 (projection method p22, CFL/Poisson cost p23, where PBM struggles p25); vessel ODE: 26-35 (RK4 order p32, sensitivity to uncertain coefficients p35); wind turbine ODE: 36-42 (Cp p38, power curve p40, turbulent wind p41, Cp sensitivity p42); strengths/limits: 44-45 |
| `L02_DDM.pdf` | 53 | Data-driven modeling: regression to deep nets, generalization, UQ, surrogates | Regression/NN/universal approximation: 8-13; training loop, loss, SGD: 15-18; train/val/test, over/underfitting: 20-22; regularization: 24-25; UQ (ensembles, MC dropout): 27-29; heat surrogate: 31-33; NS surrogate: 35-37; vessel surrogate (one-step, drift, OOD cliff, more data, rollout fine-tune): 39-44; turbine power-curve surrogate: 46-48; **limits (data-hungry, poor extrapolation, no conservation): 50**; which later method fixes which limit: 51 |
| `L03-PINN.pdf` | 74 | Physics-informed neural networks | Problem setting ODE/PDE/residual/forward vs inverse: 9-13; autodiff: 15-18; **core algorithm** (architecture 20, collocation 21, Adam then L-BFGS 22, pseudocode 23, pendulum code 24); heat PINN: 28-33; vessel PINN: 35-42 (identifiability p42); wind turbine/wake PINN: 44-49; **failure modes and cheat sheet: 51-55**; variants cPINN/XPINN/vPINN/ensembles: 57-67; **choosing a variant: 68**; workflow/libraries/open problems: 70-72 |
| `L04-COSTA.pdf` | 65 | CoSTA: Corrective Source Term Approach (hybrid PBM + learned correction) | Why PBM fails: 4-5; formulation (exact corrective source, labels): 7-9; **four error cases: 11-17 (table p17)**; using correction in solver / zero-correction property: 19-20; training/val/test protocol: 22-24; heat benchmark: 26-34; vessel: 36-42; wind turbine with synthetic erosion: 44-49; stability: 51; **limits: 52**; discovery link: 53; **checklist: 54**; one-rod six-method comparison: 59-60; appendices 61-65 |
| `L05-CSOSP.pdf` | 75 | Compressed sensing and optimal sensor placement | Motivation/Nyquist: 4-7; sparsity and compressibility: 9-17; recovery problem, incoherence, RIP, l1 relaxation: 19-31; **phase transition m ~ k log(n/k): 32-33**; algorithms (lasso, ISTA/FISTA, debiasing, OMP): 35-40; beating Nyquist: 41-43; portrait: 44-45; single-pixel camera, MRI: 47-53; **when CS fails: 54**; optimal sensor placement (gappy POD 58, D-optimal 63, QR pivoting 64-66, vs exhaustive 67, oversampling 68, wind-farm wake 69-70); **what placement buys / when it stops mattering: 71**; measured-record table: 74 |
| `L06-DataDrivenEquationDiscovery.pdf` | 74 | Equation discovery: SINDy, weak form, PDE-FIND, symbolic regression | Goal/paradigms: 4-7; SINDy + STLSQ + SINDYc: 9-14; derivative noise, weak form, ensemble-SINDy: 16-22; PDE-FIND/weak PDE: 24-25; symbolic regression (trees, GP, GEP, linear scaling, Pareto front): 27-36; SR landscape/KAN/snapping: 37-39; vessel discovery: 41-47; wind (drivetrain, closed-loop trap, wake law): 49-53; heat PDE discovery: 55-57; **failure modes: 59-64**; SINDy family (SR3, constrained, Bayesian, SINDy-PI): 66-69; **method selection table: 71**; limits: 72 |

## 2. Modules and notebooks

Convention: every notebook below has a jupytext-paired `.py` except those in modules 01 and 02 (`.ipynb` only). "Slides" = deck and physical pages. Seeds are fixed in all notebooks. Notebooks with `RUN_*`/`DEFAULTS` parameter cells can be re-run with other values. Figures go to `<module>/figures/` (git-ignored, regenerable).

### Module 00 - introduction (`python/00-introduction/`)

| Notebook | Method / system | What it demonstrates | Defines | Data | Slides |
|---|---|---|---|---|---|
| `00-introduction/notebooks/intro_three_paradigms.ipynb` | PBM vs DDM vs hybrid on the 1-DOF vessel surge coast-down `m du/dt = -d1 u - d2 u|u|` (m=10, d1=1, d2=2); PBM uses d2=1.2 (40% off) | One figure for the whole course: train on t in [0,4], predict t in [4,8]; imperfect physics extrapolates with bias, small NN fit to noisy record fails outside data, hybrid (PBM + learned correction) wins | `rk4`, `rhs_hybrid`, `err`, `plot_intro_three_paradigms`; uses `pinn_core.MLP/train` (3000 steps) | simulated in-notebook | L00 p22-27, p37 |

Assignment (not a method demo): `00-introduction/assignment/assignment_setup.ipynb` environment check.

### Module 01 - physics-based modeling (`python/01-physics-based-modeling/`)

All use the derive -> discretize -> solve -> analyze recipe and write reference arrays to `01-.../data/`. Solvers live in `pyutils/benchmarks/`.

| Notebook | Method | System | Demonstrates | Key code | Data written | Slides |
|---|---|---|---|---|---|---|
| `01-physics-based-modeling/notebooks/heat_diffusion.ipynb` | Explicit FTCS finite differences, von Neumann stability | 2-D heat PDE, unit square, alpha=1e-4, 61x61 grid; PCB with four chips and fan (segments `pcb_circuit.png`) | Field evolution, stable dt, sensitivity to alpha, PCB cooling with/without fan | `heat.HeatProblem`, `heat.solve`, `heat.stable_dt`, `solve_board`, `plot_circuit` | `data/heat_reference.npz` (x, y, T, alpha); reads `data/pcb_circuit.png` | L01 p8-15 |
| `01-physics-based-modeling/notebooks/heat_cable.ipynb` | Radial heat conduction (finite differences), steady + transient, vs analytic | Wind-farm export cable cross-section (copper conductor, XLPE insulation, water); I^2R heating | Core temperature, time to ~95% steady state, ampacity (~2160 A for 90 C XLPE limit), dynamic rating | `cable.Cable`, `cable.field_2d`, `minutes_to` | none | L01 p16-17 |
| `01-physics-based-modeling/notebooks/navier_stokes_cavity.ipynb` | Chorin projection method (predictor, pressure Poisson, corrector) | 2-D incompressible lid-driven cavity, Re=100, 41x41 | Primary vortex, streamlines, pressure | `ns.lid_driven_cavity`, `ns.cavity_dt` | `data/cavity_reference.npz` (x, y, u, v, p, Re) | L01 p18-24 |
| `01-physics-based-modeling/notebooks/navier_stokes_vortex.ipynb` | Pseudo-spectral (FFT) vorticity-streamfunction | Two like-signed vortices, periodic box, Re=2000, 128x128 (movie run on coarser grid) | Vortex merger; where PBM is costly/struggles | `ns.vortex_merger` | `data/vortex_reference.npz` (X, Y, omega, Re) | L01 p25 |
| `01-physics-based-modeling/notebooks/vessel_ode.ipynb` | Newton-Euler 3-DOF (Fossen form), RK4 / Euler / RK2, eigenvalue stability, coefficient sensitivity | Surface vessel (surge-sway-yaw), turning circle, zig-zag | Maneuvers, order of accuracy, stability, sensitivity to "amber" (uncertain) coefficients later identified/corrected | `vessel.simulate`, `vessel.turning_circle`, `vessel.zigzag`, local `integrate` | `data/vessel_reference.npz` (t, Z, maneuver) | L01 p26-35 |
| `01-physics-based-modeling/notebooks/wind_turbine.ipynb` | Rotor angular-momentum ODE `J dw/dt = T_aero - T_gen`, Heier Cp(lambda, beta), MPPT + pitch PI control, OU turbulent wind | NREL 5-MW-like turbine (model defined notebook-locally; same physics as `pyutils.benchmarks.turbine`) | Cp surface, steady power curve (region II/III), response to turbulence, power-curve sensitivity to Cp scale | `Cp`, `simulate`, `turbulent_wind`, `power_curve` | none | L01 p36-42 |

Assignments: `01-.../assignment/assignment_heat.ipynb` (plate gluing time), `assignment_navier_stokes.ipynb` (cavity Re sweep, vortex merge vs separation), `assignment_vessel.ipynb` (zig-zag overshoot, added-mass sensitivity).

### Module 02 - data-driven modeling (`python/02-data-driven-modeling/`)

All torch, float64, `pyutils.nets.MLP`-style tanh nets; data are generated in-notebook from the Module 01 solvers (no files in `02-.../data/`, which is empty).

| Notebook | Method | System | Demonstrates | Key code / training | Slides |
|---|---|---|---|---|---|
| `02-data-driven-modeling/notebooks/ddm_regression_basics.ipynb` | Polynomial regression, MLP, weight decay, gradient descent/SGD, early stopping | 1-D `y = sin(2 pi x)` + noise (n=60; 36/12/12 split) | Under/overfitting sweep, regularization, GD surface, full training loop, universal approximation | `polyfit_eval`, `fit_mlp`, `descend`, `fit_mlp_ex`; mini-batch 4, up to 4000 epochs, patience 500 | L02 p8-25 |
| `02-data-driven-modeling/notebooks/ddm_uncertainty.ipynb` | Deep ensemble (M=6) and MC dropout (p=0.1, 100 passes) | 1-D sine with a deliberate data gap (two islands) | Epistemic uncertainty grows in gap/extrapolation | `train_net` | L02 p27-29 |
| `02-data-driven-modeling/notebooks/ddm_heat_surrogate.ipynb` | MLP surrogate `T(x,y,alpha)` (3-96-96-96-1), 4-member ensemble | 2-D heat PDE, 41x41, t*=800 s, hot left/right, cold top/bottom | Interpolation good, extrapolation in alpha (1e-5, 1.7e-4) fails; train alphas {3,4,5,7,8}e-5 | `train_surrogate` (8000 Adam steps), `predict_field`, `spread`, `physics_check` | L02 p31-33 |
| `02-data-driven-modeling/notebooks/ddm_ns_surrogate.ipynb` | MLP `(x,y)->(u,v)` (2-64-64-64-2) | Lid-driven cavity Re=400, 61x61; sensors only for y<=0.65 | Fills unseen band under the lid badly (partial observation trap) | uses `lid_driven_cavity` | L02 p35-37 |
| `02-data-driven-modeling/notebooks/ddm_vessel_surrogate.ipynb` | One-step map `dnu = f(nu, tau)` (width 64), autoregressive rollout, multi-step fine-tune | 3-DOF vessel (RK4 truth), dt=0.1 s, T=40 s, random smooth thrust maneuvers | One-step accuracy vs rollout drift vs OOD maneuver cliff; more data helps interpolation not structure; rollout fine-tuning (k=12) partial fix | `Surrogate`, `train_onestep` (400 epochs), `rollout`, `finetune_multistep`, `drift_curve` | L02 p39-44 |
| `02-data-driven-modeling/notebooks/ddm_wind_ode_surrogate.ipynb` | MLP `P(v)` (1-32-32-1) | Turbine steady power curve from the rotor ODE (89 wind speeds, 3-25 m/s); trained on region II only | Predicts ~9 MW in region III while real turbine is capped at 5 MW | `predict`, uses `turbine` benchmark | L02 p46-48 |
| `02-data-driven-modeling/notebooks/ddm_wind_power_curve.ipynb` | Method of bins (IEC) vs MLP with extra features (TI, density); width sweep; curtailment OOD | **Synthetic** SCADA (2500 10-min records, Weibull wind, TI, density, noise, 6% curtailment); ideal curve `min(0.5 rho A Cp v^3, P_rated)` | Binning vs ML, bias-variance with 50-record commissioning set, curtailment cliff | `steady_power`, `make_scada`, `fit`, `fit_width` | L02 p47 (and p46-48) |

Assignments: `02-.../assignment/assignment_ddm_regression.ipynb`, `assignment_ddm_surrogate.ipynb`.

### Module 20 - corrective source term / CoSTA (`python/20-corrective-source-term/`)

Common protocol in all five notebooks: split whole trajectories into train/val/test roles before building samples; label = exact discrete corrective source (residual of reference in the runnable solver); small MLP (hidden [40,40], or [32,32] for turbine) trained on local stencil/state; validation selects checkpoint (early stopping with patience); frozen model run as recursive rollout on held-out cases; PBM = same solver without correction. Results JSON written to `20-.../data/`.

| Notebook | Method / error case | System | Demonstrates | Key code | Data written | Slides |
|---|---|---|---|---|---|---|
| `20-corrective-source-term/notebooks/costa_heat_pde.ipynb` | CoSTA, complete worked protocol; backward-Euler PBM missing a source (Case 1) | 1-D heat, D=0.5, 41 nodes, manufactured solution `T*=1+a sin(pi x) e^-t`; train a in {0.5,1,2,2.5,3}, val {0.75,1.75,2.75}, interp test {1.25,2.25}, extrap {0.35,3.75,4.25} | Labels, checkpoint selection, recursive test, extrapolation in amplitude, 5-seed and feature (`stencil` vs `stencil+xt`) ablations | `step`, `reference`, `rollout`, `label`, `features`, `dataset`, `Correction`, `fit_costa`, `evaluate` | `data/costa_heat_results.json` | L04 p25-33 (p27, 30 cite it) |
| `20-corrective-source-term/notebooks/costa_error_cases.ipynb` | CoSTA Cases 1-4: inaccurate source/boundary data, inaccurate parameter (k(x) vs constant), discretization-only, both | 1-D heat variants (same amplitudes as above, 3 seeds) | What each label contains, which inputs are needed, where the correction is valid | `make_grid`, `make_solver`, `run_case`, `evaluate`, `pbm_errors` | `data/costa_cases_results.json` | L04 p10-17 (p11 cites it) |
| `20-corrective-source-term/notebooks/costa_heat_rod.ipynb` | CoSTA Case 2: PBM alpha=0.07 vs true 0.1; plus source-term case q=20 sin(2 pi x); data-only DNN baseline from 40 sensors | Canonical heat rod (`heat1d`, 128 pts, t_f=0.05, 60 train/20 val/20 test profiles + out-of-family "bump") | Same rod as PINN (Module 22) and equation-discovery (Module 21) for cross-method comparison; difference-basis vs raw stencil ablation | `lap`, `step`, `reference`, `fit_costa`, `Correction`, `rel_l2_final` | `data/costa_heat_rod_results.json` | L04 p34, p59-60 |
| `20-corrective-source-term/notebooks/costa_vessel_ode.ipynb` | CoSTA on ODE: learn missing drag force from velocity, tau; compared with a fair black-box DDM; filter chosen on validation for noisy data; alpha-scaling of correction | 3-DOF vessel, hidden drag `-[3 u|u|, 8 v|v|, 2 r|r|]`, dt=0.02, T=30 s; maneuvers: turning circles train, zig-zags test; extrapolation ZZ(160,80,0.35), TC(300,-25) | Transfer across maneuvers; interpolation vs leaving training state box; noise 0.004 m/s | `drag`, `accel_pbm`, `accel_ref`, `rk4_step`, `simulate_reference`, `replay`, `samples`, `residual_force` | `data/costa_vessel_results.json` (seed sweep ~2 min, off by default) | L04 p35-42, p61-65 |
| `20-corrective-source-term/notebooks/costa_wind_turbine.ipynb` | CoSTA on torque deficit; coverage and input (pitch) ablations | NREL-5MW-like turbine with **synthetic** erosion `Cp_true = Cp_nom (1 - 0.015 lambda)`; OU winds 9 m/s train, 11.5 m/s transfer test | Same-regime gain large; transfer above rated needs training coverage + right inputs | `T_nom`, `T_true`, `deficit`, `simulate`, `wind`, uses `TurbineParams` | `data/costa_turbine_results.json` (seed sweep ~1.5 min) | L04 p43-49 |

Assignments: `20-.../assignment/assignment_costa_heat.ipynb`, `assignment_costa_vessel.ipynb`.

### Module 21 - equation discovery (`python/21-equation-discovery/`)

All implemented FROM SCRATCH in numpy/scipy (own `stlsq`, `library`, GP/GEP); no notebook imports `pysindy` or PySR (they are only mentioned as production tools). No torch.

| Notebook | Method | System | Demonstrates | Key code | Slides |
|---|---|---|---|---|---|
| `21-equation-discovery/notebooks/ed_sindy_vessel.ipynb` | SINDy with STLSQ, threshold/Pareto sweep, ensemble-SINDy, weak form vs finite differences, validation by rollout, closing the loop on a learned correction | 3-DOF vessel, library `{1,u,v,r,|u|u,|v|v,|r|r,uv,ur,vr,tau1..3}` per equation; noise 0-10% | Recovers benchmark coefficients; inclusion probabilities; derivative error budget | `library`, `stlsq`, `weak_system`, `score`, `rk4`, `discovered_deriv` | L06 p9-22, p41-47 |
| `21-equation-discovery/notebooks/ed_failure_modes.ipynb` | Deliberately broken SINDy: incomplete library, symptoms, hidden variable (v unmeasured), fit is not the test | Same vessel/maneuver | Aliasing of missing |u|u term (omitted-variable formula), no plateau, structured residuals, extrapolated top speed 2.5x too high, energy check on Coriolis terms | `library`, `stlsq`, `rollout`, `sweep`, `steady`, `r2`, `coriolis_power` | L06 p59-64 (p60 cites it) |
| `21-equation-discovery/notebooks/ed_sindy_variants.ipynb` | SR3, constrained SINDy (energy identity), Bayesian SINDy vs bootstrap, SINDy-PI (implicit/rational) | Vessel (noise 2/5/10%, 5 seeds) | When each variant helps | `sr3`, `stlsq_c`, `posterior`, `weak_system`, `xd_pi` | L06 p66-69 (p69 cites it) |
| `21-equation-discovery/notebooks/ed_sindy_turbine.ipynb` | Parametric SINDYc (wind speed enters library), weak form | Turbine drivetrain `J dw/dt = T_aero - T_gen` region II (unknowns J and Cp(lambda)) | Closed-loop MPPT data cannot identify J (torque is collinear with a library column); dithering fixes it; test with unseen controller gain | `dither`, `simulate`, `design`, `cp_hat`, `model_rhs`, `weak` | L06 p49-51 |
| `21-equation-discovery/notebooks/ed_sindy_wind.ipynb` | STLSQ + ensemble over small library | Wake centreline deficit (U_inf=8, eddy viscosity 0.5, synthetic, 4% noise) and learned erosion correction (slope 0.015) | Rediscovers decay law x^-1/2 and erosion law delta = slope * lambda | `stlsq`, `erosion` | L06 p52-53 (cites it) |
| `21-equation-discovery/notebooks/ed_heat_rod.ipynb` | PDE-FIND with STLSQ (library {1,T,Tx,Txx,Txxx,T^2,T Tx}), weak form, source regression on functions of x | Canonical heat rod (`heat1d`, alpha=0.1, 20 training profiles); optional source q=20 sin(2 pi x) | Recovers alpha; single sine mode cannot identify alpha; missing source discovered; errors-in-variables attenuation, fixed by weak form | `trajectories`, `derivatives`, `library`, `stlsq`, `choose_threshold`, `march`, `identifiability`, `source_library`, `weak_system` | L06 p54-57, p62 |
| `21-equation-discovery/notebooks/ed_symbolic_regression.ipynb` | Tree genetic programming with linear scaling, constant optimization, Pareto archive/knee (PySR rule), bloat; snapping learned curve to formula | Hidden `x^2+2x`; wake decay law; vessel drag correction `-3 u|u|` | Library-free discovery; success rate over 10 seeds | `gp`, `fitness`, `crossover`, `mutate`, `front`, `knee`, `snap`, `n_trees` | L06 p27-36, p39 (p30, 39 cite it) |
| `21-equation-discovery/notebooks/ed_gene_expression_programming.ipynb` | Gene expression programming (head/tail rule, Karva decoding) | Same wake and drag tasks as above | Valid-by-construction chromosomes, neutral mutations | `decode`, `GEP` class, `scaled`, `orf_length` | L06 p32-33 (p33 cites it) |

Data written: `21-.../data/heat_rod_results.json`. Assignments: `assignment_ed_sindy.ipynb` (Duffing oscillator), `assignment_ed_symbolic.ipynb` (GEP), `assignment_ed_detective.ipynb` (three datasets with hidden defects), `assignment_ed_costa_link.ipynb` (turn learned correction into equation).

### Module 22 - physics-informed neural networks (`python/22-physics-informed-nn/`)

Torch float64, tanh MLP from `pyutils.pinn_core` (`MLP`, `grad`, `train`). Hard-IC ansatz in ODE cases. Seed-sweep JSON files are optional (loaded if sweeps not re-run).

| Notebook | Method | System | Demonstrates | Training cost | Slides |
|---|---|---|---|---|---|
| `22-physics-informed-nn/notebooks/pinn_autodiff.ipynb` | Autodiff mechanics (AD vs FD vs analytic; input derivs vs parameter grads; `create_graph`; ReLU failure on 2nd-order residual; Laplacian bugs) | Heat residual, wake residual | How residuals are built; why tanh not ReLU | small (ReLU/tanh heat PINNs 1500 steps) | L03 p15-18 |
| `22-physics-informed-nn/notebooks/pinn_heat_pde.ipynb` | Forward PINN (4 residual terms), Adam then L-BFGS test, inverse PINN for alpha, seed sweep | Heat rod u_t = alpha u_xx, `sin(pi x)` IC, exact solution known; inverse: 60 noisy interior points, alpha init 0.5 | Mesh-free solve scored on unseen grid; whether L-BFGS helps; recover alpha | 4000 Adam + 300 L-BFGS; inverse 4000; sweep ~6 min | L03 p27-33 |
| `22-physics-informed-nn/notebooks/pinn_vessel_ode.ipynb` | Forward ODE-PINN, inverse PINN (recover drag), identifiability, 3-DOF forward by time-marching windows, grey-box (misspecified term) PINN | Vessel: 1-DOF surge coast-down and 3-DOF | Excitation decides identifiability; time slabs for long horizon | fwd 4000; inv 3000; 12 windows x 1400; grey-box 6000 | L03 p34-42, p68 |
| `22-physics-informed-nn/notebooks/pinn_ensemble_vessel.ipynb` | Deep-ensemble inverse PINN (M=8, bootstrap noisy data) | 1-DOF coast-down m=10, d1=1 known, d2=2 unknown, 25 obs, sigma 0.03 | Uncertainty on identified drag | 8 x 5000 steps | L03 p66-67 |
| `22-physics-informed-nn/notebooks/pinn_wind_turbine.ipynb` | Forward ODE-PINN and inverse (inertia J, J_true=38.6e6) | Turbine drivetrain spin-up at 9 m/s, T=120 s (local formulas, same physics as benchmark) | Commissioning/inertia identification from noisy omega | fwd 14000; inv 6000 | L03 p47 |
| `22-physics-informed-nn/notebooks/pinn_wind_pde.ipynb` | Inverse PDE-PINN | Parabolic wake `U d_x d = nu_t d_yy d` (analytic heat-kernel truth), sparse noisy data, unknown eddy viscosity | Reconstruct deficit field and recover nu_t | 5000 steps; sweep ~5 min | L03 p48 |
| `22-physics-informed-nn/notebooks/xpinn_heat.ipynb` | cPINN/XPINN domain decomposition with interface losses | Steady two-material rod `(k u')'=0` with kink at interface | Single net reaches low loss but wrong answer; interface flux continuity fixes it | 4000 steps (x2 runs) | L03 p58-62, p68 |
| `22-physics-informed-nn/notebooks/vpinn_heat.ipynb` | vPINN (sin test functions, Gauss quadrature) vs strong form | `-u'' = delta(x-a)`, a=0.5 | Point load unrepresentable by collocation; variational form handles it | 5000 steps each | L03 p63-65, p68 |

Data: `22-.../data/pinn_heat_pde_seed_sweep.json`, `pinn_wind_pde_seed_sweep.json`. Assignments: `assignment_pinn_heat`, `assignment_pinn_vessel`, `assignment_pinn_wind`.

### Module 24 - compressed sensing and optimal sensor placement (`python/24-compressed-sensing/`)

numpy/scipy/sklearn only (no torch); each runs in seconds (phase transition ~40 s). Data: `24-.../data/portrait.png`.

| Notebook | Method | System | Demonstrates | Key code | Slides |
|---|---|---|---|---|---|
| `24-compressed-sensing/notebooks/cs_illustrations.ipynb` | Measurement-model pictures `y=Cx`, `x=Psi s` | Synthetic tones, portrait image | Matrix-block diagrams of CS; Fourier basis 1-D/2-D | `draw_vector`, `draw_matrix`, `plot_block_*`, `plot_fourier_*` | L05 p10-13, 19-22 |
| `24-compressed-sensing/notebooks/cs_sparsity.ipynb` | Sparsity/compressibility test (DCT, keep top 1/5/20%) | 2-tone signal (1024 nonzeros in time, 2 in DCT); portrait photo | Dense in one basis, sparse in another; JPEG argument | `keep_top`, `psnr` | L05 p14-17 |
| `24-compressed-sensing/notebooks/cs_regularization.ipynb` | l0/l1/l2/l_inf norms, lasso vs ridge geometry, soft threshold | 2-D norm balls; correlated-feature regression (60 samples) | Why l1 gives zeros, l2 does not | `soft_threshold`, `minimizers`, `plot_*` | L05 p26-31 |
| `24-compressed-sensing/notebooks/cs_phase_transition.ipynb` | l1 recovery (FISTA) over (m,k) grid; fit `m = C k log(n/k)` | Random sparse vectors, n=256 | Success phase transition; C=1.47 (R2=0.98) | `soft_threshold`, `fista`, `l1_recover` | L05 p32-33 |
| `24-compressed-sensing/notebooks/cs_recovery_fista.ipynb` | ISTA, FISTA, OMP from scratch (checked vs sklearn Lasso), debiasing, beating Nyquist | n=1024, k=20, m=256; two-tone 97/337 Hz sub-Nyquist | FISTA ~3.9x fewer iterations than ISTA; debiased error ~1e-15 | `ista`, `fista`, `omp`, `lasso_objective`, `iters_to` | L05 p35-43 |
| `24-compressed-sensing/notebooks/cs_uniform_vs_random.ipynb` | Uniform sinc vs random sampling + basis pursuit (linprog) | 4-tone signal, 256 points, 48 samples | Same budget: error 1.41 vs 6e-13 | `uniform_times`, `sinc_reconstruct`, `basis_pursuit` | L05 p41-43 |
| `24-compressed-sensing/notebooks/cs_portrait_fista.ipynb` | FISTA in 2-D DCT with matrix-free operators; vs interpolation; single-pixel camera with noisy detector | 64x64 portrait, random pixels | CS loses to interpolation on clean pixels, wins with noisy detector | `fista_pixels`, `fista_patterns`, `interp_pixels`, `psnr`, `relerr` | L05 p44-45, p47-51 |
| `24-compressed-sensing/notebooks/osp_qr_illustrations.ipynb` | POD (SVD) basis, gappy POD, D-optimal sensors, pivoted QR vs exhaustive search, oversampling, noise ellipses | Tiny synthetic 1-D field (n=16, r=5; n=32, r=4) | Greedy QR matches exhaustive optimum (35,960 sets) | `make_field`, `qr_sensors`, `logdet`, `recon_err` | L05 p56-68 |
| `24-compressed-sensing/notebooks/cs_sensor_placement.ipynb` | QR-pivot sensors on POD basis vs random sensors | Simulated meandering wind-farm wake, 3 turbines, 2560 grid points | 13 QR sensors: error 0.061 vs 4.22 random; random needs 48 sensors to match | `qr_sensors`, `reconstruct_error`, `recon_field` | L05 p69-71 |

Assignment: `24-.../assignment/assignment_cs.ipynb`.

## 3. Recurring systems (compare methods on the same system)

| System | Notebooks (module/name) |
|---|---|
| Vessel, 3-DOF (RK4 `vessel` benchmark) | PBM: 01 vessel_ode; DDM: 02 ddm_vessel_surrogate; CoSTA: 20 costa_vessel_ode; discovery: 21 ed_sindy_vessel, ed_failure_modes, ed_sindy_variants, ed_symbolic_regression (drag correction), ed_gene_expression_programming; PINN: 22 pinn_vessel_ode (3-DOF part) |
| Vessel surge coast-down, 1-DOF (`m du/dt=-d1 u-d2 u|u|`) | 00 intro_three_paradigms; 22 pinn_vessel_ode, pinn_ensemble_vessel |
| 2-D heat, unit square (FTCS) | PBM: 01 heat_diffusion; DDM: 02 ddm_heat_surrogate |
| Canonical 1-D heat rod, alpha=0.1, `heat1d` | PINN: 22 pinn_heat_pde; CoSTA: 20 costa_heat_rod; discovery: 21 ed_heat_rod (L04 p59-60 compares six methods on it, incl. non-present Modules 25-26) |
| Other 1-D heat variants | 20 costa_heat_pde (D=0.5 manufactured), costa_error_cases; 22 xpinn_heat (steady composite), vpinn_heat (point load) |
| Lid-driven cavity (Navier-Stokes) | PBM: 01 navier_stokes_cavity; DDM: 02 ddm_ns_surrogate |
| Vortex merger | 01 navier_stokes_vortex only |
| Wind turbine rotor ODE (NREL 5-MW-like) | PBM: 01 wind_turbine; DDM: 02 ddm_wind_ode_surrogate, ddm_wind_power_curve (synthetic SCADA); CoSTA: 20 costa_wind_turbine; discovery: 21 ed_sindy_turbine; PINN: 22 pinn_wind_turbine |
| Wind-turbine wake | PINN: 22 pinn_wind_pde, pinn_autodiff (residual); discovery: 21 ed_sindy_wind, ed_symbolic_regression, ed_gene_expression_programming; sensing: 24 cs_sensor_placement (meandering 3-turbine wake field) |
| Power-cable cross-section | 01 heat_cable only |
| Images / synthetic signals | 24 cs_* (portrait.png, tones) |
| Toy 1-D regression | 02 ddm_regression_basics, ddm_uncertainty |

## 4. pyutils (`python/pyutils/`)

Import with `sys.path.insert(0, '../..')` (notebooks do this) then `from pyutils ...`. From a project script in `agentic-research-toolkit/projects/<slug>/code/`, use `ROOT = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(ROOT / 'python'))`.

| File | Public API (one line each) | Used by notebooks here |
|---|---|---|
| `plotstyle.py` (re-exported by `pyutils`) | `set_plot_style(mode)`, `rc_params`, `save_dual(plot_fn, name, figdir)` light+dark figures, `fg()`, `muted()`, `COLORS_LIGHT/DARK` | all |
| `nets.py` | `MLP(layers)` tanh MLP, e.g. `[1,32,32,1]` | module 02 |
| `pinn_core.py` | `MLP` (re-export), `grad(y,x)` graph-retaining derivative, `train(loss_fn, params, steps, lr, decay_every, factor, log_every)` Adam loop | 00, 20, 22 |
| `dl.py` | `set_dl_defaults`, `get_device`, `count_params`, `TrainHistory`, `train_supervised(model,x,y)` mini-batch with val split/early stop | none in this checkout |
| `operator_card.py` | `operator_card`, `experiment_card`, `function_card`, `train_test_figure` (operator-learning figures) | none in this checkout |
| `benchmarks/heat.py` | `HeatProblem`, `stable_dt`, `solve(p,t_end,dt,save_every)` 2-D FTCS | 01, 02 |
| `benchmarks/heat1d.py` | Canonical rod (ALPHA=0.1, T_F=0.05, NX=128, NMODE=8): `grid`, `sample_family`, `profile`, `profile_xx`, `exact`, `steps`, `ftcs`, `solve_family`, `dataset`, `relative_l2`, `check_reference`. Use `exact` (analytic, fast) for many-seed / Monte Carlo studies; `ftcs` for solver studies | 20, 21, 22 |
| `benchmarks/navier_stokes.py` | `cavity_dt`, `lid_driven_cavity(nx,ny,nt,nit,Re,...)`, `vortex_merger(n,nt,Re,...)` | 01, 02 |
| `benchmarks/vessel.py` | `VesselParams`, `rotation`, `state_derivative`, `linear_stability`, `simulate` (RK4), `turning_circle`, `zigzag`, `random_surge_thrust`, `heading_autopilot`, `derivative_batch`, `step`, `commissioning_record`, `commissioning_split` | 01, 02, 20, 21, 22 |
| `benchmarks/turbine.py` | `TurbineParams`, `simulate`, `steady_operating_point`, `power_curve`, `turbulent_wind` | 02, 20, 21 |
| `benchmarks/cable.py` | `Cable` dataclass, `field_2d` | 01 |
| `control*.py` (`control`, `control_experiments`, `control_field`, `control_identification`, `control_models`, `control_results`, `control_rl`, `control_safety`) | MPC, closed-loop logging, system-ID models (DMDc, SINDy, Koopman, DNN, CoSTA), FNO/DeepONet field models, PPO agent, safety shield | NONE: belong to Modules 27-30 which are not in this checkout; do not use for proposals |

## 5. Environment (`python/environment.yml`, env name `ham4digitaltwin`)

python 3.11; conda: numpy, scipy, matplotlib, pillow, scikit-image, pandas, scikit-learn, statsmodels, jupyterlab, jupytext, nbgrader; pip: torch, pysindy. Notebooks actually import: torch (modules 00, 02, 20, 22), scipy/sklearn/PIL (24), numpy+scipy only (21). `pysindy`, `statsmodels`, `scikit-image` are listed but pysindy is not imported by any notebook here; no PySR/gplearn installed.

## 6. Method chooser

| Task | Course methods | Notebooks | Slides |
|---|---|---|---|
| Forward simulation of known physics | FTCS, projection method, pseudo-spectral, RK4; PINN as mesh-free alternative | 01 all; 22 pinn_heat_pde, pinn_vessel_ode forward | L01; L03 p28-31 |
| Surrogate / regression from data | Poly/MLP regression, MLP surrogates; know the extrapolation cliff | 02 ddm_* | L02 p31-48, limits p50 |
| Uncertainty quantification | Deep ensembles, MC dropout, ensemble PINN, ensemble-SINDy, Bayesian SINDy | 02 ddm_uncertainty; 22 pinn_ensemble_vessel; 21 ed_sindy_vessel, ed_sindy_variants | L02 p27-29; L03 p66-67; L06 p22, 68 |
| Inverse parameter estimation | Inverse PINN (alpha, drag, inertia, eddy viscosity); grey-box PINN; check identifiability | 22 pinn_heat_pde, pinn_vessel_ode, pinn_wind_turbine, pinn_wind_pde | L03 p13, 32, 40-42, 46-48 |
| Correcting an imperfect model | CoSTA (learn residual source/force/torque inside the solver); four error cases; honest split by trajectory | 20 all; 22 pinn_vessel_ode (grey-box) | L04; checklist p54, limits p52 |
| Discovering equations | SINDy/STLSQ, weak form, ensemble, SINDYc, PDE-FIND, SR/GP/GEP, SR3, constrained, Bayesian, SINDy-PI; failure diagnostics | 21 all | L06 p71 (selection), p59-64 (failures) |
| Sparse sensing / reconstruction | CS with l1 (ISTA/FISTA/OMP), DCT/Fourier bases, phase transition budget (2-3x the 50% line) | 24 cs_* | L05 p19-45, p54 |
| Sensor placement | POD + pivoted QR (D-optimal greedy), gappy POD, compare with random | 24 osp_qr_illustrations, cs_sensor_placement | L05 p56-71 |
| Sensor design for estimating a parameter | Not taught directly: combine sensitivity analysis (L01 p35) with the D-optimal / log-det criterion (L05 p63) and identifiability (L03 p42); count it as a course method applied to a new objective | 24 osp_qr_illustrations (`logdet`); 22 pinn_heat_pde (inverse alpha); pyutils.benchmarks.heat1d | L01 p35; L05 p63; L03 p42 |
| Choosing a PINN variant | Vanilla first; cPINN/XPINN for kinks/material jumps; XPINN time slabs for long horizons; vPINN for rough sources; ensembles for uncertainty | 22 xpinn_heat, vpinn_heat, pinn_vessel_ode, pinn_ensemble_vessel | L03 p55, p68 |
| Diagnosing a failing PINN | Plot each loss term; adaptive weights, Fourier features, causal weights, L-BFGS, more collocation points | 22 pinn_autodiff, pinn_heat_pde | L03 p51-55 |

## 7. Datasets (all small, produced by the notebooks except images)

| Path (under `python/`) | Contents | Produced/used by |
|---|---|---|
| `01-physics-based-modeling/data/heat_reference.npz` | x, y, T, alpha (2-D heat final field) | written by heat_diffusion |
| `01-physics-based-modeling/data/cavity_reference.npz` | x, y, u, v, p, Re (cavity) | navier_stokes_cavity |
| `01-physics-based-modeling/data/vortex_reference.npz` | X, Y, omega, Re (vortex final field) | navier_stokes_vortex |
| `01-physics-based-modeling/data/vessel_reference.npz` | t, Z, maneuver (turning circle) | vessel_ode |
| `01-physics-based-modeling/data/pcb_circuit.png` | PCB photo (chips segmented) | heat_diffusion |
| `20-corrective-source-term/data/costa_{heat,heat_rod,cases,vessel,turbine}_results.json` | Recorded configs, manifests, metrics, seed sweeps (turbine includes provenance, alpha check) | written/read by module 20 notebooks; cited by slides |
| `21-equation-discovery/data/heat_rod_results.json` | Method card for the rod (what is learned, physics needed, training solves, errors) | ed_heat_rod |
| `22-physics-informed-nn/data/pinn_heat_pde_seed_sweep.json`, `pinn_wind_pde_seed_sweep.json` | Params and per-seed rows | pinn_heat_pde, pinn_wind_pde |
| `24-compressed-sensing/data/portrait.png` | Grayscale portrait for sparsity/FISTA demos | cs_illustrations, cs_sparsity, cs_portrait_fista |
| (none) | Modules 00 and 02 `data/` are empty; all other data (vessel maneuvers, SCADA, wake fields, rod profiles) are generated in-notebook from fixed seeds. No real measured data exist anywhere in the course (turbine erosion, SCADA, wake, LiDAR-like data are all synthetic). |

## 8. Caveats for agents

- Not in this checkout: reduced-order modeling, DeepONet/FNO operator learning (Modules 25-26, referenced on L02 p51 and L04 p59-60), control modules 27-30, multivariate/time-series/deep-learning modules 03-18. Do not propose them as course-supported.
- L04 p59-60 numbers for DNN, PINN, DeepONet, FNO come from notebooks not present except the CoSTA, DNN-baseline (in costa_heat_rod) and discovery rod notebooks.
- Slide footers show a different counter than PDF page numbers; use the physical pages in this file.
- Page ranges for notebooks with no on-slide citation (e.g. pinn_autodiff, costa_vessel_ode, costa_wind_turbine, ed_heat_rod, ed_sindy_turbine, heat_cable, ddm_regression_basics, ddm_uncertainty, cs_illustrations) were assigned by topic match to slide titles.
