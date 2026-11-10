# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Assignment — symbolic regression by gene expression programming
#
# You will build the three parts of a working symbolic-regression search that the lecture derived, then use them inside
# a provided evolutionary loop to discover a hidden law:
#
# 1. **Karva decoding** --- turn a fixed-length gene into an expression tree and evaluate it;
# 2. **linear scaling** --- the closed-form outer constants, $\min_{a,b}\mathrm{MSE}=\mathrm{var}(y)(1-\rho^2)$;
# 3. **the knee** of an accuracy--size front;
# 4. **the search** --- run the provided GEP loop with your functions, with restarts.
#
# Symbols: functions `+ - * /` (protected division) and `Q` $=\sqrt{|\cdot|}$; terminals `x` and `1`.
# Solution cells use `### BEGIN SOLUTION` / `### END SOLUTION`; every task ends with `assert`s.

# %%
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[_v] = '1'   # reproducible
import sys, random
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
import numpy as np, matplotlib.pyplot as plt
from pyutils import set_plot_style; set_plot_style()
ARITY = {'+': 2, '-': 2, '*': 2, '/': 2, 'Q': 1}
FN = {'+': np.add, '-': np.subtract, '*': np.multiply,
      '/': lambda a, b: np.where(np.abs(b) > 1e-6, a/np.where(np.abs(b) > 1e-6, b, 1.0), 1.0),
      'Q': lambda a: np.sqrt(np.abs(a))}

# %% [markdown]
# ## 1. Karva decoding
#
# Read the gene level by level: symbol 0 is the root; each node takes as many of the *next unread* symbols as its arity,
# level by level, left to right; stop when no argument is open. Write
#
# * `decode(gene)` returning a list of nodes `[symbol, [child indices]]` (index 0 = root) for the open reading frame;
# * `evaluate(nodes, x)` evaluating the tree on an array `x`.
#
# Test gene from the lecture: `/1QxQx+1x111x11` is $1/\sqrt{x}$ (4 coding symbols).

# %%
def decode(gene):
    ### BEGIN SOLUTION
    nodes = [[gene[0], []]]; i = 1; level = [0]
    while level:
        nxt = []
        for k in level:
            for _ in range(ARITY.get(nodes[k][0], 0)):
                nodes.append([gene[i], []]); nodes[k][1].append(len(nodes) - 1); nxt.append(len(nodes) - 1); i += 1
        level = nxt
    return nodes
    ### END SOLUTION

def evaluate(nodes, x, k=0):
    ### BEGIN SOLUTION
    s, ch = nodes[k]
    if s == 'x': return x
    if s == '1': return np.ones_like(x)
    return FN[s](*[evaluate(nodes, x, c) for c in ch])
    ### END SOLUTION

nodes = decode('/1QxQx+1x111x11')
print('coding symbols:', len(nodes), ' value at x = 4:', evaluate(nodes, np.array([4.0])))
assert len(nodes) == 4 and np.isclose(evaluate(nodes, np.array([4.0]))[0], 0.5)
assert len(decode('*A+xx1x'.replace('A', '-') + 'xx1x1xxx')) == 7
assert np.isclose(evaluate(decode('*x+x1x1' + 'x'*8), np.array([2.0]))[0], 6.0)      # x*(x+1)

# %% [markdown]
# **Question:** the head of length $h=7$ is followed by a tail of $t=8$ terminals. Prove that with binary functions
# $t=h+1$ guarantees every gene decodes (count open argument slots).

# %% [markdown]
# ## 2. Linear scaling
#
# Write `scaled(g, y)` returning `(nmse, a, b)` for the best affine fit $a+b\,g$ of the tree output $g$ to the data $y$,
# with $\mathrm{NMSE}=\mathrm{MSE}/\mathrm{var}(y)$. Use the closed form, not `np.polyfit`. A constant $g$ should return
# NMSE $=1$.

# %%
def scaled(g, y):
    ### BEGIN SOLUTION
    if not np.all(np.isfinite(g)): return np.inf, 0.0, 1.0
    vg = g.var()
    if vg < 1e-12: return 1.0, float(y.mean()), 0.0
    b = float(np.mean((g - g.mean())*(y - y.mean()))/vg); a = float(y.mean() - b*g.mean())
    return float(np.mean((a + b*g - y)**2)/y.var()), a, b
    ### END SOLUTION

xt = np.linspace(1, 8, 50); yt = 3.0 + 2.0/np.sqrt(xt)
nm, a, b = scaled(1/np.sqrt(xt), yt)
assert nm < 1e-12 and np.isclose(a, 3.0) and np.isclose(b, 2.0)
g = xt**0.3; rho = np.corrcoef(g, yt)[0, 1]
assert np.isclose(scaled(g, yt)[0], 1 - rho**2)
assert scaled(np.ones_like(xt), yt)[0] == 1.0

# %% [markdown]
# ## 3. The knee of the front
#
# A *front* is a list of `(size, nmse)` pairs, one per size, error decreasing with size. Write `knee(front, within=1.5)`
# returning the **size** of the reported model: among entries with error at most `within` times the best, the one with
# the largest drop of $\ln\mathrm{NMSE}$ per added node relative to the previous entry. Test front: the wake law of the
# lecture.

# %%
def knee(front, within=1.5):
    ### BEGIN SOLUTION
    best = min(m for _, m in front)
    score = [0.0] + [np.log(front[i - 1][1]/front[i][1])/(front[i][0] - front[i - 1][0]) for i in range(1, len(front))]
    ok = [i for i in range(len(front)) if front[i][1] <= within*best]
    return front[max(ok, key=lambda i: score[i])][0]
    ### END SOLUTION

WAKE_FRONT = [(1, 0.1737), (2, 0.0980), (3, 0.0257), (4, 0.0130), (5, 0.0128), (6, 0.0124), (13, 0.0123)]
print('knee at size', knee(WAKE_FRONT))
assert knee(WAKE_FRONT) == 4
assert knee([(1, 0.5), (3, 0.01), (7, 0.0099)]) == 3

# %% [markdown]
# ## 4. The search
#
# The hidden law behind the data below is unknown to you. The provided GEP loop (selection, mutation, transposition,
# recombination) calls your `decode`, `evaluate` and `scaled`. One run can get stuck in a local optimum, so run **ten
# seeds** and keep the best. Store the best gene in `best_gene`, its NMSE in `best_nmse`, and the number of seeds that
# reached the noise floor in `n_success`.

# %%
rs = np.random.default_rng(1)
X_DATA = np.linspace(-3, 3, 80); Y_DATA = 2/(1 + X_DATA**2) + 0.02*rs.standard_normal(80)   # (hidden law)
FLOOR = 0.00084                                                        # NMSE of the true law on these data

def gep_run(x, y, seed, funcs=('+', '-', '*', '/'), terms=('x', '1'), h=7, pop_n=200, gens=60, gamma=1e-3):
    # gene expression programming: head h (any symbol), tail h+1 (terminals); returns (fitness, gene)
    r = random.Random(seed); t_len = h + 1; F, T = list(funcs), list(terms)
    rand = lambda: [r.choice(F + T) for _ in range(h)] + [r.choice(T) for _ in range(t_len)]
    def fit(g):
        nodes = decode(g); return scaled(evaluate(nodes, x), y)[0] + gamma*len(nodes)
    def mutate(g):
        return [(r.choice(F + T) if i < h else r.choice(T)) if r.random() < 0.044 else s for i, s in enumerate(g)]
    def transpose(g):
        g = g[:]; ln = r.choice([1, 2, 3]); st = r.randrange(len(g) - ln + 1); tgt = r.randrange(1, h)
        g[:h] = (g[:tgt] + g[st:st + ln] + g[tgt:h])[:h]; return g
    pop = [rand() for _ in range(pop_n)]; best = (np.inf, None)
    for _ in range(gens):
        sc = [fit(g) for g in pop]; order = np.argsort(sc); pop = [pop[i] for i in order]
        if sc[order[0]] < best[0]: best = (sc[order[0]], pop[0])
        new = [pop[0]]
        while len(new) < pop_n:
            a = pop[min(r.sample(range(pop_n), 3))]; b = pop[min(r.sample(range(pop_n), 3))]
            if r.random() < 0.3:
                k = r.randrange(1, h + t_len); a, b = a[:k] + b[k:], b[:k] + a[k:]
            for c in (a, b):
                c = mutate(c)
                if r.random() < 0.1: c = transpose(c)
                new.append(c)
        pop = new[:pop_n]
    return best

best_gene = best_nmse = n_success = None
### BEGIN SOLUTION
results = []
for sd in range(10):
    _, g = gep_run(X_DATA, Y_DATA, sd); nodes = decode(g); m, a, b = scaled(evaluate(nodes, X_DATA), Y_DATA)
    results.append((m, ''.join(g), a, b))
results.sort(); best_nmse, best_gene = results[0][0], results[0][1]
n_success = sum(m < 1.1*FLOOR for m, *_ in results)
### END SOLUTION
nodes = decode(best_gene); _, a, b = scaled(evaluate(nodes, X_DATA), Y_DATA)
print(f'best gene {best_gene}  ->  {a:+.3f} {b:+.3f} * tree;  NMSE {best_nmse:.5f}  (noise floor {FLOOR});  '
      f'{n_success}/10 seeds reached the floor')
assert best_nmse < 1.1*FLOOR and n_success >= 1

# %% [markdown]
# *Questions:* (1) Write the law you found as a formula. If your tree is $x/(x+1/x)$, show that after linear scaling
# it is the same law as $c/(1+x^2)$. (2) Why do several seeds stop at $x\cdot x$? (3) What does running ten seeds cost,
# and what would you change in the loop to make a single run more reliable?
