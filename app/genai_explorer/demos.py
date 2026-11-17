"""Interactive demos of the ROSE survey's mathematics, one per stage of the chain
route -> objective -> sampler -> evaluation -> systems.

Pure numpy + matplotlib so that the same file runs in a Gradio server (app/app.py) and in the browser through
Gradio-Lite / Pyodide (site/index.html). Every demo returns (figure, markdown); the markdown reports the numbers
the figure shows, so the explanation never depends on reading a plot. The mathematics of each demo is verified
in the equation notebooks (notebooks/NN_*.ipynb); the chapter is named in each demo's text.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito, the survey's figure palette
BLUE, ORANGE, GREEN, RED, PURPLE, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#7f7f7f", "#56B4E9"
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 10, "legend.fontsize": 8, "legend.frameon": False})


def _fig(w=9.0, h=3.4, n=2):
    fig, axs = plt.subplots(1, n, figsize=(w, h))
    return fig, np.atleast_1d(axs)


def _gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * math.sqrt(2 * math.pi))


# ===================================================================================== 1. divergences
def divergences(separation: float = 4.0, weight: float = 0.5):
    """Fit one Gaussian to a two-mode target by forward KL (maximum likelihood) and by reverse KL."""
    x = np.linspace(-9, 9, 3601); dx = x[1] - x[0]
    s0 = 0.7
    p = weight * _gauss(x, -separation / 2, s0) + (1 - weight) * _gauss(x, separation / 2, s0)
    # forward KL(p || q) over Gaussians q is minimised by moment matching (= maximum likelihood with infinite data)
    m_f = float(np.sum(x * p) * dx)
    s_f = float(math.sqrt(np.sum((x - m_f) ** 2 * p) * dx))
    # reverse KL(q || p): grid search over (mean, sd)
    logp = np.log(p + 1e-300)
    best = (np.inf, 0.0, 1.0)
    for m in np.linspace(-separation, separation, 121):
        for s in np.linspace(0.3, max(1.0, separation), 60):
            q = _gauss(x, m, s)
            kl = float(np.sum(q * (np.log(q + 1e-300) - logp)) * dx)
            if kl < best[0]:
                best = (kl, m, s)
    kl_r, m_r, s_r = best
    q_f, q_r = _gauss(x, m_f, s_f), _gauss(x, m_r, s_r)
    kl_f = float(np.sum(p * (logp - np.log(q_f + 1e-300))) * dx)
    fig, (a, b) = _fig()
    for ax, q, col, title in ((a, q_f, BLUE, "forward KL  $D(p\\,\\|\\,q)$: covers both modes"),
                              (b, q_r, RED, "reverse KL  $D(q\\,\\|\\,p)$: picks one mode")):
        ax.fill_between(x, p, color=GREY, alpha=0.3, label="target $p$ (the data)")
        ax.plot(x, q, color=col, lw=2, label="best single Gaussian $q$")
        ax.set_xlim(-separation - 3, separation + 3); ax.set_yticks([]); ax.set_title(title); ax.legend(loc="upper left")
    fig.tight_layout()
    md = (f"**Forward KL** (what maximum likelihood minimises): mean {m_f:+.2f}, sd {s_f:.2f}, "
          f"$D_{{\\mathrm{{KL}}}}(p\\|q)$ = {kl_f:.3f} nats. It must put mass wherever the data are, so it spreads over "
          f"both modes and puts mass *between* them where there are no data.\n\n"
          f"**Reverse KL**: mean {m_r:+.2f}, sd {s_r:.2f}, $D_{{\\mathrm{{KL}}}}(q\\|p)$ = {kl_r:.3f} nats. It is "
          f"punished for mass where the data are absent, so it locks onto one mode and ignores the other.\n\n"
          "This is the survey's first fact: likelihood-trained models (autoregressive, flows, VAEs, diffusion) are "
          "*mode-covering*; adversarial and reward-tilted ones lean *mode-seeking*. "
          "Verified in notebook chapters 1 and 2.")
    return fig, md


# ===================================================================================== 2. decoding
_CORPUS = ("the cat sat on the mat . the dog sat on the rug . the cat saw the dog . the dog saw a bird . "
           "a bird sat on the cat . the cat ran to the mat . the dog ran to the park . a cat sat in the sun . "
           "the bird sang in the park . the sun was warm . the mat was red . the dog was happy . "
           "the cat was happy in the sun . a dog ran in the park . the bird saw the sun .").split()


def _bigram():
    vocab = sorted(set(_CORPUS)); idx = {w: i for i, w in enumerate(vocab)}
    C = np.full((len(vocab), len(vocab)), 0.05)                       # add-0.05 smoothing: every word is possible
    for a, b in zip(_CORPUS[:-1], _CORPUS[1:]):
        C[idx[a], idx[b]] += 1
    return vocab, idx, np.log(C / C.sum(1, keepdims=True))


def _reshape(logp, temperature, top_k, top_p):
    if temperature <= 0:                                               # greedy
        out = np.zeros_like(logp); out[np.argmax(logp)] = 1.0; return out
    l = logp / temperature
    pr = np.exp(l - l.max()); pr /= pr.sum()
    order = np.argsort(-pr)
    keep = np.zeros_like(pr, dtype=bool)
    k = len(pr) if top_k <= 0 else int(top_k)
    cum = np.cumsum(pr[order])
    n_p = int(np.searchsorted(cum, top_p) + 1) if top_p < 1 else len(pr)
    keep[order[:max(1, min(k, n_p))]] = True
    pr = np.where(keep, pr, 0); return pr / pr.sum()


def decoding(temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, seed: int = 0):
    """A bigram language model: how temperature, top-k and top-p reshape the next-token distribution."""
    vocab, idx, L = _bigram()
    rng = np.random.default_rng(int(seed))
    base = np.exp(L[idx["the"]])
    new = _reshape(L[idx["the"]], temperature, top_k, top_p)
    order = np.argsort(-base)[:10]
    fig, (a, b) = _fig(9.5, 3.2)
    xs = np.arange(len(order))
    a.bar(xs - 0.2, base[order], 0.4, color=GREY, label="model $p(\\cdot\\mid$the$)$")
    a.bar(xs + 0.2, new[order], 0.4, color=BLUE, label="after decoding rule")
    a.set_xticks(xs, [vocab[i] for i in order], rotation=40); a.set_title('next word after "the"'); a.legend()
    # entropy of the sampling distribution as temperature varies (fixed top-k / top-p)
    Ts = np.linspace(0.05, 2.0, 60)
    H = [-(lambda q: np.sum(q[q > 0] * np.log(q[q > 0])))(_reshape(L[idx["the"]], t, top_k, top_p)) for t in Ts]
    b.plot(Ts, H, color=BLUE); b.axvline(max(temperature, 0.05), color=RED, ls="--")
    b.set_xlabel("temperature"); b.set_ylabel("entropy (nats)"); b.set_title("diversity of the next word")
    fig.tight_layout()
    lines = []
    for _ in range(3):
        w = "the"; out = [w]
        for _ in range(14):
            q = _reshape(L[idx[w]], temperature, top_k, top_p)
            w = vocab[int(np.argmax(q))] if temperature <= 0 else vocab[rng.choice(len(vocab), p=q)]
            out.append(w)
        lines.append("> " + " ".join(out))
    rule = "greedy (temperature 0)" if temperature <= 0 else f"temperature {temperature:.2f}, top-k {top_k or 'off'}, top-p {top_p:.2f}"
    md = (f"**Three continuations of \"the\" under {rule}:**\n\n" + "\n\n".join(lines) + "\n\n"
          "The model is unchanged: only the *sampler* changes. Greedy decoding repeats itself (the same loop every "
          "time); high temperature produces word salad; truncation (top-k, top-p) cuts the unlikely tail. "
          "The survey's E6b study finds the same pattern on Pythia models of 70M to 6.9B parameters: greedy decoding "
          "is worst at every size, and its gap shrinks as models grow. The chain rule behind this model is verified in "
          "notebook chapter 4.")
    return fig, md


# ===================================================================================== 3. GAN divergences
def gan_divergences(shift: float = 1.5):
    """Jensen-Shannon saturates when supports do not overlap; Wasserstein-1 keeps growing."""
    grid = np.linspace(-1, 9, 2001); dx = grid[1] - grid[0]
    def box(a):
        return np.where((grid >= a) & (grid <= a + 1), 1.0, 0.0)
    p = box(0.0); p /= p.sum()

    def js_w1(sh):
        q = box(sh); q /= q.sum()
        m = 0.5 * (p + q)
        kl = lambda u, v: float(np.sum(u[u > 0] * np.log(u[u > 0] / v[u > 0])))
        js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
        w1 = float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))) * dx)
        return js, w1
    shifts = np.linspace(0, 7, 141)
    vals = np.array([js_w1(s) for s in shifts])
    js, w1 = js_w1(shift)
    fig, (a, b) = _fig()
    a.fill_between(grid, box(0.0), step="mid", color=GREY, alpha=0.4, label="data $p_{\\mathrm{data}}$")
    a.fill_between(grid, box(shift), step="mid", color=BLUE, alpha=0.4, label="generator $p_g$")
    a.set_ylim(0, 1.4); a.set_yticks([]); a.legend(loc="upper right"); a.set_title(f"generator shifted by {shift:.2f}")
    b.plot(shifts, vals[:, 0] / math.log(2), color=RED, label="Jensen-Shannon / log 2")
    b.plot(shifts, vals[:, 1] / 7, color=GREEN, label="Wasserstein-1 / 7")
    b.axvline(shift, color=GREY, ls="--"); b.set_xlabel("shift"); b.legend(); b.set_title("what the critic measures")
    fig.tight_layout()
    md = (f"At shift {shift:.2f}: **JS** = {js:.3f} nats (max log 2 = {math.log(2):.3f}), **W1** = {w1:.3f}.\n\n"
          "Once the two supports stop overlapping (shift > 1), the Jensen-Shannon divergence that the original GAN "
          "minimises is stuck at log 2: moving the generator closer does not change it, so the generator gets no "
          "gradient. The Wasserstein-1 distance (the WGAN critic's target) keeps decreasing as the generator "
          "approaches the data. Real images and generator outputs live on thin, rarely overlapping manifolds, which is "
          "why this matters. Verified in notebook chapter 6.")
    return fig, md


# ===================================================================================== 4. Langevin
def langevin(step: float = 0.05, n_steps: int = 200, barrier: float = 2.0, seed: int = 0):
    """Unadjusted Langevin dynamics on a double-well energy E(x) = barrier (x^2 - 1)^2."""
    rng = np.random.default_rng(int(seed))
    E = lambda x: barrier * (x ** 2 - 1) ** 2
    gradE = lambda x: 4 * barrier * x * (x ** 2 - 1)
    x = rng.uniform(-3, 3, size=4000)
    x[:] = -1.0 + 0.1 * rng.normal(size=x.size)                         # all chains start in the LEFT well
    trace = [x[:5].copy()]
    for _ in range(int(n_steps)):
        x = x - 0.5 * step * gradE(x) + math.sqrt(step) * rng.normal(size=x.size)   # x + (k/2) score + sqrt(k) eps
        x = np.clip(x, -4, 4)
        trace.append(x[:5].copy())
    grid = np.linspace(-2.5, 2.5, 500); dens = np.exp(-E(grid)); dens /= float(np.sum(0.5 * (dens[1:] + dens[:-1]) * np.diff(grid)))
    right = float(np.mean(x > 0))
    fig, (a, b) = _fig()
    a.hist(x, bins=80, range=(-2.5, 2.5), density=True, color=BLUE, alpha=0.6, label="Langevin samples")
    a.plot(grid, dens, color="black", label="target $e^{-E(x)}/Z$"); a.legend(); a.set_title(f"after {int(n_steps)} steps")
    tr = np.array(trace)
    b.plot(tr, lw=0.8); b.set_xlabel("step"); b.set_ylabel("x"); b.set_title("five chains")
    fig.tight_layout()
    md = (f"Share of samples in the right-hand well: **{right:.2f}** (the target has 0.50).\n\n"
          "Langevin dynamics follows the *score* $-\\nabla E$ uphill in probability and adds noise of size "
          "$\\sqrt{\\text{step}}$. It needs only the score, never the normaliser $Z$, which is why score-based and "
          "energy-based models can sample at all. But with a high barrier the chains, all started in the left well, "
          "rarely cross: the samples look right locally and miss a whole mode. Annealing through noise levels, the idea "
          "that becomes diffusion, is the cure. Try barrier 4 with 200 steps, then barrier 0.5. Verified in notebook "
          "chapter 7.")
    return fig, md


# ===================================================================================== 5. diffusion
_K = 8
_MU = np.stack([np.cos(2 * np.pi * np.arange(_K) / _K), np.sin(2 * np.pi * np.arange(_K) / _K)], 1) * 2.0
_S = 0.12
_T = 1000
_BETA = np.linspace(1e-4, 0.02, _T)
_ABAR = np.cumprod(1 - _BETA)


def _eps_exact(x, t, cls=None):
    """Bayes-optimal noise prediction E[eps | x_t] for the 8-Gaussian ring (optionally the upper-half 'class')."""
    ab = _ABAR[t]; v = ab * _S ** 2 + (1 - ab)
    modes = np.arange(_K) if cls is None else np.where(_MU[:, 1] > 1e-9)[0]
    d = x[:, None, :] - math.sqrt(ab) * _MU[None, modes, :]
    logw = -0.5 * np.sum(d ** 2, -1) / v
    w = np.exp(logw - logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    score = -np.sum(w[:, :, None] * d, 1) / v
    return -math.sqrt(1 - ab) * score


def _ddim(n, nfe, eta, guidance, rng):
    ts = np.linspace(_T - 1, 0, int(nfe)).round().astype(int)
    x = rng.normal(size=(n, 2))
    for i, t in enumerate(ts):
        e_u = _eps_exact(x, t)
        e = e_u if guidance == 0 else e_u + guidance * (_eps_exact(x, t, cls=1) - e_u)
        ab = _ABAR[t]; ab_prev = _ABAR[ts[i + 1]] if i + 1 < len(ts) else 1.0
        x0 = (x - math.sqrt(1 - ab) * e) / math.sqrt(ab)
        sig = eta * math.sqrt(max((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev), 0.0))
        x = math.sqrt(ab_prev) * x0 + math.sqrt(max(1 - ab_prev - sig ** 2, 0.0)) * e + sig * rng.normal(size=x.shape)
    return x


def diffusion(t_view: int = 300, nfe: int = 20, eta: float = 0.0, guidance: float = 0.0, seed: int = 0):
    """Forward noising of a ring of 8 Gaussians, and DDIM sampling with the exact (Bayes-optimal) noise predictor."""
    rng = np.random.default_rng(int(seed))
    n = 1500
    x0 = _MU[rng.integers(0, _K, n)] + _S * rng.normal(size=(n, 2))
    t = int(np.clip(t_view, 0, _T - 1)); ab = _ABAR[t]
    xt = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * rng.normal(size=x0.shape)
    xs = _ddim(n, nfe, eta, guidance, rng)
    dist = np.sqrt(((xs[:, None, :] - _MU[None]) ** 2).sum(-1))
    nearest = dist.argmin(1); near = dist.min(1)
    on_mode = float(np.mean(near < 3 * _S))
    covered = int(np.sum(np.bincount(nearest[near < 3 * _S], minlength=_K) > 0.02 * n / _K))
    upper = float(np.mean(_MU[nearest][:, 1] > 1e-9))
    fig, (a, b) = _fig(9.5, 4.2)
    a.scatter(xt[:, 0], xt[:, 1], s=2, color=GREY, alpha=0.5)
    a.scatter(_MU[:, 0], _MU[:, 1], marker="x", color=RED, s=30, label="data modes")
    a.set_title(f"forward: $x_t$ at t = {t}  ($\\bar\\alpha_t$ = {ab:.3f})"); a.set_aspect("equal"); a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.legend(loc="lower left")
    b.scatter(xs[:, 0], xs[:, 1], s=2, color=BLUE, alpha=0.5)
    b.scatter(_MU[:, 0], _MU[:, 1], marker="x", color=RED, s=30)
    b.set_title(f"reverse: DDIM, {int(nfe)} steps, $\\eta$ = {eta:.1f}, guidance w = {guidance:.1f}")
    b.set_aspect("equal"); b.set_xlim(-4, 4); b.set_ylim(-4, 4)
    fig.tight_layout()
    md = (f"Samples within 3 sd of a mode: **{on_mode:.0%}**; modes covered: **{covered}/8**; "
          f"share in the upper half (the 'class' that guidance favours): **{upper:.0%}**.\n\n"
          "Left: the forward process destroys the data by mixing in Gaussian noise; by t = 999 it is pure noise. "
          "Right: the sampler runs the process backwards using a *noise predictor*; here we use the exact one, so "
          "every error you see is the **sampler's**, not the network's. "
          "Fewer steps blur the samples; $\\eta$ = 0 is the deterministic (ODE) sampler, $\\eta$ = 1 adds fresh noise "
          "each step; guidance moves samples into the upper-half 'class' (w = 0 unconditional, w = 1 the plain "
          "conditional, w > 1 an exaggerated conditional that trades diversity for agreement with the class). The survey's E4 and E6a studies show the same knobs moving "
          "FID several-fold on fixed trained weights. Verified in notebook chapters 8 and 9.")
    return fig, md


# ===================================================================================== 6. post-training tilt
def tilt(beta: float = 0.5, reward_error: bool = False, k: int = 16):
    """KL-regularised post-training: the optimum is pi_ref * exp(r / beta) / Z."""
    rng = np.random.default_rng(3)
    n = 12
    pref = rng.dirichlet(np.ones(n) * 0.8)
    correct = np.zeros(n, bool); correct[[1, 4, 7]] = True                   # three correct answers, rarely sampled
    pref[correct] *= 0.25; pref /= pref.sum()
    r = correct.astype(float)
    if reward_error:                                                         # the reward model loves one wrong answer
        r = r.copy(); r[9] = 1.4
    pistar = pref * np.exp(r / beta); pistar /= pistar.sum()
    kl = float(np.sum(pistar * np.log(pistar / pref)))
    p0, p1 = float(pref[correct].sum()), float(pistar[correct].sum())
    ks = np.arange(1, 65)
    fig, (a, b) = _fig(9.5, 3.4)
    xs = np.arange(n)
    cols = [GREEN if c else (RED if (reward_error and i == 9) else GREY) for i, c in enumerate(correct)]
    a.bar(xs - 0.2, pref, 0.4, color=cols, alpha=0.35, label="reference $\\pi_{\\mathrm{ref}}$")
    a.bar(xs + 0.2, pistar, 0.4, color=cols, label="tilted $\\pi^*$")
    a.set_xticks(xs, [f"y{i}" for i in xs], fontsize=7); a.set_title("green = correct answer" + ("; red = reward-model error" if reward_error else ""))
    a.legend()
    b.plot(ks, 1 - (1 - p0) ** ks, color=GREY, label="reference")
    b.plot(ks, 1 - (1 - p1) ** ks, color=BLUE, label="exact tilt")
    b.axvline(k, color=RED, ls="--"); b.set_xscale("log", base=2); b.set_xlabel("k samples"); b.set_ylabel("pass@k"); b.legend(); b.set_title("success with k tries")
    fig.tight_layout()
    md = (f"$\\beta$ = {beta:.2f}: KL to the reference **{kl:.2f} nats**; probability of a correct answer "
          f"**{p0:.2f} → {p1:.2f}**; pass@{k}: **{1 - (1 - p0) ** k:.3f} → {1 - (1 - p1) ** k:.3f}**.\n\n"
          "RLHF, DPO and GRPO all aim at this optimum: multiply the reference model by $e^{r/\\beta}$ and renormalise. "
          "Small $\\beta$ (a cheap KL penalty) tilts hard. Two consequences the survey builds on: the tilt can only "
          "*move* probability, never create it (answers the reference never gives stay impossible), and with a correct/"
          "incorrect reward it can never lower pass@k. "
          + ("With the reward-model error switched on, a small $\\beta$ piles probability onto the wrong answer the reward "
             "model likes and pass@k *falls*: **reward over-optimisation**. The guarantee needs the reward to be right. " if reward_error else "Switch on the reward-model error and lower $\\beta$. ")
          + "The survey's E7 study finds that trained policies (GRPO, DPO) land below this exact tilt and spend 16 to 110 "
          "times the KL it needs. Verified in notebook chapter 11.")
    return fig, md


# ===================================================================================== 7. metrics
def _fid(a, b):
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    ev = np.linalg.eigvals(ca @ cb).real
    return float(np.sum((ma - mb) ** 2) + np.trace(ca) + np.trace(cb) - 2 * np.sum(np.sqrt(np.clip(ev, 0, None))))


def _pr(real, fake, k=3):
    def radii(X):
        d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); d.sort(1); return d[:, k]
    rr, rf = radii(real), radii(fake)
    d = np.sqrt(((fake[:, None] - real[None]) ** 2).sum(-1))
    precision = float(np.mean((d <= rr[None]).any(1)))
    recall = float(np.mean((d.T <= rf[None]).any(1)))
    return precision, recall


def metrics(n_samples: int = 500, model: str = "perfect copy of the data", seed: int = 0):
    """FID's upward bias at small N, and what precision and recall see that FID does not."""
    rng = np.random.default_rng(int(seed))
    centres = np.array([[-2, 0], [2, 0], [0, 2.5], [0, -2.5]], float)
    def data(n, which=None, blur=0.0):
        c = centres if which is None else centres[which]
        return c[rng.integers(0, len(c), n)] + 0.35 * rng.normal(size=(n, 2)) * (1 + blur)
    ref = data(2000)                                   # the real images the metric compares against

    def gen(n):
        if model == "drops two of four modes":
            return data(n, which=[0, 1])
        if model == "blurry (too much spread)":
            return data(n, blur=1.2)
        if model == "memorises the training set":         # copies of real images: what the metrics cannot see (E5)
            return ref[rng.integers(0, len(ref), n)]
        return data(n)                                   # "perfect copy of the data": fresh samples from p_data
    Ns = np.array([50, 100, 200, 400, 800, 1600])
    fids = [np.mean([_fid(ref, gen(int(N))) for _ in range(4)]) for N in Ns]
    fid_perfect = [np.mean([_fid(ref, data(int(N))) for _ in range(4)]) for N in Ns]
    g = gen(int(n_samples))
    prec, rec = _pr(ref[:600], g[:600])
    fig, (a, b) = _fig(9.5, 3.6)
    a.scatter(ref[:600, 0], ref[:600, 1], s=3, color=GREY, alpha=0.4, label="real")
    a.scatter(g[:600, 0], g[:600, 1], s=3, color=BLUE, alpha=0.5, label="generated")
    a.set_aspect("equal"); a.legend(loc="upper right"); a.set_title(model)
    b.plot(Ns, fid_perfect, "o-", color=GREY, label="perfect copy")
    b.plot(Ns, fids, "s-", color=BLUE, label="this model")
    b.axvline(n_samples, color=RED, ls="--"); b.set_xscale("log"); b.set_xlabel("number of generated samples N")
    b.set_ylabel("FID (2-D toy features)"); b.legend(); b.set_title("FID depends on N")
    fig.tight_layout()
    md = (f"At N = {int(n_samples)}: **FID {_fid(ref, g):.3f}**, **precision {prec:.2f}**, **recall {rec:.2f}**.\n\n"
          "FID is never zero at finite N even for a perfect copy of the data (grey curve): it is biased upward by "
          "roughly c/N, so FIDs computed with different sample counts cannot be compared. FID compresses two different "
          "failures into one number; precision (are samples realistic?) and recall (is every part of the data covered?) "
          "separate them: a model that drops modes keeps precision and loses recall, a blurry one the reverse, and "
          "a model that memorises the training set scores as well as a perfect one while generating nothing new. The survey's E2 study shows the feature "
          "extractor alone reverses some model rankings, and E8 checks the metrics against human judgments. Verified "
          "in notebook chapter 10.")
    return fig, md


# ===================================================================================== registry for the app
TABS = [
    ("Divergences", divergences, 1, "Route and objective: which divergence does training minimise?"),
    ("Language-model decoding", decoding, 4, "Sampler: the same model, different decoding rules."),
    ("GAN critics", gan_divergences, 6, "Objective: what an adversarial critic measures."),
    ("Score and Langevin", langevin, 7, "Sampler: sampling with only the score."),
    ("Diffusion sampler", diffusion, 9, "Sampler: forward noising and reverse sampling."),
    ("Post-training tilt", tilt, 11, "Systems: RLHF/DPO/GRPO as a tilt of the reference."),
    ("Evaluation metrics", metrics, 10, "Evaluation: what FID, precision and recall see."),
]
