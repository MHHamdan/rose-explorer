"""Chapter 9: discrete-time diffusion (DDPM), DDIM, classifier-free guidance and flow matching.

Covers the 31 registry ids listed under this chapter in chapter_ids.json. Build with
python notebooks/src/ch09_discrete_diffusion_ddim_guidance_flow_matching.py
"""
from nbkit import Chapter

# Shared helper code, defined once in the first entry and reused by later cells of the notebook.
SCHEDULES = r'''
def make_schedule(T=1000, kind="linear", beta_start=1e-4, beta_end=0.02):
    """The two schedules used in the paper's experiments: Ho et al. (2020) linear and Nichol & Dhariwal (2021) cosine.
    Returns betas, alphas, abar, abar_prev as float64 arrays indexed 0..T-1 (code index i is the paper's t = i + 1)."""
    if kind == "linear":
        betas = np.linspace(beta_start, beta_end, T)
    elif kind == "cosine":
        s = 0.008
        f = lambda t: np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
        ts = np.arange(T + 1, dtype=float)
        ab = f(ts) / f(0.0)
        betas = np.minimum(1 - ab[1:] / ab[:-1], 0.999)
    else:
        raise ValueError(kind)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    abar_prev = np.concatenate([[1.0], abar[:-1]])
    return betas, alphas, abar, abar_prev

betas, alphas, abar, abar_prev = make_schedule()
T = len(betas)
'''


def build():
    c = Chapter(
        9, "discrete_diffusion_ddim_guidance_flow_matching",
        "Discrete-time diffusion: DDPM, DDIM, guidance and flow matching",
        summary=(
            "A denoising diffusion model destroys data with a fixed Markov chain of Gaussian noise and learns to "
            "reverse it. Everything the paper says about this family rests on a handful of Gaussian identities: the "
            "closed-form marginal of the forward chain, the posterior of one step given the clean image, the "
            "variational bound that these give, the noise-prediction parameterisation that turns the bound into a "
            "weighted regression, and the DDIM family of samplers that share the marginals but not the chain. The "
            "chapter verifies each identity symbolically or on a Gaussian data distribution where the optimal network "
            "is known in closed form, then does the same for classifier-free guidance and for the flow-matching "
            "objective that generalises the straight-line path."),
        prerequisites="Chapters 1 (KL divergence), 3 (the evidence lower bound) and 8 (the continuous-time view)",
        paper_sections="main §3.3; Supplement S1.11, S4.6, S4.7 and the sampler study of §8.5",
    )

    # ------------------------------------------------------------------------------------------------ forward
    c.section("The forward chain", (
        "Fix a variance schedule $\\beta_1<\\dots<\\beta_T$ in $(0,1)$. At each step the chain shrinks the signal "
        "slightly and adds a little Gaussian noise. Two facts make the family tractable: the distribution after $t$ "
        "steps is Gaussian with a closed form, and so is the distribution of step $t-1$ once both the noisy image and "
        "the clean image are known."))

    c.eq(["EQ-M-027", "EQ-S1-042"], "The forward kernel and the noise schedule",
         r"q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=\mathcal{N}\big(\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\;\beta_t\mathbf{I}\big),"
         r"\qquad \alpha_t=1-\beta_t,\qquad \bar\alpha_t=\prod_{s\le t}\alpha_s .",
         plain=(
             "$\\mathbf{x}_0$ is a data point (an image) and $\\mathbf{x}_t$ is the same image after $t$ rounds of "
             "corruption. Each round multiplies the current image by a number slightly below one and adds Gaussian "
             "noise of variance $\\beta_t$. Because $(\\sqrt{1-\\beta_t})^2+\\beta_t=1$, a unit-variance input stays at "
             "unit variance: the process is *variance preserving*. $\\bar\\alpha_t$ is the product of all the shrink "
             "factors squared so far; it starts near one and falls towards zero, and it is the single number that "
             "says how much of the original image survives at step $t$."),
         refresher=(
             "This is the Euler--Maruyama discretisation of the variance-preserving SDE of Chapter 8 with "
             "$\\beta(t)\\,\\mathrm{d}t\\mapsto\\beta_t$. The paper's experiments use two schedules: the linear schedule "
             "of Ho et al. (2020), $\\beta_t$ from $10^{-4}$ to $0.02$ over $T=1000$ steps, and the cosine schedule of "
             "Nichol and Dhariwal (2021), which defines $\\bar\\alpha_t$ directly as $\\cos^2$ of a shifted time and "
             "derives $\\beta_t=1-\\bar\\alpha_t/\\bar\\alpha_{t-1}$, clipped at $0.999$. Nothing is learnt in the forward "
             "chain; the schedule is a design choice that fixes the signal-to-noise ratio "
             "$\\mathrm{SNR}_t=\\bar\\alpha_t/(1-\\bar\\alpha_t)$ at every step and, through it, the weighting of the "
             "training objective (entry 9.5) and the discretisation error of every sampler (entry 9.9). The "
             "repository implementation is `diffusion/model.py`; the helper below reproduces its two schedules."),
         code=SCHEDULES + r'''
for kind in ("linear", "cosine"):
    b, a, ab, abp = make_schedule(kind=kind)
    check(f"{kind}: betas in (0,1) and increasing", np.all((b > 0) & (b < 1)) and np.all(np.diff(b) >= -1e-15),
          f"beta_1={b[0]:.2e}, beta_T={b[-1]:.3f}")
    check(f"{kind}: abar is the cumulative product of alphas", *close(ab, np.cumprod(1 - b), 1e-15))
    check(f"{kind}: abar decreases from near 1 to near 0", ab[0] > 0.99 and ab[-1] < 1e-3 and np.all(np.diff(ab) < 0),
          f"abar_1={ab[0]:.5f}, abar_T={ab[-1]:.2e}")
# variance preservation: a unit-variance input stays at unit variance after one step
x = rng.normal(size=200_000)
x1 = np.sqrt(1 - betas[500]) * x + np.sqrt(betas[500]) * rng.normal(size=x.size)
check("one step preserves unit variance", abs(x1.var() - 1) < 0.01, f"var after one step = {x1.var():.4f}")
'''),

    c.eq(["EQ-M-028", "EQ-S1-043", "EQ-S4-044"], "The closed-form marginal after $t$ steps",
         r"q(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}\big(\sqrt{\bar\alpha_t}\,\mathbf{x}_0,\;(1-\bar\alpha_t)\mathbf{I}\big),"
         r"\qquad \mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon},"
         r"\quad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).",
         plain=(
             "You do not have to simulate the $t$ corruption steps one by one. Composing Gaussian steps gives a "
             "Gaussian, so a noisy image at any level is one multiplication and one addition away from the clean "
             "image: keep a fraction $\\sqrt{\\bar\\alpha_t}$ of the image and add noise with standard deviation "
             "$\\sqrt{1-\\bar\\alpha_t}$. This is what makes training cheap: pick a random $t$, make $\\mathbf{x}_t$ in "
             "one shot, and ask the network to undo it."),
         refresher=(
             "Induction on $t$: if $\\mathbf{x}_{t-1}=\\sqrt{\\bar\\alpha_{t-1}}\\mathbf{x}_0+\\sqrt{1-\\bar\\alpha_{t-1}}"
             "\\boldsymbol{\\epsilon}'$, then $\\mathbf{x}_t=\\sqrt{\\alpha_t\\bar\\alpha_{t-1}}\\mathbf{x}_0+"
             "\\sqrt{\\alpha_t(1-\\bar\\alpha_{t-1})}\\boldsymbol{\\epsilon}'+\\sqrt{\\beta_t}\\boldsymbol{\\epsilon}''$, and "
             "the two independent Gaussian terms merge into one of variance "
             "$\\alpha_t(1-\\bar\\alpha_{t-1})+\\beta_t=1-\\bar\\alpha_t$. The mean coefficient and the variance are "
             "therefore the recursion $c_t=\\sqrt{1-\\beta_t}\\,c_{t-1}$, $v_t=(1-\\beta_t)v_{t-1}+\\beta_t$ started at "
             "$c_0=1,v_0=0$. The Supplement S4 statement (`EQ-S4-044`) adds the reverse model "
             "$p_\\theta(\\mathbf{x}_{t-1}\\mid\\mathbf{x}_t)=\\mathcal{N}(\\boldsymbol{\\mu}_\\theta,\\sigma_t^2\\mathbf{I})$, "
             "verified in entry 9.4. The repository check `ddpm_forward_marginal_by_composing_kernels` in "
             "`verification/verify_equations.py` additionally confirms that `GaussianDiffusion.q_sample` implements "
             "this formula; that part is not repeated here."),
         code=r'''
# (a) symbolic: composing three generic kernels gives the product form
b1, b2, b3, x0 = sp.symbols("beta_1 beta_2 beta_3 x_0", positive=True)
cs, vs = sp.Integer(1), sp.Integer(0)
for bt in (b1, b2, b3):
    cs, vs = sp.sqrt(1 - bt) * cs, (1 - bt) * vs + bt
abar3 = (1 - b1) * (1 - b2) * (1 - b3)
check("sympy: mean coefficient after 3 steps = sqrt(abar_3)", sp.simplify(cs**2 - abar3) == 0)
check("sympy: variance after 3 steps = 1 - abar_3", sp.simplify(vs - (1 - abar3)) == 0)
# (b) numeric recursion for both schedules, all 1000 steps
for kind in ("linear", "cosine"):
    b, a, ab, _ = make_schedule(kind=kind)
    cc, vv, cs_, vs_ = 1.0, 0.0, [], []
    for bt in b:
        cc, vv = cc * math.sqrt(1 - bt), (1 - bt) * vv + bt
        cs_.append(cc); vs_.append(vv)
    check(f"{kind}: recursion mean coef = sqrt(abar)", *close(cs_, np.sqrt(ab), 1e-12))
    check(f"{kind}: recursion variance = 1 - abar", *close(vs_, 1 - ab, 1e-12))
# (c) Monte Carlo: simulate the chain step by step and compare with the one-shot formula at t = 300
t = 300; n = 200_000
x = np.full(n, 0.8)
for i in range(t):
    x = np.sqrt(1 - betas[i]) * x + np.sqrt(betas[i]) * rng.normal(size=n)
se_m, se_v = math.sqrt((1 - abar[t - 1]) / n), (1 - abar[t - 1]) * math.sqrt(2 / n)
check("MC chain mean = sqrt(abar_t) x0", abs(x.mean() - math.sqrt(abar[t - 1]) * 0.8) < 4 * se_m,
      f"{x.mean():.4f} vs {math.sqrt(abar[t-1])*0.8:.4f} (SE {se_m:.1e})")
check("MC chain variance = 1 - abar_t", abs(x.var() - (1 - abar[t - 1])) < 4 * se_v,
      f"{x.var():.4f} vs {1-abar[t-1]:.4f} (SE {se_v:.1e})")
'''),

    c.eq(["EQ-S1-044"], "The forward posterior given the clean image",
         r"q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}\big(\tilde{\boldsymbol\mu}_t,\tilde\beta_t\mathbf{I}\big),\quad"
         r"\tilde{\boldsymbol\mu}_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\mathbf{x}_0"
         r"+\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t,\quad"
         r"\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t .",
         plain=(
             "Suppose you know both the clean image $\\mathbf{x}_0$ and its noisy version $\\mathbf{x}_t$. Where was "
             "the chain one step earlier? The answer is again Gaussian: its mean is a weighted average of the clean "
             "and the noisy image, with weights that depend only on the schedule, and its variance $\\tilde\\beta_t$ "
             "is a little smaller than $\\beta_t$. This is the target that the learnt reverse step tries to match."),
         refresher=(
             "Bayes' rule with two Gaussians: $q(\\mathbf{x}_{t-1}\\mid\\mathbf{x}_t,\\mathbf{x}_0)\\propto "
             "q(\\mathbf{x}_t\\mid\\mathbf{x}_{t-1})\\,q(\\mathbf{x}_{t-1}\\mid\\mathbf{x}_0)$, the first factor from entry 9.1 "
             "(the Markov property removes $\\mathbf{x}_0$ from it) and the second from entry 9.2. The log of the "
             "product is quadratic in $\\mathbf{x}_{t-1}$; completing the square gives the precision "
             "$\\alpha_t/\\beta_t+1/(1-\\bar\\alpha_{t-1})=(1-\\bar\\alpha_t)/(\\beta_t(1-\\bar\\alpha_{t-1}))$ and the mean "
             "above. At $t=1$ the weight on $\\mathbf{x}_0$ is one and $\\tilde\\beta_1=0$: the first step is "
             "deterministic given the clean image, which is why the bound's $L_0$ term is handled separately "
             "(entry 9.4). The check below derives mean and variance symbolically and confirms them by conditioning "
             "simulated pairs $(\\mathbf{x}_{t-1},\\mathbf{x}_t)$ on a thin slice of $\\mathbf{x}_t$."),
         code=r'''
abp_s, at_s, x0_s, xt_s, y = sp.symbols("abar_prev alpha_t x_0 x_t y", positive=True)
bt_s = 1 - at_s; abt_s = abp_s * at_s
logpost = -(y - sp.sqrt(abp_s) * x0_s) ** 2 / (2 * (1 - abp_s)) - (xt_s - sp.sqrt(at_s) * y) ** 2 / (2 * bt_s)
prec = sp.simplify(-sp.diff(logpost, y, 2))
mean = sp.solve(sp.diff(logpost, y), y)[0]
var_ok = sp.simplify(1 / prec - (1 - abp_s) / (1 - abt_s) * bt_s) == 0
mean_ok = sp.simplify(mean - (sp.sqrt(abp_s) * bt_s / (1 - abt_s) * x0_s + sp.sqrt(at_s) * (1 - abp_s) / (1 - abt_s) * xt_s)) == 0
check("sympy: posterior variance = beta_tilde", var_ok)
check("sympy: posterior mean = c0 x0 + ct x_t with the stated coefficients", mean_ok)
# numeric coefficients for the whole linear schedule (code index i <-> paper t = i + 1)
post_c0 = np.sqrt(abar_prev) * betas / (1 - abar)
post_ct = np.sqrt(alphas) * (1 - abar_prev) / (1 - abar)
beta_tilde = (1 - abar_prev) / (1 - abar) * betas
check("t = 1: weight on x0 is 1, weight on x_t is 0, beta_tilde = 0",
      abs(post_c0[0] - 1) < 1e-12 and abs(post_ct[0]) < 1e-12 and abs(beta_tilde[0]) < 1e-12)
check("beta_tilde < beta for every t >= 2", np.all(beta_tilde[1:] < betas[1:]))
# Monte Carlo: condition simulated (x_{t-1}, x_t) pairs on x_t in a thin slice
t = 300; x0v = 0.7; n = 2_000_000
xprev = math.sqrt(abar_prev[t]) * x0v + math.sqrt(1 - abar_prev[t]) * rng.normal(size=n)
xtv = math.sqrt(alphas[t]) * xprev + math.sqrt(betas[t]) * rng.normal(size=n)
sel = np.abs(xtv - 0.2) < 0.002
mc_mean, mc_var = xprev[sel].mean(), xprev[sel].var()
th_mean = post_c0[t] * x0v + post_ct[t] * 0.2
check("MC conditional mean matches mu_tilde", abs(mc_mean - th_mean) < 4 * math.sqrt(beta_tilde[t] / sel.sum()),
      f"{mc_mean:.4f} vs {th_mean:.4f} from {sel.sum()} pairs in the slice")
check("MC conditional variance matches beta_tilde (10%)", abs(mc_var / beta_tilde[t] - 1) < 0.1,
      f"{mc_var:.3e} vs {beta_tilde[t]:.3e}")
'''),

    # ------------------------------------------------------------------------------------------------ bound
    c.section("The variational bound and the noise-prediction objective", (
        "The reverse chain is a latent-variable model with $\\mathbf{z}=\\mathbf{x}_{1:T}$ and the forward chain as "
        "its fixed approximate posterior. The evidence lower bound of Chapter 3 then splits into per-step KL "
        "divergences between Gaussians, which is why diffusion training is a regression."))

    c.eq(["EQ-M-029", "EQ-S1-045", "EQ-S1-046"], "The reverse model and the variational bound",
         r"p_\theta(\mathbf{x}_{0:T})=p(\mathbf{x}_T)\prod_{t=1}^{T}p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t),\qquad"
         r"-\log p_\theta(\mathbf{x}_0)\le\underbrace{D_{\mathrm{KL}}\big(q(\mathbf{x}_T\mid\mathbf{x}_0)\,\|\,p(\mathbf{x}_T)\big)}_{L_T}"
         r"+\sum_{t=2}^{T}\underbrace{\mathbb{E}_q\,D_{\mathrm{KL}}\big(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,\|\,p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\big)}_{L_{t-1}}"
         r"\;\underbrace{-\,\mathbb{E}_q\log p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)}_{L_0}.",
         plain=(
             "The model generates by starting from pure noise $p(\\mathbf{x}_T)=\\mathcal{N}(\\mathbf{0},\\mathbf{I})$ and "
             "applying $T$ learnt denoising steps, each a Gaussian with mean $\\boldsymbol\\mu_\\theta(\\mathbf{x}_t,t)$ and "
             "a fixed variance $\\sigma_t^2$. We cannot compute $\\log p_\\theta(\\mathbf{x}_0)$ exactly (it would mean "
             "integrating over all $T$ intermediate images), but we can bound it from below. The bound has one term "
             "per step: how far the learnt step is from the ideal step of entry 9.3, measured by a KL divergence, "
             "plus a term for how far the fully noised image is from pure noise ($L_T$, tiny when $\\bar\\alpha_T\\approx0$) "
             "and a term for the final reconstruction ($L_0$)."),
         refresher=(
             "Apply the ELBO (Chapter 3) with $q(\\mathbf{x}_{1:T}\\mid\\mathbf{x}_0)$ as the variational posterior, "
             "write $q(\\mathbf{x}_t\\mid\\mathbf{x}_{t-1})=q(\\mathbf{x}_{t-1}\\mid\\mathbf{x}_t,\\mathbf{x}_0)\\,"
             "q(\\mathbf{x}_t\\mid\\mathbf{x}_0)/q(\\mathbf{x}_{t-1}\\mid\\mathbf{x}_0)$ for $t\\ge2$ and telescope the ratio "
             "of marginals (Sohl-Dickstein et al., 2015; Ho et al., 2020, Appendix A). Each $L_{t-1}$ is a KL between "
             "two Gaussians with the same covariance when $\\sigma_t^2=\\tilde\\beta_t$, so it reduces to "
             "$\\|\\tilde{\\boldsymbol\\mu}_t-\\boldsymbol\\mu_\\theta\\|^2/(2\\sigma_t^2)$ (entry 9.5). The bound is tight "
             "only if the reverse Gaussians match the true reverse transitions, which needs both the optimal mean "
             "and the correct reverse variance; with Ho et al.'s fixed $\\sigma_t^2=\\tilde\\beta_t$ a gap remains "
             "(below). The check uses one-dimensional Gaussian data $x_0\\sim\\mathcal{N}(0,s^2)$, for which the "
             "Bayes-optimal mean is known in closed form, computes every term analytically and confirms that the "
             "bound lies above the true negative log-likelihood $\\tfrac12\\log(2\\pi s^2)+\\tfrac12$. With the exact "
             "reverse variance $\\mathrm{Var}(x_{t-1}\\mid x_t)$ the bound is tight up to $L_T$; with Ho et al.'s fixed "
             "$\\sigma_t^2=\\tilde\\beta_t$, which is the exact reverse variance only for a point-mass data distribution, "
             "a gap of a few tenths of a nat remains however finely the chain is discretised. A Monte-Carlo estimate "
             "of one $L_{t-1}$ confirms the closed form. Bits per dimension in the paper "
             "(Chapter 10) are this bound with a discretised Gaussian for $L_0$; the repository check "
             "`repo_vlb_bpd_assembly_oracle_delta_data` covers that assembly."),
         code=r'''
def gaussian_data_bound(s2, betas, alphas, abar, abar_prev, reverse_var="beta_tilde"):
    """All terms of the bound for x0 ~ N(0, s2) with the Bayes-optimal reverse mean. Every expectation is over a
    linear-Gaussian model, so it is exact. reverse_var = "beta_tilde": sigma_t^2 = beta_tilde_t (Ho et al.'s choice,
    sigma_1^2 = beta_1 for L_0); "exact": sigma_t^2 = Var(x_{t-1} | x_t), the true reverse transition's variance."""
    beta_tilde = (1 - abar_prev) / (1 - abar) * betas
    post_c0 = np.sqrt(abar_prev) * betas / (1 - abar)
    v = s2 * (1 - abar) / (abar * s2 + 1 - abar)            # Var(x0 | x_t): the irreducible error of the optimal mean
    abT = abar[-1]
    L_T = 0.5 * (abT * s2 + (1 - abT) - 1 - math.log(1 - abT))      # KL(q(x_T|x0) || N(0,1)) averaged over x0
    if reverse_var == "beta_tilde":
        L_mid = post_c0[1:] ** 2 * v[1:] / (2 * beta_tilde[1:])      # equal variances: KL = E[(mu_tilde - mu_theta)^2] / (2 sigma^2)
        L_0 = 0.5 * math.log(2 * math.pi * betas[0]) + v[0] / (2 * betas[0])
    else:
        sig2 = beta_tilde + post_c0 ** 2 * v                          # Var(x_{t-1} | x_t) = beta_tilde + c0^2 Var(x0 | x_t)
        L_mid = 0.5 * np.log(sig2[1:] / beta_tilde[1:])               # Gaussian KL with the mean term absorbed: 0.5 log(sigma^2 / beta_tilde)
        L_0 = 0.5 * math.log(2 * math.pi * math.e * v[0])             # -E log N(x0; E[x0|x1], Var(x0|x1))
    return L_T, L_mid, L_0

s2 = 0.25
nll_true = 0.5 * math.log(2 * math.pi * s2) + 0.5
for kind in ("linear", "cosine"):
    b, a, ab, abp = make_schedule(kind=kind)
    for rv in ("beta_tilde", "exact"):
        L_T, L_mid, L_0 = gaussian_data_bound(s2, b, a, ab, abp, rv)
        bound = L_T + L_mid.sum() + L_0
        check(f"{kind}, sigma^2 = {rv}: bound >= true NLL on Gaussian data", bound >= nll_true - 1e-12,
              f"bound {bound:.5f} nats vs NLL {nll_true:.5f} (gap {bound - nll_true:.2e}; L_T {L_T:.1e})")
    L_T, L_mid, L_0 = gaussian_data_bound(s2, b, a, ab, abp, "exact")
    gap = L_T + L_mid.sum() + L_0 - nll_true
    check(f"{kind}: with the exact reverse variance the bound is tight up to L_T (0 <= gap <= L_T)", -1e-12 <= gap <= L_T + 1e-12,
          f"gap = {gap:.1e}, L_T = {L_T:.1e} (the prior N(0,1) is not exactly q(x_T))")
# Ho et al.'s sigma_t^2 = beta_tilde_t is the exact reverse variance only when the data distribution is a point mass:
# the gap it leaves on Gaussian data does not close as the same continuous schedule is sampled more finely
abar_cont = lambda u: np.exp(-(0.1 * u + 0.5 * 19.9 * u**2))      # one continuous VP schedule, abar(1) = e^{-10.05}
gaps = []
for T_ in (10, 100, 1000, 10_000):
    ab = abar_cont(np.arange(1, T_ + 1) / T_); abp = np.concatenate([[1.0], ab[:-1]]); b = 1 - ab / abp; a = 1 - b
    L_T, L_mid, L_0 = gaussian_data_bound(s2, b, a, ab, abp, "beta_tilde")
    gaps.append(L_T + L_mid.sum() + L_0 - nll_true)
check("sigma^2 = beta_tilde: the gap stays positive at every refinement (not the optimal variance for spread-out data)",
      all(g > 0.1 for g in gaps), "gaps at T = 10, 100, 1000, 10000: " + ", ".join(f"{g:.3f}" for g in gaps) + " nats")
beta_tilde = (1 - abar_prev) / (1 - abar) * betas
c0 = np.sqrt(abar_prev) * betas / (1 - abar)
rels = []
for s2_small in (1e-2, 1e-4, 1e-6):
    v = s2_small * (1 - abar) / (abar * s2_small + 1 - abar)
    rels.append(np.max((c0 ** 2 * v)[1:] / beta_tilde[1:]))
check("as the data distribution shrinks to a point, the exact reverse variance tends to beta_tilde",
      rels[0] > rels[1] > rels[2] and rels[2] < 1e-2,
      "max relative excess over beta_tilde at s^2 = 1e-2, 1e-4, 1e-6: " + ", ".join(f"{r:.3g}" for r in rels) + " (largest at t = 2)")
# Monte-Carlo check of one L_{t-1} term against the closed form
t = 400; n = 400_000
x0 = math.sqrt(s2) * rng.normal(size=n); eps = rng.normal(size=n)
xt = math.sqrt(abar[t]) * x0 + math.sqrt(1 - abar[t]) * eps
x0_hat = abar[t] ** 0.5 * s2 / (abar[t] * s2 + 1 - abar[t]) * xt            # E[x0 | x_t] for Gaussian data
bt_ = (1 - abar_prev[t]) / (1 - abar[t]) * betas[t]
c0 = math.sqrt(abar_prev[t]) * betas[t] / (1 - abar[t])
mc = np.mean((c0 * (x0 - x0_hat)) ** 2) / (2 * bt_)
th = gaussian_data_bound(s2, betas, alphas, abar, abar_prev)[1][t - 1]
check("MC estimate of L_{t-1} matches the closed form (t = 401)", abs(mc / th - 1) < 0.02, f"{mc:.4e} vs {th:.4e}")
'''),

    c.eq(["EQ-S1-047", "EQ-S4-048", "EQ-M-043"], "The noise-prediction mean and the weight of each bound term",
         r"\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)=\frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\Big),\qquad "
         r"L_{t-1}=\underbrace{\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}}_{\lambda_t}\;"
         r"\mathbb{E}\big\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\big\|^2+\text{const}.",
         plain=(
             "Instead of predicting the mean of the previous image directly, the network predicts the noise "
             "$\\boldsymbol\\epsilon$ that was added to make $\\mathbf{x}_t$. Plugging that prediction into the posterior "
             "mean of entry 9.3 gives the first formula. The second says that each term of the bound is then just "
             "a squared error between the true and the predicted noise, multiplied by a weight $\\lambda_t$ that "
             "depends only on the schedule. The weight is largest at the smallest $t$, where the image is almost "
             "clean, so the exact bound cares most about fine detail."),
         refresher=(
             "Substitute $\\mathbf{x}_0=(\\mathbf{x}_t-\\sqrt{1-\\bar\\alpha_t}\\boldsymbol\\epsilon)/\\sqrt{\\bar\\alpha_t}$ into "
             "$\\tilde{\\boldsymbol\\mu}_t$ of entry 9.3; the $\\mathbf{x}_t$ coefficients combine to $1/\\sqrt{\\alpha_t}$ "
             "and the $\\boldsymbol\\epsilon$ coefficient to $-\\beta_t/(\\sqrt{\\alpha_t}\\sqrt{1-\\bar\\alpha_t})$ (first "
             "check, exact to rounding). With the same form for $\\boldsymbol\\mu_\\theta$, "
             "$\\|\\tilde{\\boldsymbol\\mu}_t-\\boldsymbol\\mu_\\theta\\|^2/(2\\sigma_t^2)=\\lambda_t\\|\\boldsymbol\\epsilon-"
             "\\boldsymbol\\epsilon_\\theta\\|^2$ (second check). The main paper (`EQ-M-043`) says that $\\lambda_t$ "
             "*decreases* in $t$ so that the uniform weighting of entry 9.6 down-weights small $t$. On the linear "
             "schedule with $\\sigma_t^2=\\beta_t$ this is true only up to $t\\approx350$: $\\lambda_t$ then rises "
             "slightly, because $\\beta_t$ keeps growing while $1-\\bar\\alpha_t$ saturates. The conclusion the paper "
             "draws still holds in the form that matters: every one of the first several dozen steps outweighs "
             "every later step, so dropping the weight shifts emphasis from small to large $t$. The check states "
             "the precise version; the registry carries this as a wording caveat. The figure shows "
             "$\\beta_t$, $\\bar\\alpha_t$, the signal-to-noise ratio and $\\lambda_t$ for both schedules."),
         code=r'''
t = 400
a, b, ab = alphas[t], betas[t], abar[t]
x0, eps, eps_hat = rng.normal(size=5), rng.normal(size=5), rng.normal(size=5)
xt = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * eps
post_c0 = np.sqrt(abar_prev) * betas / (1 - abar); post_ct = np.sqrt(alphas) * (1 - abar_prev) / (1 - abar)
beta_tilde = (1 - abar_prev) / (1 - abar) * betas
mu_tilde = post_c0[t] * x0 + post_ct[t] * xt
mu_eps = (xt - b / math.sqrt(1 - ab) * eps) / math.sqrt(a)
check("mu_tilde rewritten through eps", *close(mu_tilde, mu_eps, 1e-12))
mu_theta = (xt - b / math.sqrt(1 - ab) * eps_hat) / math.sqrt(a)
s2_t = beta_tilde[t]
lhs = np.sum((mu_tilde - mu_theta) ** 2) / (2 * s2_t)
lam = b**2 / (2 * s2_t * a * (1 - ab))
check("KL mean term = lambda_t ||eps - eps_hat||^2", *close(lhs, lam * np.sum((eps - eps_hat) ** 2), 1e-12))
# x0-prediction form of the mean agrees with the eps form
x0_hat = (xt - math.sqrt(1 - ab) * eps_hat) / math.sqrt(ab)
check("x0-form mean = eps-form mean", *close(post_c0[t] * x0_hat + post_ct[t] * xt, mu_theta, 1e-12))
# the weight lambda_t along the schedule, for sigma_t^2 = beta_t and sigma_t^2 = beta_tilde_t
lam_beta = betas**2 / (2 * betas * alphas * (1 - abar))           # sigma^2 = beta_t  ->  beta/(2 alpha (1-abar))
lam_tilde = betas[1:] ** 2 / (2 * beta_tilde[1:] * alphas[1:] * (1 - abar[1:]))
tmin = int(np.argmin(lam_beta))
tstar = max(k for k in range(1, T) if lam_beta[:k].min() > lam_beta[k:].max())
check("lambda_t (sigma^2 = beta_t) decreases strictly up to its minimum", np.all(np.diff(lam_beta[:tmin]) < 0),
      f"minimum at t = {tmin + 1}; lambda_1 = {lam_beta[0]:.3f}, lambda_min = {lam_beta[tmin]:.4f}, lambda_T = {lam_beta[-1]:.4f}")
check("lambda_t is NOT monotone over the whole linear schedule (paper wording is a simplification)",
      lam_beta[-1] > lam_beta[tmin], f"rises again after t = {tmin + 1}")
check("each of the first t* steps outweighs every later step", tstar >= 50 and lam_tilde[:49].min() > lam_tilde[49:].max(),
      f"t* = {tstar} (sigma^2 = beta_t); first 49 steps dominate for sigma^2 = beta_tilde as well")
# figure: the schedule and the weight
fig, axs = plt.subplots(1, 4, figsize=(12, 2.6))
for kind, ls in (("linear", "-"), ("cosine", "--")):
    b_, a_, ab_, abp_ = make_schedule(kind=kind)
    tt = np.arange(1, T + 1)
    axs[0].plot(tt, b_, ls, label=kind); axs[1].plot(tt, ab_, ls)
    axs[2].semilogy(tt, ab_ / (1 - ab_), ls)
    axs[3].semilogy(tt, b_ / (2 * a_ * (1 - ab_)), ls)
for ax, name in zip(axs, (r"$\beta_t$", r"$\bar\alpha_t$", r"SNR$_t=\bar\alpha_t/(1-\bar\alpha_t)$", r"$\lambda_t$ ($\sigma_t^2=\beta_t$)")):
    ax.set_title(name); ax.set_xlabel("t")
axs[0].legend(); plt.tight_layout(); plt.show()
'''),

    c.eq(["EQ-M-030", "EQ-S1-048", "EQ-S1-049"], "The simple objective and its reading as denoising score matching",
         r"\mathcal{L}_{\mathrm{simple}}(\theta)=\mathbb{E}_{t\sim\mathcal{U}\{1,\dots,T\},\,\mathbf{x}_0,\,\boldsymbol{\epsilon}}"
         r"\Big[\big\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\big(\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon},\,t\big)\big\|^2\Big],"
         r"\qquad \mathbf{s}_\theta(\mathbf{x}_t,t)=-\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)}{\sqrt{1-\bar\alpha_t}}.",
         plain=(
             "Drop the weights $\\lambda_t$ and average the squared noise error uniformly over random steps: that is "
             "the loss used in practice, and it gives better-looking samples than the exact bound. The second formula "
             "connects two views: predicting the noise is the same as predicting the direction in which the density "
             "of noisy images increases (the *score*), up to a scale. So a diffusion model is also a score model, "
             "and Chapter 7's score-matching results apply."),
         refresher=(
             "Because $\\nabla_{\\mathbf{x}_t}\\log q(\\mathbf{x}_t\\mid\\mathbf{x}_0)=-(\\mathbf{x}_t-\\sqrt{\\bar\\alpha_t}"
             "\\mathbf{x}_0)/(1-\\bar\\alpha_t)=-\\boldsymbol\\epsilon/\\sqrt{1-\\bar\\alpha_t}$, the squared noise error equals "
             "$(1-\\bar\\alpha_t)$ times the denoising score-matching loss of Chapter 7 at noise level "
             "$\\sqrt{1-\\bar\\alpha_t}$ applied to the scaled data $\\sqrt{\\bar\\alpha_t}\\mathbf{x}_0$. "
             "$\\mathcal{L}_{\\mathrm{simple}}$ is therefore $\\sum_t L_{t-1}/\\lambda_t$ up to constants: a reweighting "
             "of the bound, not a bound, and the relative weight it gives step $t$ compared with the bound is "
             "$1/\\lambda_t$, which grows with $t$ over most of the schedule. Ho et al. (2020) report that this trade "
             "improves FID and worsens bits per dimension (entry 9.14). The repository's trainer "
             "(`diffusion/model.py`) implements exactly this loss."),
         code=r'''
t = 250; n = 100_000
x0 = rng.normal(size=n) * 0.5; eps = rng.normal(size=n)
ab = abar[t]
xt = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * eps
score_true = -(xt - math.sqrt(ab) * x0) / (1 - ab)             # grad_x log q(x_t | x_0)
check("grad log q(x_t|x_0) = -eps / sqrt(1 - abar_t)", *close(score_true, -eps / math.sqrt(1 - ab), 1e-12))
# an imperfect predictor: noise error and score error differ by the factor (1 - abar_t)
eps_hat = eps + 0.3 * rng.normal(size=n)
s_hat = -eps_hat / math.sqrt(1 - ab)
check("||eps - eps_hat||^2 = (1 - abar_t) ||s_true - s_hat||^2",
      *close(np.mean((eps - eps_hat) ** 2), (1 - ab) * np.mean((score_true - s_hat) ** 2), 1e-10))
# L_simple reweights the bound: uniform weight 1/T versus lambda_t
lam_beta = betas / (2 * alphas * (1 - abar))
ratio = (1.0 / T) / lam_beta                                      # weight given by L_simple relative to the bound
check("relative to the bound, L_simple up-weights large t (ratio at t=T over t=1 > 10)", ratio[-1] / ratio[0] > 10,
      f"ratio_T / ratio_1 = {ratio[-1] / ratio[0]:.0f} on the linear schedule")
'''),

    c.eq(["EQ-S4-045"], "Velocity parameterisation, the cosine schedule and the posterior variance",
         r"\mathbf{v}=\sqrt{\bar\alpha_t}\,\boldsymbol{\epsilon}-\sqrt{1-\bar\alpha_t}\,\mathbf{x}_0,\qquad"
         r"\bar\alpha_t^{\cos}=\frac{f(t)}{f(0)},\ f(t)=\cos^2\!\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big),\qquad"
         r"\tilde\beta_t=\beta_t\,\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}.",
         plain=(
             "The network can be asked to predict the noise, the clean image, or a mixture of the two called the "
             "velocity $\\mathbf{v}$. All three carry the same information: given $\\mathbf{x}_t$, any one of them "
             "determines the other two by a rotation. The cosine schedule is a different recipe for "
             "$\\bar\\alpha_t$ that keeps more of the image for longer. The third formula is the posterior variance "
             "of entry 9.3, which Nichol and Dhariwal use as one end of a learnt interpolation for $\\sigma_t^2$."),
         refresher=(
             "Write $\\sqrt{\\bar\\alpha_t}=\\cos\\phi_t$ and $\\sqrt{1-\\bar\\alpha_t}=\\sin\\phi_t$. Then "
             "$\\mathbf{x}_t=\\cos\\phi_t\\,\\mathbf{x}_0+\\sin\\phi_t\\,\\boldsymbol\\epsilon$ and "
             "$\\mathbf{v}=\\cos\\phi_t\\,\\boldsymbol\\epsilon-\\sin\\phi_t\\,\\mathbf{x}_0$ are an orthogonal rotation of "
             "$(\\mathbf{x}_0,\\boldsymbol\\epsilon)$, so $\\mathbf{x}_0=\\cos\\phi_t\\,\\mathbf{x}_t-\\sin\\phi_t\\,\\mathbf{v}$ and "
             "$\\boldsymbol\\epsilon=\\sin\\phi_t\\,\\mathbf{x}_t+\\cos\\phi_t\\,\\mathbf{v}$ (Salimans and Ho, 2022). The three "
             "parameterisations give the same optimum but different implicit weightings and conditioning: "
             "$\\boldsymbol\\epsilon$-prediction is ill-conditioned for recovering $\\mathbf{x}_0$ as $t\\to T$ (division by "
             "$\\sqrt{\\bar\\alpha_t}\\to0$), $\\mathbf{x}_0$-prediction as $t\\to0$. The cosine schedule with "
             "$s=0.008$ (Nichol and Dhariwal, 2021) has $\\bar\\alpha_t$ nearer one at the same $t/T$, which "
             "the check quantifies at a quarter of the way through the chain."),
         code=r'''
t = 600; ab = abar[t]
x0, eps = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
xt = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * eps
v = math.sqrt(ab) * eps - math.sqrt(1 - ab) * x0
check("x0 = sqrt(abar) x_t - sqrt(1-abar) v", *close(math.sqrt(ab) * xt - math.sqrt(1 - ab) * v, x0, 1e-12))
check("eps = sqrt(1-abar) x_t + sqrt(abar) v", *close(math.sqrt(1 - ab) * xt + math.sqrt(ab) * v, eps, 1e-12))
check("(x_t, v) is a rotation of (x0, eps): norms preserved", *close(xt**2 + v**2, x0**2 + eps**2, 1e-12))
# the cosine schedule keeps more signal at the same relative time
_, _, ab_lin, _ = make_schedule(kind="linear"); _, _, ab_cos, _ = make_schedule(kind="cosine")
q = T // 4
check("cosine abar > linear abar at t = T/4", ab_cos[q] > ab_lin[q], f"cosine {ab_cos[q]:.3f} vs linear {ab_lin[q]:.3f}")
# posterior variance formula agrees with the Bayes derivation of entry 9.3 and lies in [0, beta_t]
beta_tilde = betas * (1 - abar_prev) / (1 - abar)
check("beta_tilde formula = (1 - abar_{t-1}) beta_t / (1 - abar_t)", *close(beta_tilde, (1 - abar_prev) / (1 - abar) * betas, 1e-15))
check("0 <= beta_tilde <= beta", np.all(beta_tilde >= 0) and np.all(beta_tilde <= betas))
'''),

    c.eq(["EQ-S1-050"], "Schedule invariance of the continuous-time bound",
         r"\mathcal{L}_\infty=\tfrac12\int_{\mathrm{SNR}_{\min}}^{\mathrm{SNR}_{\max}}"
         r"\mathbb{E}\big\|\mathbf{x}_0-\hat{\mathbf{x}}_\theta(\mathbf{z}_\gamma,\gamma)\big\|^2\,\mathrm{d}\gamma"
         r"\quad\text{depends on the schedule only through }\mathrm{SNR}_{\min},\mathrm{SNR}_{\max}.",
         plain=(
             "In the limit of infinitely many steps the bound stops caring *how fast* noise is added, only *how "
             "much* noise there is at the start and at the end. Two schedules that begin and end at the same "
             "signal-to-noise ratio give the same bound, whatever path they take in between. The finite-step "
             "bound does depend on the path, which is one reason schedules still matter in practice."),
         refresher=(
             "Kingma et al. (2021) change the integration variable from time to the log signal-to-noise ratio and "
             "show that the diffusion loss becomes the integral above with the reconstruction error "
             "$\\mathbb{E}\\|\\mathbf{x}_0-\\hat{\\mathbf{x}}_\\theta\\|^2$ evaluated at noise level $\\gamma$; the time "
             "parameterisation cancels. The same holds for any weighted objective whose weight is a function of the "
             "signal-to-noise ratio. Kingma and Gao (2023) then show that a weighted objective with a weighting "
             "*monotone* in the noise level is the ELBO of a model trained on Gaussian-noise-augmented data, which "
             "the paper is careful not to read as making $\\mathcal{L}_{\\mathrm{simple}}$ a bound on the clean data's "
             "likelihood. The check uses Gaussian data, for which $\\mathbb{E}\\|\\mathbf{x}_0-\\hat{\\mathbf{x}}\\|^2="
             "s^2/(1+\\gamma s^2)$ and the integral is $\\tfrac12\\log\\big((1+\\gamma_{\\max}s^2)/(1+\\gamma_{\\min}s^2)\\big)$: "
             "two different time parameterisations between the same endpoints give the same value when integrated "
             "in $t$ with the factor $\\mathrm{d}\\gamma/\\mathrm{d}t$, while the discrete 20-step bounds differ."),
         code=r'''
s2 = 0.25
g_min, g_max = 1e-3, 1e3                                     # SNR endpoints shared by both schedules
closed = 0.5 * math.log((1 + g_max * s2) / (1 + g_min * s2))
# two schedules in t in [0,1] between the same endpoints: log-linear SNR and a cosine-like path
lg_min, lg_max = math.log(g_min), math.log(g_max)
snr_A = lambda t: np.exp(lg_max + (lg_min - lg_max) * t)
snr_B = lambda t: np.exp(lg_max + (lg_min - lg_max) * (1 - np.cos(np.pi * t)) / 2)
mse = lambda g: s2 / (1 + g * s2)
vals = []
for snr in (snr_A, snr_B):
    dsnr = lambda t, h=1e-6: (snr(t + h) - snr(t - h)) / (2 * h)
    val, _ = scipy.integrate.quad(lambda t: 0.5 * mse(snr(t)) * (-dsnr(t)), 1e-4, 1 - 1e-4, limit=400)
    vals.append(val)
check("continuous-time bound: schedule A = closed form", *close(vals[0], closed, 2e-3, "A "))
check("continuous-time bound: schedule B = closed form", *close(vals[1], closed, 2e-3, "B "))
check("the two schedules agree with each other", *close(vals[0], vals[1], 2e-3))
# the discrete-time bound (20 steps, same endpoints) does depend on the path
def discrete_sum(snr, K=20):
    ts = np.linspace(0, 1, K + 1)
    g = snr(ts); ab_ = g / (1 + g)                          # SNR = abar / (1 - abar)
    tot = 0.0
    for i in range(1, K + 1):                                # KL terms with Bayes-optimal mean, equal variances
        ab_t, ab_p = ab_[i], ab_[i - 1]
        bt = 1 - ab_t / ab_p
        c0 = math.sqrt(ab_p) * bt / (1 - ab_t); btil = (1 - ab_p) / (1 - ab_t) * bt
        tot += c0**2 * s2 * (1 - ab_t) / (ab_t * s2 + 1 - ab_t) / (2 * btil)
    return tot
dA, dB = discrete_sum(snr_A), discrete_sum(snr_B)
check("20-step discrete sums differ between the two paths", abs(dA - dB) / dA > 0.01, f"{dA:.4f} vs {dB:.4f}")
'''),

    # ------------------------------------------------------------------------------------------------ DDIM
    c.section("DDIM: one network, a family of samplers", (
        "The training loss of entry 9.6 depends on the forward process only through the marginals "
        "$q(\\mathbf{x}_t\\mid\\mathbf{x}_0)$. Song et al. (2021) exploit this: any process with the same marginals "
        "can be reversed with the same trained network, and a one-parameter family of such processes gives "
        "samplers from fully stochastic to fully deterministic, on any sub-sequence of steps."))

    c.eq(["EQ-M-031", "EQ-S1-051", "EQ-S4-046"], "The DDIM update and its noise level $\\sigma_{\\tau_i}(\\eta)$",
         r"\mathbf{x}_{\tau_{i-1}}=\sqrt{\bar\alpha_{\tau_{i-1}}}\;\underbrace{\frac{\mathbf{x}_{\tau_i}-\sqrt{1-\bar\alpha_{\tau_i}}\,\boldsymbol{\epsilon}_\theta}{\sqrt{\bar\alpha_{\tau_i}}}}_{\hat{\mathbf{x}}_0}"
         r"+\sqrt{1-\bar\alpha_{\tau_{i-1}}-\sigma_{\tau_i}^2}\;\boldsymbol{\epsilon}_\theta+\sigma_{\tau_i}\boldsymbol{\epsilon},\qquad"
         r"\sigma_{\tau_i}=\eta\sqrt{\frac{1-\bar\alpha_{\tau_{i-1}}}{1-\bar\alpha_{\tau_i}}\Big(1-\frac{\bar\alpha_{\tau_i}}{\bar\alpha_{\tau_{i-1}}}\Big)}.",
         plain=(
             "Each DDIM step does three things: it uses the predicted noise to guess the clean image "
             "$\\hat{\\mathbf{x}}_0$, it re-noises that guess to the earlier level $\\tau_{i-1}$ partly with the *same* "
             "predicted noise and partly with *fresh* noise, and the knob $\\eta$ sets the mix. With $\\eta=0$ no fresh "
             "noise is added and the path from noise to image is a fixed function; with $\\eta=1$ the step is as "
             "random as the original chain. The steps $\\tau_1<\\dots<\\tau_S$ can be any subset of the $T$ training "
             "steps, which is how 1000-step models are sampled in 20 or 50 steps."),
         refresher=(
             "The family comes from a non-Markovian forward process "
             "$q_\\sigma(\\mathbf{x}_{\\tau_{i-1}}\\mid\\mathbf{x}_{\\tau_i},\\mathbf{x}_0)=\\mathcal{N}\\big(\\sqrt{\\bar\\alpha_{\\tau_{i-1}}}"
             "\\mathbf{x}_0+\\sqrt{1-\\bar\\alpha_{\\tau_{i-1}}-\\sigma^2}\\,\\tfrac{\\mathbf{x}_{\\tau_i}-\\sqrt{\\bar\\alpha_{\\tau_i}}\\mathbf{x}_0}"
             "{\\sqrt{1-\\bar\\alpha_{\\tau_i}}},\\sigma^2\\mathbf{I}\\big)$, chosen so that the marginals remain "
             "$q(\\mathbf{x}_t\\mid\\mathbf{x}_0)$ of entry 9.2 for every $\\sigma$ (Song et al., 2021, Lemma 1). Replacing "
             "$\\mathbf{x}_0$ by the network's estimate gives the update. Two facts are checked: when the network is "
             "the true noise (or the Bayes-optimal predictor on Gaussian data), every $\\eta$ maps "
             "$q(\\mathbf{x}_{\\tau_i}\\mid\\mathbf{x}_0)$ onto $q(\\mathbf{x}_{\\tau_{i-1}}\\mid\\mathbf{x}_0)$ exactly; and "
             "$\\sigma^2(\\eta=1)$ equals the posterior variance $\\tilde\\beta$ of the Markov chain restricted to the "
             "sub-sequence (entry 9.10). The quoted numbers in `EQ-S4-046` (FID 4.67 at 50 steps against 3.17 at "
             "1000 on CIFAR-10) are literature values from Song et al. and Ho et al., not reproduced here; the "
             "paper's own measurement of this trade-off is §8.5. The repository check "
             "`repo_samplers_with_bayes_optimal_eps_on_gaussian_data` confirms that `GaussianDiffusion.ddim_sample` "
             "implements the update. The figure shows the family on a two-dimensional mixture, where the "
             "Bayes-optimal $\\boldsymbol\\epsilon_\\theta$ is known in closed form."),
         code=r'''
def sigma_ddim(eta, ab_t, ab_prev):
    return eta * math.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))

# (a) eta = 1 on the full chain reproduces beta_tilde; on a sub-sequence, the sub-chain's posterior variance
sig2_full = np.array([sigma_ddim(1.0, abar[t], abar_prev[t]) ** 2 for t in range(1, T)])
beta_tilde = (1 - abar_prev) / (1 - abar) * betas
check("sigma^2(eta=1, S=T) = beta_tilde", *close(sig2_full, beta_tilde[1:], 1e-15))
tau = np.array([0, 99, 299, 599, 999]); ok_sub = True
for i in range(1, len(tau)):
    a_t, a_p = abar[tau[i]], abar[tau[i - 1]]
    b_sub = 1 - a_t / a_p                                        # beta of the one merged step tau_{i-1} -> tau_i
    ok_sub &= abs(sigma_ddim(1.0, a_t, a_p) ** 2 - (1 - a_p) / (1 - a_t) * b_sub) < 1e-15
check("sigma^2(eta=1) on a sub-sequence = posterior variance of the merged step", ok_sub)
# (b) every eta preserves the marginal when eps_theta is the true noise (exact linear-Gaussian propagation + MC)
x0v, n = 0.8, 400_000; t, tp = 700, 300; ok_m = True; msgs = []
for eta in (0.0, 0.5, 1.0):
    e = rng.normal(size=n)
    xt = math.sqrt(abar[t]) * x0v + math.sqrt(1 - abar[t]) * e
    s = sigma_ddim(eta, abar[t], abar[tp])
    x0_hat = (xt - math.sqrt(1 - abar[t]) * e) / math.sqrt(abar[t])
    xp = math.sqrt(abar[tp]) * x0_hat + math.sqrt(1 - abar[tp] - s**2) * e + s * rng.normal(size=n)
    exact_var = (1 - abar[tp] - s**2) + s**2                     # the two noise terms are independent
    ok_m &= abs(xp.mean() - math.sqrt(abar[tp]) * x0v) < 4 * math.sqrt((1 - abar[tp]) / n)
    ok_m &= abs(xp.var() - (1 - abar[tp])) < 4 * (1 - abar[tp]) * math.sqrt(2 / n) and abs(exact_var - (1 - abar[tp])) < 1e-15
    msgs.append(f"eta={eta}: var {xp.var():.4f}")
check("marginal q(x_{t'}|x0) preserved for eta in {0, 0.5, 1}", ok_m, "; ".join(msgs) + f" (target {1-abar[tp]:.4f})")
# (c) Bayes-optimal eps for a 2-D Gaussian mixture, and the eta family from the same x_T
pis = np.array([0.5, 0.5]); mus = np.array([[-2.0, 0.0], [2.0, 0.0]]); s2c = 0.3
def eps_opt(x, t):
    ab = abar[t]; var = ab * s2c + 1 - ab
    d2 = ((x[:, None, :] - math.sqrt(ab) * mus[None]) ** 2).sum(-1)
    w = pis * np.exp(-0.5 * d2 / var); w /= w.sum(1, keepdims=True)
    # E[x0|x_t] within each component shrinks x_t towards the component mean; combine with the responsibilities w
    x0_hat = (w[:, :, None] * ((s2c * math.sqrt(ab) * x[:, None, :] + (1 - ab) * mus[None]) / var)).sum(1)
    return (x - math.sqrt(ab) * x0_hat) / math.sqrt(1 - ab)
def ddim_run(xT, eta, S=20):
    seq = np.round(np.linspace(0, T - 1, S)).astype(int); x = xT.copy(); traj = [x.copy()]
    for i in reversed(range(S)):
        t = seq[i]; ab_p = abar[seq[i - 1]] if i > 0 else 1.0
        e = eps_opt(x, t); s = sigma_ddim(eta, abar[t], ab_p) if i > 0 else 0.0
        x0_hat = (x - math.sqrt(1 - abar[t]) * e) / math.sqrt(abar[t])
        x = math.sqrt(ab_p) * x0_hat + math.sqrt(max(1 - ab_p - s**2, 0)) * e + s * rng.normal(size=x.shape)
        traj.append(x.copy())
    return x, np.array(traj)
xT = rng.normal(size=(3000, 2))
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True, sharey=True)
for ax, eta in zip(axs, (0.0, 0.5, 1.0)):
    xs, traj = ddim_run(xT, eta)
    ax.scatter(xs[:, 0], xs[:, 1], s=2, alpha=0.4, label="samples")
    for k in range(6):
        ax.plot(traj[:, k, 0], traj[:, k, 1], "-", lw=0.8, color="C3", alpha=0.8)
    ax.set_title(fr"DDIM, $\eta={eta}$, 20 steps"); ax.set_aspect("equal")
    # the samples should recover the mixture: half the mass near each centre, component variance s2c
    left = xs[xs[:, 0] < 0]; right = xs[xs[:, 0] >= 0]
    check(f"eta={eta}: both modes recovered with equal mass (20 steps)",
          abs(len(left) / len(xs) - 0.5) < 0.05 and abs(left.mean(0)[0] + 2) < 0.15 and abs(right.mean(0)[0] - 2) < 0.15,
          f"share left {len(left)/len(xs):.3f}, means {left.mean(0)[0]:.2f}/{right.mean(0)[0]:.2f}, within-mode var {left.var(0).mean():.3f} (data {s2c})")
axs[0].legend(loc="upper left"); plt.tight_layout(); plt.show()
# with the exact eps both samplers under-disperse within each mode at 20 steps (a discretisation error, larger for the
# stochastic update) and converge to the data's spread as the number of steps grows
xT_big = rng.normal(size=(6000, 2))
vs = {eta: {S: (lambda x: x[x[:, 0] < 0].var(0).mean())(ddim_run(xT_big, eta, S=S)[0]) for S in (20, 1000)} for eta in (0.0, 1.0)}
check("20 steps: deterministic DDIM is closer to the data's within-mode variance than the stochastic update",
      abs(vs[0.0][20] - s2c) < abs(vs[1.0][20] - s2c), f"eta=0: {vs[0.0][20]:.3f}, eta=1: {vs[1.0][20]:.3f}, data {s2c}")
for eta in (0.0, 1.0):
    check(f"eta = {eta}: within-mode variance converges to the data's (20 -> 1000 steps; within 5% at 1000)",
          abs(vs[eta][20] - s2c) > abs(vs[eta][1000] - s2c) and abs(vs[eta][1000] / s2c - 1) < 0.05,
          ", ".join(f"S={S}: {v:.3f}" for S, v in vs[eta].items()) + f" (SE of each estimate about {s2c * math.sqrt(2 / 3000):.3f})")
'''),

    c.eq(["EQ-M-032", "EQ-S1-052"], "$\\eta=1$ is ancestral sampling, $\\eta=0$ is an Euler step of the probability-flow ODE",
         r"\eta=1:\ \sigma_{\tau_i}^2=\tilde\beta_{\tau_i}\text{ on the sub-sequence};\qquad"
         r"\eta=0:\ \bar{\mathbf{x}}_{\tau_{i-1}}=\bar{\mathbf{x}}_{\tau_i}+\big(\bar\sigma_{\tau_{i-1}}-\bar\sigma_{\tau_i}\big)\boldsymbol{\epsilon}_\theta,"
         r"\quad \bar{\mathbf{x}}=\frac{\mathbf{x}}{\sqrt{\bar\alpha}},\ \bar\sigma=\sqrt{\frac{1-\bar\alpha}{\bar\alpha}} .",
         plain=(
             "At one end of the family, $\\eta=1$, a DDIM step is exactly the original DDPM denoising step "
             "(restricted to the chosen sub-sequence), with the posterior variance of entry 9.3 as its noise. At the "
             "other end, $\\eta=0$, the step is deterministic and, after dividing everything by "
             "$\\sqrt{\\bar\\alpha}$, it is a straight-line (Euler) step along the predicted noise direction: the "
             "simplest numerical integrator of the ordinary differential equation of Chapter 8."),
         refresher=(
             "For $\\eta=1$, substitute $\\sigma^2=\\tilde\\beta$ into the update of entry 9.9 and compare with "
             "$\\tilde{\\boldsymbol\\mu}(\\mathbf{x}_t,\\hat{\\mathbf{x}}_0)$ of entry 9.3: the coefficients of "
             "$\\hat{\\mathbf{x}}_0$ and $\\boldsymbol\\epsilon_\\theta$ agree algebraically, so the DDIM step is the DDPM "
             "ancestral step with $\\sigma_t^2=\\tilde\\beta_t$ (first check, to rounding). For $\\eta=0$, divide the "
             "update by $\\sqrt{\\bar\\alpha_{\\tau_{i-1}}}$: $\\bar{\\mathbf{x}}_{\\tau_{i-1}}=\\hat{\\mathbf{x}}_0+"
             "\\bar\\sigma_{\\tau_{i-1}}\\boldsymbol\\epsilon_\\theta$ and $\\hat{\\mathbf{x}}_0=\\bar{\\mathbf{x}}_{\\tau_i}-"
             "\\bar\\sigma_{\\tau_i}\\boldsymbol\\epsilon_\\theta$, which is Euler's method for "
             "$\\mathrm{d}\\bar{\\mathbf{x}}/\\mathrm{d}\\bar\\sigma=\\boldsymbol\\epsilon_\\theta$, the probability-flow ODE in "
             "the variance-exploding parameterisation of Karras et al. (2022). This is why the paper treats DDIM "
             "with $\\eta=0$, DPM-Solver and Heun's method as integrators of one ODE differing in order and time "
             "grid (§8.5), and why the deterministic sampler carries network error along the trajectory while the "
             "stochastic one partly contracts it."),
         code=r'''
post_c0 = np.sqrt(abar_prev) * betas / (1 - abar); post_ct = np.sqrt(alphas) * (1 - abar_prev) / (1 - abar)
beta_tilde = (1 - abar_prev) / (1 - abar) * betas
t = 500
xt, eps_hat = rng.normal(size=4), rng.normal(size=4)
x0_hat = (xt - math.sqrt(1 - abar[t]) * eps_hat) / math.sqrt(abar[t])
s = math.sqrt(beta_tilde[t])
ddim_mean = math.sqrt(abar_prev[t]) * x0_hat + math.sqrt(1 - abar_prev[t] - s**2) * eps_hat
ddpm_mean = post_c0[t] * x0_hat + post_ct[t] * xt
check("eta = 1: DDIM mean = DDPM ancestral mean (sigma^2 = beta_tilde)", *close(ddim_mean, ddpm_mean, 1e-12))
# eta = 0 equals one Euler step in the rescaled variables
t, tp = 600, 400
x0_hat = (xt - math.sqrt(1 - abar[t]) * eps_hat) / math.sqrt(abar[t])
ddim0 = math.sqrt(abar[tp]) * x0_hat + math.sqrt(1 - abar[tp]) * eps_hat
sbar = lambda a: math.sqrt((1 - a) / a)
euler = math.sqrt(abar[tp]) * (xt / math.sqrt(abar[t]) + (sbar(abar[tp]) - sbar(abar[t])) * eps_hat)
check("eta = 0: DDIM step = Euler step in (x/sqrt(abar), sqrt((1-abar)/abar))", *close(ddim0, euler, 1e-12))
# eta = 0 is deterministic: two runs from the same x_T coincide; eta = 1 does not
xT = rng.normal(size=(200, 2))
state = rng.bit_generator.state
a0, _ = ddim_run(xT, 0.0); rng.bit_generator.state = state; b0, _ = ddim_run(xT, 0.0)
a1, _ = ddim_run(xT, 1.0); b1, _ = ddim_run(xT, 1.0)
check("eta = 0: identical outputs from identical x_T", np.max(np.abs(a0 - b0)) == 0.0)
check("eta = 1: fresh noise makes outputs differ", np.max(np.abs(a1 - b1)) > 0.1, f"max |diff| = {np.max(np.abs(a1 - b1)):.2f}")
'''),

    # ------------------------------------------------------------------------------------------------ guidance
    c.section("Classifier-free guidance"),

    c.eq(["EQ-M-050", "EQ-S1-053", "EQ-S4-047", "EQ-S1-054"], "Classifier-free guidance and the tilted score",
         r"\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t,t,\mathbf{c})=\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t,\varnothing)"
         r"+w\big[\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t,\mathbf{c})-\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t,\varnothing)\big],\qquad"
         r"-\frac{\tilde{\boldsymbol{\epsilon}}_\theta}{\sqrt{1-\bar\alpha_t}}=\nabla_{\mathbf{x}_t}\log\big[p_t(\mathbf{x}_t\mid\mathbf{c})\,p_t(\mathbf{c}\mid\mathbf{x}_t)^{w-1}\big]"
         r"=\nabla\log p_t(\mathbf{x}_t\mid\mathbf{c})+(w-1)\nabla\log p_t(\mathbf{c}\mid\mathbf{x}_t).",
         plain=(
             "A conditional model is trained with the condition (a class, a caption) sometimes blanked out, so one "
             "network gives both a conditional and an unconditional noise estimate. Guidance moves from the "
             "unconditional estimate *past* the conditional one by a factor $w$. With $w=1$ nothing changes; with "
             "$w>1$ samples match the condition more strongly and vary less. The second formula explains why: the "
             "guided step follows the conditional density times an extra factor, 'how clearly does this image show "
             "the condition?', raised to the power $w-1$; that factor is what a classifier would supply."),
         refresher=(
             "In score units ($\\mathbf{s}=-\\boldsymbol\\epsilon/\\sqrt{1-\\bar\\alpha_t}$), Bayes' rule gives "
             "$\\nabla\\log p_t(\\mathbf{c}\\mid\\mathbf{x}_t)=\\nabla\\log p_t(\\mathbf{x}_t\\mid\\mathbf{c})-\\nabla\\log p_t(\\mathbf{x}_t)$, "
             "so $\\mathbf{s}_\\varnothing+w(\\mathbf{s}_c-\\mathbf{s}_\\varnothing)=\\mathbf{s}_c+(w-1)\\nabla\\log p_t(\\mathbf{c}\\mid\\mathbf{x}_t)$, "
             "the classifier-guidance form of Dhariwal and Nichol (2021) with the classifier replaced by the "
             "difference of two scores (Ho and Salimans, 2022). Two conventions coexist: the paper's $w$ with $w=1$ "
             "unguided (`EQ-M-050`, the convention of the diffusers pipeline used in §8), and Ho and Salimans's "
             "$w_{\\mathrm{HS}}=w-1$. The paper states a caveat worth checking: the guided score is the score of a "
             "tilted density *at each noise level*, but the distribution of the final samples is not in general "
             "$p(\\mathbf{x}_0\\mid\\mathbf{c})\\,p(\\mathbf{c}\\mid\\mathbf{x}_0)^{w-1}$, because the tilted marginals are not "
             "the marginals of any single diffusion of a tilted data distribution. The last check demonstrates this "
             "on a one-dimensional mixture, where everything is available in closed form: integrating the "
             "probability-flow ODE with the guided score from $t=1$ to $0$ gives samples whose mean and variance "
             "differ markedly from those of the tilted density, while $w=1$ recovers $p(x\\mid c)$."),
         code=r'''
pis = np.array([0.3, 0.7]); ms = np.array([-1.0, 2.0]); c_ = 1
def logp_x(x):  return np.log(np.sum(pis * scipy.stats.norm.pdf(x, ms, 1.0)))
def logp_xc(x, c): return scipy.stats.norm.logpdf(x, ms[c], 1.0)
def logp_cx(x, c): return np.log(pis[c]) + logp_xc(x, c) - logp_x(x)
grad = lambda fn, x, e=1e-6: (fn(x + e) - fn(x - e)) / (2 * e)
x = 0.37; errs, errs_hs = [], []
for w in (1.0, 1.5, 3.0, 7.5):
    s_u, s_c = grad(logp_x, x), grad(lambda y: logp_xc(y, c_), x)
    guided = s_u + w * (s_c - s_u)
    tilt = grad(lambda y: logp_xc(y, c_) + (w - 1) * logp_cx(y, c_), x)
    hs = (1 + (w - 1)) * s_c - (w - 1) * s_u
    errs.append(abs(guided - tilt)); errs_hs.append(abs(guided - hs))
check("guided score = score of p(x|c) p(c|x)^(w-1) for w in {1, 1.5, 3, 7.5}", max(errs) < 1e-6, f"max |diff| = {max(errs):.1e}")
check("illustration (algebra): Ho & Salimans form with w_HS = w - 1 is the same estimate", max(errs_hs) < 1e-12)
check("w = 1 is the unguided conditional score", abs((s_u + 1.0 * (s_c - s_u)) - s_c) < 1e-12)
# final-sample law under guidance vs the tilted density, on the VP SDE in continuous time
bmin, bmax = 0.1, 20.0
abar_c = lambda t: math.exp(-(bmin * t + 0.5 * (bmax - bmin) * t * t))      # abar(t) = exp(-int_0^t beta); abar(1) = e^{-10.05}
beta_c = lambda t: bmin + (bmax - bmin) * t
def guided_score(x, t, w):
    a = abar_c(t); mu = math.sqrt(a) * ms; var = 1.0            # component variance a*1 + (1-a) = 1
    pdfs = scipy.stats.norm.pdf(x[:, None], mu, 1.0); s_comp = -(x[:, None] - mu) / var
    px = (pis * pdfs).sum(1); s_u = (pis * pdfs * s_comp).sum(1) / px; s_c = s_comp[:, c_]
    return s_u + w * (s_c - s_u)
def pf_ode_samples(w, n=20_000, steps=400):
    x = rng.normal(size=n)
    for i in range(steps):
        t = 1 - (i + 0.5) / steps; dt = 1.0 / steps
        x = x + (0.5 * beta_c(t) * x + 0.5 * beta_c(t) * guided_score(x, t, w)) * dt   # dx/dt = f - g^2 s / 2, run backwards
    return x
xs = np.linspace(-8, 10, 36001)
def tilt_moments(w):
    d = scipy.stats.norm.pdf(xs, ms[c_], 1) * (pis[c_] * scipy.stats.norm.pdf(xs, ms[c_], 1) / (pis * scipy.stats.norm.pdf(xs[:, None], ms, 1)).sum(1)) ** (w - 1)
    d /= np.trapezoid(d, xs); m = np.trapezoid(xs * d, xs)
    return m, np.trapezoid((xs - m) ** 2 * d, xs)
s1 = pf_ode_samples(1.0)
check("w = 1: PF-ODE samples follow p(x|c) = N(2, 1)", abs(s1.mean() - 2) < 0.05 and abs(s1.var() - 1) < 0.05,
      f"mean {s1.mean():.3f}, var {s1.var():.3f}")
s3 = pf_ode_samples(3.0); m3, v3 = tilt_moments(3.0)
check("w = 3: final samples are NOT distributed as p(x|c) p(c|x)^(w-1)", abs(s3.mean() - m3) > 0.5,
      f"guided samples mean {s3.mean():.2f}, var {s3.var():.2f}; tilted density mean {m3:.2f}, var {v3:.2f}")
'''),

    # ------------------------------------------------------------------------------------------------ flow matching
    c.section("Flow matching"),

    c.eq(["EQ-M-033", "EQ-S1-055", "EQ-S1-056"], "The straight path and the conditional flow-matching loss",
         r"\mathbf{x}_t=(1-t)\,\mathbf{x}_0+t\,\mathbf{x}_1,\quad \mathbf{x}_0\sim p_{\mathrm{data}},\ \mathbf{x}_1=\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),\qquad"
         r"\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t,\mathbf{x}_0,\mathbf{x}_1}\big\|\mathbf{v}_\theta(\mathbf{x}_t,t)-(\mathbf{x}_1-\mathbf{x}_0)\big\|^2 .",
         plain=(
             "Draw a data point and a noise point and join them by a straight line; at time $t$ you are a fraction "
             "$t$ of the way from data to noise. The network $\\mathbf{v}_\\theta$ is trained to predict the constant "
             "velocity of that line, $\\mathbf{x}_1-\\mathbf{x}_0$, from the current position. At test time, start at "
             "noise and integrate $\\mathrm{d}\\mathbf{x}/\\mathrm{d}t=\\mathbf{v}_\\theta$ backwards to $t=0$. Note the "
             "convention: here $t=0$ is data and $t=1$ is noise, the same direction as the diffusion chain above "
             "(data at 0), but the opposite of Lipman et al. (2023), who start from noise at $t=0$."),
         refresher=(
             "Regressing on the *conditional* velocity $\\mathbf{x}_1-\\mathbf{x}_0$ has the same minimiser as regressing on "
             "the intractable *marginal* velocity $\\mathbb{E}[\\mathbf{x}_1-\\mathbf{x}_0\\mid\\mathbf{x}_t]$ because the two "
             "losses differ by a constant (Lipman et al., 2023, Theorem 2; Liu et al., 2023; Albergo and "
             "Vanden-Eijnden, 2023). The marginal velocity transports $p_{\\mathrm{data}}$ to the Gaussian, so the "
             "learnt ODE is a continuous normalising flow (Chapter 5) trained without simulating it; its likelihood "
             "is available through the instantaneous change of variables. The variance-preserving diffusion path "
             "$\\mathbf{x}_t=\\sqrt{\\bar\\alpha_t}\\mathbf{x}_0+\\sqrt{1-\\bar\\alpha_t}\\boldsymbol\\epsilon$ is another "
             "admissible path; the straight line gives trajectories that reflow makes straighter and cheaper to "
             "integrate. For Gaussian data $\\mathbf{x}_0\\sim\\mathcal{N}(0,s^2)$ the marginal velocity is linear, "
             "$v(t,x)=\\frac{t-(1-t)s^2}{(1-t)^2s^2+t^2}\\,x$; the checks compare it with the Monte-Carlo regression "
             "slope and integrate it from the Gaussian back to the data law."),
         code=r'''
s2 = 0.3
v_marg = lambda t, x: (t - (1 - t) * s2) / ((1 - t) ** 2 * s2 + t**2) * x    # E[x1 - x0 | x_t = x] for Gaussian data
# path endpoints
x0, x1 = rng.normal(size=5) * math.sqrt(s2), rng.normal(size=5)
path = lambda t: (1 - t) * x0 + t * x1
check("x_t at t=0 is the data point and at t=1 the noise point", np.allclose(path(0.0), x0) and np.allclose(path(1.0), x1))
check("the conditional velocity d x_t / dt is x1 - x0", *close((path(0.5 + 1e-6) - path(0.5 - 1e-6)) / 2e-6, x1 - x0, 1e-8))
# marginal velocity = regression of (x1 - x0) on x_t
t = 0.4; n = 1_000_000
X0 = math.sqrt(s2) * rng.normal(size=n); X1 = rng.normal(size=n); Xt = (1 - t) * X0 + t * X1
slope = np.cov(X1 - X0, Xt)[0, 1] / Xt.var()
check("MC regression slope = closed-form marginal velocity at t = 0.4", *close(slope, v_marg(t, 1.0), 3e-3))
# the conditional loss and the marginal loss differ by a constant: same minimiser (checked on the linear family v = k x)
ks = np.linspace(-2, 2, 81)
L_cond = [np.mean((k * Xt - (X1 - X0)) ** 2) for k in ks]
L_marg = [np.mean((k * Xt - v_marg(t, Xt)) ** 2) for k in ks]
check("conditional and marginal losses have the same minimiser over v = k x",
      ks[int(np.argmin(L_cond))] == ks[int(np.argmin(L_marg))], f"argmin k = {ks[int(np.argmin(L_cond))]:.2f} vs closed form {v_marg(t, 1.0):.2f}")
check("the two losses differ by a constant", np.std(np.array(L_cond) - np.array(L_marg)) < 1e-3)
# integrating the marginal velocity from t = 1 to 0 maps N(0,1) onto N(0, s2)
z = np.linspace(-3, 3, 13)
sol = scipy.integrate.solve_ivp(v_marg, (1.0, 1e-9), z, rtol=1e-10, atol=1e-12)
check("ODE from noise to data maps N(0,1) quantiles to N(0, s2) quantiles", *close(sol.y[:, -1], math.sqrt(s2) * z, 1e-6))
'''),

    # ------------------------------------------------------------------------------------------------ foundation-scale
    c.section("Conditional denoising at foundation scale"),

    c.eq(["EQ-S4-051"], "Conditional denoising with a frozen text encoder, and the contrastive objective it relies on",
         r"\mathcal{L}=\mathbb{E}_{t,\mathbf{z}_0,\boldsymbol{\epsilon}}\big\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta\big(\mathbf{z}_t,t,\tau(\mathbf{c})\big)\big\|^2,\qquad"
         r"\mathcal{L}_{\mathrm{InfoNCE}}=\tfrac12\Big[\mathbb{E}_i\,\mathrm{CE}\big(\mathrm{row}_i(S/\kappa),i\big)+\mathbb{E}_j\,\mathrm{CE}\big(\mathrm{col}_j(S/\kappa),j\big)\Big],\ S_{ij}=\langle E_I(\mathbf{x}_i),E_T(\mathbf{c}_j)\rangle .",
         plain=(
             "A text-to-image diffusion model is the same noise-prediction loss as entry 9.6, with two changes: it "
             "runs on compressed latents $\\mathbf{z}$ from an autoencoder, and the network also sees the caption "
             "through a text encoder $\\tau$ that is trained separately and then frozen. That encoder usually comes "
             "from a contrastive model (CLIP): given a batch of $N$ matching image--text pairs, score all $N\\times N$ "
             "combinations and train so that each image picks out its own caption and each caption its own image."),
         refresher=(
             "Freezing $\\tau$ makes the conditional denoiser a standard conditional diffusion model whose "
             "Bayes-optimal predictor is $\\boldsymbol\\epsilon^*(\\mathbf{z}_t,t,\\mathbf{c})=-\\sqrt{1-\\bar\\alpha_t}\\,\\nabla\\log "
             "p_t(\\mathbf{z}_t\\mid\\mathbf{c})$, so everything in entries 9.6 and 9.11 applies with $p_t(\\cdot\\mid\\mathbf{c})$ in "
             "place of $p_t$; the first check confirms this for class-conditional Gaussian data. The symmetric "
             "InfoNCE loss (Radford et al., 2021) is the mean of two $N$-way cross-entropies over the similarity "
             "matrix, image-to-text and text-to-image, with a learnt temperature $\\kappa$; it is not a generative "
             "objective and its optimum is reached when matching pairs dominate every row and column. The paper's "
             "reading is that the text encoder contributes conditioning, not a route around the likelihood: the "
             "density access of the system is that of its denoiser (a bound), whatever the encoder."),
         code=r'''
# (a) conditional denoiser: the optimal eps for class c is the conditional score, not the marginal one
t = 500; ab = abar[t]; mus = np.array([-2.0, 2.0]); s2c = 0.5
z0 = mus[1] + math.sqrt(s2c) * rng.normal(size=200_000); eps = rng.normal(size=z0.size)
zt = math.sqrt(ab) * z0 + math.sqrt(1 - ab) * eps
var_t = ab * s2c + (1 - ab)
eps_cond = (zt - math.sqrt(ab) * mus[1]) / var_t * math.sqrt(1 - ab)          # -sqrt(1-ab) * grad log p_t(z_t | c=1)
eps_marg = zt / (ab * (s2c + 4.0) + 1 - ab) * math.sqrt(1 - ab)                 # single-Gaussian fit to the two-class marginal
mse_cond, mse_marg = np.mean((eps - eps_cond) ** 2), np.mean((eps - eps_marg) ** 2)
check("conditional eps = -sqrt(1-abar) grad log p_t(z_t|c) beats the unconditional predictor", mse_cond < mse_marg,
      f"MSE conditional {mse_cond:.4f} < unconditional {mse_marg:.4f}")
check("conditional MSE = Var(eps | z_t, c) = abar s^2 / var_t (1%)", abs(mse_cond / (ab * s2c / var_t) - 1) < 0.01,
      f"{mse_cond:.4f} vs {ab * s2c / var_t:.4f}")
# (b) symmetric InfoNCE on a toy batch
def info_nce(S, kappa=0.07):
    logits = torch.as_tensor(S / kappa); N = S.shape[0]; target = torch.arange(N)
    return 0.5 * (torch.nn.functional.cross_entropy(logits, target) + torch.nn.functional.cross_entropy(logits.T, target)).item()
N = 8
check("all-equal similarities: loss = log N (chance level)", abs(info_nce(np.zeros((N, N))) - math.log(N)) < 1e-9, f"log {N} = {math.log(N):.4f}")
aligned = np.eye(N) * 1.0
check("perfectly aligned pairs: loss below 0.01 at kappa = 0.07", info_nce(aligned) < 0.01, f"loss = {info_nce(aligned):.2e}")
'''),

    # ------------------------------------------------------------------------------------------------ literature
    c.section("Numbers quoted from the literature"),

    c.eq(["EQ-S4-049"], "Reported likelihoods and sample quality on CIFAR-10 (quoted values)",
         r"\text{bound-trained DDPM: }\le3.70\ \text{bits/dim at FID }13.51;\quad \mathcal{L}_{\mathrm{simple}}\text{: }\le3.75\ \text{bits/dim at FID }3.17;\quad"
         r"\text{VDM: }2.65\ (\text{no augmentation}),\ 2.49\ (\text{with});\quad \text{autoregressive: }2.80 .",
         plain=(
             "These numbers are *quoted* from the original papers (Ho et al., 2020; Kingma et al., 2021), not "
             "measured here. They illustrate the trade the simple objective makes: training on the exact bound "
             "gives a better likelihood (lower bits per dimension) but worse samples (higher FID); the simple "
             "objective reverses that; and a likelihood-oriented weighting with a learnt schedule overtakes "
             "the best autoregressive model (2.65 vs 2.80 bits/dim)."),
         refresher=(
             "Ho et al. (2020, Table 1) report $\\le3.70$ bits/dim and FID $13.51$ for the model trained on the "
             "bound and $\\le3.75$ bits/dim with FID $3.17$ for $\\mathcal{L}_{\\mathrm{simple}}$; both are written "
             "as inequalities because both reported values are bounds. Kingma et al. (2021) report $2.65$ without and "
             "$2.49$ with data augmentation; the $2.80$ reference is Sparse Transformer (Child et al., 2019). The "
             "check verifies only what the paper asserts about these numbers: their ordering and that the "
             "comparison of $2.65$ with $2.80$ is the like-for-like one (both without augmentation). Whether the "
             "values hold is a matter for the cited papers, which the bibliography audit records."),
         code=r'''
quoted = {"DDPM, bound-trained (Ho 2020)": (3.70, 13.51), "DDPM, L_simple (Ho 2020)": (3.75, 3.17),
          "VDM, no augmentation (Kingma 2021)": (2.65, None), "VDM, augmentation (Kingma 2021)": (2.49, None),
          "Sparse Transformer, autoregressive (Child 2019)": (2.80, None)}
print(f"{'model (source)':48s} {'bits/dim':>9s} {'FID':>7s}")
for k, (bpd, fid) in quoted.items():
    print(f"{k:48s} {bpd:9.2f} {'' if fid is None else f'{fid:7.2f}'}")
check("illustration (quoted values): bound-trained model has the better likelihood and the worse FID", 3.70 < 3.75 and 13.51 > 3.17)
check("illustration (quoted values): ordering 2.49 < 2.65 < 2.80 < 3.70 <= 3.75", 2.49 < 2.65 < 2.80 < 3.70 <= 3.75)
check("illustration (quoted values): VDM without augmentation is within 0.15 bits/dim of the autoregressive reference", abs(2.80 - 2.65) <= 0.15,
      "quoted values; not measured in this repository")
'''),

    return c.build()


if __name__ == "__main__":
    print(build().path)
