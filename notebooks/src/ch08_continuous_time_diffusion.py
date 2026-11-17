"""Chapter 8: continuous-time diffusion (main paper §3.6; Supplement S1.10, S4.5)."""
from nbkit import Chapter


def build():
    c = Chapter(
        8, "continuous_time_diffusion", "Continuous-time diffusion: SDEs, their reversal and the probability-flow ODE",
        summary=(
            "Diffusion models add noise to data by a stochastic differential equation and generate by running it "
            "backwards. This chapter verifies, on a case where everything is known in closed form (Gaussian data "
            "under the variance-preserving SDE), that the forward SDE produces the stated marginals and satisfies the "
            "Fokker--Planck equation, that the time reversal needs the score with exactly the factor $g^2$, that a "
            "deterministic ODE shares the same marginals and yields an exact likelihood, and that the weighting "
            "$\\lambda=g^2$ makes the training loss an upper bound on the negative log-likelihood. The paper uses these "
            "facts to say that one trained score network admits three distinct likelihood statements."),
        prerequisites="Chapter 7 (score functions and denoising score matching)",
        paper_sections="main paper §3.6; Supplement S1.10 and S4.5",
    )

    c.section("A worked example that runs through the chapter", (
        "Take one-dimensional data $x_0\\sim\\mathcal N(m_0,v_0)$ and the variance-preserving SDE with constant "
        "$\\beta$. Every marginal $p_t$ is then Gaussian with mean $m_0e^{-\\beta t/2}$ and variance "
        "$v_0e^{-\\beta t}+1-e^{-\\beta t}$, so the true score is $-(x-m_t)/v_t$ and every claim below can be checked "
        "against an exact answer. The checks are ported from `vp_sde_fokker_planck_and_reverse_sde_and_pf_ode` and "
        "extended."))

    c.code('''m0, v0, beta, T = 1.5, 0.09, 4.0, 3.0

def vp_moments(t):
    """mean and variance of p_t for Gaussian data under the VP SDE with constant beta"""
    a = math.exp(-0.5 * beta * t)
    return a * m0, a * a * v0 + 1 - a * a

def score(x, t):
    m, v = vp_moments(t); return -(x - m) / v

f = lambda x, t: -0.5 * beta * x          # VP drift
g2 = beta                                 # g(t)^2
print("p_T mean, var:", vp_moments(T), "(a standard Gaussian in the limit T -> infinity)")''')

    c.eq(
        ["EQ-M-023", "EQ-S1-037", "EQ-S4-040"], "The forward SDE and the variance-preserving / variance-exploding choices",
        r"\mathrm d\mathbf x=\mathbf f(\mathbf x,t)\,\mathrm dt+g(t)\,\mathrm d\mathbf w;\qquad "
        r"\text{VP: }\mathbf f=-\tfrac12\beta(t)\mathbf x,\ g=\sqrt{\beta(t)};\qquad \text{VE: }\mathbf f=\mathbf 0,\ g=\sqrt{\mathrm d\sigma^2(t)/\mathrm dt}",
        plain=(
            "A stochastic differential equation describes a point that drifts (the $\\mathbf f\\,\\mathrm dt$ term "
            "pulls it in a fixed direction) and jitters (the $g\\,\\mathrm d\\mathbf w$ term adds a little Gaussian "
            "noise at every instant). Start the point at a data sample and run the equation: the sample is gradually "
            "destroyed into noise. The variance-preserving choice shrinks the point towards zero while adding noise so "
            "that the total variance stays near one; the variance-exploding choice only adds noise, so the variance "
            "grows to $\\sigma^2(t)$. Either way, after enough time the result is a known Gaussian that we can sample "
            "from directly, which is the starting point for generation."),
        refresher=(
            "Song et al. (2021). $\\mathbf w$ is a Wiener process; the solution is in the Itô sense. For the VP SDE "
            "with constant $\\beta$ the transition kernel is $p_{0t}(x_t|x_0)=\\mathcal N(x_0e^{-\\beta t/2},"
            "1-e^{-\\beta t})$, the continuum of the DDPM kernel of Chapter 9 ($\\bar\\alpha_t=e^{-\\int_0^t\\beta}$); "
            "for the VE SDE it is $\\mathcal N(x_0,\\sigma^2(t)-\\sigma^2(0))$, the continuum of the NCSN noise levels. "
            "The supplement (S4.5, EQ-S4-040) adds that with the weighting $\\lambda(t)=g(t)^2$ the score-matching "
            "loss bounds the likelihood, verified in the last entry of this chapter. The check integrates the VP SDE "
            "with the Euler--Maruyama scheme for $10^5$ paths and compares the empirical mean and variance at $T$ with "
            "the closed form; the VE SDE is checked the same way with $\\sigma^2(t)=t$."),
        code='''n, K = 100_000, 3000; dt = T / K
x = m0 + math.sqrt(v0) * rng.normal(size=n)
for i in range(K):
    x = x + f(x, i * dt) * dt + math.sqrt(g2 * dt) * rng.normal(size=n)
mT, vT = vp_moments(T)
check("VP SDE: empirical mean at T matches m0 exp(-beta T/2)", *close(x.mean(), mT, 0.01))
check("VP SDE: empirical variance at T matches v0 e^{-beta T} + 1 - e^{-beta T}", *close(x.var(), vT, 0.01))
# VE SDE with sigma^2(t) = t: f = 0, g^2 = 1 -> variance grows by T
xe = m0 + math.sqrt(v0) * rng.normal(size=n)
for i in range(K):
    xe = xe + math.sqrt(1.0 * dt) * rng.normal(size=n)
check("VE SDE: mean unchanged, variance grows by sigma^2(T) - sigma^2(0) = T", abs(xe.mean() - m0) < 0.01 and abs(xe.var() - (v0 + T)) < 0.03,
      f"mean {xe.mean():.3f} (m0 {m0}), var {xe.var():.3f} (v0 + T = {v0 + T})")''')

    c.eq(
        "EQ-S1-036", "The Fokker--Planck equation for the marginals",
        r"\partial_t p_t(\mathbf x)=-\nabla\!\cdot\!\big(\mathbf f(\mathbf x,t)\,p_t(\mathbf x)\big)+\tfrac12 g(t)^2\,\Delta p_t(\mathbf x)",
        plain=(
            "Instead of following one jittering point, follow the whole cloud of points. Its density $p_t$ changes "
            "in time by two effects: the drift carries density along (the first term, a transport term) and the "
            "noise spreads it out (the second term, a diffusion term like heat spreading in a metal bar). The "
            "equation is the bookkeeping of those two effects; the Gaussian we found in the previous entry is a "
            "solution of it."),
        refresher=(
            "Also called the forward Kolmogorov equation; it follows from Itô's lemma applied to a test function and "
            "integration by parts (Øksendal, 2003). It is what makes 'the same marginals' a meaningful statement in "
            "the next entries: two processes with different noise but the same Fokker--Planck equation have the same "
            "$p_t$. The check substitutes the Gaussian $p_t=\\mathcal N(m_t,v_t)$ with the VP moments into the "
            "equation with SymPy and simplifies the residual to zero."),
        code='''x_, t_ = sp.symbols("x t", real=True)
b_, m0_, v0_ = sp.symbols("beta m_0 v_0", positive=True)
a_ = sp.exp(-b_ * t_ / 2)
m_t, v_t = a_ * m0_, a_**2 * v0_ + 1 - a_**2
p_t = sp.exp(-(x_ - m_t)**2 / (2 * v_t)) / sp.sqrt(2 * sp.pi * v_t)
f_ = -b_ * x_ / 2
residual = sp.diff(p_t, t_) + sp.diff(f_ * p_t, x_) - sp.Rational(1, 2) * b_ * sp.diff(p_t, x_, 2)
check("Gaussian with the VP moments satisfies the Fokker-Planck equation (symbolic residual = 0)", sp.simplify(residual) == 0)''')

    c.eq(
        ["EQ-M-024", "EQ-S1-038"], "The reverse-time SDE",
        r"\mathrm d\mathbf x=\Big[\mathbf f(\mathbf x,t)-g(t)^2\,\nabla_{\mathbf x}\log p_t(\mathbf x)\Big]\mathrm dt+g(t)\,\mathrm d\bar{\mathbf w},\qquad t:T\to0",
        plain=(
            "Run the noising process backwards and you get a generator: start from pure noise, follow the reverse "
            "equation, and end at a data-like sample. The reverse equation is almost the forward one with time "
            "reversed, except for one new term: the drift is corrected by the *score*, the direction in which the "
            "noisy density increases. The score is the only unknown, and a neural network is trained to estimate it "
            "(Chapter 7). Get the factor in front of the score wrong and the samples come out with the wrong spread, "
            "which is what the third check shows."),
        refresher=(
            "Anderson (1982). The reversal of an Itô diffusion is a diffusion with drift "
            "$\\mathbf f-g^2\\nabla\\log p_t$ and the same diffusion coefficient, driven by a Wiener process "
            "$\\bar{\\mathbf w}$ in reversed time; the derivation equates the Fokker--Planck equations of the forward "
            "and the reversed processes. With an estimated score $\\mathbf s_\\theta$ in place of $\\nabla\\log p_t$ "
            "the reverse SDE defines the model $p^{\\mathrm{SDE}}_\\theta$, one of the three likelihood statements the "
            "paper distinguishes. The checks integrate the reverse SDE from $p_T$ with the exact score and recover "
            "$(m_0,v_0)$; with the factor $\\tfrac12 g^2$ (the probability-flow coefficient, wrong for the SDE) the "
            "variance is visibly off."),
        code='''def reverse_sde(coef, n=100_000, K=3000):
    dt = T / K
    xr = mT + math.sqrt(vT) * rng.normal(size=n)
    for i in range(K, 0, -1):
        t = i * dt
        xr = xr - (f(xr, t) - coef * g2 * score(xr, t)) * dt + math.sqrt(g2 * dt) * rng.normal(size=n)
    return xr
xr = reverse_sde(1.0)
check("reverse SDE with the exact score recovers the data mean", *close(xr.mean(), m0, 0.01))
check("reverse SDE with the exact score recovers the data variance", *close(xr.var(), v0, 0.015))
xw = reverse_sde(0.5)
check("the factor matters: with g^2/2 in the SDE the variance is wrong", abs(xw.var() - v0) > 0.05, f"var {xw.var():.3f} vs v0 = {v0}")

fig, ax = plt.subplots(figsize=(6, 2.6))
bins = np.linspace(-1, 4, 80)
ax.hist(xr, bins, density=True, alpha=0.5, label="reverse SDE, exact score")
ax.hist(xw, bins, density=True, alpha=0.5, label="reverse SDE, coefficient g$^2$/2 (wrong)")
ax.plot(bins, scipy.stats.norm.pdf(bins, m0, math.sqrt(v0)), "k", label="data $p_0$")
ax.legend(fontsize=8, frameon=False); plt.tight_layout(); plt.show()''')

    c.eq(
        ["EQ-S1-039", "EQ-S4-041"], "The weighted denoising objective in continuous time",
        r"\mathcal J(\theta)=\mathbb E_{t}\Big[\lambda(t)\,\mathbb E_{\mathbf x_0,\,\mathbf x_t\mid\mathbf x_0}"
        r"\big\|\mathbf s_\theta(\mathbf x_t,t)-\nabla_{\mathbf x_t}\log p_{0t}(\mathbf x_t\mid\mathbf x_0)\big\|^2\Big]",
        plain=(
            "To learn the score at every noise level at once, pick a random time $t$, a data point, and a noisy "
            "version of it; ask the network to predict the direction back towards the clean point (that direction "
            "is the gradient of the *log* of the Gaussian noising density, which we can write down); and weight the error "
            "by $\\lambda(t)$. Although the network is only ever shown directions to *individual* clean points, the "
            "function that minimises this loss is the score of the whole noisy data distribution, the quantity the "
            "reverse SDE needs."),
        refresher=(
            "The continuum of denoising score matching (Vincent, 2011; Chapter 7, EQ-M-022): for each $t$ the "
            "minimiser over all functions is $\\nabla\\log p_t$, because "
            "$\\mathbb E_{\\mathbf x_0\\mid\\mathbf x_t}[\\nabla_{\\mathbf x_t}\\log p_{0t}(\\mathbf x_t|\\mathbf x_0)]"
            "=\\nabla\\log p_t(\\mathbf x_t)$. The weighting $\\lambda(t)$ does not change the population minimiser "
            "but changes how a finite network allocates its error, which is why the paper insists that a diffusion "
            "model's reported likelihood depends on the training weighting (next entries). The check fits a linear "
            "score $s(x,t)=a(t)x+b(t)$ at three fixed times by least squares on the denoising target and recovers "
            "the exact score coefficients $-1/v_t$ and $m_t/v_t$."),
        code='''for t in (0.1, 0.5, 2.0):
    m, v = vp_moments(t)
    a = math.exp(-0.5 * beta * t); kv = 1 - a * a                      # transition kernel N(a x0, kv)
    x0 = m0 + math.sqrt(v0) * rng.normal(size=200_000)
    xt = a * x0 + math.sqrt(kv) * rng.normal(size=x0.size)
    target = -(xt - a * x0) / kv                                       # grad_{x_t} log p_{0t}(x_t | x_0)
    A = np.stack([xt, np.ones_like(xt)], 1)
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)                # least squares = population minimiser of the DSM loss (linear class)
    check(f"t = {t}: DSM minimiser is the marginal score, slope -1/v_t and intercept m_t/v_t",
          abs(coef[0] + 1 / v) < 0.02 and abs(coef[1] - m / v) < 0.02, f"fit ({coef[0]:.3f}, {coef[1]:.3f}) vs ({-1 / v:.3f}, {m / v:.3f})")''')

    c.eq(
        ["EQ-M-025", "EQ-S1-040"], "The probability-flow ODE and the instantaneous change of variables",
        r"\frac{\mathrm d\mathbf x}{\mathrm dt}=\mathbf f(\mathbf x,t)-\tfrac12 g(t)^2\nabla_{\mathbf x}\log p_t(\mathbf x);\qquad "
        r"\frac{\mathrm d\log p_t(\mathbf x_t)}{\mathrm dt}=-\nabla\!\cdot\!\Big(\mathbf f-\tfrac12 g^2\nabla\log p_t\Big)(\mathbf x_t,t)",
        plain=(
            "There is also a *deterministic* way to move the noise cloud back to the data cloud: an ordinary "
            "differential equation with no randomness at all, using half the score correction and no noise term. "
            "Every noise point then lands on a definite data point, so the model becomes an invertible map, a "
            "continuous normalizing flow (Chapter 5). The second equation is the flow's bookkeeping: as a point "
            "moves, its log-density changes by minus the divergence of the velocity field, so integrating that "
            "quantity along the path gives an exact log-likelihood for the model."),
        refresher=(
            "Song et al. (2021), via the Fokker--Planck equation: the diffusion term $\\tfrac12g^2\\Delta p_t$ equals "
            "$-\\nabla\\cdot(-\\tfrac12g^2 p_t\\nabla\\log p_t)$, so the SDE's marginals are also those of an ODE with "
            "drift $\\mathbf f-\\tfrac12g^2\\nabla\\log p_t$. With $\\mathbf s_\\theta$ in place of the score the ODE "
            "defines $p^{\\mathrm{ODE}}_\\theta$, a different model from $p^{\\mathrm{SDE}}_\\theta$ unless the score is "
            "exact; its likelihood is computed with Hutchinson's trace estimator (Chapter 5) and one ODE solve per "
            "datum, and the paper warns against mixing these values with variational bounds in one comparison. The "
            "checks: (i) the ODE maps the quantiles of $p_T$ onto those of $p_0$ (for Gaussians the standardised "
            "coordinate is preserved), and (ii) integrating the instantaneous change of variables along a trajectory "
            "reproduces $\\log p_0(x)$ exactly."),
        code='''ode = lambda t, y: f(y, t) - 0.5 * g2 * score(y, t)
y0 = mT + math.sqrt(vT) * np.linspace(-3, 3, 13)                        # 13 quantiles of p_T
sol = scipy.integrate.solve_ivp(ode, (T, 1e-9), y0, rtol=1e-10, atol=1e-12)
yT = sol.y[:, -1]
check("PF-ODE maps quantiles of p_T to the same quantiles of p_0", *close((yT - m0) / math.sqrt(v0), (y0 - mT) / math.sqrt(vT), 1e-6))
xq = 1.1
def aug(t, y):                                                           # state: (x_t, accumulated divergence)
    _, v = vp_moments(t)
    div = -0.5 * beta + 0.5 * g2 / v                                      # d/dx of (f - g^2/2 score) for the Gaussian case
    return [ode(t, y[0]), div]
s2 = scipy.integrate.solve_ivp(aug, (0.0, T), [xq, 0.0], rtol=1e-10, atol=1e-12)
xT_, I = s2.y[0, -1], s2.y[1, -1]
ll = scipy.stats.norm.logpdf(xT_, mT, math.sqrt(vT)) + I                 # log p_0(x) = log p_T(x_T) + int div dt
check("instantaneous change of variables along the PF-ODE gives the exact log p_0(x)",
      *close(ll, scipy.stats.norm.logpdf(xq, m0, math.sqrt(v0)), 1e-6))''')

    c.eq(
        ["EQ-M-026", "EQ-S1-041"], "Likelihood weighting: $\\lambda=g^2$ makes the loss an upper bound on the negative log-likelihood",
        r"D_{\mathrm{KL}}\big(p_0\,\|\,p^{\mathrm{SDE}}_\theta\big)\;\le\;\tfrac12\int_0^T g(t)^2\,"
        r"\mathbb E_{p_t}\big\|\nabla\log p_t-\mathbf s_\theta\big\|^2\mathrm dt+D_{\mathrm{KL}}(p_T\,\|\,\pi)",
        plain=(
            "If you weight the training loss by $g(t)^2$, the loss (plus a constant that does not depend on the "
            "network) is never below the model's negative log-likelihood. So a model trained this way can report its "
            "loss as a guaranteed upper bound on how many bits it needs to code the data, in the same way a VAE "
            "reports its ELBO. With a different weighting, such as the one that gives the best-looking samples, no "
            "such guarantee holds."),
        refresher=(
            "Song, Durkan, Murray and Ermon (2021), Theorem 1, via Girsanov's theorem on path space: the KL between "
            "the data and the reverse-SDE model is bounded by the $g^2$-weighted score-matching error plus the "
            "prior mismatch at $T$. The denoising form adds a $\\theta$-independent constant, so 'training loss plus "
            "constant $\\ge$ NLL' as the main paper states (EQ-M-026). Equality holds for the exact score. Kingma and "
            "Gao (2023) later showed that weightings monotone in the noise level are also ELBOs, under Gaussian noise "
            "augmentation of the data (Chapter 10). The check uses the Gaussian example with a mis-scaled score "
            "$\\mathbf s_c=c\\,\\nabla\\log p_t$: the left side is the KL between $p_0$ and the Gaussian that the "
            "reverse SDE with $\\mathbf s_c$ generates (estimated by simulation), the right side is computed in closed "
            "form, and the inequality holds for every $c$ with equality within Monte-Carlo error at $c=1$."),
        code='''def kl_gauss(m1, v1, m2, v2):
    return 0.5 * (math.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1)
def reverse_with_scaled_score(cs, n=100_000, K=2000):
    dt = T / K; xr = rng.normal(size=n)                                  # start from the prior pi = N(0, 1)
    for i in range(K, 0, -1):
        t = i * dt
        xr = xr - (f(xr, t) - g2 * cs * score(xr, t)) * dt + math.sqrt(g2 * dt) * rng.normal(size=n)
    return xr
kl_prior = kl_gauss(mT, vT, 0.0, 1.0)
for cs in (0.7, 1.0, 1.3):
    # J_SM = 1/2 int g^2 E_{p_t} ||(1 - c) grad log p_t||^2 dt = 1/2 int beta (1 - c)^2 / v_t dt   (E (x - m)^2 / v^2 = 1 / v)
    J = 0.5 * scipy.integrate.quad(lambda t: beta * (1 - cs) ** 2 / vp_moments(t)[1], 0, T)[0]
    xg = reverse_with_scaled_score(cs)
    lhs = kl_gauss(m0, v0, xg.mean(), xg.var())                           # KL(p_0 || generated Gaussian)
    rhs = J + kl_prior
    check(f"c = {cs}: KL(p_0 || p_theta^SDE) <= J_SM(g^2 weighting) + KL(p_T || pi)", lhs <= rhs + 0.01,
          f"lhs {lhs:.4f} <= rhs {rhs:.4f}")
    if cs == 1.0:
        check("c = 1 (exact score): the bound is tight, lhs = rhs up to Monte-Carlo and discretisation error",
              abs(lhs - rhs) < 0.01, f"|lhs - rhs| = {abs(lhs - rhs):.4f}")''')

    return c.build()


if __name__ == "__main__":
    print(build().path)
