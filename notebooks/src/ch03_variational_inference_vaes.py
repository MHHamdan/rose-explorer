"""Chapter 3: variational inference and variational autoencoders (EQ-M-007..010, EQ-M-035, EQ-S1-014..019, EQ-S4-008..013)."""
from nbkit import Chapter


def build():
    c = Chapter(
        3, "variational_inference_vaes", "Variational inference and variational autoencoders",
        summary=(
            "A latent-variable model explains each observation $\\mathbf{x}$ through an unobserved $\\mathbf{z}$. Once the "
            "decoder is a neural network, the marginal likelihood $p_\\theta(\\mathbf{x})=\\int p_\\theta(\\mathbf{x}\\mid\\mathbf{z})"
            "p(\\mathbf{z})\\,d\\mathbf{z}$ has no closed form, so it is *bounded* instead of computed. This chapter follows that "
            "route: the evidence lower bound and its gap, the reparameterisation trick that makes the bound trainable, the "
            "importance-weighted bound that tightens it, the rate--distortion reading that explains posterior collapse and the "
            "$\\beta$-VAE, the discrete VQ-VAE objective, and the conditional-mean argument behind blurry samples. Every "
            "identity is checked on a one-dimensional Gaussian model where the exact posterior and evidence are known, so the "
            "bound, its gap and its limit can be compared with the truth to machine precision."),
        prerequisites="Chapters 1 (KL divergence) and 2 (maximum likelihood, Bayes' rule)",
        paper_sections="main §3.2 and §5.2; Supplement S1.5 and S4.2")

    c.section(
        "A model where everything is known",
        "Most checks below use the linear-Gaussian model $\\mathbf{z}\\sim\\mathcal{N}(0,1)$, "
        "$\\mathbf{x}\\mid\\mathbf{z}\\sim\\mathcal{N}(a\\mathbf{z},s^2)$. It is the simplest latent-variable model in which "
        "the three quantities that variational inference juggles are all available in closed form: the evidence "
        "$p(x)=\\mathcal{N}(x;0,a^2+s^2)$, the exact posterior $p(z\\mid x)=\\mathcal{N}\\!\\big(\\tfrac{ax}{a^2+s^2},"
        "\\tfrac{s^2}{a^2+s^2}\\big)$, and, for any Gaussian $q(z)=\\mathcal{N}(m,v)$, the expected reconstruction term "
        "$\\mathbb{E}_q[\\log p(x\\mid z)]=-\\tfrac12\\log(2\\pi s^2)-\\tfrac{(x-am)^2+a^2v}{2s^2}$. A real VAE replaces "
        "$a z$ by a network and $q$ by an encoder network; the identities do not change.")
    c.code('''# Closed forms for the linear-Gaussian model used throughout this chapter.
A, S2 = 1.3, 0.4          # x | z ~ N(A z, S2),  z ~ N(0, 1)

def log_evidence(x):
    return scipy.stats.norm.logpdf(x, 0.0, math.sqrt(A * A + S2))

def exact_posterior(x):
    """mean and variance of p(z | x)"""
    return A * x / (A * A + S2), S2 / (A * A + S2)

def expected_reconstruction(x, m, v):
    """E_{z ~ N(m, v)} [ log N(x; A z, S2) ]  (exact: a Gaussian expectation of a quadratic)"""
    return -0.5 * math.log(2 * math.pi * S2) - ((x - A * m) ** 2 + A * A * v) / (2 * S2)

def kl_gauss(m1, v1, m2, v2):
    """KL( N(m1, v1) || N(m2, v2) ) in one dimension"""
    return 0.5 * (math.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1.0)

x0 = 0.9
pm, pv = exact_posterior(x0)
print(f"x = {x0}: log p(x) = {log_evidence(x0):.6f}, posterior N({pm:.4f}, {pv:.4f})")''')

    # ------------------------------------------------------------------ ELBO and its gap
    c.eq(["EQ-M-007", "EQ-S1-014", "EQ-S1-015", "EQ-S4-008"], "The evidence lower bound and its gap",
         r"""\begin{aligned}
\log p_\theta(\mathbf{x}) &\ge \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\big[\log p_\theta(\mathbf{x}\mid\mathbf{z})\big]}_{\text{reconstruction}}
 - \underbrace{D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)}_{\text{rate}} \;=:\; \mathcal{L}(\mathbf{x};\theta,\phi),\\
\log p_\theta(\mathbf{x}) - \mathcal{L} &= D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x})\big)\;\ge 0 .
\end{aligned}""",
         plain=(
             "$p_\\theta(\\mathbf{x})$ is the probability the model gives to a data point once the hidden $\\mathbf{z}$ has "
             "been averaged out; it is called the *evidence*, and we work with its logarithm. We cannot compute it, so we introduce a guess "
             "$q_\\phi(\\mathbf{z}\\mid\\mathbf{x})$ for where $\\mathbf{z}$ probably is (the *encoder*) and get a number "
             "$\\mathcal{L}$ that is *always at most* the evidence. $\\mathcal{L}$ has two parts: how well $\\mathbf{x}$ is "
             "rebuilt from a guessed $\\mathbf{z}$ (reconstruction) minus how far the guess strays from the prior $p(\\mathbf{z})$ "
             "(rate). The second line says exactly how much is lost: the gap is the KL divergence between our guess and the true "
             "posterior. If the guess is perfect, the bound is tight."),
         refresher=(
             "Write $\\log p_\\theta(\\mathbf{x})=\\log\\mathbb{E}_q[p_\\theta(\\mathbf{x},\\mathbf{z})/q(\\mathbf{z}\\mid\\mathbf{x})]$ "
             "for any $q$ with $q>0$ wherever $p_\\theta(\\mathbf{x},\\mathbf{z})>0$; Jensen's inequality for the concave "
             "$\\log$ gives the bound, and rearranging $\\log p_\\theta(\\mathbf{x},\\mathbf{z})=\\log p_\\theta(\\mathbf{z}\\mid\\mathbf{x})"
             "+\\log p_\\theta(\\mathbf{x})$ inside the expectation gives the gap identity exactly, with no approximation. "
             "Splitting the joint the other way, $\\log p_\\theta(\\mathbf{x}\\mid\\mathbf{z})+\\log p(\\mathbf{z})$, gives the "
             "reconstruction/rate form. Maximising $\\mathcal{L}$ over $\\phi$ is inference (it minimises the gap); maximising "
             "over $\\theta$ is maximum likelihood on a lower bound, so a VAE's reported likelihood is a bound whose slack is "
             "model-dependent, which is why the paper reports IWAE bounds and warns against reading ELBO differences as "
             "likelihood differences (main §7). The *rate* name comes from Alemi et al. (2018): the KL term is the number of nats "
             "needed to transmit $\\mathbf{z}$ under the prior. The check below reproduces the repository's "
             "`elbo_plus_posterior_kl_equals_log_evidence`: for four different $q$ the ELBO plus the gap equals the evidence to "
             "$10^{-12}$, and only $q=$ posterior makes the gap zero."),
         code='''errs, elbos = [], []
for m, v in [(0.0, 1.0), (0.5, 0.2), (pm, pv), (-1.0, 3.0)]:
    rec = expected_reconstruction(x0, m, v)
    rate = kl_gauss(m, v, 0.0, 1.0)                    # KL(q || prior)
    elbo = rec - rate
    gap = kl_gauss(m, v, pm, pv)                       # KL(q || exact posterior)
    errs.append(abs(elbo + gap - log_evidence(x0)))
    elbos.append(elbo)
    print(f"q = N({m:+.3f}, {v:.3f}):  ELBO = {elbo:.6f}  gap = {gap:.6f}  ELBO+gap = {elbo+gap:.6f}")
check("ELBO + KL(q || p(z|x)) = log p(x) for every q", max(errs) < 1e-12, f"max error {max(errs):.1e}")
check("ELBO <= log p(x) for every q", all(e <= log_evidence(x0) + 1e-12 for e in elbos))
check("bound is tight exactly at q = posterior", abs(elbos[2] - log_evidence(x0)) < 1e-12)''')

    # ------------------------------------------------------------------ reparameterisation
    c.eq(["EQ-M-008", "EQ-S1-016"], "The reparameterisation trick",
         r"""\mathbf{z}=\boldsymbol{\mu}_\phi(\mathbf{x})+\boldsymbol{\sigma}_\phi(\mathbf{x})\odot\boldsymbol{\epsilon},\quad
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),\qquad
\nabla_\phi\,\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}[g(\mathbf{z})]=\mathbb{E}_{\boldsymbol{\epsilon}}\big[\nabla_\phi\, g\big(\mathbf{z}(\boldsymbol{\epsilon},\phi)\big)\big]""",
         plain=(
             "To train the encoder we need the gradient of an *average over random draws* with respect to the parameters "
             "that shape those draws. The trick: draw the randomness $\\boldsymbol{\\epsilon}$ first from a fixed standard "
             "Gaussian, then shift and scale it with $\\boldsymbol{\\mu}_\\phi$ and $\\boldsymbol{\\sigma}_\\phi$. Now the "
             "sample is an ordinary function of $\\phi$, and the gradient can be pushed inside the average and computed by "
             "backpropagation through the sample itself."),
         refresher=(
             "For a location--scale family the map $\\boldsymbol{\\epsilon}\\mapsto\\mathbf{z}$ is a change of variables that "
             "does not depend on $\\phi$ once $\\boldsymbol{\\epsilon}$ is fixed, so differentiation and expectation commute "
             "whenever $g$ is differentiable and dominated. The resulting *pathwise* estimator typically has far lower variance "
             "than the score-function (REINFORCE) estimator $\\mathbb{E}_q[g(\\mathbf{z})\\nabla_\\phi\\log q_\\phi(\\mathbf{z})]$, "
             "which is what made VAEs trainable by SGD (Kingma and Welling 2014; Rezende et al. 2014). It does not apply to "
             "discrete $\\mathbf{z}$, which is why the VQ-VAE below needs a straight-through estimator. The check ports the "
             "repository's `reparameterisation_gradient_unbiased`: for $g(z)=z^2$ and $z=\\mu+s\\epsilon$ the exact gradients "
             "are $\\partial_\\mu\\mathbb{E}[z^2]=2\\mu$ and $\\partial_s\\mathbb{E}[z^2]=2s$; autograd through the samples "
             "recovers them, and the Monte-Carlo standard error is reported alongside."),
         code='''torch.manual_seed(1)
mu = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
s = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
n = 400_000
eps = torch.randn(n, dtype=torch.float64)
g = (mu + s * eps).pow(2)                      # g(z) evaluated on reparameterised samples
g.mean().backward()
# per-sample gradients 2 z and 2 z eps give the standard error of the two estimates
se_mu = (2 * (mu + s * eps)).std().item() / math.sqrt(n)
se_s = (2 * (mu + s * eps) * eps).std().item() / math.sqrt(n)
check("d/dmu E[z^2] = 2 mu via pathwise gradient", abs(mu.grad.item() - 2 * 0.7) < 4 * se_mu,
      f"{mu.grad.item():.4f} vs 1.4 (SE {se_mu:.1e})")
check("d/ds E[z^2] = 2 s via pathwise gradient", abs(s.grad.item() - 2 * 1.3) < 4 * se_s,
      f"{s.grad.item():.4f} vs 2.6 (SE {se_s:.1e})")
# compare with the score-function estimator on the same samples: same mean, far larger variance
with torch.no_grad():
    z = mu + s * eps
    score_mu = (z - mu) / s**2                    # d/dmu log N(z; mu, s^2)
    sf = (z.pow(2) * score_mu)
check("score-function estimator has the same mean but larger variance",
      abs(sf.mean().item() - 1.4) < 4 * sf.std().item() / math.sqrt(n) and sf.var().item() > (2 * z).var().item(),
      f"variance ratio score-function / pathwise = {sf.var().item() / (2 * z).var().item():.1f}")''')

    # ------------------------------------------------------------------ Gaussian rate term
    c.eq(["EQ-S1-017", "EQ-S4-009"], "The closed-form Gaussian rate term",
         r"""D_{\mathrm{KL}}\big(\mathcal{N}(\boldsymbol{\mu},\mathrm{diag}\,\boldsymbol{\sigma}^2)\,\|\,\mathcal{N}(\mathbf{0},\mathbf{I})\big)
=\tfrac12\sum_j\big(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\big)""",
         plain=(
             "When the encoder's guess is a Gaussian with independent coordinates and the prior is the standard Gaussian, "
             "the rate term needs no sampling at all: it is a simple sum over coordinates. Each coordinate pays for being "
             "shifted away from zero ($\\mu_j^2$) and for having a spread different from one ($\\sigma_j^2-1-\\log\\sigma_j^2$, "
             "which is zero only at $\\sigma_j=1$). Together with the reparameterisation, this is why one sample of "
             "$\\mathbf{z}$ per data point is enough to train a VAE."),
         refresher=(
             "Integrate $\\log q-\\log p$ under $q$ coordinate by coordinate: $\\mathbb{E}_q[-\\tfrac12\\log(2\\pi\\sigma_j^2)"
             "-\\tfrac12]+\\mathbb{E}_q[\\tfrac12\\log 2\\pi+\\tfrac12 z_j^2]$ with $\\mathbb{E}_q[z_j^2]=\\mu_j^2+\\sigma_j^2$. "
             "The term is non-negative by the inequality $t-1-\\log t\\ge0$. The repository check "
             "`kl_gaussian_closed_form_vs_mc` compares this formula with a Monte-Carlo estimate and with `vae/model.py`; the "
             "cell below does the Monte-Carlo comparison and checks the formula against the general two-Gaussian KL."),
         code='''d = 4
mu_v = rng.normal(size=d)
sig = np.exp(rng.normal(scale=0.3, size=d))
closed = 0.5 * np.sum(mu_v**2 + sig**2 - 1 - np.log(sig**2))
general = sum(kl_gauss(mu_v[j], sig[j]**2, 0.0, 1.0) for j in range(d))
check("sum formula equals the general Gaussian KL", abs(closed - general) < 1e-12, f"{closed:.6f} vs {general:.6f}")
zs = mu_v + sig * rng.normal(size=(400_000, d))
lq = scipy.stats.norm.logpdf(zs, mu_v, sig).sum(1)
lp = scipy.stats.norm.logpdf(zs, 0.0, 1.0).sum(1)
mc, se = (lq - lp).mean(), (lq - lp).std() / math.sqrt(len(zs))
check("closed form agrees with Monte Carlo", abs(mc - closed) < 5 * se, f"MC {mc:.4f} +- {se:.1e}, closed {closed:.4f}")
rate = lambda m, s_: 0.5 * np.sum(m**2 + s_**2 - 1 - np.log(s_**2))
grid_m, grid_s = np.meshgrid(np.linspace(-2, 2, 81), np.linspace(0.2, 3, 81))
rates = 0.5 * (grid_m**2 + grid_s**2 - 1 - np.log(grid_s**2))                 # one dimension, over a grid of (mu, sigma)
i_min = np.unravel_index(np.argmin(rates), rates.shape)
check("rate term is non-negative and its minimum 0 is at (mu, sigma) = (0, 1)",
      closed > 0 and rates.min() >= -1e-12 and abs(grid_m[i_min]) < 1e-9 and abs(grid_s[i_min] - 1) < 0.02,
      f"min over the grid {rates.min():.2e} at mu = {grid_m[i_min]:.2f}, sigma = {grid_s[i_min]:.3f}")''')

    # ------------------------------------------------------------------ IWAE
    c.eq(["EQ-M-009", "EQ-S1-018", "EQ-S4-010"], "The importance-weighted bound (IWAE)",
         r"""\mathcal{L}_K(\mathbf{x})=\mathbb{E}_{\mathbf{z}_1,\dots,\mathbf{z}_K\sim q_\phi(\mathbf{z}\mid\mathbf{x})}
\left[\log\frac1K\sum_{k=1}^K\frac{p_\theta(\mathbf{x},\mathbf{z}_k)}{q_\phi(\mathbf{z}_k\mid\mathbf{x})}\right],\qquad
\mathcal{L}_1\le\mathcal{L}_K\le\mathcal{L}_{K+1}\le\log p_\theta(\mathbf{x})""",
         plain=(
             "Instead of one guessed $\\mathbf{z}$, draw $K$ of them, weight each by how plausible it is, and take the log of "
             "the average weight. With $K=1$ this is the ordinary ELBO. Using more draws can only raise the bound, never lower "
             "it, and as $K$ grows the bound climbs towards the true evidence. The paper reports $-\\mathcal{L}_{100}$ as the "
             "VAE's negative log-likelihood bound in its experiments."),
         refresher=(
             "The average of $K$ importance weights $w_k=p_\\theta(\\mathbf{x},\\mathbf{z}_k)/q(\\mathbf{z}_k\\mid\\mathbf{x})$ "
             "is an unbiased estimate of $p_\\theta(\\mathbf{x})$, so Jensen gives $\\mathcal{L}_K\\le\\log p_\\theta(\\mathbf{x})$. "
             "Monotonicity follows because the $(K+1)$-sample average equals the average, over random $K$-subsets, of the "
             "$K$-sample averages, and $\\log$ is concave (Jensen gives $\\mathcal{L}_K\\le\\mathcal{L}_{K+1}$) (Burda et al. 2016, Theorem 1). Convergence $\\mathcal{L}_K\\to\\log p_\\theta(\\mathbf{x})$ holds "
             "when the weights are bounded; the registry marks the main-paper wording `YES-WITH-WORDING-FIX` because the "
             "paper's one-line statement omits this condition, which the supplement states. The estimator's variance grows "
             "with the KL between $q$ and the posterior, so IWAE narrows but does not close a bad encoder's gap. The check uses "
             "a deliberately poor $q$ (the prior) and $K$ from 1 to 1000: the estimates are non-decreasing within their standard "
             "errors, stay below the evidence, and reach it to within 0.02 nats, as in the repository check "
             "`iwae_monotone_in_K_and_converges` (whose `vae/model.py` part is not ported here)."),
         code='''rng_iw = np.random.default_rng(4)
x1 = 2.5                                   # a point in the tail, where the prior is a poor q
logpx1 = log_evidence(x1)
m, v = 0.0, 1.0                            # q = prior: a poor posterior approximation
Ks = [1, 2, 5, 10, 50, 200, 1000]
R = 4000
vals, ses = [], []
for K in Ks:
    z = m + math.sqrt(v) * rng_iw.normal(size=(R, K))
    lw = (scipy.stats.norm.logpdf(x1, A * z, math.sqrt(S2)) + scipy.stats.norm.logpdf(z, 0, 1)
          - scipy.stats.norm.logpdf(z, m, math.sqrt(v)))
    lk = np.logaddexp.reduce(lw, axis=1) - math.log(K)
    vals.append(lk.mean()); ses.append(lk.std() / math.sqrt(R))
elbo_exact = expected_reconstruction(x1, m, v) - kl_gauss(m, v, 0, 1)
check("L_1 equals the ELBO", abs(vals[0] - elbo_exact) < 4 * ses[0], f"{vals[0]:.4f} vs exact ELBO {elbo_exact:.4f}")
check("L_K is non-decreasing in K (within standard errors)",
      all(vals[i + 1] >= vals[i] - 3 * (ses[i] + ses[i + 1]) for i in range(len(Ks) - 1)))
check("L_K <= log p(x) for every K", all(vk <= logpx1 + 3 * s for vk, s in zip(vals, ses)))
check("L_1000 is within 0.02 nats of log p(x)", abs(vals[-1] - logpx1) < 0.02, f"{vals[-1]:.4f} vs {logpx1:.4f}")

fig, ax = plt.subplots(figsize=(5, 3))
ax.errorbar(Ks, vals, yerr=[2 * s for s in ses], marker="o", label=r"$\\mathcal{L}_K$ (q = prior)")
ax.axhline(logpx1, color="k", ls="--", label=r"$\\log p(x)$")
ax.axhline(elbo_exact, color="grey", ls=":", label=r"ELBO $=\\mathcal{L}_1$")
ax.set_xscale("log"); ax.set_xlabel("K importance samples"); ax.set_ylabel("nats"); ax.legend(); ax.set_title(f"IWAE bound at x = {x1}")
plt.tight_layout(); plt.show()''')

    # ------------------------------------------------------------------ posterior collapse
    c.eq(["EQ-M-010", "EQ-S1-019"], "Posterior collapse: the rate can go to zero",
         r"""q_\phi(\mathbf{z}\mid\mathbf{x})=p(\mathbf{z})\ \Longrightarrow\ \text{rate}=D_{\mathrm{KL}}\big(q_\phi\,\|\,p\big)=0,\qquad
\mathcal{L}_\beta=\text{reconstruction}-\beta\cdot\text{rate}""",
         plain=(
             "If the decoder is strong enough to model $\\mathbf{x}$ on its own, the cheapest thing the encoder can do is "
             "ignore $\\mathbf{x}$ and output the prior: then the rate term is exactly zero and the latent carries no "
             "information. Nothing in the objective forbids this. Multiplying the rate by a weight $\\beta$ changes how "
             "expensive information in $\\mathbf{z}$ is, which moves the solution along a trade-off curve between rate and "
             "reconstruction quality rather than fixing the collapse."),
         refresher=(
             "In the rate--distortion reading of Alemi et al. (2018) every $(\\text{rate},\\text{distortion})$ pair on a "
             "convex curve is attainable, and the plain ELBO ($\\beta=1$) picks the point of slope $-1$; a decoder that "
             "can represent $p_{\\mathrm{data}}$ without $\\mathbf{z}$ makes the zero-rate point optimal (Chen et al. 2017 give "
             "the bits-back argument). The check builds a two-component discrete toy: a decoder family rich enough to fit the "
             "data without $\\mathbf{z}$ and a free encoder; maximising the ELBO exactly (by enumeration) returns rate zero, "
             "while a decoder forced to depend on $\\mathbf{z}$ returns a positive rate, and sweeping $\\beta$ traces a "
             "monotone rate curve."),
         code='''# Toy: x in {0,1} with p_data(x=1) = 0.3; z in {0,1} with prior p(z=1) = 0.5.
# Encoder q(z=1|x) = (q0, q1); decoder p(x=1|z) = (d0, d1).  Everything is enumerable, so the ELBO is exact.
pdat = np.array([0.7, 0.3])
def elbo_beta(q0, q1, d0, d1, beta=1.0):
    q = np.array([[1 - q0, q0], [1 - q1, q1]])            # q[x, z]
    dec = np.array([[1 - d0, d0], [1 - d1, d1]])          # dec[z, x]
    total = 0.0
    for x in (0, 1):
        rec = sum(q[x, z] * math.log(dec[z, x]) for z in (0, 1))
        rate = sum(q[x, z] * math.log(q[x, z] / 0.5) for z in (0, 1) if q[x, z] > 0)
        total += pdat[x] * (rec - beta * rate)
    return total
def best(beta, tie_decoder=False):
    grid = np.linspace(0.001, 0.999, 61)
    bestv, arg = -1e9, None
    for q0 in grid:
        for q1 in grid:
            for d0 in grid:
                for d1 in ([d0] if tie_decoder else grid):
                    v = elbo_beta(q0, q1, d0, d1, beta)
                    if v > bestv: bestv, arg = v, (q0, q1, d0, d1)
    q0, q1, d0, d1 = arg
    rate = sum(pdat[x] * sum(p * math.log(p / 0.5) for p in ([1 - qq, qq])) for x, qq in zip((0, 1), (q0, q1)))
    return bestv, rate, arg
v_free, rate_free, arg_free = best(1.0)
check("a decoder that can ignore z is optimal with rate 0 (encoder = prior)", rate_free < 1e-3 and abs(arg_free[0] - 0.5) < 0.02 and abs(arg_free[1] - 0.5) < 0.02,
      f"optimal q(z=1|x) = ({arg_free[0]:.3f}, {arg_free[1]:.3f}), rate = {rate_free:.4f} nats, ELBO = {v_free:.4f} = entropy bound {-(pdat*np.log(pdat)).sum():.4f}")
# the same objective when the decoder must depend on z: p(x=1|z) = (d0, 1-d0) -- information in z becomes worth paying for
def best_tied(beta):
    grid = np.linspace(0.001, 0.999, 61); bestv, arg = -1e9, None
    for q0 in grid:
        for q1 in grid:
            for d0 in grid:
                v = elbo_beta(q0, q1, d0, 1 - d0, beta)
                if v > bestv: bestv, arg = v, (q0, q1, d0)
    q0, q1, d0 = arg
    rate = sum(pdat[x] * sum(p * math.log(p / 0.5) for p in ([1 - qq, qq])) for x, qq in zip((0, 1), (q0, q1)))
    return rate
rates = [best_tied(b) for b in (0.5, 1.0, 2.0, 4.0)]
check("with a z-dependent decoder the rate is positive and falls as beta grows",
      rates[0] > 0.05 and all(rates[i + 1] <= rates[i] + 1e-9 for i in range(3)), "rate at beta=0.5,1,2,4: " + ", ".join(f"{r:.3f}" for r in rates))''')

    # ------------------------------------------------------------------ beta-VAE
    c.eq(["EQ-M-035", "EQ-S4-011"], "The $\\beta$-VAE objective and the direction of the bound",
         r"""\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}\mid\mathbf{z})]-\beta\,D_{\mathrm{KL}}\big(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\big)
=\mathcal{L}-(\beta-1)\,D_{\mathrm{KL}}\big(q_\phi\,\|\,p\big),\qquad
\beta>1:\ \mathcal{L}_\beta\le\mathcal{L}\le\log p_\theta(\mathbf{x});\quad \beta<1:\ \text{may exceed }\log p_\theta(\mathbf{x})""",
         plain=(
             "The $\\beta$-VAE multiplies the rate by $\\beta$. Because the rate is never negative, a $\\beta$ above one "
             "subtracts something extra from the ELBO, so the objective is still a valid (but looser) lower bound on the "
             "evidence. A $\\beta$ below one *adds* something back and the objective can rise above the evidence, so it is no "
             "longer a bound and must not be reported as a likelihood."),
         refresher=(
             "The identity is one line of algebra. The consequence matters for reporting: Higgins et al. (2017) proposed "
             "$\\beta>1$ for disentanglement, Locatello et al. (2019) showed the disentanglement claim does not survive "
             "controlled evaluation, and the paper reads $\\beta$ as rate control (main §5.2 and the $\\beta=4$ runs in §8). "
             "The registry marks this `YES-WITH-WORDING-FIX`: the main paper says 'none guaranteed for $\\beta<1$', which the "
             "supplement states precisely as 'can exceed $\\log p$'. The check ports `beta_vae_bound_direction` and adds a "
             "computed instance in the Gaussian model where $\\beta=0.2$ makes the objective exceed the evidence."),
         code='''# exact values in the linear-Gaussian model with a mediocre q
m, v = 0.5, 0.2
rec = expected_reconstruction(x0, m, v); rate = kl_gauss(m, v, 0, 1); lp = log_evidence(x0)
Lb = lambda b: rec - b * rate
check("identity L_beta = ELBO - (beta-1) rate", all(abs(Lb(b) - ((rec - rate) - (b - 1) * rate)) < 1e-12 for b in (0.2, 1, 1.5, 4)))
check("beta > 1: L_beta <= ELBO <= log p(x)", all(Lb(b) <= rec - rate <= lp for b in (1.5, 4.0)),
      f"L_4 = {Lb(4):.4f} <= ELBO {rec-rate:.4f} <= log p {lp:.4f}")
# beta < 1 can exceed log p: choose q close to the posterior so the gap is tiny but the rate is not
m2, v2 = pm, pv
rec2 = expected_reconstruction(x0, m2, v2); rate2 = kl_gauss(m2, v2, 0, 1)
check("beta < 1: the objective can exceed log p(x) (not a bound)", rec2 - 0.2 * rate2 > lp,
      f"beta=0.2 at q=posterior: {rec2 - 0.2*rate2:.4f} > log p {lp:.4f}")
# the ported repository check on fixed numbers
rec_, kl_, logp_ = -80.0, 12.0, -85.0
check("ported check: fixed numbers", all(rec_ - b * kl_ <= rec_ - kl_ <= logp_ for b in (1.5, 4.0)) and rec_ - 0.2 * kl_ > logp_)''')

    # ------------------------------------------------------------------ VQ-VAE
    c.eq("EQ-S4-012", "The VQ-VAE objective and the straight-through estimator",
         r"""\mathcal{L}_{\mathrm{VQ}}=\log p_\theta(\mathbf{x}\mid\mathbf{z}_q)
-\big\|\operatorname{sg}[\mathbf{z}_e(\mathbf{x})]-\mathbf{e}_{k^*}\big\|_2^2
-\gamma\,\big\|\mathbf{z}_e(\mathbf{x})-\operatorname{sg}[\mathbf{e}_{k^*}]\big\|_2^2,\qquad
k^*=\arg\min_k\|\mathbf{z}_e(\mathbf{x})-\mathbf{e}_k\|_2""",
         plain=(
             "A VQ-VAE replaces the Gaussian guess by a *lookup*: the encoder output $\\mathbf{z}_e$ is snapped to the nearest "
             "entry $\\mathbf{e}_{k^*}$ of a learned table (the codebook), and the decoder sees only that entry. The second "
             "term pulls the chosen table entry towards the encoder output; the third ('commitment') pulls the encoder towards "
             "the entry. $\\operatorname{sg}[\\cdot]$ means 'treat as a constant when differentiating', so each term updates "
             "only one side. Snapping has no gradient, so the decoder's gradient at $\\mathbf{z}_q$ is simply copied to "
             "$\\mathbf{z}_e$ (the straight-through estimator)."),
         refresher=(
             "With a uniform prior over $K$ codes the KL term is the constant $\\log K$, so the rate is fixed by the codebook "
             "size and posterior collapse cannot occur in the continuous sense; generation needs a second-stage prior "
             "over code indices (PixelCNN, later transformers), which is why the paper codes a VQ-VAE system by its sampling "
             "family. The check implements the objective in torch with `detach()` as the stop-gradient and confirms, by "
             "autograd on a toy codebook: the codebook term moves only $\\mathbf{e}_{k^*}$ (gradient $2(\\mathbf{e}_{k^*}-\\mathbf{z}_e)$, "
             "zero on other entries, zero on the encoder); the commitment term moves only the encoder; and the "
             "straight-through estimator delivers to $\\mathbf{z}_e$ exactly the decoder gradient at $\\mathbf{z}_q$."),
         code='''torch.manual_seed(0)
D_, K_ = 3, 5
codebook = torch.randn(K_, D_, dtype=torch.float64, requires_grad=True)
z_e = torch.randn(D_, dtype=torch.float64, requires_grad=True)
x = torch.randn(D_, dtype=torch.float64)
W = torch.randn(D_, D_, dtype=torch.float64)                       # a linear "decoder": x_hat = W z_q
with torch.no_grad():
    k_star = int(torch.argmin(((z_e[None] - codebook) ** 2).sum(1)))
e_k = codebook[k_star]
z_q = z_e + (e_k - z_e).detach()                                   # forward: z_q == e_k; backward: dz_q/dz_e = I (straight-through)
rec = -0.5 * ((x - W @ z_q) ** 2).sum()                            # log p(x | z_q) up to a constant (unit-variance Gaussian)
codebook_term = -((z_e.detach() - e_k) ** 2).sum()
gamma = 0.25
commit_term = -gamma * ((z_e - e_k.detach()) ** 2).sum()
loss = -(rec + codebook_term + commit_term)
loss.backward()
# expected gradients
with torch.no_grad():
    g_dec = W.T @ (W @ e_k - x)                                    # gradient of -rec w.r.t. z_q
    exp_codebook = torch.zeros_like(codebook); exp_codebook[k_star] = 2 * (e_k - z_e)        # from -codebook_term
    exp_ze = g_dec + 2 * gamma * (z_e - e_k)                      # straight-through copy + commitment
check("z_q equals the selected code in the forward pass", torch.allclose(z_q.detach(), e_k.detach()))
check("codebook term updates only e_{k*}, by 2(e_k* - z_e); encoder untouched by it",
      torch.allclose(codebook.grad, exp_codebook), f"max |diff| = {(codebook.grad - exp_codebook).abs().max():.1e}")
check("encoder receives decoder gradient at z_q (straight-through) + commitment gradient",
      torch.allclose(z_e.grad, exp_ze), f"max |diff| = {(z_e.grad - exp_ze).abs().max():.1e}")
onehot = np.eye(K_)[2]; unif = np.full(K_, 1 / K_)
kl_oh = float(np.sum(onehot[onehot > 0] * np.log(onehot[onehot > 0] / unif[onehot > 0])))   # KL(one-hot posterior || uniform prior)
check("uniform prior over K codes: rate is the constant log K", abs(kl_oh - math.log(K_)) < 1e-12,
      f"KL(one-hot || uniform) = log K = {math.log(K_):.4f} nats")''')

    # ------------------------------------------------------------------ aggregate posterior / blur
    c.eq("EQ-S4-013", "Aggregate posterior and the conditional-mean argument for blur",
         r"""q(\mathbf{z})=\mathbb{E}_{p_{\mathrm{data}}(\mathbf{x})}\big[q_\phi(\mathbf{z}\mid\mathbf{x})\big],\qquad
\arg\max_{\boldsymbol{\mu}}\ \mathbb{E}\big[\log\mathcal{N}(\mathbf{x};\boldsymbol{\mu},\sigma^2\mathbf{I})\,\big|\,\mathbf{z}\big]=\mathbb{E}[\mathbf{x}\mid\mathbf{z}]""",
         plain=(
             "The *aggregate posterior* is where the encoder sends the whole data set, averaged over data points; it rarely "
             "matches the prior, so some regions of $\\mathbf{z}$ decode to nothing realistic. The second statement explains "
             "blurry samples: if several different images are all mapped near the same $\\mathbf{z}$, a Gaussian decoder with "
             "one mean per $\\mathbf{z}$ does best by outputting their *average*, and the average of sharp images is a blurry "
             "image. The culprit is the single-mode decoder, not the squared error as such."),
         refresher=(
             "For a fixed $\\mathbf{z}$ the expected Gaussian log-likelihood is a quadratic in $\\boldsymbol{\\mu}$ whose "
             "maximiser is the conditional mean, for any output variance; with a factorised, unimodal decoder the optimum "
             "averages every $\\mathbf{x}$ the posterior could map to that $\\mathbf{z}$ (Dai and Wipf 2019; Alemi et al. 2018). "
             "A mismatch between $q(\\mathbf{z})$ and $p(\\mathbf{z})$ is the second failure listed in S4.2 (Tomczak and Welling "
             "2018 fit the prior to the aggregate). The check (i) computes the aggregate posterior of the linear-Gaussian model "
             "in closed form (a Gaussian whose variance is $\\neq1$ for an imperfect encoder) and compares it with a Monte-Carlo "
             "mixture, and (ii) verifies on a two-point conditional that the maximiser of the expected log-likelihood is the mean "
             "and that the log-likelihood of the *mode* is strictly lower."),
         code='''# (i) aggregate posterior of the linear-Gaussian model with the exact posterior encoder: q(z) = N(0, A^2/(A^2+S2) * ... ) computed exactly
# p_data(x) = N(0, A^2 + S2);  q(z|x) = N(c x, pv) with c = A/(A^2+S2)  =>  q(z) = N(0, c^2 (A^2+S2) + pv) = N(0, 1) exactly
c_ = A / (A * A + S2)
agg_var_exact_encoder = c_**2 * (A * A + S2) + S2 / (A * A + S2)
check("exact-posterior encoder: aggregate posterior equals the prior N(0,1)", abs(agg_var_exact_encoder - 1.0) < 1e-12, f"var = {agg_var_exact_encoder:.12f}")
# an amortised encoder with the wrong scale (c' = 0.5 c, same variance) gives an aggregate that is NOT the prior
agg_var_bad = (0.5 * c_)**2 * (A * A + S2) + S2 / (A * A + S2)
xs = rng.normal(0, math.sqrt(A * A + S2), size=200_000)
zs = 0.5 * c_ * xs + math.sqrt(S2 / (A * A + S2)) * rng.normal(size=xs.size)
check("imperfect encoder: aggregate posterior variance != 1 (closed form matches Monte Carlo)",
      abs(zs.var() - agg_var_bad) < 0.01 and abs(agg_var_bad - 1) > 0.2, f"var = {agg_var_bad:.3f} (MC {zs.var():.3f})")
# (ii) conditional-mean argument: two images x_a, x_b both decoded from the same z
xa, xb = np.array([0.0, 1.0]), np.array([1.0, 0.0])
def expected_ll(mu, s2=0.1):
    return 0.5 * (scipy.stats.norm.logpdf(xa, mu, math.sqrt(s2)).sum() + scipy.stats.norm.logpdf(xb, mu, math.sqrt(s2)).sum())
mean = 0.5 * (xa + xb)
res = scipy.optimize.minimize(lambda m: -expected_ll(m), x0=np.array([0.2, 0.7]))
check("maximiser of the expected Gaussian log-likelihood is the conditional mean", np.allclose(res.x, mean, atol=1e-4), f"argmax = {res.x.round(4)}")
check("the mean (blurry average) beats either sharp image as decoder output", expected_ll(mean) > expected_ll(xa) and expected_ll(mean) > expected_ll(xb),
      f"E[log lik] at mean {expected_ll(mean):.2f} vs at x_a {expected_ll(xa):.2f}")''')

    return c.build()


if __name__ == "__main__":
    print(build().path)
