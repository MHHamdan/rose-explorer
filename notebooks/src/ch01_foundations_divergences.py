"""Chapter 1: probability, entropy and divergences (main paper §3.1, Supplement S1.1--S1.2, S1.12 table)."""
from nbkit import Chapter


def build():
    c = Chapter(
        1, "foundations_divergences", "Probability, entropy and divergences",
        summary=(
            "Every generative model is a probability distribution $p_\\theta$ that is fitted to a data distribution "
            "$p_{\\mathrm{data}}$, and fitting needs a way to say how far two distributions are apart. This chapter "
            "introduces the quantities the paper uses for that purpose (entropy, the Kullback--Leibler and "
            "Jensen--Shannon divergences, the Wasserstein-1 distance) and the one fact that organises the whole "
            "survey: for a neural model the likelihood is usually an integral that cannot be computed, so each "
            "family must take a route around it."),
        prerequisites="elementary probability (densities, expectations) and a little calculus",
        paper_sections="main paper §3.1--3.2 (foundations); Supplement S1.1, S1.2, Table S1",
    )

    c.section("Why these quantities", (
        "The paper's central question is what a model can and cannot compute about its own likelihood. The answer "
        "decides which objective it can minimise, and every objective in the survey is a divergence or a bound on one. "
        "We therefore start with the divergences themselves, on small discrete examples where every sum is exact, "
        "and keep the continuous cases (Gaussians, a bimodal target) for the places where a closed form exists."))

    # ------------------------------------------------------------------ EQ-S1-001
    c.eq(
        "EQ-S1-001", "Why a table or a Gaussian is not enough",
        r"\#\text{params(table over }28\times28\text{ binary images)}=2^{784}-1,\qquad "
        r"\#\text{params}(\mathcal N(\mu,\Sigma)\text{ on }\mathbb R^D)=\tfrac{D(D+3)}{2}=O(D^2)",
        plain=(
            "A $28\\times28$ black-and-white image has $784$ pixels, each $0$ or $1$, so there are $2^{784}$ possible "
            "images. A table that stores the probability of every image needs one number per image, minus one because "
            "the probabilities must sum to one. That is about $10^{236}$ numbers: more than the atoms in the universe. "
            "A Gaussian, at the other extreme, needs only a mean ($D$ numbers) and a covariance matrix "
            "($D(D+1)/2$ numbers), but it can only describe one blob, not the sharp, structured set that real images "
            "occupy. Deep generative models sit in between: a neural network with millions of parameters that is "
            "flexible but no longer has a computable likelihood."),
        refresher=(
            "The table count is the dimension of the probability simplex over $2^{784}$ outcomes. The Gaussian count "
            "is $D$ (mean) plus $D(D+1)/2$ (symmetric covariance), $D(D+3)/2$ in total, which is $O(D^2)$; for "
            "$D=784$ it is $308{,}504$. The paper's point is not the arithmetic but the gap: natural images live near a "
            "low-dimensional structured set in $\\mathbb R^D$, which neither a full table (too many parameters, no "
            "generalisation) nor a Gaussian (one unimodal blob) describes. The three routes of the next entry are the "
            "three ways to parameterise a flexible $p_\\theta$ and still train it."),
        code='''D = 28 * 28
table_params = 2**D - 1
gauss_params = D + D * (D + 1) // 2
check("table over 28x28 binary images has 2^784 - 1 parameters",
      table_params == 2**784 - 1 and len(str(table_params)) == 237,
      f"2^784 - 1 has {len(str(table_params))} decimal digits")
check("Gaussian on R^784 has D(D+3)/2 = O(D^2) parameters",
      gauss_params == D * (D + 3) // 2 and gauss_params == 308504,
      f"D(D+3)/2 = {gauss_params:,}")
# O(D^2): the ratio to D^2 stays bounded as D grows
ratios = [(d + d * (d + 1) // 2) / d**2 for d in (10, 100, 1000, 10000)]
check("Gaussian parameter count / D^2 tends to 1/2", all(0.5 <= r <= 0.7 for r in ratios) and abs(ratios[-1] - 0.5) < 1e-3,
      "ratios: " + ", ".join(f"{r:.4f}" for r in ratios))''')

    # ------------------------------------------------------------------ EQ-M-001 + EQ-S1-002
    c.eq(
        ["EQ-M-001", "EQ-S1-002"], "The intractable likelihood and the routes around it",
        r"p_\theta(\mathbf x)=\int p_\theta(\mathbf x\mid\mathbf z)\,p(\mathbf z)\,d\mathbf z"
        r"\quad\text{or}\quad p_\theta(\mathbf x)=\frac{\exp(-E_\theta(\mathbf x))}{Z_\theta},\qquad "
        r"Z_\theta=\int\exp(-E_\theta(\mathbf x))\,d\mathbf x",
        plain=(
            "A neural generative model usually defines the probability of a data point $\\mathbf x$ in one of two "
            "ways. Either it introduces a hidden variable $\\mathbf z$ (a latent) and says how $\\mathbf x$ is produced "
            "from it, in which case the probability of $\\mathbf x$ alone is the integral over all the $\\mathbf z$ "
            "that could have produced it. Or it defines an energy $E_\\theta(\\mathbf x)$, low for plausible points, "
            "and turns it into a probability by dividing by the total $Z_\\theta$. In both cases the integral runs "
            "over a huge space and cannot be computed exactly once a neural network is inside it. The survey's "
            "taxonomy follows from the three ways out: (i) *factorise* $\\mathbf x$ into a product of simple "
            "conditionals, or push simple random noise through a reversible function, so the probability can still "
            "be computed exactly; (ii) *bound* the likelihood from below, or *match scores* (gradients of the log-density) instead "
            "of densities; (iii) give up the density altogether and train a sampler *adversarially*."),
        refresher=(
            "The marginal $\\int p_\\theta(\\mathbf x\\mid\\mathbf z)p(\\mathbf z)\\,d\\mathbf z$ is tractable only when "
            "the conditional and prior are conjugate or $\\mathbf z$ is low-dimensional and discrete; with a decoder "
            "network it is a high-dimensional integral with no closed form. The normaliser $Z_\\theta$ has the same "
            "status for an energy network. Figure S1 of the supplement lists the five constructions: transport "
            "$p_Z(f(\\mathbf x))\\,|\\det J_f|$ (flows, Chapter 5), factorisation $\\prod_i p(x_i\\mid\\mathbf x_{<i})$ "
            "(autoregressive models, Chapter 4), the bound $\\log p_\\theta(\\mathbf x)\\ge\\mathcal L$ (VAEs, "
            "Chapter 3; diffusion, Chapter 9), score matching $\\nabla_{\\mathbf x}\\log p(\\mathbf x)$ (Chapters 7--8) "
            "and the adversarial game $\\min_\\theta\\max_\\psi V$ (Chapter 6). The verification below builds a tiny "
            "model where the marginal and the normaliser *can* be computed (two latent states; a $7$-point energy), "
            "confirms that the two routes give a normalised density, then checks the three exact-likelihood "
            "identities from the figure on the same discrete example: the chain rule, the change of variables for a "
            "linear map (the sign of the log-determinant, and that composing maps adds their log-determinants, "
            "the check `change_of_variables_sign_and_composition` of the repository), and that a lower bound "
            "is one."),
        code='''# (1) A latent-variable model small enough to marginalise exactly: z in {0,1}, x in {0,..,4}.
prior = np.array([0.3, 0.7])
lik = np.array([[0.5, 0.3, 0.1, 0.05, 0.05],        # p(x | z=0)
                [0.05, 0.05, 0.2, 0.3, 0.4]])        # p(x | z=1)
marginal = prior @ lik                               # p(x) = sum_z p(x|z) p(z)
check("latent marginal sums to one", abs(marginal.sum() - 1) < 1e-12, f"sum p(x) = {marginal.sum():.12f}")

# (2) An energy-based model on 7 points: p = exp(-E)/Z needs the sum Z over every point.
E = np.array([0.0, 1.2, -0.3, 2.0, 0.7, -1.1, 0.4])
Z = np.exp(-E).sum()
p_ebm = np.exp(-E) / Z
check("energy model normalised by Z", abs(p_ebm.sum() - 1) < 1e-12 and np.argmax(p_ebm) == np.argmin(E),
      f"Z = {Z:.4f}; the lowest energy has the highest probability")

# (3) Route 'factorise': the chain rule on a 3x3x3 joint table.
P = rng.dirichlet(np.ones(27)).reshape(3, 3, 3)
x = (2, 0, 1)
p1 = P.sum((1, 2))[x[0]]
p2 = P[x[0]].sum(1)[x[1]] / P[x[0]].sum()
p3 = P[x[0], x[1], x[2]] / P[x[0], x[1]].sum()
check("factorise: log p(x) = sum_i log p(x_i | x_<i)", *close(math.log(p1) + math.log(p2) + math.log(p3), math.log(P[x]), 1e-12))

# (4) Route 'transport': z = f(x) = A2 (A1 x + b1) + b2 with z ~ N(0, I). The density of x is
#     p_Z(f(x)) * |det A1| * |det A2| (log-determinants ADD under composition, with a PLUS sign in the
#     normalising direction x -> z). Compare with the exact Gaussian density of x = A^{-1}(z - b).
A1 = rng.normal(size=(3, 3)) + 2 * np.eye(3); b1 = rng.normal(size=3)
A2 = rng.normal(size=(3, 3)) + 2 * np.eye(3); b2 = rng.normal(size=3)
xs = rng.normal(size=(5, 3))
z = (xs @ A1.T + b1) @ A2.T + b2
logp_flow = scipy.stats.multivariate_normal(np.zeros(3), np.eye(3)).logpdf(z) \\
            + np.log(abs(np.linalg.det(A1))) + np.log(abs(np.linalg.det(A2)))
A = A2 @ A1; b = A2 @ b1 + b2; Ainv = np.linalg.inv(A)
logp_true = scipy.stats.multivariate_normal(-Ainv @ b, Ainv @ Ainv.T).logpdf(xs)
check("transport: log p_Z(f(x)) + sum_k log|det J_k| = exact log density", *close(logp_flow, logp_true, 1e-10))
logp_wrong_sign = scipy.stats.multivariate_normal(np.zeros(3), np.eye(3)).logpdf(z) - np.log(abs(np.linalg.det(A)))
check("transport: the minus-sign variant is wrong", np.max(np.abs(logp_wrong_sign - logp_true)) > 1e-3,
      f"max error of the wrong sign = {np.max(np.abs(logp_wrong_sign - logp_true)):.3f}")

# (5) Route 'bound': for the model of (1), Jensen gives log p(x) >= E_q[log p(x,z) - log q(z)] for ANY q(z).
x0 = 3
for q0 in (0.1, 0.5, 0.9):
    q = np.array([q0, 1 - q0])
    elbo = np.sum(q * (np.log(lik[:, x0] * prior) - np.log(q)))
    check(f"bound: ELBO <= log p(x) for q(z=0)={q0}", elbo <= math.log(marginal[x0]) + 1e-12,
          f"ELBO = {elbo:.4f} <= log p(x) = {math.log(marginal[x0]):.4f}")''')

    # ------------------------------------------------------------------ EQ-S1-003
    c.eq(
        "EQ-S1-003", "Entropy and cross-entropy",
        r"H(p)=-\mathbb E_{\mathbf x\sim p}\big[\log p(\mathbf x)\big],\qquad "
        r"H(p,q)=-\mathbb E_{\mathbf x\sim p}\big[\log q(\mathbf x)\big]",
        plain=(
            "Entropy measures how unpredictable a distribution is: the average surprise $-\\log p(\\mathbf x)$ of "
            "its own outcomes. A fair coin has entropy $\\log 2$; a coin that always lands heads has entropy $0$. "
            "Cross-entropy is the surprise you feel if the outcomes come from $p$ but you had bet according to $q$. "
            "Betting on the truth is always best, so $H(p,q)\\ge H(p)$, with equality only when $q=p$. The training "
            "loss of a language model or a classifier is a cross-entropy: the data supply $p$, the network supplies $q$."),
        refresher=(
            "In nats when the logarithm is natural, in bits when it is base 2. For a discrete distribution "
            "$0\\le H(p)\\le\\log K$ with $K$ outcomes, the maximum at the uniform distribution; for a density the "
            "differential entropy can be negative and is not invariant under reparameterisation, which is one reason "
            "the paper compares likelihoods only within a fixed data representation (Chapter 10). The Gibbs "
            "inequality $H(p,q)\\ge H(p)$ is the statement $D_{\\mathrm{KL}}(p\\|q)\\ge0$ of the next entry. The "
            "check uses the Gaussian closed form $H(\\mathcal N(\\mu,\\sigma^2))=\\tfrac12\\log(2\\pi e\\sigma^2)$ as "
            "a continuous example and the uniform maximum as a discrete one."),
        code='''def H(p):                       # entropy of a discrete distribution, in nats
    p = np.asarray(p, float); m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))

def H_cross(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float); m = p > 0
    return float(-np.sum(p[m] * np.log(q[m])))

check("fair coin: H = log 2", *close(H([0.5, 0.5]), math.log(2), 1e-12))
check("certain outcome: H = 0", H([1.0, 0.0]) == 0.0)
K = 6
worst = max(H(rng.dirichlet(np.ones(K))) for _ in range(3000))
check("uniform maximises entropy over 3000 random 6-point distributions", worst <= math.log(K) + 1e-12,
      f"max H = {worst:.4f} <= log 6 = {math.log(K):.4f}")
p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
check("cross-entropy >= entropy (Gibbs), equality iff q = p",
      H_cross(p, q) >= H(p) and abs(H_cross(p, p) - H(p)) < 1e-12,
      f"H(p,q) = {H_cross(p, q):.4f} >= H(p) = {H(p):.4f}")
# continuous example: differential entropy of N(mu, sigma^2) = 0.5 log(2 pi e sigma^2), by quadrature
mu, sigma = 0.3, 1.7
f = lambda x: -scipy.stats.norm.pdf(x, mu, sigma) * scipy.stats.norm.logpdf(x, mu, sigma)
h_num = scipy.integrate.quad(f, mu - 12 * sigma, mu + 12 * sigma)[0]
check("Gaussian differential entropy by quadrature", *close(h_num, 0.5 * math.log(2 * math.pi * math.e * sigma**2), 1e-8))
check("differential entropy can be negative (sigma = 0.1)", 0.5 * math.log(2 * math.pi * math.e * 0.01) < 0,
      f"H(N(0, 0.1^2)) = {0.5 * math.log(2 * math.pi * math.e * 0.01):.3f} nats")''')

    # ------------------------------------------------------------------ EQ-M-002 + EQ-S1-004
    c.eq(
        ["EQ-M-002", "EQ-S1-004"], "The Kullback--Leibler divergence",
        r"D_{\mathrm{KL}}(p\,\|\,q)=\mathbb E_{\mathbf x\sim p}\!\left[\log\frac{p(\mathbf x)}{q(\mathbf x)}\right]"
        r"=H(p,q)-H(p)\;\ge0,\qquad\text{finite only when }p\ll q",
        plain=(
            "The KL divergence is the extra surprise you pay for believing $q$ when the truth is $p$: cross-entropy "
            "minus entropy. It is never negative and is zero only when the two distributions agree. It is *not* a "
            "distance: $D_{\\mathrm{KL}}(p\\|q)$ and $D_{\\mathrm{KL}}(q\\|p)$ differ, often a lot. The condition "
            "$p\\ll q$ (\"$p$ is absolutely continuous with respect to $q$\") means that wherever $p$ puts probability, "
            "$q$ must put some too; if $q$ says an event is impossible and it happens, the surprise is infinite."),
        refresher=(
            "Non-negativity is Jensen's inequality applied to $-\\log$, or equivalently $\\log u\\le u-1$: "
            "$-D_{\\mathrm{KL}}(p\\|q)=\\mathbb E_p[\\log(q/p)]\\le\\mathbb E_p[q/p-1]\\le0$. Equality holds iff "
            "$p=q$ almost everywhere. Between two diagonal Gaussians in $d$ dimensions the closed form is "
            "$\\sum_j\\big[\\log(\\sigma_{2j}/\\sigma_{1j})+(\\sigma_{1j}^2+(\\mu_{1j}-\\mu_{2j})^2)/(2\\sigma_{2j}^2)"
            "-\\tfrac12\\big]$, which the VAE rate term uses (Chapter 3). The check compares that closed form with a "
            "Monte-Carlo estimate (the repository check `kl_gaussian_closed_form_vs_mc`, within five standard "
            "errors), verifies $D_{\\mathrm{KL}}=H(p,q)-H(p)$ exactly on a discrete example, shows the asymmetry, and "
            "shows that the divergence is infinite when $p\\not\\ll q$."),
        code='''def kl_disc(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float); m = p > 0
    return math.inf if np.any(q[m] == 0) else float(np.sum(p[m] * np.log(p[m] / q[m])))

# closed form between diagonal Gaussians versus Monte Carlo (ported from verify_equations.py)
d = 4
mu1, mu2 = rng.normal(size=d), rng.normal(size=d)
s1, s2 = np.exp(rng.normal(scale=0.3, size=d)), np.exp(rng.normal(scale=0.3, size=d))
closed = np.sum(np.log(s2 / s1) + (s1**2 + (mu1 - mu2)**2) / (2 * s2**2) - 0.5)
xs = mu1 + s1 * rng.normal(size=(400_000, d))
diff = scipy.stats.norm.logpdf(xs, mu1, s1).sum(1) - scipy.stats.norm.logpdf(xs, mu2, s2).sum(1)
mc, se = diff.mean(), diff.std() / math.sqrt(len(xs))
check("KL(N1||N2): Monte Carlo = closed form within 5 s.e.", *close(mc, closed, 5 * se, f"MC {mc:.4f}, closed {closed:.4f}, s.e. {se:.1e}; "))

p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
check("KL = H(p,q) - H(p) exactly", *close(kl_disc(p, q), H_cross(p, q) - H(p), 1e-12))
check("KL >= 0 and KL(p||p) = 0", kl_disc(p, q) >= 0 and kl_disc(p, p) == 0.0, f"KL(p||q) = {kl_disc(p, q):.4f}")
check("KL is asymmetric", abs(kl_disc(p, q) - kl_disc(q, p)) > 1e-6,
      f"KL(p||q) = {kl_disc(p, q):.4f}, KL(q||p) = {kl_disc(q, p):.4f}")
check("KL = inf when p puts mass where q has none (p not << q)",
      math.isinf(kl_disc([0.5, 0.5], [1.0, 0.0])) and kl_disc([1.0, 0.0], [0.5, 0.5]) == math.log(2),
      "KL([.5,.5] || [1,0]) = inf but KL([1,0] || [.5,.5]) = log 2")''')

    # ------------------------------------------------------------------ EQ-M-004 + EQ-S1-010
    c.eq(
        ["EQ-M-004", "EQ-S1-010"], "Forward KL covers modes, reverse KL seeks one",
        r"D_{\mathrm{KL}}(p_{\mathrm{data}}\,\|\,p_\theta)=\infty\ \text{if }p_\theta=0\text{ where }p_{\mathrm{data}}>0"
        r"\quad\text{(mode covering)};\qquad "
        r"D_{\mathrm{KL}}(p_\theta\,\|\,p_{\mathrm{data}})\ \text{tolerates missing modes (mode seeking)}",
        plain=(
            "Because KL is asymmetric, which way round you write it changes what a fitted model looks like. "
            "Minimising the *forward* divergence $D_{\\mathrm{KL}}(p_{\\mathrm{data}}\\|p_\\theta)$ (which is what "
            "maximum likelihood does, Chapter 2) punishes the model infinitely for giving zero probability to "
            "anything the data do, so the model spreads out to cover every cluster, even at the price of putting "
            "mass in between them where nothing real lives. Minimising the *reverse* divergence "
            "$D_{\\mathrm{KL}}(p_\\theta\\|p_{\\mathrm{data}})$ punishes the model for putting mass where the data "
            "have none, so a model too simple to cover everything picks one cluster and ignores the rest. The "
            "figure below fits a single Gaussian to a two-humped target both ways."),
        refresher=(
            "With a bimodal target $p=\\tfrac12\\mathcal N(-3,0.7^2)+\\tfrac12\\mathcal N(3,0.7^2)$ and a Gaussian "
            "family $q=\\mathcal N(m,s^2)$: the forward-KL minimiser is the moment match, $m=0$ and "
            "$s^2=\\mathrm{Var}_p=9+0.49$ (because $\\arg\\min_q D_{\\mathrm{KL}}(p\\|q)$ over an exponential family "
            "matches sufficient statistics), while the reverse-KL minimiser sits on one mode, $m\\approx\\pm3$, "
            "$s\\approx0.7$. The paper uses this asymmetry (with the discriminator as a learned perceptual loss) to "
            "explain diverse but occasionally implausible samples from likelihood-trained models and sharp but "
            "mode-dropping samples from adversarial ones, and it is why log-likelihood and sample quality can "
            "disagree (Theis et al., 2016). The check is the repository's "
            "`kl_forward_mode_covering_vs_reverse_mode_seeking`: both divergences are computed by quadrature on a "
            "grid and minimised with Nelder--Mead. Caveat the paper states in S1: the claims are about the "
            "population divergences; with finite data a maximum-likelihood fit can still miss a mode that the "
            "sample does not contain."),
        code='''xg = np.linspace(-12, 12, 6001); dx = xg[1] - xg[0]
p = 0.5 * scipy.stats.norm.pdf(xg, -3, 0.7) + 0.5 * scipy.stats.norm.pdf(xg, 3, 0.7)

def both_kls(par):
    m, ls = par
    q = scipy.stats.norm.pdf(xg, m, np.exp(ls)) + 1e-300
    fwd = np.sum(p * (np.log(p + 1e-300) - np.log(q))) * dx     # KL(p_data || q)
    rev = np.sum(q * (np.log(q) - np.log(p + 1e-300))) * dx     # KL(q || p_data)
    return fwd, rev

fwd = scipy.optimize.minimize(lambda a: both_kls(a)[0], [0.3, 0.0], method="Nelder-Mead").x
rev = scipy.optimize.minimize(lambda a: both_kls(a)[1], [0.3, 0.0], method="Nelder-Mead").x
sf, sr = np.exp(fwd[1]), np.exp(rev[1])
moment_sd = math.sqrt(9 + 0.49)
check("forward KL: fit is the moment match (mean 0, sd = sqrt(9.49)), covering both modes",
      abs(fwd[0]) < 1e-2 and abs(sf - moment_sd) < 1e-2, f"N({fwd[0]:.3f}, {sf:.3f}^2) vs sd {moment_sd:.3f}")
check("reverse KL: fit locks on one mode (mean +-3, sd 0.7)",
      abs(abs(rev[0]) - 3) < 0.05 and abs(sr - 0.7) < 0.05, f"N({rev[0]:.3f}, {sr:.3f}^2)")
# the forward divergence of the mode-seeking fit is much larger than that of the covering fit (near-infinite)
check("the mode-seeking fit has a far larger forward KL than the covering fit",
      both_kls(rev)[0] > 10 * both_kls(fwd)[0], f"forward KL: covering {both_kls(fwd)[0]:.3f}, seeking {both_kls(rev)[0]:.3f}")

fig, ax = plt.subplots(figsize=(6, 2.6))
ax.plot(xg, p, "k", lw=2, label="target $p_{\\\\mathrm{data}}$ (two modes)")
ax.plot(xg, scipy.stats.norm.pdf(xg, fwd[0], sf), label="forward-KL fit (covers both)")
ax.plot(xg, scipy.stats.norm.pdf(xg, rev[0], sr), "--", label="reverse-KL fit (one mode)")
ax.set_xlim(-7, 7); ax.set_xlabel("x"); ax.set_ylabel("density"); ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()''')

    # ------------------------------------------------------------------ EQ-S1-005
    c.eq(
        "EQ-S1-005", "$f$-divergences: one recipe for KL, reverse KL, $\\chi^2$ and JS",
        r"D_f(p\,\|\,q)=\mathbb E_{\mathbf x\sim q}\!\left[f\!\left(\frac{p(\mathbf x)}{q(\mathbf x)}\right)\right],"
        r"\qquad f\text{ convex},\ f(1)=0;\qquad f(u)=u\log u\ \Rightarrow\ D_f=D_{\mathrm{KL}}(p\,\|\,q)",
        plain=(
            "Many divergences share one shape: look at the ratio $p/q$ at each point, pass it through a fixed "
            "convex function $f$ that is zero at $1$, and average under $q$. Different choices of $f$ give "
            "different divergences; $u\\log u$ gives KL, $-\\log u$ gives reverse KL, $(u-1)^2$ gives Pearson's "
            "$\\chi^2$, and a particular combination of logarithms gives the Jensen--Shannon divergence of the next "
            "entry. The family matters in the survey because GANs can be set up to minimise any of them "
            "(Chapter 6), and because every member fails in the same way when the two distributions do not overlap."),
        refresher=(
            "Convexity of $f$ with $f(1)=0$ gives $D_f\\ge0$ by Jensen under $q$: "
            "$\\mathbb E_q[f(p/q)]\\ge f(\\mathbb E_q[p/q])=f(1)=0$. Generators: KL $u\\log u$; reverse KL $-\\log u$; "
            "$\\chi^2$ $(u-1)^2$; total variation $\\tfrac12|u-1|$; JS "
            "$\\tfrac12[u\\log u-(u+1)\\log\\tfrac{u+1}{2}]$. The f-GAN of Nowozin et al. (2016) estimates $D_f$ from "
            "samples through its convex conjugate (Chapter 6). The check evaluates the recipe with $f(u)=u\\log u$ "
            "and with the JS generator on a random pair of discrete distributions and compares with the direct "
            "formulas (ported from `js_bounds_and_disjoint_supports`)."),
        code='''def js_disc(p, q):
    m = 0.5 * (np.asarray(p, float) + np.asarray(q, float))
    return 0.5 * kl_disc(p, m) + 0.5 * kl_disc(q, m)

p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
u = p / q
check("f(u) = u log u reproduces KL(p||q)", *close(np.sum(q * u * np.log(u)), kl_disc(p, q), 1e-12))
check("f(u) = -log u reproduces KL(q||p)", *close(np.sum(q * (-np.log(u))), kl_disc(q, p), 1e-12))
check("f(u) = (u-1)^2 reproduces Pearson chi^2 = sum (p-q)^2/q", *close(np.sum(q * (u - 1)**2), np.sum((p - q)**2 / q), 1e-12))
f_js = 0.5 * (u * np.log(u) - (u + 1) * np.log((u + 1) / 2))
check("JS generator reproduces the Jensen-Shannon divergence", *close(np.sum(q * f_js), js_disc(p, q), 1e-12))
# non-negativity for every generator, on 500 random pairs
gens = {"KL": lambda u: u * np.log(u), "revKL": lambda u: -np.log(u), "chi2": lambda u: (u - 1)**2,
        "JS": lambda u: 0.5 * (u * np.log(u) - (u + 1) * np.log((u + 1) / 2))}
mins = {k: min(np.sum(q_ * g(p_ / q_)) for p_, q_ in (tuple(rng.dirichlet(np.ones(5)) for _ in range(2)) for _ in range(500)))
        for k, g in gens.items()}
check("every f-divergence is >= 0 on 500 random pairs", all(v >= -1e-12 for v in mins.values()),
      ", ".join(f"min {k} = {v:.2e}" for k, v in mins.items()))''')

    # ------------------------------------------------------------------ EQ-M-005 + EQ-S1-006
    c.eq(
        ["EQ-M-005", "EQ-S1-006"], "The Jensen--Shannon divergence and the problem of disjoint supports",
        r"\mathrm{JS}(p\,\|\,q)=\tfrac12 D_{\mathrm{KL}}(p\,\|\,m)+\tfrac12 D_{\mathrm{KL}}(q\,\|\,m),\quad "
        r"m=\tfrac12(p+q);\qquad 0\le\mathrm{JS}\le\log2,\ \text{and }\mathrm{JS}=\log 2\text{ when the supports are disjoint}",
        plain=(
            "The JS divergence fixes two inconveniences of KL: it is symmetric, and it is never infinite, because "
            "it compares each distribution with their average $m$, which covers everything either of them covers. "
            "Its largest possible value is $\\log 2$, reached exactly when $p$ and $q$ live on completely separate "
            "sets. That maximum is also its weakness: if a generated distribution and the real one do not overlap "
            "at all, which is the usual situation early in GAN training, JS is flat at $\\log 2$ no matter how close "
            "the two are in space, so it gives the generator no direction to move in."),
        refresher=(
            "JS is the $f$-divergence with generator $\\tfrac12[u\\log u-(u+1)\\log\\tfrac{u+1}{2}]$; its square root "
            "is a metric. The bound $\\log2$ follows from $p\\le2m$, so $\\log(p/m)\\le\\log2$, and is attained iff "
            "$p\\perp q$. For every $f$-divergence, disjoint supports give a constant ($\\log 2$ for JS, total "
            "variation $1$) or infinity (KL), which Arjovsky and Bottou (2017) argue is generic for distributions on "
            "low-dimensional manifolds in $\\mathbb R^D$; this is the motivation for the Wasserstein distance of the "
            "next entry. The check is the repository's `js_bounds_and_disjoint_supports`: the maximum of JS over "
            "2000 random pairs stays below $\\log2$, and a disjoint pair gives JS $=\\log2$, TV $=1$ and KL $=\\infty$. "
            "In the paper the GAN value at the optimal discriminator is $2\\,\\mathrm{JS}-\\log4$ (Chapter 6)."),
        code='''mx = 0.0
for _ in range(2000):
    a, b = rng.dirichlet(np.ones(6) * 0.3), rng.dirichlet(np.ones(6) * 0.3)
    mx = max(mx, js_disc(a, b))
check("JS <= log 2 over 2000 random (spiky) pairs", mx <= math.log(2) + 1e-12, f"max JS = {mx:.4f}, log 2 = {math.log(2):.4f}")
p, q = np.array([0.2, 0.8, 0.0, 0.0]), np.array([0.0, 0.0, 0.6, 0.4])
tv = 0.5 * np.abs(p - q).sum()
check("disjoint supports: JS = log 2, TV = 1, KL = inf",
      abs(js_disc(p, q) - math.log(2)) < 1e-12 and abs(tv - 1) < 1e-12 and math.isinf(kl_disc(p, q)),
      f"JS = {js_disc(p, q):.6f}, TV = {tv}, KL = inf")
a, b = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
check("JS is symmetric and zero at p = q", abs(js_disc(a, b) - js_disc(b, a)) < 1e-12 and js_disc(a, a) == 0.0)
# flatness: two point masses that move apart; JS is constant as soon as they no longer coincide
grid = np.round(np.arange(-200, 201) * 0.005, 6)                 # a fine grid containing 0 and every theta below
def point_mass(at):
    v = np.zeros(grid.size); v[np.argmin(np.abs(grid - at))] = 1.0; return v
js_at = [js_disc(point_mass(0.0), point_mass(theta)) for theta in (1.0, 0.1, 0.01)]
check("JS between delta_0 and delta_theta is log 2 for every theta != 0 (no gradient signal)",
      np.allclose(js_at, math.log(2)), "JS = log 2 at theta = 1, 0.1, 0.01")''')

    # ------------------------------------------------------------------ EQ-M-006 + EQ-S1-007
    c.eq(
        ["EQ-M-006", "EQ-S1-007"], "The Wasserstein-1 distance and Kantorovich--Rubinstein duality",
        r"W_1(p,q)=\inf_{\gamma\in\Pi(p,q)}\mathbb E_{(\mathbf x,\mathbf y)\sim\gamma}\big[\|\mathbf x-\mathbf y\|\big]"
        r"=\sup_{\|f\|_L\le1}\Big(\mathbb E_{\mathbf x\sim p}[f(\mathbf x)]-\mathbb E_{\mathbf x\sim q}[f(\mathbf x)]\Big)",
        plain=(
            "Imagine $p$ and $q$ as two piles of sand of equal weight. The Wasserstein-1 (earth mover) distance is "
            "the least total effort, mass times distance moved, needed to reshape one pile into the other; "
            "$\\gamma$ is a transport plan saying how much sand goes from each place to each other place. The "
            "second form says the same number can be found by a different game: choose a function $f$ that never "
            "changes faster than slope $1$ and ask how differently it scores the two piles on average. Unlike KL "
            "and JS, $W_1$ keeps changing smoothly as two non-overlapping distributions move closer, which is why "
            "the Wasserstein GAN uses it."),
        refresher=(
            "$\\Pi(p,q)$ is the set of couplings with marginals $p$ and $q$; the primal is a linear programme in "
            "$\\gamma$, the dual is a linear programme over $1$-Lipschitz $f$, and strong duality "
            "(Kantorovich--Rubinstein) makes them equal. In one dimension $W_1=\\int|F_p(x)-F_q(x)|\\,dx$, the area "
            "between the CDFs. The paper's claim, in its corrected wording (registry status YES-WITH-WORDING-FIX), "
            "is that $W_1(p_{\\mathrm{data}},p_\\theta)$ is continuous in $\\theta$ whenever the generator "
            "$G_\\theta$ is continuous in $\\theta$ (on a compact space, or for a locally Lipschitz generator with "
            "integrable constant), and differentiable almost everywhere when $G_\\theta$ is locally Lipschitz "
            "(Arjovsky et al., 2017); the uncorrected form omitted the integrability condition. The check "
            "(`w1_kantorovich_rubinstein_dual_and_continuity`) solves the primal and dual LPs on seven points with "
            "`scipy.optimize.linprog`, compares both with the CDF formula, and contrasts $W_1(\\delta_0,\\delta_\\theta)"
            "=|\\theta|\\to0$ with JS $\\equiv\\log2$."),
        code='''xs = np.sort(rng.normal(size=7))
p, q = rng.dirichlet(np.ones(7)), rng.dirichlet(np.ones(7))
n = len(xs)
C = np.abs(xs[:, None] - xs[None, :])
# primal: min <C, gamma> over couplings with marginals p and q
A_eq = np.vstack([np.kron(np.eye(n), np.ones(n)), np.kron(np.ones(n), np.eye(n))])
primal = scipy.optimize.linprog(C.ravel(), A_eq=A_eq, b_eq=np.r_[p, q], bounds=(0, None), method="highs").fun
# dual: max sum_i f_i (p_i - q_i) subject to |f_i - f_j| <= |x_i - x_j| (1-Lipschitz on the support)
rows, b = [], []
for i in range(n):
    for j in range(n):
        if i != j:
            r = np.zeros(n); r[i], r[j] = 1, -1
            rows.append(r); b.append(C[i, j])
dual = -scipy.optimize.linprog(-(p - q), A_ub=np.array(rows), b_ub=b, bounds=(None, None), method="highs").fun
cdf_area = np.sum(np.abs(np.cumsum(p - q)[:-1]) * np.diff(xs))
check("primal (transport plan) = dual (1-Lipschitz critic) = area between CDFs",
      *close([primal, dual], [cdf_area, cdf_area], 1e-8, f"primal {primal:.6f}, dual {dual:.6f}, CDF {cdf_area:.6f}; "))
# continuity: point masses at 0 and theta
w1 = {th: scipy.stats.wasserstein_distance([0.0], [th]) for th in (1.0, 0.1, 0.01)}      # computed, not assumed
gridw = np.round(np.arange(-200, 201) * 0.005, 6)
def pm(at):
    v = np.zeros(gridw.size); v[np.argmin(np.abs(gridw - at))] = 1.0; return v
js = {th: js_disc(pm(0.0), pm(th)) for th in (1.0, 0.1, 0.01)}
check("W1(delta_0, delta_theta) = |theta| -> 0 while JS stays at log 2",
      w1[0.01] < w1[0.1] < w1[1.0] and np.allclose(list(js.values()), math.log(2)),
      "W1: " + ", ".join(f"{v:.2f}" for v in w1.values()) + "; JS: log 2 throughout")
# W1 >= 0, symmetric, and zero at p = q (a metric)
check("W1 is symmetric, zero at p = q, and scipy agrees with the LP",
      abs(scipy.stats.wasserstein_distance(xs, xs, p, q) - primal) < 1e-8
      and abs(scipy.stats.wasserstein_distance(xs, xs, q, p) - primal) < 1e-8
      and scipy.stats.wasserstein_distance(xs, xs, p, p) == 0.0, "W1(p,q) = W1(q,p) = LP value; W1(p,p) = 0")''')

    # ------------------------------------------------------------------ EQ-S1-060
    c.eq(
        "EQ-S1-060", "Which divergence each family minimises (Table S1)",
        r"\text{AR, flow, VAE: forward KL (exact or as a bound)};\quad\text{GAN: JS}/f\text{-divergence}/W_1;"
        r"\quad\text{EBM: forward KL (biased gradient) or Fisher};\quad\text{score/diffusion: weighted Fisher divergence}",
        plain=(
            "The supplement's summary table ends each row with the divergence that the family's training objective "
            "minimises. Autoregressive models and flows maximise the exact likelihood, so they minimise forward KL "
            "(Chapter 2). A VAE maximises a lower bound on the likelihood, so it minimises forward KL only up to "
            "the gap of the bound. A GAN plays a game whose value, at the best possible discriminator, is a JS "
            "divergence, an $f$-divergence, or $W_1$ depending on the loss. Score models, and energy models trained by "
            "score matching, minimise the Fisher divergence, the average squared difference between the gradients of the two log-densities, which "
            "never needs the normaliser $Z_\\theta$. This chapter has defined every one of these except the Fisher "
            "divergence, which is introduced below and derived in Chapter 7."),
        refresher=(
            "The Fisher divergence is $F(p\\|q)=\\tfrac12\\mathbb E_p\\|\\nabla_{\\mathbf x}\\log p-\\nabla_{\\mathbf x}"
            "\\log q\\|^2$; it is zero iff the scores agree, hence iff $p=q$ when both are normalised densities on a "
            "connected support, and it is blind to the normaliser because $\\nabla_{\\mathbf x}\\log Z_\\theta=0$. The "
            "table's qualifiers matter: the VAE's forward KL is \"(bound)\", the EBM's is \"(biased)\" because the "
            "contrastive-divergence gradient truncates the MCMC chain, and diffusion's weighted Fisher divergence "
            "becomes a bound on the negative log-likelihood only with the likelihood weighting $\\lambda=g^2$ "
            "(Chapter 8). The check verifies the three facts the row rests on: forward KL and Fisher divergence "
            "vanish together for Gaussians and differ otherwise, the Fisher divergence ignores the normaliser, and "
            "the GAN value at the optimal discriminator equals $2\\,\\mathrm{JS}-\\log4$ (computed exactly on a "
            "discrete example; Chapter 6 gives the derivation)."),
        code='''# Fisher divergence between N(mu1,s1^2) and N(mu2,s2^2): 0.5 E_p[(d/dx log p - d/dx log q)^2] in closed form
def fisher_gauss(mu1, s1, mu2, s2):
    # scores: -(x-mu1)/s1^2 and -(x-mu2)/s2^2; the difference is affine in x, so E_p of its square is exact
    a = -1 / s1**2 + 1 / s2**2               # coefficient of x
    b = mu1 / s1**2 - mu2 / s2**2            # constant
    return 0.5 * (a**2 * (s1**2 + mu1**2) + 2 * a * b * mu1 + b**2)

def fisher_quad(mu1, s1, mu2, s2):
    g = lambda x: scipy.stats.norm.pdf(x, mu1, s1) * 0.5 * ((-(x - mu1) / s1**2) - (-(x - mu2) / s2**2))**2
    return scipy.integrate.quad(g, mu1 - 12 * s1, mu1 + 12 * s1)[0]

check("Fisher divergence closed form = quadrature", *close(fisher_gauss(0.3, 1.2, -0.4, 0.8), fisher_quad(0.3, 1.2, -0.4, 0.8), 1e-8))
kl_gauss = lambda m1, s1, m2, s2: math.log(s2 / s1) + (s1**2 + (m1 - m2)**2) / (2 * s2**2) - 0.5
check("Fisher = 0 and KL = 0 iff the Gaussians coincide; both positive otherwise",
      fisher_gauss(0.3, 1.2, 0.3, 1.2) == 0.0 and abs(kl_gauss(0.3, 1.2, 0.3, 1.2)) < 1e-15
      and fisher_gauss(0.3, 1.2, 0.31, 1.2) > 0 and kl_gauss(0.3, 1.2, 0.31, 1.2) > 0,
      f"at a small offset: Fisher {fisher_gauss(0.3, 1.2, 0.31, 1.2):.2e}, KL {kl_gauss(0.3, 1.2, 0.31, 1.2):.2e}")
# blind to the normaliser: the score of exp(-E)/Z does not depend on Z (symbolically)
x, Z = sp.symbols("x Z", positive=True)
Eexpr = x**4 / 4 - x**2
score_unnorm = sp.diff(-Eexpr, x)
score_norm = sp.diff(sp.log(sp.exp(-Eexpr) / Z), x)
check("the score of exp(-E)/Z does not depend on Z (symbolic)", sp.simplify(score_norm - score_unnorm) == 0,
      f"d/dx log p = {score_unnorm}")
# GAN row: value at the optimal discriminator is 2 JS - log 4 (exact on a discrete example)
pd_, pg_ = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
Dstar = pd_ / (pd_ + pg_)
V = np.sum(pd_ * np.log(Dstar)) + np.sum(pg_ * np.log(1 - Dstar))
check("GAN value at D* = 2 JS(p_data||p_g) - log 4", *close(V, 2 * js_disc(pd_, pg_) - math.log(4), 1e-12))
# AR/flow row: exact MLE minimises forward KL; the next chapter proves it, here the identity -E_p log q = H(p) + KL(p||q)
check("-E_p[log q] = H(p) + KL(p||q): maximising likelihood minimises forward KL",
      *close(H_cross(pd_, pg_), H(pd_) + kl_disc(pd_, pg_), 1e-12))''')

    return c.build()


if __name__ == "__main__":
    print(build().path)
