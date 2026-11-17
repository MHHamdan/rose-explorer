"""Chapter 2: maximum likelihood as divergence minimisation; latent variables, Bayes and EM
(main paper §3.2, Supplement S1.3--S1.4, S4.1, S10.1)."""
from nbkit import Chapter


def build():
    c = Chapter(
        2, "likelihood_latents_bayes", "Maximum likelihood, latent variables and Bayes",
        summary=(
            "Almost every objective in the survey is maximum likelihood or a bound on it, so this chapter establishes "
            "what maximum likelihood does: with enough data it minimises the forward KL divergence from the data to the "
            "model, which is why autoregressive models, flows and (through a bound) VAEs are mode-covering. The second "
            "half introduces latent variables, Bayes' rule for the posterior over them, and the "
            "expectation--maximisation algorithm, whose monotonicity is the ancestor of the variational bound of "
            "Chapter 3."),
        prerequisites="Chapter 1 (entropy, KL divergence)",
        paper_sections="main paper §3.2, §5.1 (autoregressive models); Supplement S1.3, S1.4, S4.1, S10.1",
    )

    c.section("From a loss on data to a divergence between distributions", (
        "Maximum likelihood looks like a statement about a finite sample. The two identities below turn it into a "
        "statement about distributions, and that is the form in which the paper uses it throughout: the route a "
        "model takes decides whether it can evaluate the likelihood at all, and if it can, maximising it is "
        "minimising forward KL."))

    # ------------------------------------------------------------------ EQ-S1-008
    c.eq(
        "EQ-S1-008", "Maximum likelihood estimation and its population limit",
        r"\hat\theta_{\mathrm{ML}}=\arg\max_\theta\frac1N\sum_{n=1}^N\log p_\theta(\mathbf x^{(n)})"
        r"\;\xrightarrow[N\to\infty]{}\;\arg\max_\theta\,\mathbb E_{\mathbf x\sim p_{\mathrm{data}}}\big[\log p_\theta(\mathbf x)\big]",
        plain=(
            "Maximum likelihood picks the parameters $\\theta$ under which the data we saw were most probable. We "
            "score $\\theta$ by the average log-probability it assigns to the $N$ training points and take the best. "
            "The arrow says what happens with more and more data: the average over the sample becomes the average "
            "over the true data distribution (the law of large numbers), so with infinite data we would be maximising "
            "the expected log-probability of a fresh data point."),
        refresher=(
            "Fisher (1922). The sample average $\\frac1N\\sum_n\\log p_\\theta(\\mathbf x^{(n)})$ converges to "
            "$\\mathbb E_{p_{\\mathrm{data}}}[\\log p_\\theta]$ pointwise in $\\theta$ by the strong law of large numbers; "
            "for the *maximisers* to converge one needs uniform convergence over a compact parameter set (or an "
            "equivalent regularity condition) and identifiability, the conditions the supplement cites from Bishop "
            "(2006) for consistency and asymptotic efficiency. The check uses a misspecified family, "
            "$\\mathrm{Binomial}(7,\\theta)$ fitted to an arbitrary distribution on $\\{0,\\dots,7\\}$, so that the "
            "population maximiser is not the truth but is still well defined: the binomial MLE from $2\\times10^5$ "
            "samples lands on it (ported from `mle_equals_forward_kl_minimiser`)."),
        code='''def kl_disc(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float); m = p > 0
    return math.inf if np.any(q[m] == 0) else float(np.sum(p[m] * np.log(p[m] / q[m])))

pdata = rng.dirichlet(np.ones(8))                  # an arbitrary distribution on {0,...,7}
k = np.arange(8)
thetas = np.linspace(0.01, 0.99, 981)
logp = np.array([scipy.stats.binom.logpmf(k, 7, t) for t in thetas])   # misspecified family Binomial(7, theta)
expected_ll = logp @ pdata                          # E_{p_data}[log p_theta(x)] for every theta
theta_pop = thetas[np.argmax(expected_ll)]
# finite-sample MLEs for growing N (binomial MLE = sample mean / 7)
for N in (100, 10_000, 200_000):
    sample = rng.choice(8, size=N, p=pdata)
    mle = sample.mean() / 7
    tol = 4 * sample.std() / 7 / math.sqrt(N) + 1e-3          # four standard errors of the mean/7, plus the 0.001 grid step
    check(f"MLE from N = {N:>7,} samples is within {tol:.4f} (4 s.e.) of the population maximiser",
          abs(mle - theta_pop) < tol, f"MLE {mle:.4f}, population argmax {theta_pop:.4f}")''')

    # ------------------------------------------------------------------ EQ-M-003 + EQ-S1-009 + EQ-S10-002
    c.eq(
        ["EQ-M-003", "EQ-S1-009", "EQ-S10-002"], "Maximum likelihood minimises the forward KL divergence",
        r"\arg\max_\theta\,\mathbb E_{p_{\mathrm{data}}}\big[\log p_\theta(\mathbf x)\big]"
        r"=\arg\min_\theta D_{\mathrm{KL}}(p_{\mathrm{data}}\,\|\,p_\theta),\qquad\text{because}\quad "
        r"D_{\mathrm{KL}}(p_{\mathrm{data}}\,\|\,p_\theta)=-H(p_{\mathrm{data}})-\mathbb E_{p_{\mathrm{data}}}[\log p_\theta]",
        plain=(
            "The KL divergence from the data to the model is the entropy of the data (a constant we cannot change) "
            "minus the expected log-likelihood (the thing maximum likelihood maximises). Making one larger makes the "
            "other smaller, so the best-fitting parameters are exactly the ones closest to the data in forward KL. "
            "This is the single most used fact in the paper: it is why likelihood-trained models cover every mode "
            "(Chapter 1) and why a language model's pretraining loss is a divergence between the corpus and the "
            "model over whole sequences (Supplement S10)."),
        refresher=(
            "Expand $D_{\\mathrm{KL}}(p_{\\mathrm{data}}\\|p_\\theta)=\\mathbb E_{p_{\\mathrm{data}}}[\\log p_{\\mathrm{data}}]"
            "-\\mathbb E_{p_{\\mathrm{data}}}[\\log p_\\theta]$; the first term is $-H(p_{\\mathrm{data}})$ and does not "
            "depend on $\\theta$. The identity needs $p_{\\mathrm{data}}\\ll p_\\theta$, otherwise both sides are "
            "$-\\infty$ and $+\\infty$ respectively, which is the mode-covering force. The main-paper statement "
            "(`eq:mle_kl`, status YES-WITH-WORDING-FIX) combines this with the previous entry and now says explicitly "
            "that the limit holds \"under standard identifiability and regularity conditions\". The check shows, on "
            "the misspecified binomial example, that the $\\theta$ maximising the expected log-likelihood is the "
            "$\\theta$ minimising the KL, and that their sum is the constant $H(p_{\\mathrm{data}})$ at every $\\theta$ "
            "to machine precision."),
        code='''kl = np.array([kl_disc(pdata, np.exp(l)) for l in logp])
check("argmax E[log p_theta] == argmin KL(p_data||p_theta) on the same grid",
      np.argmax(expected_ll) == np.argmin(kl), f"theta* = {thetas[np.argmin(kl)]:.3f}")
H_data = -np.sum(pdata * np.log(pdata))
check("-E[log p_theta] - KL(p_data||p_theta) = H(p_data) for every theta",
      *close(-expected_ll - kl, H_data, 1e-10, f"H(p_data) = {H_data:.4f}; "))
check("KL is minimal but not zero: the family is misspecified",
      kl.min() > 1e-3, f"min KL = {kl.min():.4f} nats")

fig, ax = plt.subplots(figsize=(6, 2.6))
ax.plot(thetas, -expected_ll, label=r"$-\\mathbb{E}_{p_{\\mathrm{data}}}[\\log p_\\theta]$ (NLL)")
ax.plot(thetas, kl, "--", label=r"$D_{\\mathrm{KL}}(p_{\\mathrm{data}}\\|p_\\theta)$")
ax.axvline(theta_pop, color="k", lw=0.8); ax.set_xlabel(r"$\\theta$"); ax.set_ylabel("nats")
ax.set_title(r"the two curves differ by the constant $H(p_{\\mathrm{data}})$ and share their minimiser", fontsize=9)
ax.legend(fontsize=8, frameon=False); plt.tight_layout(); plt.show()''')

    # ------------------------------------------------------------------ EQ-S4-004
    c.eq(
        "EQ-S4-004", "Cross-entropy training is maximum likelihood, and it is convex in the logits",
        r"-\mathbb E_{p_{\mathrm{data}}}[\log p_\theta(\mathbf x)]=H(p_{\mathrm{data}})+D_{\mathrm{KL}}(p_{\mathrm{data}}\,\|\,p_\theta);"
        r"\qquad \ell(\mathbf z)=\operatorname{logsumexp}(\mathbf z)-z_y\ \text{is convex in the logits }\mathbf z",
        plain=(
            "A classifier or a next-token predictor is trained with the cross-entropy loss: minus the log-probability "
            "the network gives to the right answer. The first identity says this loss is the data's entropy, which "
            "nobody can reduce, plus the KL divergence to the model, which training reduces: so cross-entropy "
            "training *is* maximum likelihood. The second says that, as a function of the raw network outputs "
            "(the logits $\\mathbf z$, turned into probabilities by softmax), the loss is bowl-shaped, with no local "
            "minima to get stuck in. That is one reason the supplement gives for the stable, predictable training of "
            "autoregressive models compared with adversarial ones."),
        refresher=(
            "The first identity is the previous entry rearranged. For the second, "
            "$\\ell(\\mathbf z)=\\log\\sum_j e^{z_j}-z_y$: the log-sum-exp is convex (its Hessian is "
            "$\\mathrm{diag}(\\mathbf s)-\\mathbf s\\mathbf s^\\top$ with $\\mathbf s=\\mathrm{softmax}(\\mathbf z)$, a "
            "positive semidefinite matrix because $\\mathbf s^\\top\\mathbf 1=1$) and $-z_y$ is linear. Convexity is in "
            "the *logits*, not in the network weights; the end-to-end loss of a deep network is non-convex. The check "
            "computes the Hessian with `torch.autograd`, verifies it equals the closed form and has no negative "
            "eigenvalue, tests the Jensen inequality $\\ell(\\tfrac12(\\mathbf z_1+\\mathbf z_2))\\le\\tfrac12"
            "(\\ell(\\mathbf z_1)+\\ell(\\mathbf z_2))$ on random pairs, and shows the gradient "
            "$\\mathrm{softmax}(\\mathbf z)-\\mathbf e_y$."),
        code='''K, y = 6, 2
def loss(z):               # cross-entropy of class y as a function of the logits
    return torch.logsumexp(z, 0) - z[y]

z0 = torch.randn(K, dtype=torch.float64)
Hs = torch.autograd.functional.hessian(loss, z0)
s = torch.softmax(z0, 0)
check("Hessian of logsumexp(z) - z_y equals diag(s) - s s^T", *close(Hs.numpy(), (torch.diag(s) - torch.outer(s, s)).numpy(), 1e-12))
check("Hessian is positive semidefinite (convex in the logits)", torch.linalg.eigvalsh(Hs).min().item() > -1e-12,
      f"min eigenvalue = {torch.linalg.eigvalsh(Hs).min().item():.2e}")
g = torch.autograd.functional.jacobian(loss, z0)
e_y = torch.zeros(K, dtype=torch.float64); e_y[y] = 1
check("gradient = softmax(z) - e_y", *close(g.numpy(), (s - e_y).numpy(), 1e-12))
viol = 0
for _ in range(500):
    z1, z2 = torch.randn(K, dtype=torch.float64), torch.randn(K, dtype=torch.float64)
    if loss((z1 + z2) / 2) > (loss(z1) + loss(z2)) / 2 + 1e-12:
        viol += 1
check("midpoint convexity holds on 500 random logit pairs", viol == 0, f"violations: {viol}")
# first identity on a discrete example: expected cross-entropy loss = H(p_data) + KL
p_true = rng.dirichlet(np.ones(K)); q_model = torch.softmax(torch.randn(K, dtype=torch.float64), 0).numpy()
check("E_p[-log q] = H(p) + KL(p||q)", *close(-np.sum(p_true * np.log(q_model)), -np.sum(p_true * np.log(p_true)) + kl_disc(p_true, q_model), 1e-12))''')

    # ------------------------------------------------------------------ EQ-M-034
    c.eq(
        "EQ-M-034", "Autoregressive training is exact maximum likelihood, hence forward KL",
        r"\text{exact MLE over sequences}\ \Rightarrow\ \arg\min_\theta D_{\mathrm{KL}}(p_{\mathrm{data}}\,\|\,p_\theta)"
        r"\ \Rightarrow\ \text{mode covering in the capacity limit}",
        plain=(
            "An autoregressive model (Chapter 4) can compute the exact probability of a whole sequence, so it can be "
            "trained by maximum likelihood with nothing approximated. By the previous entries, that means it "
            "minimises the forward KL divergence, and so, if the model is flexible enough, it must put probability on "
            "every kind of sequence the data contain: it covers every mode. The paper lists this as a property of the "
            "route, not of the architecture, and it is the reason the family produces diverse samples."),
        refresher=(
            "The claim has two parts: (i) summing the exact conditionals gives the exact joint log-likelihood, so the "
            "sample objective is the finite-sample version of EQ-S1-008 with no bound or surrogate; (ii) forward KL "
            "over sequences is the divergence minimised, so the mode-covering argument of EQ-M-004 applies, in the "
            "capacity limit (a model that cannot represent the data may still drop modes, and a finite sample contains "
            "only the modes it contains). The check fits a bigram model on a tiny alphabet by exact maximum "
            "likelihood (count normalisation) and shows that (a) its sequence likelihood is the product of the "
            "conditionals, (b) the fitted conditionals are the forward-KL minimisers, and (c) every sequence with "
            "positive data probability keeps positive model probability, whereas a one-regime model has a strictly "
            "larger forward KL (it would be infinite if it gave zero probability to any data sequence)."),
        code='''# data distribution over length-3 sequences on {a, b, c}, generated by a hidden two-regime process
A = 3; L = 3
seqs = np.array(np.meshgrid(*[range(A)] * L, indexing="ij")).reshape(L, -1).T
def regime_prob(seq, T):
    p = 1 / A
    for i in range(1, L): p *= T[seq[i - 1], seq[i]]
    return p
T1 = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])      # 'stay' regime
T2 = np.array([[0.1, 0.45, 0.45], [0.45, 0.1, 0.45], [0.45, 0.45, 0.1]]) # 'switch' regime
pdata_seq = np.array([0.5 * regime_prob(s, T1) + 0.5 * regime_prob(s, T2) for s in seqs])
check("data distribution over 27 sequences sums to one", abs(pdata_seq.sum() - 1) < 1e-12)

# exact MLE of a bigram model from a large sample: normalised transition counts
sample = seqs[rng.choice(len(seqs), size=300_000, p=pdata_seq)]
counts = np.zeros((A, A))
for i in range(1, L):
    np.add.at(counts, (sample[:, i - 1], sample[:, i]), 1)
T_hat = counts / counts.sum(1, keepdims=True)
# (a) the model's sequence probability is the product of its conditionals (exact likelihood)
p_model = np.array([regime_prob(s, T_hat) for s in seqs])
check("illustration: products of the fitted conditionals form a normalised distribution over sequences", abs(p_model.sum() - 1) < 1e-12)
# (b) the population forward-KL minimiser over bigram models is the true conditional p(x_i | x_{i-1}); compare
T_pop = np.zeros((A, A))
for s, pr in zip(seqs, pdata_seq):
    for i in range(1, L): T_pop[s[i - 1], s[i]] += pr
T_pop /= T_pop.sum(1, keepdims=True)
check("MLE transition matrix matches the population forward-KL minimiser", *close(T_hat, T_pop, 5e-3))
# (c) mode covering: every data-supported sequence keeps positive probability; a one-regime model does not minimise KL
kl_mle = kl_disc(pdata_seq, p_model)
kl_one = kl_disc(pdata_seq, np.array([regime_prob(s, T1) for s in seqs]))
check("forward KL of the MLE bigram is below that of the sharper one-regime model",
      kl_mle < kl_one and np.all(p_model[pdata_seq > 0] > 0), f"KL(MLE) = {kl_mle:.4f} < KL(one regime) = {kl_one:.4f}")''')

    c.section("Latent variables", (
        "Deep generative models explain data through hidden causes. The next three entries state the two "
        "operations this needs, marginalising the latent out and inverting the model with Bayes' rule, and the "
        "classical algorithm that alternates between them, whose guarantee is the seed of the variational bound in "
        "Chapter 3."))

    # ------------------------------------------------------------------ EQ-S1-011
    c.eq(
        "EQ-S1-011", "A latent-variable model: marginal likelihood and posterior",
        r"p_\theta(\mathbf x)=\int p_\theta(\mathbf x\mid\mathbf z)\,p(\mathbf z)\,d\mathbf z,\qquad "
        r"p_\theta(\mathbf z\mid\mathbf x)=\frac{p_\theta(\mathbf x\mid\mathbf z)\,p(\mathbf z)}{p_\theta(\mathbf x)}",
        plain=(
            "A latent-variable model tells a two-step story: first draw a hidden cause $\\mathbf z$ from a simple "
            "prior $p(\\mathbf z)$, then draw the observation $\\mathbf x$ from a simple conditional "
            "$p_\\theta(\\mathbf x\\mid\\mathbf z)$. The probability of seeing $\\mathbf x$ is the sum (integral) over "
            "every cause that could have produced it, weighted by how likely each cause is. The posterior runs the "
            "story backwards: given that we saw $\\mathbf x$, how plausible is each cause? Both simple pieces can "
            "compose into a very flexible distribution over $\\mathbf x$, which is the point; the price is that both "
            "integrals become intractable once the conditional is a neural network."),
        refresher=(
            "The marginal is the normaliser of the posterior, which is why evaluating the posterior exactly is as "
            "hard as evaluating the likelihood. Tractable cases: discrete $\\mathbf z$ with few states (mixtures), "
            "linear-Gaussian models (factor analysis, PPCA), and conjugate pairs. The verification uses the "
            "linear-Gaussian model $\\mathbf z\\sim\\mathcal N(0,1)$, $\\mathbf x\\mid\\mathbf z\\sim\\mathcal N(az,s^2)$, "
            "whose marginal is $\\mathcal N(0,a^2+s^2)$ and whose posterior is "
            "$\\mathcal N\\big(ax/(a^2+s^2),\\,s^2/(a^2+s^2)\\big)$, and checks both against numerical integration; "
            "it also shows that the marginal of a Gaussian mixture is not Gaussian (the composition is more "
            "expressive than its parts)."),
        code='''a, s2, x_obs = 1.3, 0.4, 0.9
prior = scipy.stats.norm(0, 1)
lik = lambda z: scipy.stats.norm.pdf(x_obs, a * z, math.sqrt(s2))
marg_num = scipy.integrate.quad(lambda z: lik(z) * prior.pdf(z), -12, 12)[0]
marg_closed = scipy.stats.norm.pdf(x_obs, 0, math.sqrt(a**2 + s2))
check("marginal p(x) = int p(x|z) p(z) dz equals the closed form N(0, a^2 + s^2)", *close(marg_num, marg_closed, 1e-10))
post_mean, post_var = a * x_obs / (a**2 + s2), s2 / (a**2 + s2)
post_num = lambda z: lik(z) * prior.pdf(z) / marg_num
m1 = scipy.integrate.quad(lambda z: z * post_num(z), -12, 12)[0]
m2 = scipy.integrate.quad(lambda z: (z - m1)**2 * post_num(z), -12, 12)[0]
check("posterior p(z|x) = p(x|z) p(z) / p(x) has the closed-form mean and variance", *close([m1, m2], [post_mean, post_var], 1e-8))
check("posterior integrates to one", *close(scipy.integrate.quad(post_num, -12, 12)[0], 1.0, 1e-10))
# expressiveness: a two-state latent with Gaussian conditionals gives a bimodal, non-Gaussian marginal
zs = np.array([-2.5, 2.5]); w = np.array([0.5, 0.5])
mix = lambda x: np.sum(w * scipy.stats.norm.pdf(x[:, None], zs, 0.7), 1)
xg = np.linspace(-6, 6, 1201)
mode_count = int(np.sum((mix(xg)[1:-1] > mix(xg)[:-2]) & (mix(xg)[1:-1] > mix(xg)[2:])))
check("a mixture marginal is bimodal although prior and conditional are simple", mode_count == 2, f"{mode_count} local maxima")''')

    # ------------------------------------------------------------------ EQ-S1-012
    c.eq(
        "EQ-S1-012", "Bayes' rule in log form",
        r"\log p(\mathbf z\mid\mathbf x)=\log p(\mathbf x\mid\mathbf z)+\log p(\mathbf z)-\log p(\mathbf x)",
        plain=(
            "Taking logarithms turns the products and quotients of Bayes' rule into sums and differences, which is "
            "how it is always implemented: log-posterior equals log-likelihood plus log-prior minus log-evidence. "
            "The last term, $\\log p(\\mathbf x)$, is the same for every $\\mathbf z$; it only makes the posterior sum "
            "to one. Rearranged, the identity says $\\log p(\\mathbf x)=\\log p(\\mathbf x\\mid\\mathbf z)+\\log p(\\mathbf z)"
            "-\\log p(\\mathbf z\\mid\\mathbf x)$ for *any* $\\mathbf z$, a fact Chapter 3 turns into the evidence "
            "lower bound."),
        refresher=(
            "Valid wherever $p(\\mathbf x)>0$. The rearranged form, averaged under an arbitrary $q(\\mathbf z)$, gives "
            "$\\log p(\\mathbf x)=\\mathbb E_q[\\log p(\\mathbf x,\\mathbf z)-\\log q(\\mathbf z)]+D_{\\mathrm{KL}}(q\\|p(\\mathbf z\\mid\\mathbf x))$, "
            "the ELBO identity (EQ-S1-015). The check verifies the log identity on the linear-Gaussian model at "
            "several $\\mathbf z$ (the left-hand side from the closed-form posterior, the right-hand side from its "
            "three parts) and confirms that the right-hand side is constant in $\\mathbf z$ once the posterior term "
            "is moved across."),
        code='''post = scipy.stats.norm(post_mean, math.sqrt(post_var))
zs_test = np.array([-2.0, -0.3, 0.0, 0.7, 2.4])
lhs = post.logpdf(zs_test)
rhs = scipy.stats.norm.logpdf(x_obs, a * zs_test, math.sqrt(s2)) + prior.logpdf(zs_test) - math.log(marg_closed)
check("log p(z|x) = log p(x|z) + log p(z) - log p(x) at five values of z", *close(lhs, rhs, 1e-10))
evidence_from_any_z = scipy.stats.norm.logpdf(x_obs, a * zs_test, math.sqrt(s2)) + prior.logpdf(zs_test) - post.logpdf(zs_test)
check("log p(x) = log p(x|z) + log p(z) - log p(z|x) is the same for every z",
      *close(evidence_from_any_z, math.log(marg_closed), 1e-10, f"log p(x) = {math.log(marg_closed):.6f}; "))''')

    # ------------------------------------------------------------------ EQ-S1-013
    c.eq(
        "EQ-S1-013", "Expectation--maximisation never decreases the likelihood",
        r"\sum_{n}\log p_{\theta^{(k+1)}}(\mathbf x^{(n)})\;\ge\;\sum_{n}\log p_{\theta^{(k)}}(\mathbf x^{(n)}),"
        r"\qquad\theta^{(k+1)}=\arg\max_\theta\sum_n\mathbb E_{p_{\theta^{(k)}}(\mathbf z\mid\mathbf x^{(n)})}\big[\log p_\theta(\mathbf x^{(n)},\mathbf z)\big]",
        plain=(
            "When the latent $\\mathbf z$ is simple enough that the posterior can be computed, the "
            "expectation--maximisation (EM) algorithm fits a latent-variable model by repeating two steps: "
            "E, work out how responsible each hidden cause is for each data point under the current parameters; "
            "M, re-fit the parameters as if those responsibilities were observed. The guarantee is that the "
            "log-likelihood of the data goes up, or at worst stays the same, at every round. The plot below shows it "
            "for a mixture of two Gaussians started from a poor guess."),
        refresher=(
            "EM (Dempster, Laird and Rubin, 1977) is coordinate ascent on the bound "
            "$\\mathcal L(q,\\theta)=\\mathbb E_q[\\log p_\\theta(\\mathbf x,\\mathbf z)]+H(q)\\le\\log p_\\theta(\\mathbf x)$: "
            "the E-step sets $q=p_{\\theta^{(k)}}(\\mathbf z\\mid\\mathbf x)$, which makes the bound tight, and the "
            "M-step maximises it over $\\theta$; since the bound is tight before the M-step and a lower bound after, "
            "the likelihood cannot fall. Convergence is to a stationary point, not necessarily the global maximum. "
            "Variational inference (Chapter 3) is EM with the E-step restricted to a tractable family, which is "
            "exactly where the bound stops being tight. The check runs 60 EM iterations for a two-component Gaussian "
            "mixture and asserts that every increment of the log-likelihood is $\\ge-10^{-9}$ "
            "(`em_monotone_gmm`)."),
        code='''xd = np.r_[rng.normal(-2, 1, 300), rng.normal(3, 0.5, 200)]
pi, mu, sd = np.array([0.5, 0.5]), np.array([0.0, 0.5]), np.array([2.0, 2.0])   # a poor start
lls = []
for _ in range(60):
    comp = pi * scipy.stats.norm.pdf(xd[:, None], mu, sd)        # joint p(x, z) for each component
    lls.append(np.log(comp.sum(1)).sum())                        # log-likelihood before the update
    r = comp / comp.sum(1, keepdims=True)                        # E-step: responsibilities p(z | x)
    nk = r.sum(0)                                                # M-step: weighted MLE
    pi, mu = nk / len(xd), (r * xd[:, None]).sum(0) / nk
    sd = np.sqrt((r * (xd[:, None] - mu)**2).sum(0) / nk)
incr = np.diff(lls)
check("log-likelihood never decreases over 60 EM steps", bool(np.all(incr >= -1e-9)), f"min increment = {incr.min():.2e}")
check("EM recovers the two components (means near -2 and 3)", np.allclose(np.sort(mu), [-2, 3], atol=0.3), f"means {np.sort(mu).round(3)}")
# the bound is tight after the E-step: E_q[log p(x,z)] + H(q) equals the log-likelihood, with q = p(z|x)
comp = pi * scipy.stats.norm.pdf(xd[:, None], mu, sd); r = comp / comp.sum(1, keepdims=True)
bound = np.sum(r * np.log(comp)) - np.sum(r * np.log(r))
check("after the E-step the bound equals the log-likelihood (tight)", *close(bound, np.log(comp.sum(1)).sum(), 1e-8))

fig, ax = plt.subplots(figsize=(6, 2.4))
ax.plot(lls, marker=".", ms=4); ax.set_xlabel("EM iteration"); ax.set_ylabel("log-likelihood")
ax.set_title("monotone increase from a poor initialisation", fontsize=9); plt.tight_layout(); plt.show()''')

    return c.build()


if __name__ == "__main__":
    print(build().path)
