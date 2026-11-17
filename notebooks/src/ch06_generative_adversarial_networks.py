"""Chapter 6: generative adversarial networks (main paper §3.4, §5.3; Supplement S1.7, S4.3)."""
from nbkit import Chapter


def build():
    c = Chapter(
        6, "generative_adversarial_networks", "Generative adversarial networks: implicit models and their divergences",
        summary=(
            "A GAN never writes down a density: it defines a sampler and trains it against a critic. This chapter "
            "verifies what the critic estimates (a Jensen--Shannon divergence at the optimum of the original game, a "
            "Wasserstein-1 distance for the WGAN critic), why the original objective fails when the data lie on a "
            "manifold (the divergence saturates at $\\log 2$), what the non-saturating loss changes, and what the "
            "gradient penalties that made training stable actually constrain. These facts are what the paper means "
            "when it says the adversarial route gives up density access and inherits a different divergence."),
        prerequisites="Chapter 1 (KL, Jensen--Shannon, Wasserstein-1); Chapter 2 (maximum likelihood)",
        paper_sections="main paper §3.4 and §5.3; Supplement S1.7 and S4.3",
    )

    c.section("The game and what its optimum measures", (
        "The original GAN objective is a two-player game. The entries below show that, if the discriminator is "
        "allowed to be optimal, the game reduces to a divergence between the data and the generator, and that this "
        "divergence is the wrong one when the two distributions do not overlap. The checks use small discrete "
        "distributions so that every expectation is an exact sum."))

    c.eq(
        ["EQ-M-014", "EQ-S1-023", "EQ-S4-014"], "The minimax game",
        r"\min_\theta\max_\psi\;V(D_\psi,G_\theta)=\mathbb E_{\mathbf x\sim p_{\mathrm{data}}}\big[\log D_\psi(\mathbf x)\big]"
        r"+\mathbb E_{\mathbf z\sim p(\mathbf z)}\big[\log\big(1-D_\psi(G_\theta(\mathbf z))\big)\big]",
        plain=(
            "Two networks play against each other. The generator $G_\\theta$ turns random noise $\\mathbf z$ into a "
            "sample. The discriminator $D_\\psi$ looks at a sample and outputs a number between 0 and 1, its belief "
            "that the sample is real. The value $V$ rewards the discriminator for saying 'real' ($D$ near 1) on real "
            "data and 'fake' ($D$ near 0) on generated data; the discriminator maximises $V$, the generator minimises "
            "it. Nothing here needs the generator's probability of a sample, only the ability to draw samples, which is "
            "why the paper files GANs under the route that *avoids* the likelihood."),
        refresher=(
            "Goodfellow et al. (2014). Write $p_g$ for the distribution of $G_\\theta(\\mathbf z)$. For fixed $G$ the "
            "inner problem separates pointwise: at each $\\mathbf x$ one maximises "
            "$p_{\\mathrm{data}}(\\mathbf x)\\log D+p_g(\\mathbf x)\\log(1-D)$ over $D\\in(0,1)$, a strictly concave "
            "function of $D$ (next entry). The game is stated over all measurable $D$; with a parametric $D_\\psi$ "
            "the inner maximum is a lower bound on the divergence that the next entry identifies, so a weak "
            "discriminator under-estimates it. The check computes $V$ on two discrete distributions and confirms "
            "that for any fixed $D$ the value is a sum of two expectations, and that $V$ is concave in $D$ "
            "coordinate-wise (second derivative $-p_{\\mathrm{data}}/D^2-p_g/(1-D)^2<0$)."),
        code='''def kl_disc(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float); m = p > 0
    return math.inf if np.any(q[m] == 0) else float(np.sum(p[m] * np.log(p[m] / q[m])))
def js_disc(p, q):
    m = 0.5 * (np.asarray(p) + np.asarray(q)); return 0.5 * kl_disc(p, m) + 0.5 * kl_disc(q, m)

pd_ = rng.dirichlet(np.ones(10)); pg = rng.dirichlet(np.ones(10))        # data and generator on 10 points
V = lambda D: float(np.sum(pd_ * np.log(D)) + np.sum(pg * np.log(1 - D)))
D = rng.uniform(0.05, 0.95, size=10)
check("V(D,G) is the sum of the two expectations", *close(V(D), pd_ @ np.log(D) + pg @ np.log(1 - D), 1e-14))
d2 = -pd_ / D**2 - pg / (1 - D)**2
check("V is strictly concave in each D(x): second derivative < 0 everywhere", np.all(d2 < 0), f"max d2V/dD2 = {d2.max():.3f}")''')

    c.eq(
        ["EQ-M-015", "EQ-S1-024", "EQ-S4-015"], "The optimal discriminator and the Jensen--Shannon identity",
        r"D^*(\mathbf x)=\frac{p_{\mathrm{data}}(\mathbf x)}{p_{\mathrm{data}}(\mathbf x)+p_g(\mathbf x)},\qquad "
        r"V(D^*,G)=2\,\mathrm{JS}(p_{\mathrm{data}}\,\|\,p_g)-\log 4,\qquad \min_G V(D^*,G)=-\log4\iff p_g=p_{\mathrm{data}}",
        plain=(
            "If the discriminator could be perfect, its best answer at a point is the share of real density there: "
            "$D^*=\\tfrac12$ where real and generated are equally likely, near 1 where only real data occur. Plugging "
            "that answer back into the game turns the whole thing into a single number, twice the Jensen--Shannon "
            "divergence between the data and the generator minus a constant. So the generator, playing against a "
            "perfect critic, is trying to make the Jensen--Shannon divergence zero, which happens only when it "
            "matches the data exactly. Then $D^*$ is $\\tfrac12$ everywhere: the critic can only guess."),
        refresher=(
            "Setting the pointwise derivative $p_{\\mathrm{data}}/D-p_g/(1-D)$ to zero gives $D^*$. Substituting, "
            "$V(D^*,G)=\\mathbb E_{p_{\\mathrm{data}}}\\log\\frac{p_{\\mathrm{data}}}{p_{\\mathrm{data}}+p_g}"
            "+\\mathbb E_{p_g}\\log\\frac{p_g}{p_{\\mathrm{data}}+p_g}$; writing $p_{\\mathrm{data}}+p_g=2m$ with "
            "$m$ the mixture yields $D_{\\mathrm{KL}}(p_{\\mathrm{data}}\\|m)+D_{\\mathrm{KL}}(p_g\\|m)-2\\log2"
            "=2\\,\\mathrm{JS}-\\log4$. The identity holds where both densities exist with respect to a common "
            "measure; for distributions on disjoint manifolds it still holds in the discrete-sum form used here, and "
            "gives $\\log4-\\log4=0$ at $\\mathrm{JS}=\\log2$, the saturation discussed two entries below. Ported "
            "from `gan_optimal_discriminator_value_minus_log4_plus_2js`: $D^*$ is a maximiser (200 random "
            "perturbations lower $V$), the identity holds to machine precision, and $V=-\\log4$ at $p_g=p_{\\mathrm{data}}$."),
        code='''Ds = pd_ / (pd_ + pg)
check("V(D*,G) = 2 JS(p_data||p_g) - log 4", *close(V(Ds), 2 * js_disc(pd_, pg) - math.log(4), 1e-12))
worse = all(V(np.clip(Ds + 1e-3 * rng.normal(size=10), 1e-9, 1 - 1e-9)) <= V(Ds) + 1e-15 for _ in range(200))
check("D* maximises V: 200 random perturbations all lower it", worse)
check("at p_g = p_data: D* = 1/2 and V = -log 4", *close(V(np.full(10, 0.5)), -math.log(4), 1e-12))
check("-log 4 is the minimum over generators: V(D*,G) >= -log 4 since JS >= 0", V(Ds) >= -math.log(4) - 1e-12,
      f"V(D*,G) = {V(Ds):.4f} > {-math.log(4):.4f}")''')

    c.eq(
        ["EQ-M-016", "EQ-S1-025", "EQ-M-036"], "The non-saturating generator loss and its gradient",
        r"\mathcal L_G^{\mathrm{NS}}=-\mathbb E_{\mathbf z}\big[\log D(G(\mathbf z))\big];\qquad "
        r"\nabla_\theta\mathcal L_G^{\mathrm{NS}}\Big|_{D=D^*}=\nabla_\theta\Big[D_{\mathrm{KL}}(p_g\,\|\,p_{\mathrm{data}})-2\,\mathrm{JS}(p_{\mathrm{data}}\,\|\,p_g)\Big]",
        plain=(
            "In the original game the generator minimises $\\log(1-D(G(\\mathbf z)))$. Early in training the "
            "discriminator is confident that generated samples are fake, $D\\approx0$, and this loss is flat there: "
            "the generator gets almost no gradient and cannot improve. The non-saturating trick is to *maximise* "
            "$\\log D(G(\\mathbf z))$ instead, which is steep exactly where the generator is doing badly. Both losses "
            "want the same thing in the end (fooling the critic), but they push differently along the way. The "
            "second equation says what the new loss is secretly optimising when the critic is optimal: a *reverse* "
            "KL (which likes sharp samples and tolerates dropped modes, Chapter 1) minus a Jensen--Shannon term "
            "that pulls the other way."),
        refresher=(
            "Goodfellow et al. (2014); the gradient identity is Theorem 2.5 of Arjovsky and Bottou (2017), who use "
            "it to explain mode dropping and unstable gradients: the reverse-KL term is mode-seeking and the "
            "$-2\\mathrm{JS}$ term has the wrong sign for a divergence minimisation. The identity holds with $D^*$ "
            "*held fixed* at the current $p_g$ (the discriminator is not re-optimised inside the gradient), which is "
            "the alternating-update setting. Both losses have the same stationary point at $p_g=p_{\\mathrm{data}}$ "
            "where $D^*\\equiv\\tfrac12$. The paper's main text (EQ-M-036) cites this identity in §5.3 as the reason "
            "GANs drop modes. The check (ported from `gan_nonsaturating_gradient_identity`) parameterises "
            "$p_g=\\mathrm{softmax}(\\theta)$ on six points, compares finite-difference gradients of the two sides, "
            "verifies the stationary point, and shows the saturating loss's gradient vanishing as $D\\to0$."),
        code='''pd6 = rng.dirichlet(np.ones(6)); th0 = rng.normal(size=6)
sm = lambda t: np.exp(t - t.max()) / np.exp(t - t.max()).sum()
Dstar = pd6 / (pd6 + sm(th0))                                   # optimal D at theta0, held fixed
ns = lambda t: -np.sum(sm(t) * np.log(Dstar))                   # non-saturating loss as a function of theta
crit = lambda t: kl_disc(sm(t), pd6) - 2 * js_disc(sm(t), pd6)  # KL(p_g||p_data) - 2 JS
e = 1e-6; I6 = np.eye(6)
g1 = np.array([(ns(th0 + e * I6[i]) - ns(th0 - e * I6[i])) / (2 * e) for i in range(6)])
g2 = np.array([(crit(th0 + e * I6[i]) - crit(th0 - e * I6[i])) / (2 * e) for i in range(6)])
check("grad of NS loss = grad[KL(p_g||p_data) - 2 JS] at fixed D*", *close(g1, g2, 1e-7))
th_d = np.log(pd6); Dh = np.full(6, 0.5)
g3 = np.array([(-np.sum(sm(th_d + e * I6[i]) * np.log(Dh)) + np.sum(sm(th_d - e * I6[i]) * np.log(Dh))) / (2 * e) for i in range(6)])
check("stationary at p_g = p_data (D* = 1/2)", np.max(np.abs(g3)) < 1e-8, f"|grad| = {np.max(np.abs(g3)):.1e}")
# saturation, by autograd: gradients w.r.t. the discriminator logit l when D = sigmoid(l) is small (a confident 'fake')
l_ = torch.tensor([-8.0, -4.0, 0.0], dtype=torch.float64, requires_grad=True)
g_sat, = torch.autograd.grad(torch.log(1 - torch.sigmoid(l_)).sum(), l_)     # original (saturating) generator loss
g_ns, = torch.autograd.grad((-torch.log(torch.sigmoid(l_))).sum(), l_)      # non-saturating loss
check("saturating loss gradient vanishes as D -> 0; the non-saturating one stays O(1)",
      abs(g_sat[0].item()) < 1e-3 and abs(g_ns[0].item()) > 0.99,
      f"at logit -8: |grad| saturating {abs(g_sat[0].item()):.1e}, non-saturating {abs(g_ns[0].item()):.3f}")''')

    c.eq(
        ["EQ-S1-026", "EQ-S4-016"], "Disjoint supports: a perfect discriminator and a constant divergence",
        r"\mathrm{supp}\,p_{\mathrm{data}}\cap\mathrm{supp}\,p_g=\varnothing\;\Rightarrow\;D^*\in\{0,1\}\ \text{a.e.},\quad "
        r"\mathrm{JS}(p_{\mathrm{data}}\,\|\,p_g)=\log 2,\quad\nabla_\theta\,\mathrm{JS}=0",
        plain=(
            "Images occupy a thin sheet inside the space of all pixel arrays, and a generator's outputs occupy "
            "another thin sheet. Two thin sheets almost never overlap. When they do not, a discriminator can be "
            "perfect: it says 1 on one sheet and 0 on the other. The Jensen--Shannon divergence is then stuck at its "
            "maximum value $\\log2$ no matter how close the sheets are, so it gives the generator no direction to "
            "move in. This is the fragility that motivated the Wasserstein GAN two entries below."),
        refresher=(
            "Arjovsky and Bottou (2017), Theorems 2.1--2.3: if the supports are disjoint (or more generally "
            "contained in two closed manifolds that do not perfectly align), there is an optimal discriminator with "
            "accuracy 1 and zero gradient almost everywhere, and $\\mathrm{JS}=\\log2$, $D_{\\mathrm{KL}}=\\infty$ in "
            "both directions. All $f$-divergences share this: they depend only on the density ratio, which is 0 or "
            "$\\infty$ off the overlap. Integral probability metrics such as $W_1$ depend on distances between "
            "points and remain informative (Chapter 1, EQ-M-006). The check moves two disjoint discrete "
            "distributions apart on a line: the JS divergence stays at $\\log2$ at every separation while $W_1$ "
            "grows linearly with it, and the figure shows both."),
        code='''grid = np.arange(40)
pA = np.zeros(40); pA[[5, 6, 7]] = [0.25, 0.5, 0.25]        # p_data on three adjacent points
seps = np.arange(4, 31, 2)
js_vals, w1_vals = [], []
for s in seps:
    pB = np.roll(pA, s)                                     # p_g: the same shape shifted by s (disjoint support)
    js_vals.append(js_disc(pA, pB))
    w1_vals.append(scipy.stats.wasserstein_distance(grid, grid, pA, pB))
js_vals, w1_vals = np.array(js_vals), np.array(w1_vals)
check("JS = log 2 at every separation of two disjoint supports", *close(js_vals, math.log(2), 1e-12))
check("W1 grows linearly with the separation (W1 = s for a pure shift)", *close(w1_vals, seps, 1e-9))
Ds_dis = pA / (pA + np.roll(pA, 10) + 1e-300)
check("optimal discriminator is 0/1 on the supports", np.all(np.isin(np.round(Ds_dis[pA + np.roll(pA, 10) > 0], 12), [0.0, 1.0])))

fig, ax = plt.subplots(figsize=(6, 2.6))
ax.plot(seps, js_vals, "o-", label="JS divergence (saturated at log 2)")
ax.plot(seps, w1_vals / w1_vals.max() * math.log(2), "s--", label="$W_1$ (rescaled): still increasing")
ax.set_xlabel("separation of the two supports"); ax.set_ylabel("divergence"); ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()''')

    c.eq(
        "EQ-S1-027", "The $f$-GAN variational bound",
        r"D_f(p_{\mathrm{data}}\,\|\,p_g)\;\ge\;\sup_{T}\Big(\mathbb E_{p_{\mathrm{data}}}[T(\mathbf x)]-\mathbb E_{p_g}[f^*(T(\mathbf x))]\Big),"
        r"\qquad f^*(t)=\sup_u\,(ut-f(u))",
        plain=(
            "Any divergence of the form $\\mathbb E_q[f(p/q)]$ for a convex $f$ (the KL divergence is the case "
            "$f(u)=u\\log u$) can be written as the best value of a game: pick a scoring function $T$, reward it on "
            "real data and charge it a convex penalty $f^*$ on generated data. The discriminator of a GAN is such a "
            "$T$, so a GAN is estimating a divergence, not just classifying. Choosing a different $f$ gives a "
            "different GAN; the original one corresponds to a particular $f$ and the Wasserstein GAN to a metric that "
            "is not an $f$-divergence at all."),
        refresher=(
            "Nowozin et al. (2016), after Nguyen et al. (2010): Fenchel conjugacy $f(u)=\\sup_t(tu-f^*(t))$ applied "
            "inside $\\mathbb E_q[f(p/q)]$ and the supremum pulled outside the expectation gives the bound; it is "
            "tight when $T^*(\\mathbf x)=f'(p(\\mathbf x)/q(\\mathbf x))$ is in the class of $T$. For the KL "
            "divergence $f^*(t)=e^{t-1}$ and $T^*=1+\\log(p/q)$. A parametric $T$ (a network) only reaches a lower "
            "bound, which is why a weak discriminator under-estimates the divergence and why the paper treats the "
            "discriminator as a critic estimating a divergence rather than as a classifier. The check (ported from "
            "`wgan_critic_and_gradient_penalty`) evaluates the KL bound at $T^*$ and at a perturbed $T$."),
        code='''p5, q5 = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
T = 1 + np.log(p5 / q5)                                           # optimal T for the KL generator f(u) = u log u
bound = lambda T: float(np.sum(p5 * T) - np.sum(q5 * np.exp(T - 1)))
check("f-GAN bound is tight at T* = 1 + log(p/q) for KL", *close(bound(T), kl_disc(p5, q5), 1e-12))
check("any other T gives a lower value (it is a lower bound)",
      all(bound(T + 0.3 * rng.normal(size=5)) <= kl_disc(p5, q5) + 1e-12 for _ in range(100)))''')

    c.section("Replacing the divergence: Wasserstein critics and their constraints", (
        "The Wasserstein GAN keeps the two-player structure but changes what the critic estimates. The entries below "
        "verify the dual form it uses, the gradient penalty that enforces the critic's Lipschitz constraint, and the "
        "other regularisers the paper mentions."))

    c.eq(
        ["EQ-M-017", "EQ-S1-028", "EQ-S4-017", "EQ-S4-018"], "The WGAN critic, the gradient penalty and the hinge loss",
        r"W_1(p_{\mathrm{data}},p_g)=\sup_{\|f\|_L\le1}\;\mathbb E_{p_{\mathrm{data}}}[f(\mathbf x)]-\mathbb E_{\mathbf z}[f(G(\mathbf z))];\qquad "
        r"\mathcal L_{\mathrm{GP}}=\lambda\,\mathbb E_{\hat{\mathbf x}}\big[(\|\nabla_{\hat{\mathbf x}}f(\hat{\mathbf x})\|_2-1)^2\big];\qquad "
        r"\mathcal L_{\mathrm{hinge}}=\mathbb E_{\mathbf x}[\max(0,1-D(\mathbf x))]+\mathbb E_{\mathbf z}[\max(0,1+D(G(\mathbf z)))]",
        plain=(
            "The Wasserstein-1 distance is the cost of moving one pile of probability mass onto another. Its dual "
            "form says the same number is the largest gap between the average score on real data and on generated "
            "data over all *gentle* scoring functions $f$ (ones that cannot change by more than 1 when the input moves "
            "by 1). That gap is what a WGAN critic estimates, and unlike the Jensen--Shannon divergence it keeps "
            "growing as the generator drifts away, so the gradient never dies. The gradient penalty keeps the critic "
            "gentle by charging it whenever the slope of $f$, measured at points between real and generated samples, "
            "is not 1. The hinge loss is another critic objective that stops rewarding the critic once it is confident "
            "by a margin of 1."),
        refresher=(
            "Arjovsky et al. (2017): Kantorovich--Rubinstein duality (Chapter 1, EQ-M-006) for the Euclidean cost; "
            "the sup is over 1-Lipschitz $f$ and the critic must be constrained to that class, otherwise the gap is "
            "unbounded. Weight clipping (the original device) biases the critic towards simple functions; the "
            "gradient penalty of Gulrajani et al. (2017) on random interpolates "
            "$\\hat{\\mathbf x}=\\alpha\\mathbf x+(1-\\alpha)G(\\mathbf z)$ penalises deviation of the gradient norm "
            "from 1 (the optimal critic has unit gradient norm almost everywhere on the transport paths); spectral "
            "normalisation (Miyato et al., 2018) bounds the Lipschitz constant layer by layer instead. The hinge "
            "loss (Lim and Ye, 2017) is used with spectral normalisation in SAGAN and BigGAN. The checks: (i) the dual "
            "on a 1-D discrete support, solved as a linear programme over the critic values with the Lipschitz "
            "constraints $|f_{i+1}-f_i|\\le|x_{i+1}-x_i|$, equals the closed-form 1-D $W_1$ from SciPy; (ii) for a "
            "linear critic $f(\\mathbf x)=\\mathbf w^\\top\\mathbf x$ the penalty is exactly $(\\|\\mathbf w\\|-1)^2$, "
            "independent of the interpolates (the repository's `gan.model.gradient_penalty` is checked against this "
            "in `verify_equations.py`); (iii) the hinge loss has zero gradient outside the margin."),
        code='''# (i) Kantorovich-Rubinstein dual as a linear programme on a 1-D grid
xs = np.sort(rng.uniform(0, 10, size=12)); pA = rng.dirichlet(np.ones(12)); pB = rng.dirichlet(np.ones(12))
n = len(xs)
A_ub, b_ub = [], []
for i in range(n - 1):                                 # |f_{i+1} - f_i| <= |x_{i+1} - x_i|  (1-Lipschitz on the grid)
    row = np.zeros(n); row[i + 1], row[i] = 1, -1; A_ub += [row, -row]; b_ub += [xs[i + 1] - xs[i]] * 2
res = scipy.optimize.linprog(-(pA - pB), A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(None, None)] * n, method="highs")
w1_dual = -res.fun
w1_primal = scipy.stats.wasserstein_distance(xs, xs, pA, pB)
check("sup over 1-Lipschitz f of E_p f - E_q f  =  W1 (1-D closed form)", *close(w1_dual, w1_primal, 1e-9))

# (ii) gradient penalty for a linear critic equals (||w|| - 1)^2 whatever the interpolates are
w = torch.randn(5, dtype=torch.float64, requires_grad=False)
real, fake = torch.randn(64, 5, dtype=torch.float64), torch.randn(64, 5, dtype=torch.float64)
alpha = torch.rand(64, 1, dtype=torch.float64)
xhat = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
f = xhat @ w
grad, = torch.autograd.grad(f.sum(), xhat)
gp = ((grad.norm(dim=1) - 1) ** 2).mean().item()
check("gradient penalty of a linear critic = (||w|| - 1)^2", *close(gp, (w.norm().item() - 1) ** 2, 1e-12))
# (iii) hinge loss: zero gradient once D(real) > 1 or D(fake) < -1
d = torch.tensor([2.0, 0.5, -0.5, -2.0], requires_grad=True)
hinge_real = torch.clamp(1 - d, min=0).sum(); hinge_real.backward()
check("hinge loss on real: gradient 0 beyond the margin (D > 1), -1 inside it", torch.allclose(d.grad, torch.tensor([0.0, -1.0, -1.0, -1.0])))''')

    c.eq(
        "EQ-S4-019", "The $R_1$ penalty and the two-time-scale update",
        r"R_1=\frac{\gamma}{2}\,\mathbb E_{\mathbf x\sim p_{\mathrm{data}}}\big[\|\nabla_{\mathbf x}D_\psi(\mathbf x)\|^2\big];\qquad "
        r"\text{TTUR: }\eta_D>\eta_G",
        plain=(
            "The $R_1$ penalty punishes the discriminator for having a steep slope *on real data*. Near the end of "
            "training, when the generator is close to the data, a steep discriminator keeps pushing the generator "
            "past the target and the two oscillate; flattening the discriminator around the real data lets the game "
            "settle. The two-time-scale rule is simpler still: let the discriminator learn faster than the generator, "
            "so that the generator always faces a critic that has nearly caught up."),
        refresher=(
            "Mescheder et al. (2018) analyse the game as a dynamical system and prove, on the Dirac-GAN "
            "($p_{\\mathrm{data}}=\\delta_0$, $G=\\theta$, $D(x)=\\psi x$), that simultaneous and alternating "
            "gradient descent on the unregularised game (and on WGAN and WGAN-GP) are not locally convergent, "
            "whereas the zero-centred penalty $R_1$ on real data makes the equilibrium locally attractive. "
            "StyleGAN2 uses $R_1$ with lazy regularisation. Heusel et al. (2017) prove convergence to a local Nash "
            "equilibrium under two time scales $\\eta_D>\\eta_G$ with stochastic approximation arguments, and "
            "introduced FID in the same paper. The check reproduces the Dirac-GAN: without $R_1$ simultaneous gradient "
            "steps spiral slowly outward; with $R_1$ they spiral in. It also verifies the autograd value of $R_1$ for a "
            "logistic discriminator against the closed form $\\gamma/2\\,\\|\\mathbf w\\|^2\\,\\mathbb E\\big[(D(1-D))^2\\big]$."),
        code='''# Dirac-GAN: V(theta, psi) = f(psi*theta) + f(0) with f = -softplus(-t); simultaneous gradient descent
def dirac_gan(gamma, steps=2000, lr=0.1):
    th, ps = 1.0, 1.0; radii = []
    for _ in range(steps):
        sig = 1 / (1 + math.exp(-ps * th))               # D = sigmoid(psi*x) at the generated point x = theta
        g_th = (1 - sig) * ps                               # dV/dtheta = f'(psi*theta) psi with f' = 1 - sigmoid; the generator descends
        g_ps = (1 - sig) * th - gamma * ps                 # discriminator ascent on V minus R1; for logit psi*x, grad_x = psi so R1 = gamma/2 psi^2
        th, ps = th - lr * g_th, ps + lr * g_ps
        radii.append(math.hypot(th, ps))
    return np.array(radii)
r0, r1 = dirac_gan(0.0), dirac_gan(1.0)
check("unregularised Dirac-GAN does not converge: simultaneous steps spiral outward", r0[-1] > r0[0], f"radius {r0[0]:.2f} -> {r0[-1]:.2f}")
check("with R1 the iterates converge to the equilibrium (theta, psi) = (0, 0)", r1[-1] < 1e-2, f"radius {r1[0]:.2f} -> {r1[-1]:.2e}")
# autograd R1 for D(x) = sigmoid(w.x + b) equals gamma/2 * ||w||^2 * E[(D(1-D))^2]
wv, b, gamma = torch.randn(4, dtype=torch.float64), 0.3, 10.0
X = torch.randn(256, 4, dtype=torch.float64, requires_grad=True)
Dx = torch.sigmoid(X @ wv + b)
gx, = torch.autograd.grad(Dx.sum(), X, create_graph=False)
R1 = gamma / 2 * (gx.norm(dim=1) ** 2).mean().item()
closed = gamma / 2 * (wv.norm() ** 2 * ((Dx * (1 - Dx)) ** 2).mean()).item()
check("R1 by autograd = gamma/2 ||w||^2 E[(D(1-D))^2] for a logistic discriminator", *close(R1, closed, 1e-12))''')

    c.eq(
        "EQ-S4-020", "Cycle-consistency loss",
        r"\mathcal L_{\mathrm{cyc}}=\mathbb E_{x}\big\|F(G(x))-x\big\|_1+\mathbb E_{y}\big\|G(F(y))-y\big\|_1",
        plain=(
            "CycleGAN translates between two domains (horses and zebras, summer and winter) without paired examples. "
            "Two generators go in opposite directions, $G$ from $X$ to $Y$ and $F$ back. The cycle loss says: if you "
            "translate and translate back you should get your original image. It is zero exactly when the two "
            "generators undo each other on the data, which rules out the degenerate solution where $G$ ignores its "
            "input and produces any plausible $Y$-image."),
        refresher=(
            "Zhu et al. (2017). The adversarial losses alone constrain only the *marginals* $G_\\#p_X\\approx p_Y$ and "
            "$F_\\#p_Y\\approx p_X$; infinitely many maps satisfy them. Cycle consistency adds that $F\\circ G$ and "
            "$G\\circ F$ be the identity on the respective data supports, which with the adversarial terms "
            "encourages (but does not guarantee) a bijection between domains. The $\\ell_1$ norm is a choice; the "
            "paper lists CycleGAN as an application of the implicit route to unpaired translation. The check builds "
            "two invertible toy maps, verifies the loss is zero for exact inverses and positive for a map that only "
            "matches the marginal."),
        code='''X = rng.normal(size=(500, 2))
G = lambda x: x @ np.array([[2.0, 0.5], [0.0, 1.0]]) + 1.0          # affine, invertible
Finv = lambda y: (y - 1.0) @ np.linalg.inv(np.array([[2.0, 0.5], [0.0, 1.0]]))
Y = G(rng.normal(size=(500, 2)))
cyc = lambda F_: np.abs(F_(G(X)) - X).sum(1).mean() + np.abs(G(F_(Y)) - Y).sum(1).mean()
check("cycle loss is zero when F is the exact inverse of G", *close(cyc(Finv), 0.0, 1e-10))
F_marginal = lambda y: Finv(y)[::-1]                                 # same marginal distribution, wrong pairing
check("a map with the right marginal but the wrong pairing has a positive cycle loss", cyc(F_marginal) > 1.0, f"loss = {cyc(F_marginal):.3f}")''')

    return c.build()


if __name__ == "__main__":
    print(build().path)
