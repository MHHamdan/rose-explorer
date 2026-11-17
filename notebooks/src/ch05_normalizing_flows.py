"""Chapter 5: normalizing flows. Build with `python notebooks/src/ch05_normalizing_flows.py`."""
from nbkit import Chapter


def build():
    c = Chapter(
        5, "normalizing_flows", "Normalizing flows: exact likelihood by invertible transport",
        summary=(
            "A normalizing flow pushes a simple base density through an invertible map and keeps an *exact* "
            "density by the change-of-variables formula. This chapter verifies that formula and the devices that "
            "make its Jacobian determinant cheap (coupling layers, invertible $1\\times1$ convolutions, "
            "autoregressive maps, trace estimators in continuous time), the dequantisation bound that makes "
            "bits/dim comparable across families, the family's structural limit on manifold data, and the "
            "flow-matching reformulation that drops invertibility while keeping a likelihood."),
        prerequisites="Chapters 1 and 2 (divergences, maximum likelihood); Chapter 3 for the posterior-flow entry",
        paper_sections="main §3.4 and §5.4; Supplement S1.6 and S4.4")

    c.section("Why invertible maps give exact densities",
              "If $\\mathbf{z}=f(\\mathbf{x})$ is a smooth bijection, probability mass is conserved: the mass "
              "that $p_Z$ puts on a small region around $\\mathbf{z}$ equals the mass $p_\\theta$ puts on its "
              "pre-image around $\\mathbf{x}$, and the ratio of the two volumes is $|\\det J_f(\\mathbf{x})|$. "
              "Everything in this chapter is a consequence of that one bookkeeping rule.")

    # ------------------------------------------------------------------------------------------- 5.1
    c.eq(["EQ-M-012", "EQ-S1-021", "EQ-S4-021"], "The change-of-variables formula and its composition",
         r"""\log p_\theta(\mathbf{x})=\log p_Z\big(f_\theta(\mathbf{x})\big)+\sum_{k=1}^{M}\log\big|\det J_{f_k}(\mathbf{h}_{k-1})\big|
         =\log p_Z(\mathbf{z})-\log\Big|\det\frac{\partial g_\theta}{\partial\mathbf{z}}(\mathbf{z})\Big|,
         \qquad \mathbf{h}_0=\mathbf{x},\ \mathbf{h}_M=\mathbf{z},\ g_\theta=f_\theta^{-1}""",
         plain=(
             "$f_\\theta$ turns a data point $\\mathbf{x}$ into a code $\\mathbf{z}$ that follows a simple "
             "distribution $p_Z$ (usually a standard Gaussian). The density of $\\mathbf{x}$ is the density of its "
             "code, corrected by how much $f_\\theta$ stretches or squeezes space at $\\mathbf{x}$, which is the "
             "absolute Jacobian determinant. If $f_\\theta$ is built from $M$ layers, the corrections add up as "
             "logs, one per layer. The second form says the same thing in the sampling direction $g_\\theta$: "
             "stretching on the way out is squeezing on the way in."),
         refresher=(
             "Assumptions: $f_\\theta:\\mathbb{R}^D\\to\\mathbb{R}^D$ is a $C^1$ diffeomorphism, so "
             "$\\dim\\mathbf{z}=\\dim\\mathbf{x}$ and $\\det J_f\\neq0$ everywhere. Derivation: for a measurable "
             "set $A$, $\\int_A p_\\theta=\\int_{f(A)}p_Z$; substituting $\\mathbf{z}=f(\\mathbf{x})$ in the "
             "right-hand integral introduces $|\\det J_f|$, and since $A$ is arbitrary the integrands agree. The "
             "chain rule gives $J_{f_M\\circ\\cdots\\circ f_1}=\\prod_k J_{f_k}(\\mathbf{h}_{k-1})$ and "
             "$\\det$ is multiplicative, hence the sum of logs. The inverse-function theorem gives "
             "$J_{g}(\\mathbf{z})=J_f(\\mathbf{x})^{-1}$, hence the minus sign in the generative form; the most "
             "common implementation error is using that sign in the normalising direction. The paper trains "
             "a Real NVP under this formula (Section 8, E1) and the repository check "
             "`repo_realnvp_logprob_vs_autograd_jacobian` compares `flow/model.py` against an autograd Jacobian; "
             "here we verify the identity on a two-layer affine flow where the exact density is known."),
         code=r'''# f = f2 o f1 with affine layers; x = A^{-1}(z - b) for z ~ N(0, I) has a known Gaussian density.
rng = np.random.default_rng(7)
A1 = rng.normal(size=(3, 3)) + 2 * np.eye(3); b1 = rng.normal(size=3)
A2 = rng.normal(size=(3, 3)) + 2 * np.eye(3); b2 = rng.normal(size=3)
x = rng.normal(size=(5, 3))
h1 = x @ A1.T + b1                       # layer 1
z = h1 @ A2.T + b2                       # layer 2 (normalising direction x -> z)
logdet = np.log(abs(np.linalg.det(A1))) + np.log(abs(np.linalg.det(A2)))      # sum over layers
logp_flow = scipy.stats.multivariate_normal(np.zeros(3), np.eye(3)).logpdf(z) + logdet
A = A2 @ A1; b = A2 @ b1 + b2; Ainv = np.linalg.inv(A)
logp_true = scipy.stats.multivariate_normal(-Ainv @ b, Ainv @ Ainv.T).logpdf(x)  # x = A^{-1}(z - b)
check("log p_Z(f(x)) + sum_k log|det J_k| equals the exact density", *close(logp_flow, logp_true, 1e-10))
# generative form: -log|det dg/dz| with g = f^{-1} gives the same number
logp_gen = scipy.stats.multivariate_normal(np.zeros(3), np.eye(3)).logpdf(z) - np.log(abs(np.linalg.det(Ainv)))
check("generative form  log p_Z(z) - log|det dg/dz|  agrees", *close(logp_gen, logp_true, 1e-10))
# the sign matters: the wrong sign in the normalising direction is far off
wrong = scipy.stats.multivariate_normal(np.zeros(3), np.eye(3)).logpdf(z) - logdet
check("the minus-sign variant in the normalising direction is rejected", np.max(np.abs(wrong - logp_true)) > 1e-3,
      f"max error of the wrong sign = {np.max(np.abs(wrong - logp_true)):.2f} nats")
# composition: det of the product equals the product of dets (log-sum over layers)
check("log|det(A2 A1)| = log|det A1| + log|det A2|", *close(np.log(abs(np.linalg.det(A))), logdet, 1e-12))''')

    # ------------------------------------------------------------------------------------------- 5.2
    c.eq(["EQ-S4-022"], "Additive and affine coupling layers",
         r"""\mathbf{h}'_A=\mathbf{h}_A,\qquad
         \mathbf{h}'_B=\mathbf{h}_B\odot\exp\big(s(\mathbf{h}_A)\big)+t(\mathbf{h}_A),\qquad
         \log|\det J|=\sum_j s(\mathbf{h}_A)_j,\qquad
         \mathbf{h}_B=\big(\mathbf{h}'_B-t(\mathbf{h}_A)\big)\odot\exp\big(-s(\mathbf{h}_A)\big)""",
         plain=(
             "Split the vector in two halves. Leave half $A$ alone and use it to compute a scale $e^{s}$ and a "
             "shift $t$ for half $B$ with any network you like. Because half $A$ is unchanged, you can recompute "
             "the same $s$ and $t$ from the output and undo the transformation exactly, and the Jacobian is "
             "triangular, so its determinant is just the product of the scales: the log-determinant is the sum of "
             "the $s_j$. NICE uses only the shift ($s=0$), so its determinant is 1."),
         refresher=(
             "The Jacobian in block form is $\\begin{pmatrix}I&0\\cr \\partial\\mathbf{h}'_B/\\partial\\mathbf{h}_A&"
             "\\mathrm{diag}(e^{s})\\end{pmatrix}$; the off-diagonal block, which contains all the network's "
             "derivatives, does not enter the determinant, so $s$ and $t$ can be arbitrary (non-invertible) "
             "networks and the cost is $O(D)$ instead of $O(D^3)$. Alternating the partition between layers "
             "(checkerboard and channel masks in Real NVP) is what lets every dimension be transformed. Unbounded "
             "$s$ makes $e^{s}$ overflow, which is why implementations bound it (entry 5.9). The check below builds "
             "a coupling layer with a random MLP conditioner and compares the formula with a full autograd Jacobian."),
         code=r'''torch.manual_seed(0)
D = 6; dA = 3
net = torch.nn.Sequential(torch.nn.Linear(dA, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2 * (D - dA))).double()

def coupling(h, additive=False):
    hA, hB = h[:dA], h[dA:]
    s, t = net(hA).chunk(2)
    s = torch.zeros_like(s) if additive else torch.tanh(s)      # bounded log-scale, as in the paper's implementation
    return torch.cat([hA, hB * torch.exp(s) + t]), s.sum()

def inverse(hp):
    hA, hpB = hp[:dA], hp[dA:]
    s, t = net(hA).chunk(2); s = torch.tanh(s)
    return torch.cat([hA, (hpB - t) * torch.exp(-s)])

h = torch.randn(D, dtype=torch.float64)
hp, logdet_formula = coupling(h)
J = torch.autograd.functional.jacobian(lambda v: coupling(v)[0], h)
check("affine coupling: log|det J| = sum_j s_j (autograd Jacobian)", *close(torch.linalg.slogdet(J)[1].item(), logdet_formula.item(), 1e-10))
check("affine coupling: Jacobian is block lower-triangular (upper-right block is zero)", torch.allclose(J[:dA, dA:], torch.zeros(dA, D - dA, dtype=torch.float64)))
check("affine coupling: inverse recovers the input exactly", *close(inverse(hp).detach().numpy(), h.detach().numpy(), 1e-12))
J_add = torch.autograd.functional.jacobian(lambda v: coupling(v, additive=True)[0], h)
check("additive (NICE) coupling: det J = 1", *close(torch.linalg.det(J_add).item(), 1.0, 1e-12))

# Figure: what one affine coupling layer does to a grid in 2-D (and that the inverse undoes it)
net2 = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)).double()
with torch.no_grad():
    for p in net2.parameters(): p.mul_(2.0)
g = torch.stack(torch.meshgrid(torch.linspace(-2, 2, 15, dtype=torch.float64), torch.linspace(-2, 2, 15, dtype=torch.float64), indexing="ij"), -1).reshape(-1, 2)
s2, t2 = net2(g[:, :1]).chunk(2, dim=1)
fwd = torch.cat([g[:, :1], g[:, 1:] * torch.exp(torch.tanh(s2)) + t2], 1)
back = torch.cat([fwd[:, :1], (fwd[:, 1:] - t2) * torch.exp(-torch.tanh(s2))], 1)
fig, axs = plt.subplots(1, 3, figsize=(9, 3))
for ax, pts, title in zip(axs, (g, fwd, back), ("input grid", "after coupling (x1 kept, x2 scaled and shifted)", "after the inverse")):
    ax.scatter(*pts.detach().numpy().T, s=6); ax.set_title(title, fontsize=9); ax.set_aspect("equal")
plt.tight_layout(); plt.show()
check("figure round trip: inverse(coupling(grid)) = grid", *close(back.detach().numpy(), g.numpy(), 1e-12))''')

    # ------------------------------------------------------------------------------------------- 5.3
    c.eq(["EQ-M-013", "EQ-S1-022", "EQ-S4-023"], "Cheap determinants: triangular Jacobians, invertible $1\\times1$ convolutions, ActNorm",
         r"""\text{triangular }J:\ \log|\det J|=\sum_i\log|J_{ii}|\ \ (O(D));\qquad
         1{\times}1\text{ conv with }\mathbf{W}\in\mathbb{R}^{c\times c}\text{ on an }h\times w\text{ map: }
         \log|\det J|=h\,w\,\log|\det\mathbf{W}|;\qquad
         \mathbf{W}=\mathbf{P}\mathbf{L}(\mathbf{U}+\mathrm{diag}\,\mathbf{s})\Rightarrow\log|\det\mathbf{W}|=\sum_c\log|s_c|""",
         plain=(
             "A determinant of a $D\\times D$ matrix normally costs about $D^3$ operations, which is hopeless for "
             "images. Flows only use maps whose Jacobian is triangular (determinant = product of the diagonal, "
             "$D$ operations) or, for Glow's learned channel mixing, the same small $c\\times c$ matrix "
             "$\\mathbf{W}$ applied at every one of the $h\\,w$ pixel positions, so the log-determinant is "
             "$h\\,w$ times that of $\\mathbf{W}$. Writing $\\mathbf{W}$ as a permutation times two triangular "
             "factors makes even that determinant a sum of $c$ logs."),
         refresher=(
             "A $1\\times1$ convolution is the linear map $\\mathbf{I}_{hw}\\otimes\\mathbf{W}$ on the flattened "
             "map (block-diagonal after reordering), and $\\det(\\mathbf{I}_{hw}\\otimes\\mathbf{W})=(\\det\\mathbf{W})^{hw}$. "
             "The LU form of Glow fixes $\\mathbf{P}$ at initialisation, keeps $\\mathbf{L}$ unit lower-triangular "
             "and $\\mathbf{U}$ strictly upper-triangular, and learns the diagonal $\\mathbf{s}$ separately, so "
             "$|\\det\\mathbf{W}|=\\prod_c|s_c|$ at $O(c)$ cost. ActNorm is a per-channel affine map, hence also a "
             "$1\\times1$ 'convolution' with diagonal $\\mathbf{W}$, contributing $h\\,w\\sum_c\\log|\\mathrm{scale}_c|$; "
             "the paper's implementation accumulates this term (Supplement S4.4). The checks compare each formula "
             "with a full autograd Jacobian on a tiny map."),
         code=r'''torch.manual_seed(2)
c_, h_, w_ = 3, 2, 2
W = torch.randn(c_, c_, dtype=torch.float64)
conv = lambda v: torch.nn.functional.conv2d(v.view(1, c_, h_, w_), W.view(c_, c_, 1, 1)).reshape(-1)
J = torch.autograd.functional.jacobian(conv, torch.randn(c_ * h_ * w_, dtype=torch.float64))
check("1x1 conv: log|det J| = h w log|det W|", *close(torch.linalg.slogdet(J)[1].item(), h_ * w_ * torch.linalg.slogdet(W)[1].item(), 1e-10))
# LU parameterisation: W = P L (U + diag s)  ->  log|det W| = sum log|s_c|
P, L, U = torch.linalg.lu(W)
s = torch.diagonal(U); U0 = torch.triu(U, 1)
W_lu = P @ L @ (U0 + torch.diag(s))
check("LU reconstruction W = P L (U + diag s)", *close(W_lu.numpy(), W.numpy(), 1e-12))
check("log|det W| = sum_c log|s_c|", *close(torch.linalg.slogdet(W)[1].item(), torch.log(s.abs()).sum().item(), 1e-12))
# triangular Jacobian: determinant is the product of the diagonal, O(D)
T = torch.tril(torch.randn(5, 5, dtype=torch.float64)) + 3 * torch.eye(5, dtype=torch.float64)
check("triangular J: log|det| = sum log|diag|", *close(torch.linalg.slogdet(T)[1].item(), torch.log(torch.diagonal(T).abs()).sum().item(), 1e-12))
# ActNorm: per-channel affine map on an h x w map
scale, bias = torch.rand(c_, dtype=torch.float64) + 0.5, torch.randn(c_, dtype=torch.float64)
actnorm = lambda v: ((v.view(c_, h_, w_) * scale[:, None, None]) + bias[:, None, None]).reshape(-1)
Ja = torch.autograd.functional.jacobian(actnorm, torch.randn(c_ * h_ * w_, dtype=torch.float64))
check("ActNorm: log|det J| = h w sum_c log|scale_c|", *close(torch.linalg.slogdet(Ja)[1].item(), h_ * w_ * torch.log(scale).sum().item(), 1e-12))''')

    # ------------------------------------------------------------------------------------------- 5.4
    c.eq(["EQ-S4-024"], "Autoregressive flows: a triangular map from any autoregressive conditioner",
         r"""z_i=\frac{x_i-\mu_i(\mathbf{x}_{<i})}{\sigma_i(\mathbf{x}_{<i})},\qquad
         \frac{\partial z_i}{\partial x_j}=0\ \text{for }j>i,\qquad
         \det\frac{\partial\mathbf{z}}{\partial\mathbf{x}}=\prod_i\sigma_i(\mathbf{x}_{<i})^{-1}""",
         plain=(
             "Standardise each coordinate using a mean and a scale computed from the coordinates before it. "
             "Coordinate $i$ of the output never depends on coordinates after $i$ of the input, so the Jacobian is "
             "triangular and its determinant is the product of the diagonal entries $1/\\sigma_i$. Any network "
             "that outputs $(\\mu_i,\\sigma_i)$ from $\\mathbf{x}_{<i}$ (such as a masked MLP) gives a valid flow. "
             "Computing $\\mathbf{z}$ from $\\mathbf{x}$ can be done for all $i$ at once (MAF); computing "
             "$\\mathbf{x}$ from $\\mathbf{z}$ must go coordinate by coordinate, because $x_i$ needs $x_{<i}$ "
             "first (and the reverse holds for IAF)."),
         refresher=(
             "Invertibility needs $\\sigma_i>0$; the inverse is $x_i=\\mu_i(\\mathbf{x}_{<i})+\\sigma_i(\\mathbf{x}_{<i})z_i$, "
             "a $D$-step recursion. MAF conditions on $\\mathbf{x}_{<i}$ (density parallel, sampling sequential); "
             "IAF conditions on $\\mathbf{z}_{<i}$ (sampling parallel, density of external data sequential), which "
             "is the right trade-off for a variational posterior that is only ever evaluated on its own samples "
             "(main paper §5.4). The check uses a random masked conditioner, confirms the strict lower-triangular "
             "structure with autograd and the determinant formula, and counts conditioner calls in each direction."),
         code=r'''torch.manual_seed(3)
D = 5
Wm = torch.randn(D, D, dtype=torch.float64) * torch.tril(torch.ones(D, D, dtype=torch.float64), -1)   # strictly lower mask
Ws = torch.randn(D, D, dtype=torch.float64) * torch.tril(torch.ones(D, D, dtype=torch.float64), -1)
n_pass = {"count": 0}                                   # counts conditioner evaluations
def mu(x):
    n_pass["count"] += 1; return torch.tanh(Wm @ x)      # mu_i depends on x_{<i} only
sig = lambda x: torch.exp(0.5 * torch.tanh(Ws @ x))     # sigma_i > 0, depends on x_{<i} only (evaluated with mu)
f = lambda x: (x - mu(x)) / sig(x)
x = torch.randn(D, dtype=torch.float64)
J = torch.autograd.functional.jacobian(f, x)
check("Jacobian is lower-triangular (dz_i/dx_j = 0 for j > i)", torch.allclose(torch.triu(J, 1), torch.zeros_like(J)))
check("det J = prod_i sigma_i^{-1}", *close(torch.linalg.det(J).item(), torch.prod(1 / sig(x)).item(), 1e-12))
# sequential inverse: x_i = mu_i(x_<i) + sigma_i(x_<i) z_i, one coordinate at a time
n_pass["count"] = 0; z = f(x); density_passes = n_pass["count"]        # density direction: z = f(x)
xr = torch.zeros(D, dtype=torch.float64); n_pass["count"] = 0
for i in range(D):
    xr[i] = mu(xr)[i] + sig(xr)[i] * z[i]
calls = n_pass["count"]                                                   # sampling direction: x = f^{-1}(z)
check("sequential inverse recovers x", *close(xr.detach().numpy(), x.detach().numpy(), 1e-12), )
check("density direction: 1 conditioner pass; sampling direction: D passes", density_passes == 1 and calls == D,
      f"D = {D}: density used {density_passes} pass, inverse used {calls} passes")''')

    # ------------------------------------------------------------------------------------------- 5.5
    c.eq(["EQ-S4-025"], "Continuous-time flows: the instantaneous change of variables and Hutchinson's trace estimator",
         r"""\frac{d\mathbf{h}_t}{dt}=\mathbf{v}_\theta(\mathbf{h}_t,t),\qquad
         \frac{d\log p(\mathbf{h}_t)}{dt}=-\operatorname{Tr}\frac{\partial\mathbf{v}_\theta}{\partial\mathbf{h}}(\mathbf{h}_t,t),\qquad
         \operatorname{Tr}A=\mathbb{E}_{\mathbf{v}}\big[\mathbf{v}^{\top}A\mathbf{v}\big]\ \text{for }\mathbb{E}[\mathbf{v}\mathbf{v}^{\top}]=\mathbf{I}""",
         plain=(
             "Instead of a stack of layers, let a point move continuously along a velocity field. Its "
             "log-density changes at a rate given by minus the divergence of the field (the trace of the "
             "Jacobian), so the total log-determinant of the flow is an integral of traces along the path. A trace "
             "of a $D\\times D$ matrix can be estimated with a random vector $\\mathbf{v}$ whose covariance is the "
             "identity: $\\mathbf{v}^\\top A\\mathbf{v}$ has expectation $\\operatorname{Tr}A$, and costs one "
             "vector--Jacobian product instead of $D$."),
         refresher=(
             "Derivation: for the discrete-time flow with step $\\delta$, $J\\approx I+\\delta\\,\\partial\\mathbf{v}/\\partial\\mathbf{h}$ "
             "and $\\log|\\det(I+\\delta A)|=\\delta\\operatorname{Tr}A+O(\\delta^2)$; dividing by $\\delta$ and "
             "letting $\\delta\\to0$ gives the trace ODE (Chen et al., 2018). It requires $\\mathbf{v}$ Lipschitz in "
             "$\\mathbf{h}$ so that the ODE has unique solutions (hence is a bijection). Hutchinson's identity "
             "follows from linearity of expectation: $\\mathbb{E}[\\mathbf{v}^\\top A\\mathbf{v}]=\\sum_{ij}A_{ij}\\mathbb{E}[v_iv_j]=\\operatorname{Tr}A$; "
             "Rademacher $\\pm1$ vectors have the smallest variance among such choices. The registry marks this "
             "entry 'yes with wording fix' because an earlier draft stated the estimator for Gaussian $\\mathbf{v}$ "
             "only. Checks: a linear field $\\mathbf{v}=A\\mathbf{h}$ has the exact solution "
             "$\\log p(\\mathbf{h}_T)=\\log p(\\mathbf{h}_0)-T\\operatorname{Tr}A$, compared with an ODE solve; then "
             "Hutchinson with Rademacher and Gaussian probes."),
         code=r'''rng = np.random.default_rng(13)
A = rng.normal(size=(4, 4)) * 0.5
h0 = rng.normal(size=4); T = 1.3
# integrate the augmented system (h, log p) with the trace ODE
def rhs(t, y):
    h = y[:4]; return np.concatenate([A @ h, [-np.trace(A)]])
sol = scipy.integrate.solve_ivp(rhs, (0, T), np.concatenate([h0, [0.0]]), rtol=1e-10, atol=1e-12)
hT, dlogp = sol.y[:4, -1], sol.y[4, -1]
# exact: h_T = expm(A T) h0 and the log-density changes by -log|det expm(A T)| = -T Tr A
check("instantaneous change of variables integrates to -log|det J| for a linear field",
      *close(dlogp, -np.linalg.slogdet(scipy.linalg.expm(A * T))[1], 1e-8))
check("ODE solution matches expm(A T) h0", *close(hT, scipy.linalg.expm(A * T) @ h0, 1e-7))
# Hutchinson's estimator with Rademacher and Gaussian probes (Monte Carlo: tolerance = 5 standard errors)
B = rng.normal(size=(6, 6)); n = 200_000
v = rng.choice([-1.0, 1.0], size=(n, 6)); est = np.einsum("ni,ij,nj->n", v, B, v)
g = rng.normal(size=(n, 6)); est2 = np.einsum("ni,ij,nj->n", g, B, g)
check("Hutchinson, Rademacher probes", *close(est.mean(), np.trace(B), 5 * est.std() / math.sqrt(n), f"SE {est.std()/math.sqrt(n):.2e}; "))
check("Hutchinson, Gaussian probes", *close(est2.mean(), np.trace(B), 5 * est2.std() / math.sqrt(n), f"SE {est2.std()/math.sqrt(n):.2e}; "))
check("Rademacher probes have lower variance than Gaussian probes", est.std() < est2.std(), f"std {est.std():.3f} vs {est2.std():.3f}")''')

    # ------------------------------------------------------------------------------------------- 5.6
    c.eq(["EQ-M-040", "EQ-S4-031"], "Flow matching: a regression target instead of a Jacobian, and the likelihood it still has",
         r"""\mathbf{x}_t=(1-t)\,\mathbf{x}+t\,\mathbf{z},\quad \mathbf{x}\sim p_{\mathrm{data}},\ \mathbf{z}\sim\mathcal{N}(0,I),\qquad
         \mathcal{L}_{\mathrm{CFM}}=\mathbb{E}_{t,\mathbf{x},\mathbf{z}}\big\|\mathbf{v}_\theta(\mathbf{x}_t,t)-(\mathbf{z}-\mathbf{x})\big\|^2,\qquad
         \mathbf{v}^*(\mathbf{x}_t,t)=\mathbb{E}[\mathbf{z}-\mathbf{x}\mid\mathbf{x}_t]""",
         plain=(
             "Draw a data point and a noise point, join them by a straight line, and ask a network to predict the "
             "line's direction from a point on it and the time. The best possible prediction is the average "
             "direction of all lines passing through that point, and that average velocity field moves the whole "
             "data distribution to the noise distribution along the way (and back when run in reverse). Because "
             "the network is trained by plain regression, it needs neither an inverse nor a cheap determinant, but "
             "the ODE it defines is still a continuous flow, so its likelihood can be computed with the trace "
             "formula of entry 5.5."),
         refresher=(
             "Conditional flow matching (Lipman et al., 2023; Liu et al., 2023): the conditional path "
             "$p_t(\\mathbf{x}_t\\mid\\mathbf{x},\\mathbf{z})$ is a Dirac on the line; its marginal velocity "
             "$\\mathbf{v}^*=\\mathbb{E}[\\mathbf{z}-\\mathbf{x}\\mid\\mathbf{x}_t]$ satisfies the continuity "
             "equation for the marginal $p_t$, and the least-squares property of conditional expectation makes "
             "$\\mathcal{L}_{\\mathrm{CFM}}$ and the intractable marginal objective differ by a constant. The "
             "paper's convention puts data at $t=0$ and noise at $t=1$ (Supplement S4.4 says so explicitly), the same direction as the DDPM chain (data at $t=0$) but "
             "the opposite of Lipman et al. (2023), who put noise at $t=0$. The learned ODE is a CNF 'in the sense of eq. (cov)' (EQ-M-040): its density follows "
             "from the instantaneous change of variables. Check: for Gaussian data $\\mathcal{N}(\\mu,\\sigma^2)$ "
             "the marginal velocity is available in closed form by Gaussian conditioning; integrating the ODE "
             "from noise with the trace term reproduces the data density exactly, and the regression minimiser "
             "is verified as a conditional mean on a discrete example."),
         code=r'''mu, sd = 1.5, 0.6     # data x ~ N(mu, sd^2); noise z ~ N(0, 1); x_t = (1-t) x + t z
def v_star(xt, t):
    # (x, z) jointly Gaussian; x_t is a linear function of them -> E[z - x | x_t] is linear in x_t
    m_t = (1 - t) * mu; var_t = (1 - t) ** 2 * sd ** 2 + t ** 2
    cov_zx_xt = t - (1 - t) * sd ** 2            # Cov(z - x, x_t) = t Var z - (1-t) Var x
    mean_zx = -mu                               # E[z - x]
    return mean_zx + cov_zx_xt / var_t * (xt - m_t)
def dv_dx(t):
    var_t = (1 - t) ** 2 * sd ** 2 + t ** 2
    return (t - (1 - t) * sd ** 2) / var_t
# 1. the marginal velocity transports the marginal: integrate the ODE for many noise samples from t=1 to t=0
rng = np.random.default_rng(0)
z0 = rng.normal(size=4000)
sol = scipy.integrate.solve_ivp(lambda t, y: v_star(y, t), (1.0, 0.0), z0, rtol=1e-8, atol=1e-10)
x_gen = sol.y[:, -1]
check("ODE run from noise to t=0 reproduces the data moments", abs(x_gen.mean() - mu) < 0.05 and abs(x_gen.std() - sd) < 0.03,
      f"mean {x_gen.mean():.3f} (target {mu}), sd {x_gen.std():.3f} (target {sd}); MC error ~0.01-0.02")
# 2. likelihood of a data point through the instantaneous change of variables along the flow-matching ODE
def logp_fm(x_obs):
    rhs = lambda t, y: [v_star(y[0], t), dv_dx(t)]        # d log p / dt = -Tr dv/dx along the forward (data->noise) direction
    s = scipy.integrate.solve_ivp(rhs, (0.0, 1.0), [x_obs, 0.0], rtol=1e-10, atol=1e-12)
    z_end, int_tr = s.y[0, -1], s.y[1, -1]
    return scipy.stats.norm.logpdf(z_end) + int_tr          # log p_0(x) = log p_1(z) + int_0^1 Tr dv/dx dt
xs = np.array([0.3, 1.5, 2.4])
check("flow-matching ODE + trace integral gives the exact data log-density (EQ-M-040)",
      *close([logp_fm(x) for x in xs], scipy.stats.norm.logpdf(xs, mu, sd), 1e-6))
# 3. the regression target's minimiser is the conditional mean (discrete example, exact arithmetic)
P = np.array([[0.1, 0.3], [0.4, 0.2]])            # joint of (x_t bucket, direction value index)
vals = np.array([-1.0, 2.0])
cond_mean = (P * vals).sum(1) / P.sum(1)
loss = lambda v: np.sum(P * (v[:, None] - vals[None, :]) ** 2)
opt = scipy.optimize.minimize(loss, np.zeros(2)).x
check("least-squares minimiser of the CFM loss = E[target | x_t]", *close(opt, cond_mean, 1e-6))''')

    # ------------------------------------------------------------------------------------------- 5.7
    c.eq(["EQ-S4-026", "EQ-S4-032"], "Dequantisation: a continuous density lower-bounds a discrete probability, and how bits/dim are computed",
         r"""\tilde{\mathbf{x}}=\mathbf{x}+\mathbf{u},\ \mathbf{u}\sim\mathcal{U}[0,1)^D:\qquad
         \log P(\mathbf{x})=\log\!\int_{[0,1)^D}p(\mathbf{x}+\mathbf{u})\,d\mathbf{u}\ \ge\ \mathbb{E}_{\mathbf{u}}\big[\log p(\mathbf{x}+\mathbf{u})\big];\qquad
         \text{bits/dim}=\frac{-\log p_\theta(\mathbf{y})+D\log256}{D\log2},\ \ \mathbf{y}=\frac{\mathbf{x}+\mathbf{u}}{256}""",
         plain=(
             "Pixels are integers, but a flow needs continuous inputs, so uniform noise in $[0,1)$ is added to each "
             "pixel. The probability of the integer image is the integral of the continuous density over the unit "
             "cube around it; the average of the log-density over that cube is never larger than the log of the "
             "average (Jensen), so the number a flow reports is a lower bound on the log-probability of the real "
             "discrete image, i.e. an upper bound on its code length. When the model works on pixels rescaled "
             "to $[0,1)$, every dimension was shrunk by 256, and that costs $\\log 256$ nats per dimension, which "
             "must be added back."),
         refresher=(
             "Jensen with the concave $\\log$ and the uniform measure on the cube; equality iff $p$ is constant "
             "on the cube. Flow++ replaces the uniform $q(\\mathbf{u})$ by a learned $q(\\mathbf{u}\\mid\\mathbf{x})$, "
             "which gives the variational-dequantisation bound $\\mathbb{E}_q[\\log p(\\mathbf{x}+\\mathbf{u})-\\log q(\\mathbf{u}\\mid\\mathbf{x})]$ "
             "that is tight when $q\\propto p$ on the cube (verified below). The rescaling term: if "
             "$\\mathbf{y}=\\tilde{\\mathbf{x}}/256$ then $p_{\\tilde{X}}(\\tilde{\\mathbf{x}})=p_Y(\\mathbf{y})\\,256^{-D}$, "
             "so $-\\log p_{\\tilde X}=-\\log p_Y+D\\log256$; dividing by $D\\log2$ gives bits per dimension. "
             "**Registry note.** The paper text (Supplement S4.4, 'Practical implementation' (iii)) applies the "
             "formula to the rescaled input $\\mathbf{y}$, which is correct. The registry entry `EQ-S4-032` "
             "transcribes it with $\\tilde{\\mathbf{x}}$ (the unscaled dequantised input) as the argument, and in "
             "that form it double-counts the $D\\log256$ term; that is why it is marked 'NO'. The check below "
             "shows both facts: the formula holds for $\\mathbf{y}$ and fails for $\\tilde{\\mathbf{x}}$."),
         code=r'''# 1. Jensen: discrete mass >= exp(E_u log p(x+u)) for a 1-D continuous density p and integer x
p = lambda t: scipy.stats.norm.pdf(t, loc=3.2, scale=1.1)
for x in (1, 3, 5):
    mass = scipy.integrate.quad(p, x, x + 1)[0]
    bound = scipy.integrate.quad(lambda u: np.log(p(x + u)), 0, 1)[0]
    check(f"log P(x={x}) >= E_u log p(x+u)", np.log(mass) >= bound - 1e-12, f"log P = {np.log(mass):.5f} >= {bound:.5f}")
# 2. variational dequantisation: q(u|x) proportional to p(x+u) on the cube makes the bound tight
x = 3; Z = scipy.integrate.quad(p, x, x + 1)[0]; q = lambda u: p(x + u) / Z
iw_bound = scipy.integrate.quad(lambda u: q(u) * (np.log(p(x + u)) - np.log(q(u))), 0, 1)[0]
check("optimal variational dequantiser makes the bound exact", *close(iw_bound, np.log(Z), 1e-9))
# 3. bits/dim: model density on y = (x+u)/256 vs the same model expressed on x~ = x+u
D = 4; rng = np.random.default_rng(1)
y = rng.uniform(size=D); xt = 256 * y
logp_y = scipy.stats.multivariate_normal(np.full(D, 0.5), 0.04 * np.eye(D)).logpdf(y)           # density on y
logp_xt = logp_y - D * math.log(256)                                                            # the same density, on x~ = 256 y
bpd_true = -logp_xt / (D * math.log(2))
check("bits/dim = (-log p(y) + D log 256) / (D log 2) when p is evaluated on the rescaled y", *close((-logp_y + D * math.log(256)) / (D * math.log(2)), bpd_true, 1e-12))
wrong = (-logp_xt + D * math.log(256)) / (D * math.log(2))
check("the same formula applied to the unscaled x~ double-counts log 256 (registry wording, marked NO)", abs(wrong - bpd_true) > 1, f"off by {wrong - bpd_true:.3f} bits/dim = log2(256) = 8")''')

    # ------------------------------------------------------------------------------------------- 5.8
    c.eq(["EQ-M-038"], "How much of Flow++'s gain is dequantisation? (an arithmetic check)",
         r"""\frac{0.13}{3.49-3.08}=\frac{0.13}{0.41}\approx0.32,\qquad \frac{0.13}{3.35-3.08}=\frac{0.13}{0.27}\approx0.48""",
         plain=(
             "On CIFAR-10, Real NVP reports 3.49 bits/dim, Glow 3.35 and Flow++ 3.08. Flow++'s own ablation "
             "attributes about 0.13 bits/dim to variational dequantisation. That is roughly a third of the gap to "
             "Real NVP and about half of the gap to Glow, which is why the paper says values under the two "
             "dequantisation conventions 'are not strictly comparable'."),
         refresher=(
             "The numbers are literature values (Dinh et al., 2017; Kingma and Dhariwal, 2018; Ho et al., 2019, "
             "Table 2 of the Flow++ ablation) and the entry is an arithmetic claim, not a measurement; the paper "
             "quotes it as 'about 0.13' and 'a substantial part'. Ablation gains are not additive across "
             "components, so the share is indicative. The check reproduces the arithmetic."),
         code=r'''share_realnvp = 0.13 / (3.49 - 3.08)
share_glow = 0.13 / (3.35 - 3.08)
check("0.13 / (3.49 - 3.08) ~ 0.32", round(share_realnvp, 2) == 0.32, f"{share_realnvp:.4f}")
check("0.13 / (3.35 - 3.08) ~ 0.48 (gap to Glow)", round(share_glow, 2) == 0.48, f"{share_glow:.4f}")''')

    # ------------------------------------------------------------------------------------------- 5.9
    c.eq(["EQ-M-037", "EQ-S4-027"], "Which KL a flow minimises: forward KL for density estimation, reverse KL as a posterior; the logit pre-processing and the bounded log-scale",
         r"""\text{density estimation: }\min_\theta D_{\mathrm{KL}}(p_{\mathrm{data}}\,\|\,p_\theta);\qquad
         \text{posterior flow: }\min_\phi D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z}\mid\mathbf{x})),\ \
         H[q_\phi]=H[p_Z]+\mathbb{E}\log\Big|\det\frac{\partial g_\phi}{\partial\boldsymbol{\epsilon}}\Big|;\qquad
         \mathbf{x}\mapsto\operatorname{logit}\big(\alpha+(1-2\alpha)\mathbf{x}\big),\ \alpha=0.05;\qquad s=\tanh(\hat s)\,e^{\gamma}\Rightarrow|s|\le e^{\gamma}""",
         plain=(
             "Trained by maximum likelihood on data, a flow minimises the forward KL, which punishes it wherever "
             "data exist and the model's density is low, so it covers all the data (and may spread mass where no "
             "data are). Used as a VAE posterior, a flow minimises the reverse KL instead, and its "
             "log-determinant enters through the entropy of the posterior. Two engineering details: pixel values "
             "in $[0,1]$ are mapped to the whole real line by a logit, whose own stretching must be added to the "
             "likelihood, and the log-scale of a coupling layer is passed through $\\tanh$ so that $e^{s}$ can never "
             "overflow."),
         refresher=(
             "MLE $\\Leftrightarrow$ forward KL is Chapter 2's identity. For the posterior flow, "
             "$q_\\phi$ is the push-forward of a base $p_Z$ through $g_\\phi$, and the change of variables gives "
             "$\\log q_\\phi(\\mathbf{z})=\\log p_Z(\\boldsymbol{\\epsilon})-\\log|\\det\\partial g_\\phi/\\partial\\boldsymbol{\\epsilon}|$, "
             "so $H[q_\\phi]=H[p_Z]+\\mathbb{E}\\log|\\det J_{g_\\phi}|$ (Rezende and Mohamed, 2015; Kingma et al., "
             "2016): the ELBO needs only samples from $q_\\phi$ and this entropy, never the density of external "
             "points, which is why IAF's sequential density direction is harmless there. MAF/IAF parallelism is in "
             "entry 5.4. The logit map $y=\\operatorname{logit}(\\alpha+(1-2\\alpha)x)$ has "
             "$dy/dx=(1-2\\alpha)/(p(1-p))$ with $p=\\alpha+(1-2\\alpha)x$; its log-determinant is "
             "$\\sum_i\\log(dy_i/dx_i)$ and is part of $\\log p_\\theta$ in the paper's bits/dim. "
             "Checks: the entropy identity on a linear flow; forward vs reverse KL fits of a Gaussian to a "
             "two-mode target (covering vs seeking); the logit log-determinant against autograd; the "
             "$\\tanh$ bound."),
         code=r'''# 1. entropy of a flow posterior: H[q] = H[p_Z] + E log|det J_g|  (linear flow, exact)
rng = np.random.default_rng(5)
Bm = rng.normal(size=(3, 3)) + 2 * np.eye(3)
H_base = scipy.stats.multivariate_normal(np.zeros(3), np.eye(3)).entropy()
H_q = scipy.stats.multivariate_normal(np.zeros(3), Bm @ Bm.T).entropy()          # q = push-forward of N(0,I) by g(e) = B e
check("H[q] = H[p_Z] + log|det B| for the linear flow g(e) = B e", *close(H_q, H_base + np.linalg.slogdet(Bm)[1], 1e-12))
# 2. forward KL covers both modes, reverse KL picks one (Gaussian fit to a two-mode target)
grid = np.linspace(-9, 9, 3001); dx = grid[1] - grid[0]
target = 0.5 * scipy.stats.norm.pdf(grid, -3, 0.7) + 0.5 * scipy.stats.norm.pdf(grid, 3, 0.7)
def kls(a):
    q = scipy.stats.norm.pdf(grid, a[0], np.exp(a[1])) + 1e-300
    return np.sum(target * np.log(target / q)) * dx, np.sum(q * np.log(q / (target + 1e-300))) * dx
fwd = scipy.optimize.minimize(lambda a: kls(a)[0], [0.3, 0.0], method="Nelder-Mead").x
rev = scipy.optimize.minimize(lambda a: kls(a)[1], [0.3, 0.0], method="Nelder-Mead").x
check("forward-KL fit sits between the modes (mode covering)", abs(fwd[0]) < 0.05 and np.exp(fwd[1]) > 2.5, f"mean {fwd[0]:.3f}, sd {np.exp(fwd[1]):.3f}")
check("reverse-KL fit locks onto one mode (mode seeking)", abs(abs(rev[0]) - 3) < 0.1 and np.exp(rev[1]) < 1.0, f"mean {rev[0]:.3f}, sd {np.exp(rev[1]):.3f}")
# 3. logit pre-processing log-determinant vs autograd
alpha = 0.05
pre = lambda x: torch.logit(alpha + (1 - 2 * alpha) * x)
xpix = torch.rand(4, dtype=torch.float64)
Jp = torch.autograd.functional.jacobian(pre, xpix)
pp = alpha + (1 - 2 * alpha) * xpix
check("logit pre-processing: log|det| = sum log[(1-2a)/(p(1-p))]", *close(torch.linalg.slogdet(Jp)[1].item(), torch.log((1 - 2 * alpha) / (pp * (1 - pp))).sum().item(), 1e-12))
# 4. bounded log-scale
gamma = 1.7; s_hat = torch.linspace(-50, 50, 1001, dtype=torch.float64)
s = torch.tanh(s_hat) * math.exp(gamma)
check("s = tanh(s_hat) e^gamma is bounded by e^gamma, so exp(s) cannot overflow", bool((s.abs() <= math.exp(gamma)).all()) and torch.isfinite(torch.exp(s)).all().item())''')

    # ------------------------------------------------------------------------------------------- 5.10
    c.eq(["EQ-M-039", "EQ-S4-029"], "Data on a manifold push the log-determinant to $-\\infty$",
         r"""\log\Big|\det\frac{\partial g_\theta}{\partial\mathbf{z}}\Big|\to-\infty
         \iff\log\Big|\det\frac{\partial f_\theta}{\partial\mathbf{x}}\Big|\to+\infty
         \iff\log p_\theta(\mathbf{x})\to+\infty\ \text{on the data}""",
         plain=(
             "A flow has as many latent dimensions as data dimensions. If the data really live on a thinner "
             "set (a curve in the plane, say), the only way a bijection of the whole plane can put all its mass "
             "on that curve is to squeeze space infinitely hard onto it: the generative map's volume factor goes to "
             "zero (log-determinant to $-\\infty$) and the density on the curve to infinity. That makes "
             "likelihoods ill-conditioned and is one reason flows can give higher likelihood to out-of-distribution "
             "images."),
         refresher=(
             "Formalisation: let the data be $\\mathcal{N}(0,\\mathrm{diag}(1,\\varepsilon^2))$, a Gaussian "
             "collapsing onto the $x_1$ axis as $\\varepsilon\\to0$. The maximum-likelihood linear flow is the "
             "whitening map $f(\\mathbf{x})=\\Sigma^{-1/2}\\mathbf{x}$ with "
             "$\\log|\\det J_f|=-\\tfrac12\\log\\det\\Sigma=-\\log\\varepsilon\\to+\\infty$, equivalently "
             "$\\log|\\det J_g|=\\log\\varepsilon\\to-\\infty$ by the inverse-function theorem, and the "
             "log-density on the axis grows like $-\\log\\varepsilon$. Nothing is wrong with the formula; "
             "the mismatch is between a full-dimensional density and degenerate data (Nalisnick et al., 2019 "
             "document the empirical consequence for Glow). The two statements of the entry are the same "
             "statement in the two directions of the map, which the check confirms along with the limit."),
         code=r'''eps_list = [1.0, 0.1, 0.01, 1e-3, 1e-4]
rows = []
for e in eps_list:
    Sigma = np.diag([1.0, e ** 2])
    Jf = np.linalg.inv(scipy.linalg.sqrtm(Sigma).real)        # f(x) = Sigma^{-1/2} x, the ML linear flow
    Jg = np.linalg.inv(Jf)
    logdet_f = np.linalg.slogdet(Jf)[1]; logdet_g = np.linalg.slogdet(Jg)[1]
    logp_on_axis = scipy.stats.multivariate_normal(np.zeros(2), Sigma).logpdf([0.7, 0.0])
    rows.append((e, logdet_f, logdet_g, logp_on_axis))
    check(f"eps={e:g}: log|det J_g| = -log|det J_f| (inverse-function theorem)", *close(logdet_g, -logdet_f, 1e-9))
    check(f"eps={e:g}: log|det J_f| = -log eps", *close(logdet_f, -math.log(e), 1e-9))
ld_g = [r[2] for r in rows]; lp = [r[3] for r in rows]
check("log|det J_g| decreases without bound as the data collapse onto a line", all(np.diff(ld_g) < 0) and ld_g[-1] < -9, f"{[round(v, 2) for v in ld_g]}")
closed = [-math.log(2 * math.pi) - math.log(e) - 0.5 * 0.7 ** 2 for e in eps_list]   # log N([0.7, 0]; 0, diag(1, eps^2))
check("the log-density on the data grows without bound, as -log eps + const", all(np.diff(lp) > 0) and close(lp, closed, 1e-9)[0],
      f"{[round(v, 2) for v in lp]} = -log eps - {math.log(2 * math.pi) + 0.245:.3f}")''')

    # ------------------------------------------------------------------------------------------- 5.11
    c.eq(["EQ-S4-028"], "The combination that defines the family: exact log-density and one-pass sampling",
         r"""\text{coupling flow: }\log p_\theta(\mathbf{x})\ \text{exact in one pass, and }\mathbf{x}=g_\theta(\mathbf{z})\ \text{in one pass};\qquad
         \text{autoregressive: exact }\log p\text{, }D\text{ sequential passes to sample};\quad
         \text{VAE/diffusion: a bound};\quad \text{GAN: no density}""",
         plain=(
             "The paper says flows are the only family in its taxonomy with both an exact, efficient "
             "log-density and sampling in a single network pass. Autoregressive models also have an exact "
             "likelihood but must generate one coordinate at a time; VAEs and diffusion models report a bound; "
             "GANs have no density at all. This is a statement about the taxonomy's coding, so the check is a "
             "small computational one: count network passes in each direction for a coupling flow and for an "
             "autoregressive flow of the same size."),
         refresher=(
             "'Efficient' means a cost polynomial and low in $D$ for an arbitrary external input; 'one pass' means "
             "the number of sequential conditioner evaluations is independent of $D$ (depth is fixed). The claim "
             "is qualified in the registry ('yes with wording fix') because flow matching, coded under diffusion by "
             "training signal, also has an exact density through an ODE solve, which is efficient but not "
             "one-pass, and because autoregressive flows (MAF) are exact but sequential in one direction; the "
             "paper's wording 'coupling flows' and 'in our taxonomy' carries those qualifications. The check "
             "instruments a coupling flow and a MAF-style flow with call counters."),
         code=r'''torch.manual_seed(4)
D = 8; depth = 3
class Counter:
    def __init__(self): self.n = 0
cnt = Counter()
mlps = [torch.nn.Sequential(torch.nn.Linear(D // 2, 8), torch.nn.Tanh(), torch.nn.Linear(8, D)).double() for _ in range(depth)]
def cond(k, hA):
    cnt.n += 1
    out = mlps[k](hA); s, t = out[:D // 2], out[D // 2:]
    return torch.tanh(s), t
def flow_fwd(x):            # x -> z, alternating halves
    h = x.clone(); ld = 0.0
    for k in range(depth):
        a, b = (slice(0, D // 2), slice(D // 2, D)) if k % 2 == 0 else (slice(D // 2, D), slice(0, D // 2))
        s, t = cond(k, h[a]); h = h.clone(); h[b] = h[b] * torch.exp(s) + t; ld = ld + s.sum()
    return h, ld
def flow_inv(z):            # z -> x
    h = z.clone()
    for k in reversed(range(depth)):
        a, b = (slice(0, D // 2), slice(D // 2, D)) if k % 2 == 0 else (slice(D // 2, D), slice(0, D // 2))
        s, t = cond(k, h[a]); h = h.clone(); h[b] = (h[b] - t) * torch.exp(-s)
    return h
x = torch.randn(D, dtype=torch.float64)
cnt.n = 0; z, ld = flow_fwd(x); n_density = cnt.n
cnt.n = 0; xr = flow_inv(z); n_sample = cnt.n
check("coupling flow: exact log-density uses `depth` conditioner passes, independent of D", n_density == depth, f"{n_density} passes")
check("coupling flow: sampling also uses `depth` passes (one pass per layer)", n_sample == depth, f"{n_sample} passes")
check("coupling flow round trip", *close(xr.detach().numpy(), x.detach().numpy(), 1e-12))
# autoregressive (MAF-style) flow with one masked layer: density in 1 pass, sampling in D passes
Wm = torch.randn(D, D, dtype=torch.float64) * torch.tril(torch.ones(D, D, dtype=torch.float64), -1)
maf_cnt = Counter()
def maf_cond(h):
    maf_cnt.n += 1; return torch.tanh(Wm @ h), torch.exp(0.3 * torch.tanh(Wm @ h))
maf_cnt.n = 0; m_, s_ = maf_cond(x); z2 = (x - m_) / s_; n_maf_density = maf_cnt.n
maf_cnt.n = 0; xr2 = torch.zeros(D, dtype=torch.float64)
for i in range(D):
    m_, s_ = maf_cond(xr2); xr2[i] = m_[i] + s_[i] * z2[i]
n_maf_sample = maf_cnt.n
check("autoregressive flow: density in 1 pass, sampling in D sequential passes", n_maf_density == 1 and n_maf_sample == D, f"{n_maf_density} vs {n_maf_sample} (D = {D})")
check("autoregressive round trip", *close(xr2.detach().numpy(), x.detach().numpy(), 1e-12))''')

    # ------------------------------------------------------------------------------------------- 5.12
    c.eq(["EQ-S4-030"], "Probability density distillation: a flow student trained by reverse KL from a teacher",
         r"""\min_{\phi}\ D_{\mathrm{KL}}\big(p_{\mathrm{student}}\,\|\,p_{\mathrm{teacher}}\big)
         =\min_\phi\ \Big(-H[p_{\mathrm{student}}]-\mathbb{E}_{\mathbf{x}\sim p_{\mathrm{student}}}\big[\log p_{\mathrm{teacher}}(\mathbf{x})\big]\Big)""",
         plain=(
             "Parallel WaveNet trains a fast flow (the student, an inverse autoregressive flow that samples in one "
             "pass) to imitate a slow autoregressive teacher. The objective is the reverse KL: draw samples from "
             "the student, score them under the teacher, and add the student's entropy, which a flow knows "
             "exactly from its log-determinant (entry 5.9). No teacher samples are needed, only teacher "
             "log-densities, and the minimum is reached when the student equals the teacher."),
         refresher=(
             "$D_{\\mathrm{KL}}(q\\|p)=\\mathbb{E}_q[\\log q]-\\mathbb{E}_q[\\log p]=-H[q]-\\mathbb{E}_q[\\log p]$; "
             "for a flow student $H[q]$ is available in closed form up to a Monte-Carlo average of log-determinants, "
             "and $\\log p_{\\mathrm{teacher}}$ is cheap to evaluate in parallel for a given input even though "
             "sampling the teacher is sequential. Being a reverse KL, the objective is mode-seeking (van den Oord "
             "et al., 2018, add power and perceptual losses to counter it). The student's density remains exact "
             "but, as the main paper notes, evaluating it on external data costs one sequential pass per "
             "dimension for an IAF. Check: a linear flow student distils a Gaussian teacher; the reverse-KL "
             "objective computed from student samples and the entropy identity equals the closed-form Gaussian "
             "KL, is minimised at the teacher's parameters, and its gradient needs no teacher samples."),
         code=r'''rng = np.random.default_rng(8)
mu_t, sd_t = 2.0, 1.5                                  # teacher N(mu_t, sd_t^2)
def rev_kl_from_samples(a, n=200_000):
    m, log_s = a
    eps = rng.standard_normal(n); x = m + np.exp(log_s) * eps     # student = flow g(eps) = m + s eps
    H = 0.5 * math.log(2 * math.pi * math.e) + log_s               # H[base] + log|det J_g|
    return -H - np.mean(scipy.stats.norm.logpdf(x, mu_t, sd_t))
closed = lambda a: math.log(sd_t / np.exp(a[1])) + (np.exp(a[1]) ** 2 + (a[0] - mu_t) ** 2) / (2 * sd_t ** 2) - 0.5
a0 = np.array([0.0, 0.0])
check("reverse KL from student samples + entropy identity = closed-form KL", *close(rev_kl_from_samples(a0), closed(a0), 0.01, "MC with n=2e5; "))
opt = scipy.optimize.minimize(closed, a0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-12}).x
check("the minimiser is the teacher (student = teacher)", *close([opt[0], np.exp(opt[1])], [mu_t, sd_t], 1e-4))
check("KL(student || teacher) = 0 at the optimum and > 0 elsewhere", abs(closed(opt)) < 1e-8 and closed(a0) > 0.1, f"{closed(opt):.2e} vs {closed(a0):.3f}")
# the gradient uses only student samples and the teacher's density (no teacher samples): autograd vs the closed form
at = torch.tensor([0.3, 0.2], dtype=torch.float64, requires_grad=True)
epsn = torch.randn(400_000, dtype=torch.float64)
xs_ = at[0] + torch.exp(at[1]) * epsn                                   # student samples, reparameterised
log_teacher = -0.5 * ((xs_ - mu_t) / sd_t) ** 2 - math.log(sd_t * math.sqrt(2 * math.pi))
loss = -(0.5 * math.log(2 * math.pi * math.e) + at[1]) - log_teacher.mean()
g_auto = torch.autograd.grad(loss, at)[0].numpy()
h = 1e-6; a1 = np.array([0.3, 0.2])
g_closed = np.array([(closed(a1 + h * e_) - closed(a1 - h * e_)) / (2 * h) for e_ in np.eye(2)])
check("gradient from student samples + teacher log-density = gradient of the closed-form KL", *close(g_auto, g_closed, 0.02))''')

    return c.build()


if __name__ == "__main__":
    print(build().path)
