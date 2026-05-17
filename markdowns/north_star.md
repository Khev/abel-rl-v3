# North Star: RL for Discovering Symbolic Transformations of Differential Equations

## Core idea

The big research direction is **not**:

> RL solves algebraic equations slightly worse than SymPy.

The stronger direction is:

> **RL learns to search over symbolic transformations, ansätze, reductions, and conserved quantities for exact mathematical systems.**

Equation solving is the toy domain. The real target is **AI-assisted discovery of exact reductions for ODEs and PDEs**.

---

## One-sentence north star

Build an RL/search system that can rediscover known symbolic transformations in mathematics — and eventually discover new transformations or reductions for nonlinear ODEs/PDEs.

---

## Why this is exciting

Symbolic math has properties that make it unusually well-suited for RL:

1. **Exact verification**  
   A proposed transformation either works or does not. We can substitute it back and check exactly.

2. **Replayable traces**  
   A successful solution path can be stored and reused as supervised training data.

3. **Natural search tree**  
   Mathematical reasoning is naturally a sequence of transformations.

4. **Clean difficulty ladder**  
   We can start from algebra, then move to ODE substitutions, PDE ansätze, conservation laws, and eventually new reductions.

5. **Non-LLM reasoning benchmark**  
   This gives a clean, exact testbed for learned symbolic reasoning without relying on language-model tricks.

---

## Better framing

Avoid framing the work as competing with CAS systems.

Weak framing:

> We use PPO to solve equations.

Strong framing:

> We study learned symbolic search over exact mathematical transformation systems.

Even stronger:

> We train agents to discover transformations that simplify nonlinear equations.

---

## Research ladder

### Level 1: Closed algebraic equations

Current domain.

Examples:

\[
ax+b=0,
\qquad
\frac{ax+b}{cx+d}+e=0,
\qquad
\log(ax+b)+c=0.
\]

The required terms already exist in the expression tree. The agent learns to rearrange, expand, collect, divide, and apply inverse functions.

**Goal:** show PPO/TreeMLP/search can solve closed symbolic equations.

---

### Level 2: Neural-guided symbolic search

Upgrade current PPO setup with:

- value-guided beam search,
- expert iteration,
- supervised pretraining from successful traces,
- factorized action policies,
- learned cost-to-go.

This shifts the contribution from “PPO works” to:

> A neural policy learns to guide symbolic derivation search.

This is much stronger.

---

### Level 3: Open equations and substitution discovery

This is the big jump.

Examples:

\[
ax^2 + bx + c = 0
\]

requires inventing terms like:

\[
\frac{b}{2a},
\qquad
\left(\frac{b}{2a}\right)^2.
\]

This is no longer closed-form rearrangement over existing subterms. It requires **generative symbolic reasoning**.

Potential transformations:

| Equation family | Desired learned move |
|---|---|
| Quadratic | Complete the square |
| Cubic/quartic | Depress the polynomial |
| Exponential pair | Substitute \(u=e^{kx}\) |
| Rational equation | Clear denominators / substitute denominator |
| Trig equation | Use inverse trig or identities |
| Radical equation | Square / rationalize carefully |

**Goal:** learn change-of-variable policies.

---

### Level 4: ODE transformations

Move from algebraic equations to differential equations.

Classic targets:

| ODE type | Desired transformation |
|---|---|
| Bernoulli ODE | Power substitution |
| Riccati ODE | Linearize to second-order ODE |
| Exact equations | Find integrating factor |
| Homogeneous ODEs | Ratio substitution \(v=y/x\) |
| Autonomous ODEs | Reduce order |
| Special-function ODEs | Recognize Airy/Bessel/Painlevé structure |

**Goal:** have the agent rediscover textbook ODE transformations.

---

### Level 5: PDE reductions

Classic PDE moves:

| PDE | Desired discovery |
|---|---|
| Burgers equation | Cole–Hopf transform |
| Heat/wave equation | Separation of variables |
| KdV | Traveling-wave / soliton ansatz |
| Fisher–KPP | Traveling-wave reduction |
| Nonlinear Schrödinger | Soliton ansatz |
| Reaction–diffusion equations | Similarity reductions |
| Vlasov/Boltzmann-type equations | Moment closures / entropy structures |

**Goal:** agent proposes transformations or ansätze that reduce PDEs to simpler ODEs or algebraic constraints.

---

### Level 6: Conserved quantities and Lyapunov functions

Instead of solving the equation directly, ask the agent to find quantities like:

\[
E[u] = \int \mathcal{E}(u,u_x,u_{xx},\dots)\,dx
\]

such that:

\[
\frac{dE}{dt}=0
\]

or:

\[
\frac{dE}{dt}\le 0.
\]

This may be more scientifically valuable than explicit solutions.

**Goal:** discover conserved quantities, monotone quantities, or Lyapunov-like functionals.

---

### Level 7: New reductions for swarmalator continuum models

This is the personal/research-home target.

Use RL symbolic search to look for:

- order-parameter closures,
- invariant manifolds,
- low-rank ansätze,
- exact async / phase-wave / mixed-state reductions,
- stability-relevant coordinate transforms,
- conserved or Lyapunov-like quantities,
- continuum reductions of swarmalator density PDEs.

**Goal:** discover a new exact or semi-exact reduction in a young field where Kevin has domain expertise.

---

## Important ODE/PDE families without general closed-form solutions

These are not first targets, but they are useful north-star examples.

| System | Why important | How RL could help |
|---|---|---|
| Navier–Stokes / Euler | Fluids, turbulence, aerospace, weather | Search for special solutions, invariants, reductions, not full general solution |
| Boltzmann equation | Kinetic theory, statistical mechanics | Search for entropy structures, moment closures, special solutions |
| Vlasov–Maxwell | Plasma physics, fusion, astrophysics | Search for conserved quantities and reduced ansätze |
| Three-body / N-body ODEs | Celestial mechanics and chaos | Search for special orbits, invariants, canonical transformations |
| Keller–Segel / chemotaxis | Aggregation, blow-up, active matter | Search for blow-up profiles, self-similar reductions, Lyapunov functionals |
| Nonlinear Schrödinger / Gross–Pitaevskii | Optics, BECs, waves | Search for soliton ansätze and invariant reductions |
| Painlevé equations | Nonlinear special functions | Search for parameter regimes with rational/algebraic/classical solutions |
| Swarmalator PDEs | Kevin’s home field | Search for closures, reductions, stability transforms |

---

## Proposed MDP for symbolic ODE/PDE transformation

### State

A symbolic mathematical object, for example:

- algebraic equation,
- ODE,
- PDE,
- system of equations,
- candidate ansatz,
- transformed equation,
- residual after substitution.

### Actions

Possible action classes:

| Action type | Examples |
|---|---|
| Algebraic manipulation | expand, collect, simplify, factor |
| Inverse operation | log, exp, square, sqrt, asin |
| Substitution | \(u=e^x\), \(v=y/x\), \(u=x-b/2a\) |
| Ansatz | \(u(x,t)=U(x-ct)\), \(u=t^{-\alpha}F(x/t^\beta)\) |
| Symmetry reduction | scaling, translation, rotation invariance |
| Conservation-law proposal | propose density \(\mathcal{E}\) |
| Integrating factor | multiply by \(\mu(x,y)\) |
| Linearization | Riccati-to-linear, Cole–Hopf-like transforms |
| Special-function recognition | Airy, Bessel, elliptic, Painlevé |
| Moment closure | derive finite ODE system for moments/order parameters |

### Transition

Apply the symbolic transformation using a CAS backend such as SymPy.

### Reward

Possible reward terms:

- verified solution or reduction,
- reduced differential order,
- reduced number of independent variables,
- lower expression complexity,
- transformed PDE becomes ODE,
- residual vanishes after ansatz,
- conserved quantity verified,
- shorter derivation trace,
- better generalization to parameterized equation families.

### Terminal condition

Examples:

- equation solved,
- ODE reduced to quadrature,
- PDE reduced to ODE,
- residual exactly zero,
- conserved quantity verified,
- known canonical form reached,
- new valid transformation found.

---

## First experiments to run

### Experiment 1: Rediscover textbook ODE transformations

Train on families where the correct transformation is known.

Examples:

- Bernoulli equation,
- Riccati equation,
- homogeneous first-order ODE,
- exact equations with integrating factors,
- separable equations disguised by algebraic transformations.

Success metric:

> Does the agent recover the known transformation and solve held-out parameterized instances?

---

### Experiment 2: Rediscover Cole–Hopf

Target:

\[
u_t + u u_x = \nu u_{xx}.
\]

Desired transformation:

\[
u = -2\nu \partial_x \log \phi.
\]

Success metric:

> Does the agent transform Burgers’ equation into the heat equation, or verify a Cole–Hopf-like residual reduction?

---

### Experiment 3: Traveling-wave reductions

For PDEs like KdV, Fisher–KPP, NLS, reaction–diffusion equations, ask the agent to propose:

\[
u(x,t)=U(x-ct).
\]

Success metric:

> Does the PDE reduce to an ODE with lower complexity?

---

### Experiment 4: Self-similar solutions

Ask the agent to propose:

\[
u(x,t)=t^{-\alpha}F(x/t^\beta).
\]

Success metric:

> Does it identify exponents \(\alpha,\beta\) that reduce the PDE to an ODE?

---

### Experiment 5: Conservation-law discovery

For PDE:

\[
u_t = F[u],
\]

search for densities:

\[
\mathcal{E}(u,u_x,u_{xx},\dots)
\]

such that:

\[
\frac{d}{dt}\int \mathcal{E}\,dx=0.
\]

Success metric:

> Exact symbolic verification of conservation.

---

### Experiment 6: Swarmalator continuum reduction

Apply the system to Kevin’s own continuum equations.

Search for:

- closures for \(r,s\),
- invariant manifolds,
- exact async/phase-wave/mixed-state ansätze,
- Lyapunov-like quantities,
- low-dimensional reductions.

Success metric:

> A reduction that can be verified analytically and compared against simulation.

---

## Implementation path

### Phase A: Strengthen current algebra solver

Do before jumping to ODE/PDEs.

1. Add value-guided beam search.
2. Save successful traces to disk.
3. Add supervised pretraining on traces.
4. Add expert iteration.
5. Add factorized operation/term action scoring.
6. Test generalization to deeper equations.

### Phase B: Add generative symbolic terms

Needed for open equations.

Actions should allow the agent to synthesize terms from a grammar:

```text
term :=
    constant
  | variable
  | subtree
  | term + term
  | term * term
  | term / term
  | term^2
  | sqrt(term)
  | log(term)
  | exp(term)
```

This enables completing the square, substitutions, and generated ansätze.

### Phase C: Move to ODEs

Start with known transformations.

State representation should include:

- expression tree,
- derivative nodes,
- dependent/independent variables,
- order of derivatives,
- candidate substitutions.

### Phase D: Move to PDE reductions

Add actions for:

- traveling wave,
- similarity ansatz,
- separation of variables,
- conservation-law proposal,
- dimensional reduction.

### Phase E: Apply to original science

Use the system to attack swarmalator continuum models and related active-matter PDEs.

---

## Paper sequence

### Paper 1: RL for closed symbolic equation solving

Core contribution:

> PPO/TreeMLP/replay/search can solve symbolic equations by learning transformation policies.

Likely level:

- workshop,
- AI-for-math venue,
- symbolic reasoning venue.

---

### Paper 2: Expert iteration for symbolic transformation search

Core contribution:

> Neural-guided search plus replayed exact traces improves symbolic equation solving and generalizes to deeper equations.

Likely stronger ML paper.

---

### Paper 3: Learned change-of-variables for open equations

Core contribution:

> RL discovers substitutions and generated terms needed for open symbolic equations.

This is a much more serious contribution.

---

### Paper 4: RL-discovered ODE/PDE reductions

Core contribution:

> RL rediscovers classic transformations and discovers candidate reductions for nonlinear differential equations.

This is the true north-star paper.

---

### Paper 5: RL-assisted discovery in swarmalator PDEs

Core contribution:

> The system discovers or helps verify a new reduction/closure/conserved quantity for swarmalator continuum equations.

This would connect the symbolic RL project back to Kevin’s scientific home territory.

---

## Risks

### Risk 1: “Why not just use SymPy?”

Mitigation:

- Do not frame as a CAS replacement.
- Frame as learned symbolic search.
- Emphasize generalization, trace discovery, and transformation learning.

### Risk 2: Search space explosion

Mitigation:

- Use expert iteration.
- Use value-guided beam search.
- Use grammar constraints.
- Use exact verification.
- Use curriculum over equation families.

### Risk 3: RL instability

Mitigation:

- Use supervised trace pretraining.
- Use replay.
- Use value-guided search.
- Treat PPO as one component, not the whole method.

### Risk 4: Too ambitious too soon

Mitigation:

- Start with algebra.
- Then ODE textbook transformations.
- Only then PDE reductions.
- Use known transformations as benchmarks before claiming discovery.

---

## The big-picture claim

The project is strongest if framed as:

> **A reinforcement-learning system for discovering symbolic transformations in exact mathematical environments.**

The equation solver is the first demonstration, not the endpoint.

---

## Short version for a grant/paper intro

We propose to study mathematical reasoning as reinforcement learning over exact symbolic transformation systems. In this framework, states are symbolic equations or differential equations, actions are algebraic transformations, substitutions, ansätze, and reductions, and rewards are given by exact verification of simplification, solution, or conservation. Starting from algebraic equation solving, we aim to build agents that rediscover classical ODE/PDE transformations such as Riccati linearization, Cole–Hopf, traveling-wave reductions, and similarity solutions, and eventually discover new reductions or conserved quantities in nonlinear systems such as swarmalator continuum models.

---

## The actual north star

A system that, when handed a nonlinear equation, can say:

> “Try this transformation.”

and the transformation is mathematically valid, verifiable, and useful.

That would be genuinely cool.

