## (GCS:) [SHORTEST PATHS IN GRAPHS OF CONVEX SETS](https://arxiv.org/pdf/2101.11565) - 6/18/2024
### Problem Defn.
- $G := (V, \epsilon)$
- $v \in V$
- $X_v \subset \mathbb{R}^n$ - convex set for vertex $v$
- $x_v$ - point within $X_v$
- $e = (u,v) \in \epsilon$


<br /><hr /><br />

## (Fast IRIS:) [Faster Algorithms for Growing Collision-Free Convex Polytopes in Robot Configuration Space]() - 8/7/2024
### Probabalistic Guarantees on Fraction of Polytope in Collision
- Given polytope $\mathcal{P}$:
- Assume user-defined admissible fraction of final polytope in collision $\varepsilon$, true fraction in collsion $\varepsilon_{tr}$
- Claim: if $\varepsilon_{tr} \geq \varepsilon$ (i.e. more collision than admissible), $\mathbf{Pr}[\bar{X}_M \leq (1-\tau)\varepsilon] \leq \delta$ (i.e. probability of samples falsely concluding that $\mathcal{P}$ is sufficiently collison-free is $\leq \delta$)
   - $\delta$: confidence/allowed probability of being wrong (i.e. 5%)
   - $\bar{X}_M$: number of samples in collison over $M$ samples
      - $M = 2 \frac{\text{log}(1/\delta)}{\varepsilon \tau^2}$
   - $\tau$: user-tuned constant (typically 0.5); trades off more samples vs probability of rejecting $\mathcal{P}$
   - Math Intuition: Chernoff Bound: $\Pr\left[\bar{X}_M \leq (1 - \tau) \epsilon_{tr}\right] \leq e^{-M \epsilon_{tr} \tau^2 / 2}$; therefore, set $M$ so that $e^{-M \epsilon_{tr} \tau^2 / 2} = \delta$
- UnadaptiveTest($\delta, \varepsilon, \tau$) procedure: take $M$ samples from $\mathcal{P}$; *accept* if $\bar{X}_M \leq (1 - \tau) \epsilon$, else continue adding hyperplanes

### IRIS-ZO (Zero Order Optimization), aka Fast IRIS
- Does not need to iterate through collision pairs. Instead, samples $M$ configurations from $\mathcal{P}$; uses collision-checker to check if in collision or not. If in collision, bisection/binary search used to find closest point to center of ellipse in collision. Tangent hyperplane is then drawn to this point.
- Uses UnadaptiveTest($\delta, \varepsilon, \tau$) procedure (using the samples drawn in above bullet pt) to determine when enough hyperplanes have been added

### IRIS-NP2, aka Ray IRIS
- Slight Adaptation of IRIS-NP
- Simply seeds nonlinear program (NLP) differently: samples points in $\mathcal{P}$ until finding one in collision, uses linear search outward from ellipse center to find the closest collision point on the ray from ellipse center to the sample, then uses this collision point to seed the NLP.
   - Faster than IRIS-NP since IRIS-NP seeds NLP with uniform samples, lots of time wasted on points that are not near collision --> NLP cannot find solution.
- Also uses UnadaptiveTest($\delta, \varepsilon, \tau$) procedure (using the samples drawn in above bullet pt) to determine when enough hyperplanes have been added

<br /><hr /><br />

## Hit and Run Sampling: - 8/6/24
- Goal: randomly sample from a convex set. Over a large number of samples, converges to a uniform distribution.
- Is a Markov Chain--at each step, selecting the next sample only depends on current sample, not previous samples.
- Start with initial sample $x_0$ in convex set $S$.
- `for each iteration:`
   - Select random unit-vector direction $d$. Sample uniformly from a multi-variate normal distribution and normalize the result.
   - Define a line $L$ passing through $x_0$, and find the intersection of $L$ with the bounds of the $S$, $a$ and $b$. This involves finding the two values of $\lambda$ where $x_0 + \lambda d$ intersects with a hyperplane.
   - Randomly sample a new $x_0$ from $L$ by sampling between the two values of $\lambda$. This new $x_0$ is your new sample.

<br /><hr /><br />

## (Clique Covers:) [Approximating Robot Configuration Spaces with few Convex Sets using Clique Covers of Visibility Graphs](https://groups.csail.mit.edu/robotics-center/public_papers/Werner23.pdf) - 1/18/2024
### Method
- Goal: minimum cardinality (number of nodes) to maximize $\alpha$: fraction of collision-free config. space that is occupied by convex cover.
- Goal approximation: minimum number of cliques to maximize $\alpha$.
- *Visibility Graph*: undirected graph w/vertices and edges between all 2 vertices that can "see" each other.
- *Clique Cover*: Collection of cliques where each vtx in graph is in a clique.
- Algorithm:
   - Repeat until $\alpha$ reaches coverage threshold:
      - Sample points in $C^{free}$, construct visibility graph by checking collisions along line segments for each pair of sampled points.
         - Note that the convex regions from earlier iterations of alg. are not part of $C^{free}$ (to encourage exploration).
      - While true:
         - MaxClique: NP-complete but fast in practice; finds max clique in graph.
         - Removes clique from graph and adds to Clique Cover.
         - If max clique is smaller than a threshold, break.
      - MinVolumeEllipsoids: enclose each clique in ellipsoid (defined by center point and principal radii) of minimum volume; solved with SDP.
      - Basically run an iteration of IRIS using these ellipsoids as seeds $q_0$ for each IRIS region.
      - Check/estimiate $\alpha$ by sampling points in $C^{free}$, seeing what fraction fall in convex regions.


### Limitations
- Holes in $C^{free}$: if hole can be contained in clique, then convex regions may contain collisions. In practice, very rare.
- Solution (but very slow): When building MaxClique, also include all points in the convex hull of the max clique to be part of the clique. (The math in the paper is confusing.)


<br /><hr /><br />

## (Deits14:) [Computing Large Convex Regions of Obstacle-Free Space through Semidefinite Programming](https://groups.csail.mit.edu/robotics-center/public_papers/Deits14.pdf)
### Summary:
- Similar to IRIS-NP, but w/two more assumptions: 
   1) Obtacles are convex
   2) Obstacles are known
- This allows IRIS to do counter-example search without needing forward kinematics or NP, and just using a least-distance quadratic programming problem.
- Basically, only feasible in task-space (i.e. for drones).

<br /><hr /><br />

## (IRIS-NP:) [Growing Convex Collision-Free Regions in Configuration Space using Nonlinear Programming](https://arxiv.org/pdf/2303.14737.pdf) - 1/12/2024
### Method
- Assumptions: known collision geometries (in task-space).
- Goal: generate convex polytope w/max volume inscribed ellipsoid
   - <sub><sup>Note: calculating volume of polytope itself is NP-hard → ellipsoid as a heuristic</sup></sub>.
#### Generating 1 ellipsoid:
- Ellipsoid expressed like so: $ \epsilon(C,d) = {x | (x-d)^T C^T C(x-d) \leq 1} $.
   - $d$ is the center of the ellipsoid; $C$ is a symmetric positive definite matrix; eigenvalues and eigenvectors represent scale and direction of principle axes of the ellipsoid. (Note: bc $C$ is symmetric, it can be diagonalized into $QVQ^T$ where $Q$ is an orthogonal matrix with columns being the eigenvectors of $C$).
- Polytope expresed as collections of "halfplanes": $ P(A,b) = \{x|Ax \leq b\} $
- Initialization: seed $q_0$; $P_0$ is initialized w/robot joint limits. $\epsilon_0$ initialized as tiny hyperphere.
- Adding Separating Hyperplanes: Iterate over all pairs of collision bodies. For each pair of collision bodies, repeat until "counter-example search" repeatedly fails:
   - search for configurations within polytope resulting in collision ("counter-example search"). Add plane tangent to the ellipsoid at any collision point.
   - "counter-example search": Technically, solves (non-linear) optimization for nearest point to ellipsoid center that results in collision between two given collision bodies (performs forward kinematics to detect the collision in task-space $\rightarrow$ non-linearity). Can be geometrically understood as uniformly expanding ellipsoid until collision detected.
      - mathematical description: 

$$
\begin{aligned}
& \text{minimize}_{q,t} && \|q - c\|^2_E \\
& \text{subject to} && t \in \mathcal{A}(q) \cap \mathcal{B}(q), && q \in \mathcal{P}.
\end{aligned}
$$

<sub>Finding closest configuration q in the current polytope P to center of current ellipse c such that there is a point t that is in both collision bodies A and B at q.</sub>
   - if obstacles are convex in config. space (usually not the case unless you pre-decompose non-convex obstacles into convex parts), then tangent hyperplane guaranteed to separate collision from non-collision.
   - if obstacles non-convex in config. space, back the hyperplane away by user-defined margin $\delta$.
      - <img src="ReadingNotesSupplements/nonconvex_obstacles_backup.png" alt="" style="width:50%; margin-top: 10px"/>
      - makes hyperplane extra conservative, but ensures finite number of hyperplanes can guarantee the convex set is out of collision.
   - If "counter-example search" fails multiple times, break and move onto next collision pair (probably no more collisions in convex set; this is not guaranteed bc the "counter-example search" is a non-linear optimization $\rightarrow$ not guaranteed to find all solutions).
- Analytically calculate volume of inscribed ellipse.
- Repeat until ellipse growth rate too low.

<center><img src="ReadingNotesSupplements/polytope_generation_graphic.jpg" alt="" style="width:100%; margin-top: 10px"/></center><br />

- Speed Optimizations:
   - Sorting collision bodies: considering closest collision bodies first is better bc the closer hyperplane may separate further obstacles $\rightarrow$ fewer hyperplanes needed overall. Therefore, for each seed $q_0$, first consider collision bodies with closest task-space distance. This is a heuristic for the collision body distance in config. space.

#### Decomposing the entire config. space

The goal for a single polytope = maximize volume; but, if we have multiple polytope all maximizing volume, they'll end up ignoring small crevices or smaller areas. As a heuristic to encourage expnsion, each time an polytope is generated, treat it as an obstacle when generating the next polytope.

#### Region Refinement

Note: works only with the introduction of new obstacles, not the removal of existing obstacles.

Keep original regions. When new obstacle is introduced, start with some seed (can be either the already-generated ellipsoid for a nearby region or a new hypersphere at a new nearby $q_0$); perform just one round (i.e. only 1 iteration of the outer `while` loop in the graphic above) of adding hyperplanes, considering only collision pairs between the new obstacle and all other collision bodies.

<br /><hr /><br />

## [Motion Planning around Obstacles with Convex Optimization](https://arxiv.org/pdf/2205.04422.pdf) - 1/12/2024
### Background
- Traj. Opt. is optimal, but low in high dimensions, many obstacles (many non-convexities)
- Advantages of GCS:
   - takes advantage of sampling & optimization to achieve global optimality quickly in clutter & high dimensions
   - works with differential constraints (i.e. velocity/acceleration) (sampling algorithms have trouble with this due to discreet samples)


