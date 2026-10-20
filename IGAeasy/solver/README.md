# Linear solvers

Existing calls retain their behavior. Both solvers accept `scaling=False`,
`scaling=True`, or `scaling='auto'`.

```python
from IGAeasy.solver import direct_solver, iterative_solver

u = direct_solver.solve(K, f, scaling='auto')
u = iterative_solver.solve(K, f, preconditioner='ILU', scaling='auto',
                           rtol=1e-10, atol=0.)
```

Symmetric diagonal equilibration solves `D K D y = D f`, with
`D_ii = 1/sqrt(K_ii)`, then returns `u = D y`. It preserves symmetry and
positive definiteness when these are present. A positive diagonal alone
does not establish positive definiteness.

`auto` applies this transformation whenever the active diagonal is positive;
otherwise it skips it. `True` rejects nonpositive diagonals. When scaling is
enabled, inactive detection removes only exactly zero rows and columns.
Small nonzero cut-cell DOFs are retained even if their entries fall below `tol`.

The iterative solver selects GMRES for ILU. General ILU cannot be used with CG.
With scaling enabled, the returned coefficients must also pass a residual
check in the original reduced system, normalized by its largest entry. This
check allows a factor ten relative to `max(atol, rtol*norm(f))`.
Residual convergence alone does not guarantee an accurate physical field in
ill-conditioned systems; use a direct reference or independent error norm.

For the tested trimmed membrane, scaling plus ILU substantially reduces
iterations while preserving the physical solution. This is a recommendation
for similar systems, not an automatic choice of solver or ILU parameters.
Benchmark larger systems including factorization time and memory first.
Diagonal scaling does not remove near-linear dependence of restricted bases.

`direct_solver.solve(..., condnum=True)` still reports the condition number of
the original reduced matrix before diagonal equilibration. This option
explicitly permits densifying sparse matrices and is unsuitable for large
systems.

### Overlapping additive Schwarz

```python
from IGAeasy.preconditioning import element_dof_blocks, SchwarzPreconditioner
from IGAeasy.solver.iterative_solver import solve

blocks = element_dof_blocks(RS, dof_per_cp=3)
u = solve(K, f, preconditioner='Schwarz', dof_blocks=blocks, scaling=True)
# For an externally managed solver, use indices of its actual matrix:
operator = SchwarzPreconditioner(K, blocks).as_linear_operator()
```

The solver accepts original global block indices and remaps them when inactive
DOFs are removed. After other reductions (e.g. elimination of prescribed DOFs
or multipoint constraints), supply blocks for that reduced matrix instead.
The class depends only on the matrix and integer index lists. Spline kernels,
control points and approximation spaces are unchanged.

The operator sums local inverse corrections. Duplicate blocks are ignored;
uncovered DOFs receive diagonal singleton corrections. Each block is scaled
locally and Cholesky-factorized. Singular or indefinite blocks raise an error;
there is no implicit regularization or removal of basis functions.
The matrix must be symmetric; CG additionally requires an SPD global system.
GMRES is also selectable, but does not relax the local SPD requirement.
Dense local factors can consume considerable memory for high degrees and many
overlapping blocks. This is an optional preconditioner, not an automatic
replacement for ILU, nor a remedy for inaccurate assembly or excessive Nitsche
penalties. Physical error norms remain necessary for small-cut studies.
