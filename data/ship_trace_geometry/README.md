# Coincidence and anchored traces in the ship-frame design model

The checked geometry is the undeformed spline representation. Its two moving
surfaces share a prescribed physical curve. The receiving strip is evaluated
on its transverse centreline, with its longitudinal coordinate reversed to
match the curve parameter. The trimmed web uses the affine reference trace
on its refined parent surface. Both contain the same three design modes.

On each span between trace knot crossings, the aligned trace polynomials
have degree at most six. Seven values determine their Bernstein
coefficients. The norm of each curve difference is bounded over a span by
the maximum norm of its difference coefficients. The shared curve has 44
spans and is also evaluated directly at 4044 points. For the 17 anchored
traces, zero motion is evaluated over the follower-retained portions of
the archived owner curves, using 1001 points per retained interval and 39
additional exported reference points per trace. A numerical allowance
for the difference between the affine restriction and the stored owner
curves is added using derivative coefficient bounds of the design fields.
All reported bounds use Euclidean norms in millimetres. The 1e-6 mm
threshold is a numerical geometry tolerance.

The unscaled anchor matrix is 546 by 645. At relative SVD cutoff 1e-12 its
rank is 92, leaving a null space of dimension 553. The shared-curve
restriction to this null space is 308 by 553 and has rank 173 at the same
relative cutoff. The relative residual for each target mode is
||F - U_r U_r^T F||_F / ||F||_F. The absolute coefficient residual is
the largest Euclidean norm over the Bernstein rows. The anchor bound at
a given shape uses the sum of the three mode bounds weighted by the
absolute shape coefficients.

- trace_summary.json gives the ranks, residuals and results for the
  straight frame, prepared stress-priority start, both 0.05% endpoints,
  both 0.04% stationary endpoints and the separate 0.04% first-feasible
  design.
- trace_residuals.npz contains anonymized singular spectra, the retained
  left singular vectors, target mode coefficients, difference fields
  and anchor error budgets. It supports replay of the linear algebra
  and geometry bounds without the original surface control meshes.
- check_geometry.py uses NumPy to recompute and check these reported
  quantities. Run python check_geometry.py in this directory.

The curve and anchor measures are numerical checks for the specified
design geometry, including the follower-retained part of each owner
curve. They do not specify exact opposite-side retained intervals or
post-fabrication tolerances.
