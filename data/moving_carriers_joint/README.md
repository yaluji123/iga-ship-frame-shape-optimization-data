# Moving noncoplanar shell joint

Run:

    python both_carriers_joint.py

The program requires NumPy and SciPy. The reference calculation used Python 3.12, NumPy 1.26.4 and SciPy 1.13.1. Keep both_carriers_joint.py and moving_skew_gradient.py in the same directory. The script assembles and solves the reference configuration and eight real perturbations, evaluates the transpose adjoints at the reference configuration, verifies the reduction and calibration derivatives, and writes both_carriers_joint.json. The JSON stores the real states, reduction weights, branch checks, and all centered differences.

## Geometry and discretization

The two planes meet along a straight open segment. Let the dimensionless design coordinate be beta and let (xi, eta) range over [0, 1] × [−1, 1]. In mm, the common segment is

    x(xi) = beta (0.04, 0.05, −0.08) + (1.15 xi + 0.41 beta xi²) (1, 0, 0).

For carrier i add s_i (eta − t_i) (0, cos(phi_i), sin(phi_i)). The edge parameters t_i are −0.85 + 0.11 beta and 0.12 − 0.23 beta. The transverse scales are 1 + 0.19 beta and 1 − 0.13 beta, and the angles in radians are −0.13 + 0.27 beta and 0.79 − 0.31 beta. Carrier one is retained below t_1. Carrier two is untrimmed, with t_2 inside the patch. Both carriers use cubic B-splines, with three and five equal longitudinal spans and one transverse span each. The physical frame is tied to the longitudinal and transverse axes of each plane. The material parameters are E = 210000 MPa, Poisson ratio 0.3, and thickness 0.1 mm. Physical domain, trace, and traction weights use 10-point Gauss integration per interval. The trace integration is split at the union of the two longitudinal knot partitions.

Exterior edge loads are prescribed per unit reference xi, with intensity factor 1 + 0.3 xi + 0.24 sin(2 pi xi). The Cartesian load intensity vectors are (0.025, −0.018, 0.025) on eta = −1 of carrier one and (0.013, −0.058, −0.062) on eta = 1 of carrier two, in N per unit reference xi. Fix all three translations at (0, −1) and the global Y and Z translations at (1, −1) on carrier one. Fix the global Z translation at (0, 1) on carrier two.

## Reduction and verification

Classify a control point as weak if the three stiffness row norms are at most 0.001 of the maxima in their respective Cartesian components of the same patch. Reconstruct each weak displacement from supported controls of the same patch using the weighted affine moment equations, with the squared 1-plus-grid-distance cost. The two independent moment coordinates are X and Y on the first plane. Check the reproduction of all three Cartesian coordinates and its derivative. Recompute the supported/weak classification, donors, reduction weights, projection, both trace maps, signed endpoint work, length weights, and two spectral calibrations at every real perturbation. Each calibration uses the reduced active-side trace and bulk energy operators. The transpose adjoints reuse the reference factorization.

Compare complete derivatives with centered differences at four step sizes. The direct omission removes only the reduction derivative from the reduced residual and explicit response. The full reduction omission also removes its contribution to the interface calibration derivative. Both omissions keep the reference geometry, stiffness, stress field, state, adjoints, and all other shape derivatives unchanged. Each response uses the full coupled state and the KS aggregate over both faces or the exterior-load work.
