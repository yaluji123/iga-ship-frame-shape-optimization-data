# Manufactured parabolic shell calculations

This folder provides 224 fresh assembly and equilibrium solutions from the same four method executable used for the original sweep. The reference displacement errors, signed stress errors, recovered energies, per face stress integrals and recovered energy to work differences match the original numerical records exactly. Table 1 uses the original matched forward timings in the accompanying manufactured shell table. The fresh reruns tested numerical repeatability.

## Reproducing the checks

Run the following from this directory.

    python verify_records.py
    python recompute_shells.py

The first command verifies the complete sweep against ../manufactured_shells.csv and reproduces the selection for Table 1. The second command builds the spline bases, kinematics, operators, load and response from scratch for the three manufactured fields and four methods at 2 × 2 cells and 10 mm thickness. It verifies the resulting displacement and signed stress errors and the recovered energy to work difference against the corresponding fresh records. It uses NumPy and SciPy and does not call the original C++ executable. Its numerical result comparison uses relative tolerance 5 × 10⁻⁶ and absolute tolerance 5 × 10⁻¹⁰. The recorded timings use the compiled C++ implementation and are independent of the Python runtime.

## Geometry, boundary condition and loading

The midsurface is (s, 250v, 0.0005s²/2) in mm, with s = 1000u and 0 ≤ u,v ≤ 1. Use open uniform cubic tensor splines with unit weights and simple interior knots. The coefficients in the first longitudinal row are fixed in the three Cartesian directions. The second row is fixed in the Cartesian z direction, enforcing zero normal slope at u = 0. The material is linear elastic with E = 210000 MPa and ν = 0.3. Use thicknesses of 10, 1 and 0.1 mm for the two bending fields and 10 mm for membrane shear. The mesh catalog is 2, 4, 8 or 16 longitudinal spans crossed with 2 or 4 transverse spans.

The independent manufactured load equals the continuous weak form applied to the exact membrane strain and curvature and uses the compatible displacement variations in all four methods. Assemble stiffnesses with 8 × 8 Gauss points per cell. Assemble the independent load and evaluate displacement, both signed face stresses and recovered energy with 16 × 16 Gauss points per cell. In each arm and field, a representative case checked the stiffness with 12 × 12 versus 8 × 8 and the load with 24 × 24 versus 16 × 16. All twelve checks meet 10⁻¹⁰ in relative Frobenius matrix norm or Euclidean load norm. The exact manufactured fields and material transformations are executable in recompute_shells.py.

The physical basis uses the tangential projection of the Cartesian x direction as its first axis. B_m and B_b produce membrane strains and signed changes in curvature, respectively, with engineering shear or twist in the third slot. Recovered stresses are C(e − ζκ) at ζ = ±t/2. The two face tensor stress norm uses weights (1, 1, 2) on the squares of the two normal and one shear stress components.

## The four discrete operators

* **Compatible cubic (`compatible`).** Assemble direct membrane and bending operators B_mᵀD_m B_m and B_bᵀD_b B_b with physical surface measure.
* **Material projection (`native`).** Project the orthonormal physical membrane components into Q(2,3), Q(3,2) and Q(2,2), formed by dropping one endpoint knot from each end in each reduced direction. Assemble M = ∫ΦᵀD_mΦ and G = ∫ΦᵀD_m B_m using physical surface measure. Scale the basis columns with the inverse square roots of the Gram diagonal and solve the normal block and engineering shear block. The condensed contribution is GᵀM⁻¹G. Bending remains compatible.
* **Approximate dual for membrane strains (`dual`).** Use the same physical component trial spaces and use tensor approximate dual splines for testing, with the degree two and three matrices obtained from the one dimensional knot recurrence. For a field F, with Φ its trial basis and Φ̃ its approximate dual, the modified test functions are D_F⁻ᵀΦ̃/J. The material and geometry factors cancel in the coefficient map. Let L be the diagonal of row sums of ∫Φ̃ᵀΦ du dv. Compute Z = L⁻¹∫Φ̃ᵀB_F du dv. All entries of L are positive and agree with one within 10⁻¹⁰. The stiffness contribution is GᵀZ, where Gᵀ = ∫B_FᵀD_FΦ J du dv. Bending remains compatible.
* **Approximate dual for both fields (`full_dual`).** Apply the preceding procedure to membrane strains and curvatures separately, with the same scalar spaces and physical frame and their respective material blocks. Sum the two condensed contributions. Keep the resulting unsymmetric operator.

The approximate dual arms are linear implementations of Nguyen et al., arXiv:2406.16685v4, Eqs. (42)–(43), (63)–(73), with the membrane variant specified in Remark 6.3. The same definitions appear in v5. The present calculations use the stated physical component convention throughout. [2025 version](https://arxiv.org/html/2406.16685v4) · [2026 revision](https://arxiv.org/html/2406.16685v5)

## Timings and comparison

The original Table 1 runs used the same optimized C++17 binary (Microsoft C++ with optimizations and precise floating point, Eigen 3.4.0, one numerical thread with four logical processors available). The free equilibrium uses one sparse LU factorization and up to four refinement corrections with compensated accumulation. The forward clock starts as the per case carrier is constructed. It covers carrier construction, projection, stiffness and load assembly, factorization and the equilibrium solve. It ends before stress recovery, reference integration checks and positive control work. For the full dual arm the clock includes both projection assembly passes. The original tabulation takes the lowest recorded forward seconds that meet 1% displacement and signed face stress errors together. The fresh residuals and error values qualify the same states.

## Equilibrium, internal action and energy

Let F denote the free displacement rows, U the solved coefficients and f the independent weak load.

* **Free equilibrium** (`true_free_residual`): ||(KU − f)_F||₂ / ||f_F||₂.
* **Free residual with the recovered internal action** (`actual_U_source_operator_free_residual`): ||(f_int^rec − f)_F||₂ / ||f_F||₂. Apply the projected contributions through their actual coefficient maps and physical coupling, and apply each compatible contribution through its assembled operator.
* **Discrete bilinear work** (`bilinear_work_closure_relative`): |E_d − W/2|/E_d, with W = Uᵀf and E_d = UᵀKU/2 obtained from the recorded discrete membrane and bending forms. This is an algebraic equilibrium check for homogeneous prescribed displacements and also applies to the unsymmetric operators.
* **Recovered energy to work difference** (`physical_energy_work_gap_relative_diagnostic`): |E_rec − W/2|/E_rec, with E_rec the independently integrated energy of the recovered strains and curvatures (each term is 1/2 times its material quadratic form). This quantity compares the physical strain recovery with the work of the solved state.

The first two maxima over 224 solves are 9.9154934 × 10⁻¹¹ and the third is 9.9888500 × 10⁻¹¹. The maximum recovered energy to work difference is 0.0012724851 for full dual membrane shear on 2 × 2 cells at 10 mm thickness. For this state the recovered membrane and bending energies are 2112.527321949 and 0.02358588533 N mm, and W = 4230.478194880 N mm. The three discrete consistency measures are each below 2.2 × 10⁻¹⁵ in this case.

