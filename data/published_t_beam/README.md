# Flat T-beam benchmark

The files document the moving web of the flat T-beam in Zhao, Hwang and Chen (2024), *Computer Methods in Applied Mechanics and Engineering* 431, 117322, §5.1.1. [Article](https://doi.org/10.1016/j.cma.2024.117322)

- `model.json` defines the present bicubic reconstruction and retains the model conventions.
- `comparison_results.json` records the two SLSQP runs, the geometry and loading checks against the public example, and the curve values read from Figure 7(b). The center and start energies use the same observable ratio, each divided by the energy at the starting web position. The article's published terminal position and original normalized energy are recorded separately.
- `optimization_history.csv` gives the objective and web position at each distinct design evaluated by the optimizer and at the independently checked endpoint. The cumulative count includes the independent endpoint analyses and the additional states required by center differencing.
- `span_refinement_results.json` records the longitudinal mesh check with four web spans and two flange spans. The transverse basis is unchanged.
- `verify_comparison.py` checks the counts, endpoint stationarity and energy ratios using the archived data. Run it with Python 3.

Both searches begin at a web offset of 0.50 m with the same objective, bounds and optimizer tolerances. Each starts with a new analysis. One structural evaluation comprises assembly, reduced stiffness factorization and equilibrium solution at the specified geometry. The adjoint and diagnostic right-hand sides reuse the factorization at that design. The reference's published results and the number of structural evaluations on the present shell model are separate comparisons.
