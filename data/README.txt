Data and numerical verification for
Isogeometric shape optimization of trimmed ship-frame connections using consistent stress sensitivities

The archive uses millimetres, newtons and megapascals for the ship frame.
The ship_corrected directory contains the positive endpoint norm, force
and stress records on the corrected frame, real differences, measured
optimization states and independent endpoint checks. The stress and
compliance responses use the baseline for the same frame model.
Preparation and the separate branch probes have their own records.

The fixed_candidate_fe directory uses the exact same shape in the
projected shell and paired finite element comparisons. This is a fixed
verification curve. The shape in ship_corrected/first_feasible.json was
selected by the current predictive search from the straight frame.
The ship_trace_geometry records cover the common and anchored traces
for the design modes and seven stated designs.

Run the following with Python 3, NumPy and SciPy from the archive root.

    python ship_corrected/verify_evidence.py
    python ship_trace_geometry/check_geometry.py
    python fixed_candidate_fe/verify_candidate.py
    python moving_cut_projection.py
    python controlled_T/check_model.py

Run these inside the named directory.

    forward_joint: python check_stability.py
    joint_gradient: python check_gradient.py
    moving_carriers_joint: python both_carriers_joint.py
    manufactured_shell: python verify_records.py
    manufactured_shell: python recompute_shells.py
    single_edge_cantilever: python check.py
    published_t_beam: python verify_comparison.py

The small-shell and open-joint programs reassemble and solve the
specified models. The ship programs verify numerical records, response
changes, geometry and counts. New full-frame states require the original
CAD and solver models. SHA256SUMS.txt lists the exact archive file checksums.

Citation and rights
Use the dataset citation in the repository CITATION.cff. The repository
LICENSE states the copyright notice and terms for the data and scripts.
