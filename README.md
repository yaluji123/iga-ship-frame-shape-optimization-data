# Ship-frame shape optimization data

Numerical verification data and scripts for *Isogeometric shape optimization of trimmed ship-frame connections using consistent stress sensitivities*.

Version 1.2.1 presents the reviewed files as a browsable directory and updates the dataset citation. The numerical records and verification scripts are identical to version 1.2.0. Browse the files below or download the complete release from Zenodo.

| Study | Files | What to check |
|---|---|---|
| Manufactured shell and strain projection | [manufactured shell](data/manufactured_shell/), [56-state results](data/manufactured_shells.csv), [moving cut](data/moving_cut_projection.py) | Displacement and signed face stress errors, matched cost at an error target, and differentiation of the projection on a moving retained region. |
| Controlled T and L joints | [T joint model](data/controlled_T/), [T joint penalty check](data/T_joint_penalty_test.json), [L joint points](data/L_joint_points.json), [L joint gradients](data/L_joint_gradients.json) | Shared geometry, selected design responses, and gradient checks. |
| Open joints and moving carriers | [forward transfer](data/forward_joint/), [moving joint gradient](data/joint_gradient/), [two moving carriers](data/moving_carriers_joint/) | Load transfer, coupling stability, and sensitivity to both moving carriers and the changing displacement reduction. |
| Published single-edge cantilever | [cantilever benchmark](data/single_edge_cantilever/) | Input dimensions, optimized edge profiles, matched analytic and difference searches, and independent discretization checks. |
| Published flat T-beam | [T-beam benchmark](data/published_t_beam/) | Web offset and normalized compliance, refinement of the reference comparison, and solve counts in matched searches. |
| Ship-frame geometry | [shared curve and anchors](data/ship_trace_geometry/) | Geometry of the three design modes, the common curve, and residual bounds on 17 anchored traces for seven design states. |
| Ship-frame sensitivity and optimization | [ship-frame results](data/ship_corrected/) | Endpoint calibration, gradient partition, support reaction and penalty checks, 8 versus 56 evaluations from the prepared curved design, 6 versus 11 for stationarity from the straight frame, and the separate two-evaluation first feasible design. |
| Fixed curve and paired finite element checks | [fixed curve comparisons](data/fixed_candidate_fe/) | The shared reference spline, the projected shell regional summaries, the four paired finite element comparisons, and element weights for recalculating window averages. |

## Verification

The [data README](data/README.txt) lists the 12 verification commands and dependencies. The open examples include small calculations that can be solved with the supplied scripts. The ship-frame files record the verified responses, histories and geometry. Generating a new full-frame solution requires the original CAD and solver models. The [checksum list](data/SHA256SUMS.txt) identifies the exact contents of the data directory.

## Citation

Yuheng Ji, Zhijie Dai, Deyu Wang, Yao Wang, and Chuntong Li (2026). *Replication data for Isogeometric shape optimization of trimmed ship-frame connections using consistent stress sensitivities*. Version 1.2.1. Zenodo.

Use the version DOI shown in the [Zenodo archive](https://doi.org/10.5281/zenodo.24490793). The [citation file](CITATION.cff) provides the author details.

The copyright notice and terms for the data and scripts are provided in [LICENSE](LICENSE).
