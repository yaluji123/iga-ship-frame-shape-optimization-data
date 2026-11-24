"""Check released ship-frame trace residuals against the reported geometric bounds."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def max_vector_norm(v):
    return float(np.max(np.linalg.norm(v, axis=-1)))


def mode_maximum(v):
    return np.max(np.linalg.norm(v, axis=-1), axis=0)


def compare(actual, declared):
    np.testing.assert_allclose(actual, declared, rtol=1e-8, atol=5e-14)


def main():
    results = json.loads((HERE / "trace_summary.json").read_text(encoding="utf-8"))
    with np.load(HERE / "trace_residuals.npz", allow_pickle=False) as file:
        a = {name: file[name] for name in file.files}
    assert all(np.isfinite(value).all() for value in a.values())

    for key, sigma in (("C0", a["sigma0"]), ("C_gamma_N0", a["singular"])):
        row = results["rank"][key]
        assert len(sigma) == min(row["shape"])
        cutoff = row["cutoff_relative"] * sigma[0]
        rank = int(np.sum(sigma > cutoff))
        assert rank == row["rank"]
        compare([sigma[0], cutoff, sigma[rank - 1], sigma[rank]],
                [row["sigma_max"], row["cutoff_absolute"],
                 row["sigma_last_retained"], row["sigma_first_discarded"]])
    assert results["rank"]["N0_columns"] == results["rank"]["C0"]["shape"][1] - results["rank"]["C0"]["rank"]
    up = a["left_vectors"]
    assert up.shape == (results["rank"]["C_gamma_N0"]["shape"][0], results["rank"]["C_gamma_N0"]["rank"])
    np.testing.assert_allclose(up.T @ up, np.eye(up.shape[1]), atol=1e-12, rtol=0)

    rhs = a["target"].reshape(up.shape[0], -1)
    projected_residual = (rhs - up @ (up.T @ rhs)).reshape(-1, 3, 3)
    relative_residual = np.linalg.norm(projected_residual, axis=(0, 2)) / np.linalg.norm(a["target"], axis=(0, 2))
    modes = results["per_mode"]
    compare(relative_residual, modes["retained_range_relative_residual"])
    compare(mode_maximum(projected_residual), modes["retained_range_maximum_Bernstein_residual_mm"])
    compare(mode_maximum(a["shared_coeff"]), modes["follower_to_target_maximum_Bernstein_residual_mm"])
    compare(mode_maximum(a["actual_mode_gap"]), modes["two_patch_maximum_Bernstein_gap_mm"])
    compare(mode_maximum(a["anchor_coeff"]), modes["maximum_anchor_Bernstein_residual_mm"])
    compare(mode_maximum(a["sampled_anchor_fields"]), modes["maximum_anchor_sampled_motion_mm"])
    compare(np.max(a["anchor_budget"], axis=0), modes["maximum_anchor_continuous_bound_mm"])

    assert a["sampled_shared_baseline"].shape == (results["shared_sample_count"], 3)
    assert a["anchor_budget"].shape == (results["anchor_trace_count"], 3)
    counts = [row["dense_owner_snapshot_points"] + row["direct_export_points"]
              for row in results["anchor_sample_counts"]]
    coefficient_order = np.argsort(a["anchor_row_index"])
    by_trace = a["anchor_index"][coefficient_order]
    assert np.array_equal(np.sort(a["anchor_row_index"]), np.arange(len(by_trace)))
    assert set(by_trace) == set(range(results["anchor_trace_count"]))
    for anchor_index in range(results["anchor_trace_count"]):
        assert np.all(mode_maximum(a["anchor_coeff"][by_trace == anchor_index]) <= a["anchor_budget"][anchor_index])
    assert sum(counts) == len(a["sampled_anchor_fields"])
    blocks = np.split(a["sampled_anchor_fields"], np.cumsum(counts)[:-1])
    for state, row in results["cases"].items():
        beta = np.asarray(row["beta"])
        joint_sample = max_vector_norm(a["sampled_shared_baseline"] + np.einsum("i,qij->qj", beta, a["sampled_shared_modes"]))
        joint_bound = max_vector_norm(a["actual_base_gap"] + np.einsum("i,kij->kj", beta, a["actual_mode_gap"]))
        anchor_samples = [max_vector_norm(np.einsum("i,qij->qj", beta, block)) for block in blocks]
        anchor_bounds = a["anchor_budget"] @ np.abs(beta)
        compare(joint_sample, row["shared_sampled_maximum_mm"])
        compare(joint_bound, row["shared_bernstein_upper_bound_mm"])
        compare(anchor_samples, row["anchor_sampled_maximum_mm_by_trace"])
        compare(anchor_bounds, row["anchor_continuous_bound_mm_by_trace"])
        tolerance = results["geometry_tolerance_mm"]
        assert joint_sample <= joint_bound + 1e-11 <= tolerance
        assert np.all(np.asarray(anchor_samples) <= anchor_bounds + 1e-12)
        assert np.max(anchor_bounds) <= tolerance
        print(f"{state}: coincidence {joint_bound:.3e} mm, anchor bound {max(anchor_bounds):.3e} mm")


if __name__ == "__main__":
    main()
