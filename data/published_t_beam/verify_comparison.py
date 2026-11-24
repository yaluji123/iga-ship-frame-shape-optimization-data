"""Check the reported counts, endpoint agreement, and tabulated reference ratio."""
import csv
import json
from math import isclose
from pathlib import Path

root = Path(__file__).resolve().parent
data = json.loads((root / "comparison_results.json").read_text(encoding="utf-8"))
reference = data["published_reference"]
estimate = {p["web_offset_m"]: p["figure_energy"] for p in reference["readout_samples"]}
assert abs(estimate[0.] - .18719) < 1.e-5
ratio = estimate[0.] / estimate[data["problem"]["start_m"]]
assert isclose(ratio, reference["center_to_start_energy_ratio"], rel_tol=1.e-12)
with (root / "optimization_history.csv").open(encoding="utf-8", newline="") as f:
    history = list(csv.DictReader(f))

step = data["problem"]["centred_difference_step_m"]
energies = []
locations = []
for label in ("analytical", "centered_finite_difference"):
    arm = data[label]
    center = arm["optimizer_samples"]
    perturbations = arm["finite_difference_perturbations"]
    verification = arm["fresh_terminal_checks"]
    states = center + perturbations + verification
    assert (len(center), len(perturbations), len(verification)) == (5, 0 if label == "analytical" else 10, 3)
    assert len(states) == arm["total_structural_evaluations"] == arm["reduced_stiffness_factorizations"]
    assert all(row["solve_roles"].count("state") == row["factorizations"] == 1 for row in states)
    assert all(isclose(row["volume_m3"], 4., rel_tol=1.e-12) and row["relative_residual"] < 1.e-10 for row in states)
    assert center[0]["web_offset_m"] == data["problem"]["start_m"]
    assert verification[0]["web_offset_m"] == arm["terminal_m"]
    assert isclose(verification[1]["web_offset_m"], arm["terminal_m"]-step, abs_tol=1.e-12)
    assert isclose(verification[2]["web_offset_m"], arm["terminal_m"]+step, abs_tol=1.e-12)
    fd = (verification[2]["compliance_nmm"] - verification[1]["compliance_nmm"]) / (2 * step)
    assert abs(fd)/center[0]["compliance_nmm"] <= 1.e-5
    assert isclose(verification[0]["compliance_nmm"] / center[0]["compliance_nmm"],
                   arm["normalized_compliance_to_start"], rel_tol=1.e-12)
    assert isclose(arm["internal_energy_j"], .5e-3*verification[0]["compliance_nmm"], rel_tol=1.e-12)
    rows = [row for row in history if row["method"] == label]
    assert len(rows) == len(center) + 1
    assert int(rows[-1]["cumulative_structural_evaluations"]) == len(states)
    for sample, row in zip(center, rows):
        assert sample["web_offset_m"] == float(row["web_offset_m"])
        assert isclose(sample["compliance_nmm"] / center[0]["compliance_nmm"], float(row["energy_relative_to_start"]), rel_tol=1.e-12)
    energies.append(arm["internal_energy_j"])
    locations.append(arm["terminal_m"])

assert abs(locations[0]-locations[1])<1.e-5
assert all(abs(x) < 1.e-5 for x in locations)
assert isclose(*energies, abs_tol=1.e-10)
mesh = root / "span_refinement_results.json"
if mesh.exists():
    two = json.loads(mesh.read_text(encoding="utf-8"))
    assert two["center_is_better_than_both_neighbors"]
    assert all(p["relative_residual"] < 1.e-10 for p in two["states"])
    assert two["compliance_gradient_at_start_relative_error"] < 1.e-5
print("Flat T-beam evidence checks passed")
