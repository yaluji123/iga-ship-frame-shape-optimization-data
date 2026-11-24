"""Recompute the exact-candidate FE table from its paired responses and anonymous overlap weights."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

def read(name):
    with (HERE / name).open(encoding="utf-8-sig") as f:
        return json.load(f)

def equal(value, target, atol=1.e-7):
    assert math.isclose(value, target, rel_tol=0., abs_tol=atol), (value, target)

def percent(x, y):
    return 100. * (y / x - 1.)

def main():
    r = read("case_results.json")
    assert r["candidate_section"] == "5.3.8"
    assert max(abs(a-b) for a,b in zip(r["candidate_beta"], [3.1164455700249425,-.5253978169516366,-4.3763255003709265])) < 2.e-11
    equal(.06, r["pressure_MPa"])
    groups = defaultdict(list)
    with (HERE / "fe_common_window_weights.csv").open(newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            values = tuple(float(row[n]) for n in ("common_area_mm2", "straight_face_envelope_MPa", "candidate_face_envelope_MPa"))
            assert all(math.isfinite(v) and v >= 0. for v in values)
            assert values[0] > 0.
            groups[row["pair"], row["window"]].append(values)
    assert {"face_L2", "face_L4", "line_L4", "face_L5"} <= set(r["pairs"])
    for name, pair in r["pairs"].items():
        a, b = pair["states"]
        p = percent(a["compliance_Nmm"], b["compliance_Nmm"])
        equal(p, pair["compliance_increase_percent"])
        equal(5. - p, pair["compliance_margin_percentage_points"])
        equal(percent(a["peak"]["value_MPa"], b["peak"]["value_MPa"]), pair["whole_sampled_peak_change_percent"])
        equal(percent(a["volume_mm3"], b["volume_mm3"]), pair["shell_mesh_volume_change_percent"])
        assert a["fixed_nodes"] == b["fixed_nodes"] and a["structural_elements"] == b["structural_elements"]
        assert pair["element_connectivity_and_fixed_nodes_equal"]
        equal(-18000., pair["resultant_force_N"][1])
        assert all(a["peak"][k] == b["peak"][k] for k in ("owner", "element", "face", "ip", "section_point"))
        for state in (a, b):
            assert 0 < state["peak"]["distance_to_restrained_geometry_mm"] < 6.22
        for x in (a,b):
            assert x["max_force_residual_ratio"] < 1.e-5 and x["max_moment_residual_ratio"] < 1.e-5
        for k, w in pair["common_physical_windows"].items():
            terms = groups.pop((name, k))
            area = math.fsum(t[0] for t in terms)
            a0 = math.fsum(t[0]*t[1] for t in terms)/area
            a1 = math.fsum(t[0]*t[2] for t in terms)/area
            equal(area, w["projected_overlap_mm2"])
            equal(a0, w["base_MPa"])
            equal(a1, w["changed_MPa"])
            equal(percent(a0, a1), w["percent"])
    assert not groups
    iga = read("iga_fixed_design_regions.json")
    images = [iga["states"][n] for n in ("baseline", "candidate")]
    equal(percent(*(v["compliance_Nmm"] for v in images)), iga["compliance_change_percent"])
    equal(-percent(*(v["volume_mm3"] for v in images)), r["IGA_represented_CAD_volume_reduction_percent"])
    equal(percent(*(v["maximum_von_mises_MPa"] for v in images)), iga["full_peak_change_percent"])
    for pair in iga["common_retained_labels_with_distance_cutoffs"].values():
        equal(percent(pair["straight"], pair["candidate"]), pair["relative_change_percent"])
    print("Verified all paired compliance, sampled peaks, equal-topology records and common window means")

if __name__ == "__main__":
    main()
