
"""Verify the corrected design-response and qualification records shipped with the article."""
from pathlib import Path
from math import fsum, isclose
import json
import numpy as np

HERE=Path(__file__).resolve().parent

def read(name):
    return json.loads((HERE/name).read_bytes())

def close(a,b):
    assert isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(a,b)

def test_kkt(k):
    values=[k[key] for key in ("primal_violation","stationarity_relative","complementarity")]
    assert all(0<=value<=1e-6 for value in values)
    assert k["passed"]

def run():
    sums=read("summary.json")
    fixed=read("fixed_states.json")
    straight,reference,probe=(fixed[k] for k in ("straight","reference_curve","probe"))
    close(sums["first_mode"]["ks_increment"],probe["ks"]-straight["ks"])
    close(sums["first_mode"]["peak_increment"],probe["maximum_von_mises"]-straight["maximum_von_mises"])
    assert sums["first_mode"]["ks_increment"] < 0 < sums["first_mode"]["peak_increment"]
    c_rise=100*(reference["compliance"]/straight["compliance"]-1)
    close(c_rise,sums["reference_curve"]["compliance_rise_percent"])
    assert c_rise<5 and reference["maximum_von_mises"]<straight["maximum_von_mises"]

    endpoint=read("endpoint_norm.json")
    assert endpoint["interface_count"]==183 and len(endpoint["corrected_interfaces"])==4
    for row in endpoint["corrected_interfaces"]:
        assert min(row["before_norm_weights"])<endpoint["length_resolution_mm"]<max(row["before_norm_weights"])
        close(fsum(r["original_length"] for r in row["join"]),row["joined_length"])
        for entry in row["join"]:
            close(entry["norm_length"],row["joined_length"])
        for info in row["new_spectra"].values():
            assert info["d"]>0
            close(info["g1"]+info["g2"],1.)

    support=read("support_sweep.json")
    cases=support["cases"]
    assert len(support["support_blocks"])==6 and set(cases)=={"straight","reference_curve","probe"}
    for block in support["support_blocks"]:
        close(block["penalty_N_per_mm"],1e3*block["bulk_max_N_per_mm"])
    maps={key:{r["multiplier"]:r for r in data["rows"]} for key,data in cases.items()}
    for mult in (.1,1.,10.):
        rows=[maps[k][mult] for k in ("straight","reference_curve","probe")]
        for r in rows:
            assert 0 <= r["reduced_residual"] < 2e-8
            assert np.linalg.norm(np.asarray(r["support_reaction_N"])+r["external_N"])<1e-6
            assert r["support_balance_N"]<1e-6
        s,c,p=rows
        assert p["ks_MPa"]<s["ks_MPa"] and c["sampled_peak_MPa"]<s["sampled_peak_MPa"]<p["sampled_peak_MPa"]
        assert 100*(c["compliance_Nmm"]/s["compliance_Nmm"]-1)<5

    difference=read("straight_gradient.json")
    h=difference["step"]
    for key,g in (("ks","stress_gradient"),("compliance","compliance_gradient"),("volume","volume_gradient")):
        row=difference["gradients"][key]
        values=np.array([(x["plus"][key]-x["minus"][key])/(2*h) for x in difference["pairs"]])
        a=np.asarray(row["adjoint"])
        np.testing.assert_allclose(values,row["finite_difference"],rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(a,straight[g],rtol=1e-11,atol=1e-12)
        rel=np.linalg.norm(values-a)/np.linalg.norm(a)
        close(rel,row["relative_l2_error"])
        assert rel<1e-6

    warm=read("warm_search.json")
    assert warm["problem"]["mass_reduction_fraction"]==.0005
    assert len(warm["independent_fd_preparation"])==8
    for role,tag in (("analytic","analytic"),("feasible_fd","central")):
        arm=warm["arms"][role]
        assert sum(len(s["states"]) for s in arm["stages"].values())==sums["warm_counts"][tag]
        for stage in arm["stages"].values():
            assert len(stage["states"])==stage["actual_count"]
            assert all(x["same_branch"] for x in stage["stencils"])
        test_kkt(arm["candidate_kkt"]["kkt"])
        test_kkt(arm["analytic_kkt"]["kkt"])
        assert arm["independent_final"]["ks"]<straight["ks"]

    cold=read("cold_searches.json")
    for arm in cold["routes"]:
        assert len(arm["rows"])==sums["cold_counts"][arm["route"]]
        test_kkt(arm["rows"][-1]["kkt"])
    proposal=read("first_feasible.json")
    assert proposal["measurements"]==2 and proposal["actual"]["compliance_rise_percent"]<5
    assert proposal["actual"]["mass_saved_percent"]>=.04 and proposal["actual"]["peak_change_percent"]<0
    print("Verified corrected interface calibration, nine support balances, gradients and actual optimization records.")

if __name__=="__main__":run()

