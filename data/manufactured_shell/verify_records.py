"""Verify the fresh residuals, the frozen numerical comparison and Table 1 selection."""
from collections import Counter
import csv
import math
from pathlib import Path


here = Path(__file__).resolve().parent
methods = ("native", "compatible", "dual", "full_dual")
catalog = (("generator", (10., 1., .1)), ("bending_twist", (10., 1., .1)), ("membrane_shear", (10.,)))
reduced_residual = ("true_free_residual", "actual_U_source_operator_free_residual", "bilinear_work_closure_relative")
face = ("bottom_signed_stress", "top_signed_stress")


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    a = {(r["field"], r["arm"], int(r["Nu"]), int(r["Nv"]), float(r["t_mm"])):r for r in records}
    assert len(a) == len(records)
    return a


def main():
    fresh = read_table(here/"fresh_shell_sweep.csv")
    previous = read_table(here.parent/"manufactured_shells.csv")
    expected = {(field,method,nu,nv,t) for field,thickness in catalog for t in thickness
                for method in methods for nu in (2,4,8,16) for nv in (2,4)}
    assert fresh.keys() == previous.keys() == expected
    assert len(fresh) == 224 and Counter(key[0] for key in fresh) == {
        "generator":96,"bending_twist":96,"membrane_shear":32}
    for key,r in fresh.items():
        assert int(r["dofs"]) == 3*(key[2]+3)*(key[3]+3)
        assert int(r["free_dofs"]) == int(r["dofs"])-4*(key[3]+3)
        assert all(0 <= float(r[m]) <= 1e-10 for m in reduced_residual)
        w = float(r["work"])
        es = float(r["Em_solver"])+float(r["Eb_solver"])
        ep = float(r["Em_physical_method_recovery"])+float(r["Eb_physical_method_recovery"])
        ebil = abs(es-w/2)/es
        erec = abs(ep-w/2)/ep
        assert min(w,es,ep) > 0
        assert math.isclose(ebil,float(r["bilinear_work_closure_relative"]),rel_tol=1e-10,abs_tol=2e-15)
        assert math.isclose(erec,float(r["physical_energy_work_gap_relative_diagnostic"]),rel_tol=1e-10,abs_tol=2e-15)
        face_norm = math.sqrt(sum(float(r[f+"_error_squared_integral"]) for f in face)/
                              sum(float(r[f+"_reference_squared_integral"]) for f in face))
        assert math.isclose(face_norm,float(r["signed_two_face_tensor_stress_L2_relative"]),rel_tol=1e-12)
        for name in ("u_L2_relative","signed_two_face_tensor_stress_L2_relative",
                     "physical_energy_work_gap_relative_diagnostic","Em_physical_method_recovery",
                     "Eb_physical_method_recovery") + tuple(f+"_"+s+"_squared_integral" for f in face for s in ("error","reference")):
            assert float(r[name]) == float(previous[key][name]), (key,name)
    print("Matched fresh and frozen states:",len(fresh))
    for col in reduced_residual+("physical_energy_work_gap_relative_diagnostic",):
        worst = max(fresh,key=lambda x:float(fresh[x][col]))
        print(col, fresh[worst][col], worst)
    worst = max(fresh,key=lambda x:float(fresh[x]["physical_energy_work_gap_relative_diagnostic"]))
    assert worst == ("membrane_shear","full_dual",2,2,10.)
    print("Table 1 selection from original forward timings, matched by fresh errors:")
    print("field,t_mm,"+",".join(methods))
    for field,thickness in catalog:
        for t in thickness:
            winners=[]
            for method in methods:
                qualified = [prior for key,prior in previous.items() if (key[0],key[1],key[4]) == (field,method,t)
                             and max(float(fresh[key][q]) for q in ("u_L2_relative",
                                                                      "signed_two_face_tensor_stress_L2_relative")) <= .01]
                if not qualified:
                    winners.append("not reached")
                    continue
                r = min(qualified,key=lambda q:float(q["forward"]))
                winners.append(f"{r['dofs']}/{float(r['forward']):.3f}")
            print(f"{field},{t:g},"+",".join(winners))


if __name__ == "__main__":
    main()

