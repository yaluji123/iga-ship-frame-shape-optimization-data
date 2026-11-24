"""Check recorded cantilever optimization, finite-difference stencils and profiles."""
from collections import Counter
from pathlib import Path
import csv
import json
import math

root = Path(__file__).resolve().parent


def read(name):
    return json.loads((root / (name + ".json")).read_text(encoding="utf-8"))


def close(a,b,tol=1.e-10):
    assert math.isclose(a,b,rel_tol=tol,abs_tol=tol),(a,b)


def same(a,b,tol=1.e-10):
    assert len(a)==len(b)
    for x,y in zip(a,b):
        close(x,y,tol)


model = read("model")
geom = model["geometry"]["upper_edge"]
opt = model["optimization"]
lower, upper, cap = (opt[k] for k in ("lower_bound_per_ordinate","upper_bound_per_ordinate","area_upper_bound"))
area_weights = (2.1875,3.28125,3.90625,3.75,3.75,3.75,3.90625,3.28125,2.1875)
scale = model["geometry"]["initial_height"]
knots = geom["open_quadratic_knot_vector"]


def dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def check_state(row):
    close(row["area"],dot(area_weights,row["y"]),1.e-12)
    close(row["quadrature_area"],row["area"],1.e-7)
    assert row["residual"]<2.e-8
    close(row["right_max"],row["tip"])
    close(row["right_max_y"],0.)


def cfd(rows, centre, prefix, step):
    assert len(rows)==18
    derivatives=[]
    for i in range(9):
        plus,minus=rows[2*i:2*i+2]
        assert plus["kind"]==prefix+" +" and minus["kind"]==prefix+" -"
        check_state(plus)
        check_state(minus)
        assert plus["gradient"] is None and minus["gradient"] is None
        for j, y in enumerate(centre):
            delta=step*scale if i==j else 0.
            close(plus["y"][j],y+delta)
            close(minus["y"][j],y-delta)
        derivatives.append((plus["tip"]-minus["tip"])/(2.*step))
    return derivatives


def check_run(report):
    same(report["area_coefficients"],area_weights)
    rows=report["state_records"]
    assert len(rows)==report["counted_states_including_fresh_endpoint"]
    assert len(rows)==report["counted_search_states"]+1
    assert rows[0]["kind"]=="initial" and rows[-1]==report["endpoint"]
    same(rows[0]["y"],geom["initial_y_controls"])
    counts=Counter(r["kind"] for r in rows)
    assert counts["initial"]==1 and counts["fresh endpoint"]==1
    if report["route"]=="analytic":
        assert all(r["gradient"] is not None for r in rows)
        assert counts["trial"]==len(rows)-2
    else:
        assert all(r["gradient"] is None for r in rows[:-1])
        assert counts["central +"]==counts["central -"]==9*report["iterations"]
        centres=[r["y"] for r in rows if r["kind"] in ("initial","trial")]
        i=0
        while i<len(rows)-1:
            if rows[i]["kind"]=="central +":
                centre=[(a+b)/2. for a,b in zip(rows[i]["y"],rows[i+1]["y"])]
                assert any(math.dist(centre,x)<1.e-9 for x in centres)
                cfd(rows[i:i+18],centre,"central",report["central_dimensionless_step"])
                i+=18
            else:
                i+=1
    for row in rows:
        check_state(row)
    tip0=report["initial_displacement"]
    end=report["endpoint"]
    close(tip0,rows[0]["tip"])
    close(report["normalized_tip"],end["tip"]/tip0)
    assert lower-1e-8 <= min(end["y"]) and max(end["y"])<=upper+1e-8
    close(end["area"],cap)
    lam=report["area_multiplier"]
    assert lam>=0
    g=[(scale/tip0)*v+lam*scale*w/cap for v,w in zip(end["gradient"],area_weights)]
    residual=max(abs(min(0.,z) if y<=lower+1.e-6 else max(0.,z) if y>=upper-1.e-6 else z)
                 for y,z in zip(end["y"],g))
    close(residual,report["normalized_stationarity"])
    assert residual<=1.e-5
    return counts


a,f,a14=map(read,("analytic_7","central_7","analytic_14"))
for route in (a,f,a14):
    check_run(route)
assert a["counted_states_including_fresh_endpoint"] < f["counted_states_including_fresh_endpoint"]
close(a["normalized_tip"],f["normalized_tip"])
assert max(abs(x-y) for x,y in zip(a["endpoint"]["y"],f["endpoint"]["y"]))<1.e-6

grads=read("gradient_check_7")
for role,shape in (("initial",geom["initial_y_controls"]),("endpoint",a["endpoint"]["y"])):
    record=grads[role]
    centre,*states=record["state_records"]
    same(centre["y"],shape)
    check_state(centre)
    analytic=[v*scale for v in centre["gradient"]]
    fd=cfd(states,centre["y"],role,a["central_dimensionless_step"])
    same(analytic,record["scaled_analytic"])
    same(fd,record["scaled_fd"])
    error=math.dist(analytic,fd)/math.hypot(*analytic)
    close(error,record["relative_error"])
    assert error<1.e-5

mesh=read("mesh_replay")
outputs={}
shapes={"initial":geom["initial_y_controls"],"cas_7":a["endpoint"]["y"],"cas_14":a14["endpoint"]["y"]}
shapes.update((k,v["terminal_y_controls"]) for k,v in model["published_reference"].items())
for spans,states in mesh.items():
    for name,row in states.items():
        same(row["y"],shapes[name])
        close(row["normalized_tip"],row["tip"]/states["initial"]["tip"])
        check_state(row)
    outputs[spans]=(states["cas_7"]["tip"],states["cas_14"]["tip"])
same(outputs["7"][:1],[a["endpoint"]["tip"]])
same(outputs["14"][1:],[a14["endpoint"]["tip"]])
assert all(abs(x-y)/y<.005 for x,y in zip(outputs["14"],outputs["28"]))


def spline(u,controls):
    if u<=0.:
        return controls[0]
    if u>=1.:
        return controls[-1]
    span=max(i for i,k in enumerate(knots) if k<=u)
    values=list(controls[span-2:span+1])
    for r in (1,2):
        for j in range(2,r-1,-1):
            i=span-2+j
            alpha=(u-knots[i])/(knots[i+3-r]-knots[i])
            values[j]=(1-alpha)*values[j-1]+alpha*values[j]
    return values[2]


profiles=("y_initial","y_analytic","y_lian_2016_mma","y_sun_2018_pso")
arms=("initial","cas_7","lian_mma","sun_pso")
with (root/"optimized_edge_profiles.csv").open(encoding="utf-8") as file:
    rows=list(csv.DictReader(file))
assert len(rows)==401
for i,row in enumerate(rows):
    u=float(row["u"])
    close(u,i/400.)
    close(float(row["x"]),spline(u,geom["x_controls"]))
    for column,arm in zip(profiles,arms):
        close(float(row[column]),spline(u,shapes[arm]))
print("Cantilever profiles, displacement gradients, search counts, terminal conditions, and mesh reanalysis verified.")
