"""Check the controlled T-connection geometry. Optionally export VTK surfaces for inspection."""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.interpolate import BSpline

HERE = Path(__file__).resolve().parent


def surface(patch, net, u, v):
    nu = BSpline.design_matrix(u, patch["knots_u"], patch["degree_u"]).toarray()
    nv = BSpline.design_matrix(v, patch["knots_v"], patch["degree_v"]).toarray()
    return np.einsum("ij,jkd,ik->id", nu, net, nv)


def geometry(model, beta):
    web, receiver = (model["patches"][key] for key in ("web", "receiver"))
    web_net = np.array(web["control_points_at_beta_zero_mm"], dtype=float)
    receiver_net = np.array(receiver["control_points_at_beta_zero_mm"], dtype=float)
    receiver_net[1:, :, 1] += np.asarray(beta)[:, None]
    curve = receiver_net[:, 0, :].copy()
    curve[:, 2] = 0.
    return web, receiver, web_net, receiver_net, BSpline(receiver["knots_u"], curve, 3)


def vtk(path, web, receiver, web_net, receiver_net, curve):
    # This tessellation is for viewing the exact model in model.json.
    u, v = np.meshgrid(np.linspace(0, 1, 65), np.linspace(0, 1, 9), indexing="ij")
    u, v = u.ravel(), v.ravel()
    sweep = curve(u)
    a = surface(web, web_net, u, v * sweep[:, 1] / 768)
    b = surface(receiver, receiver_net, u, v)
    line = curve(np.linspace(0, 1, 65))
    points = np.vstack((a, b, line))
    faces = [(start+i*9+j, start+(i+1)*9+j, start+(i+1)*9+j+1, start+i*9+j+1)
             for start in (0, len(a)) for i in range(64) for j in range(8)]
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("# vtk DataFile Version 3.0\nControlled T midsurfaces and shared line; mm\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(points)} double\n")
        np.savetxt(f, points, fmt="%.12g")
        f.write(f"LINES 1 {len(line)+1}\n{len(line)} ")
        f.write(" ".join(str(n) for n in range(len(a)+len(b),len(points)))+"\n")
        f.write(f"POLYGONS {len(faces)} {len(faces)*5}\n")
        for row in faces:
            f.write("4 "+" ".join(map(str,row))+"\n")
        f.write(f"CELL_DATA {len(faces)+1}\nSCALARS part int 1\nLOOKUP_TABLE default\n")
        f.write("3\n"+"1\n"*(len(faces)//2)+"2\n"*(len(faces)//2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtk-dir",type=Path,help="create three display tessellations; each VTK includes both plates and the common edge")
    opts=parser.parse_args()
    model=json.loads((HERE/"model.json").read_text("utf-8"))
    if opts.vtk_dir:
        opts.vtk_dir.mkdir(parents=True, exist_ok=True)
    ndof = 3*sum(len(s["control_points_at_beta_zero_mm"])*len(s["control_points_at_beta_zero_mm"][0]) for s in model["patches"].values())
    fixed = model["boundary_conditions"]["fixed_dofs_zero_based"]
    assert ndof==108 and len(set(fixed))==48 and ndof-len(fixed)==60
    assert fixed == list(range(24))+list(range(60,84))
    assert model["boundary_conditions"]["dead_force"]=={"global_dof_zero_based":107,"force_N":1000.0}
    for name,state in model["configurations"].items():
        beta=state["beta_mm"]
        web,receiver,wn,pn,line=geometry(model,beta)
        q=np.linspace(0.,1.,257)
        boundary=surface(web,wn,q,line(q)[:,1]/768.)
        centre=surface(receiver,pn,q,np.full(len(q),.5))
        gap=max(np.max(np.abs(boundary-line(q))),np.max(np.abs(centre-line(q))))
        assert gap<1e-9
        # The cubic ordinate reaches extrema at endpoints or derivative roots.
        from scipy.interpolate import PPoly
        roots=PPoly.from_spline(BSpline(receiver["knots_u"],pn[:,0,1],3)).derivative().roots(extrapolate=False)
        extremes=line(np.r_[0.,roots[(roots>0)&(roots<1)],1.])[:,1]
        assert np.all((extremes>0.) & (extremes<768.))
        z,w=np.polynomial.legendre.leggauss(16)
        length=.5*np.dot(w,np.linalg.norm(line.derivative()((z+1)/2),axis=1))
        volume=5.*(768.*(384.+sum(beta)/4.)+96.*length)
        reference=state["archived_response"]["volume_mm3"]
        assert np.isclose(volume,reference,rtol=1e-11,atol=1e-6)
        assert state["archived_response"]["KS_MPa"]>=state["archived_response"]["sampled_peak_MPa"]
        print(f"{name}: max shared-coordinate error {gap:.3g} mm; volume {volume:.6f} mm^3")
        if opts.vtk_dir:
            vtk(opts.vtk_dir/(name+".vtk"),web,receiver,wn,pn,line)
    print("Geometry, boundary indexing and archived volumes verified. No structural solve was run.")

if __name__=="__main__":
    main()

