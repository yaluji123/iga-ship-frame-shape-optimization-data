"""Conforming cubic finite elements for the published two-dimensional cantilever."""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

HERE = Path(__file__).resolve().parent
PARAMETER_NODES = np.linspace(0., 1., 4)
LAGRANGE = np.linalg.inv(np.polynomial.polynomial.polyvander(PARAMETER_NODES, 3))
Q, W = np.polynomial.legendre.leggauss(6)
Q = (Q + 1) / 2
W /= 2
N = np.polynomial.polynomial.polyvander(Q, 3) @ LAGRANGE
DN = np.polynomial.polynomial.polyvander(Q, 2) @ (np.arange(1, 4)[:, None] * LAGRANGE[1:])
A = np.einsum("qi,rj->qrij", DN, N).reshape(6, 6, 16)
B = np.einsum("qi,rj->qrij", N, DN).reshape(6, 6, 16)


def solve(spec, heights, nx_per_span, ny, plane_strain=True):
    shape = spec["geometry"]["upper_edge"]
    knots = shape["open_quadratic_knot_vector"]
    xfun = BSpline(knots, shape["x_controls"], 2)
    hfun = BSpline(knots, heights, 2)
    young, nu = spec["material"]["E"], spec["material"]["nu"]
    mu = young / (2 * (1 + nu))
    lam = young * nu / ((1 + nu) * (1 - 2 * nu)) if plane_strain else young * nu / (1 - nu**2)
    d = 2 * mu + lam
    breaks = np.unique(knots)
    ugrid = np.concatenate([np.linspace(a, b, nx_per_span, endpoint=False) for a, b in zip(breaks[:-1], breaks[1:])] + [breaks[-1:]])
    u = ugrid[:-1, None] + np.diff(ugrid)[:, None] * Q
    v = (np.arange(ny)[:, None] + Q) / ny
    dx = xfun.derivative()(u)
    dh = hfun.derivative()(u)
    h = hfun(u)
    nx = len(ugrid) - 1
    gids = np.arange((3 * nx + 1) * (3 * ny + 1)).reshape(3 * nx + 1, 3 * ny + 1)
    rhs = np.zeros((gids.size, 2))
    nd = nx * ny
    row, col = np.empty((2, nd, 32, 32), dtype=np.int32)
    values = np.empty((nd, 32, 32))
    n = 0
    for i, du in enumerate(np.diff(ugrid)):
        for j in range(ny):
            ids = gids[3*i:3*i+4, 3*j:3*j+4].ravel()
            dofs = (2*ids[:, None] + np.array([0, 1])).ravel()
            row[n] = dofs[:, None]
            col[n] = dofs[None, :]
            gx = (A / du - B * ny * v[j, None, :, None] * (dh[i] / h[i])[:, None, None]) / dx[i, :, None, None]
            gy = B * ny / h[i, :, None, None]
            weight = W[:, None] * W[None, :] * (dx[i] * h[i] * du / ny)[:, None]
            xx = gx.reshape(36, 16)
            yy = gy.reshape(36, 16)
            xw = weight.reshape(36, 1) * xx
            yw = weight.reshape(36, 1) * yy
            kxx = xx.T @ xw
            kyy = yy.T @ yw
            kxy = xx.T @ yw
            k = values[n]
            k[::2, ::2] = d * kxx + mu * kyy
            k[1::2, 1::2] = mu * kxx + d * kyy
            k[::2, 1::2] = lam * kxy + mu * kxy.T
            k[1::2, ::2] = k[::2, 1::2].T
            n += 1
            if j == 0 and xfun(ugrid[i]) >= spec["load"]["x_range"][0] - 1e-12:
                rhs[ids[::4]] += (N.T @ (W * dx[i] * du))[:, None] * spec["load"]["traction_components"]
    k = coo_matrix((values.ravel(), (row.ravel(), col.ravel())), shape=(rhs.size, rhs.size)).tocsr()
    f = rhs.ravel()
    uflat = np.zeros_like(f)
    free = slice(2*(3*ny+1), None)
    uflat[free] = spsolve(k[free, free], f[free])
    residual = k @ uflat - f
    relative_residual = np.linalg.norm(residual[free]) / np.linalg.norm(f[free])
    np.testing.assert_allclose(rhs.sum(axis=0), [0., -11.25], atol=3e-12, rtol=0.)
    assert relative_residual < 1e-8, relative_residual
    np.testing.assert_allclose(residual.reshape(-1, 2)[:3*ny+1].sum(axis=0), [0., 11.25], atol=3e-7, rtol=0.)
    right = uflat.reshape(-1, 2)[gids[-1], 1]
    max_abs, y_at_max = -1., None
    for j in range(ny):
        polynomial = np.polynomial.Polynomial(LAGRANGE @ right[3*j:3*j+4])
        extrema = [0., 1.] + [z.real for z in polynomial.deriv().roots() if abs(z.imag) < 1e-12 and 0 < z.real < 1]
        for z in extrema:
            score = abs(float(polynomial(z)))
            if score > max_abs:
                max_abs, y_at_max = score, heights[-1] * (j + z) / ny
    return {
        "longitudinal_elements": nx,
        "transverse_elements": ny,
        "degrees_of_freedom": rhs.size,
        "signed_lower_tip_displacement": float(right[0]),
        "signed_upper_tip_displacement": float(right[-1]),
        "max_abs_right_edge_displacement": max_abs,
        "max_abs_right_edge_y": y_at_max,
        "load_work": float(f @ uflat),
        "relative_equilibrium_residual": float(relative_residual)
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extra-heights", type=float, nargs=9, help="Nine optimizer ordinates from clamp to tip")
    p.add_argument("--output", type=Path, default=HERE / "independent_fe_results.json")
    args = p.parse_args()
    spec = json.loads((HERE / "model.json").read_text(encoding="utf-8"))
    heights = {"initial": spec["geometry"]["upper_edge"]["initial_y_controls"]}
    heights.update({key: val["terminal_y_controls"] for key, val in spec["published_reference"].items()})
    if args.extra_heights:
        heights["current_method"] = args.extra_heights
    result = {"constitutive": "Plane strain, E=210000, Poisson ratio=0.3, unit thickness", "grids": {}}
    for nx, ny in [(1, 2), (2, 4), (4, 6), (8, 10), (16, 16)]:
        key = f"{7*nx}x{ny}"
        grid = {name: solve(spec, values, nx, ny) for name, values in heights.items()}
        for name, row in grid.items():
            row["lower_tip_ratio_to_initial"] = row["signed_lower_tip_displacement"] / grid["initial"]["signed_lower_tip_displacement"]
            row["max_edge_ratio_to_initial"] = row["max_abs_right_edge_displacement"] / grid["initial"]["max_abs_right_edge_displacement"]
            print(key, name, row["signed_lower_tip_displacement"], row["max_abs_right_edge_displacement"], row["lower_tip_ratio_to_initial"], row["max_edge_ratio_to_initial"], flush=True)
        result["grids"][key] = grid
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
