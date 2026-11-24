"""Independent plane Kirchhoff–Love joint with moving carriers and affine reduction."""

import json
from pathlib import Path
import sys

import moving_skew_gradient as shell
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import block_diag, eigvalsh, lu_factor, lu_solve, solve

HERE = Path(__file__).resolve().parent
BETA = 0.2
WEAK_LIMIT = 1e-3
DONOR_SIZES = (8, 12, 16, 24, 32, 64, 128, 256)
CX = 1.15
BETA_X = 0.41
CLAMPS = (0, 1, 2)
EDGE_LOADS = ((.025, -.018, .025), (.013, -.058, -.062))
# beta is dimensionless and all physical lengths are in mm.


class Carrier(shell.Patch):
    def __init__(self, n, beta, active, trim):
        self.n = n
        self.beta = beta
        self.cubic_knots = np.r_[np.zeros(4), np.arange(1, n) / n, np.ones(4)]
        self.dofs = (n + 3) * 4 * 3
        self.transverse_scale = 1 + (0.19 if active else -0.13) * beta
        angle = (-0.13 + 0.27 * beta) if active else (0.79 - 0.31 * beta)
        c, s = np.cos(angle), np.sin(angle)
        self.frame = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        self.global_to_local = np.kron(np.eye(self.dofs // 3), self.frame.T)
        self.cut = trim + .11 * beta if active else .12 - .23 * beta
        x, wx = shell.quadrature(np.linspace(0., 1., n + 1))
        self.x = np.repeat(x, len(shell.TR))
        upper = self.cut if active else 1.
        width = 1 + upper
        self.y = np.tile(-1 + width * shell.TR, len(x))
        self.weights = (wx[:, None] * self.dxdxi(x)[:, None] *
                        self.transverse_scale * width * shell.TW).ravel()
        self.b = self.jets(self.x, self.y, ((1, 0), (0, 1), (2, 0), (0, 2), (1, 1)))

        a, b, c = self.strain_bases(self.x, self.y)
        wa, wb, wc = (v * self.weights[:, None] for v in (a, b, c))
        ab = shell.POISSON * a.T @ wb
        gram = np.block([[a.T @ wa, ab], [ab.T, b.T @ wb]])
        bx = shell.spread(self.b[1, 0], 0)
        by = shell.spread(self.b[0, 1], 1)
        shear = shell.spread(self.b[0, 1], 0) + shell.spread(self.b[1, 0], 1)
        normal_rhs = np.concatenate((wa.T @ (bx + shell.POISSON * by),
                                     wb.T @ (by + shell.POISSON * bx)))
        shear_rhs = wc.T @ shear
        maps = gram_solve(gram, normal_rhs)
        shear_map = gram_solve(c.T @ wc, shear_rhs)
        self.maps = maps[:a.shape[1]], maps[a.shape[1]:], shear_map
        membrane = shell.THICKNESS * (shell.NORMAL * normal_rhs.T @ maps
                                     + shell.SHEAR * shear_rhs.T @ shear_map)
        xx, yy, xy = (shell.spread(self.b[k], 2) for k in ((2, 0), (0, 2), (1, 1)))
        bend = shell.BEND * (xx.T @ (self.weights[:, None] * (xx + shell.POISSON * yy))
                            + yy.T @ (self.weights[:, None] * (yy + shell.POISSON * xx))
                            + 2 * (1 - shell.POISSON) * xy.T @ (self.weights[:, None] * xy))
        self.bulk = membrane + bend

    def dxdxi(self, xi):
        return CX + 2 * BETA_X * self.beta * xi

    def jets(self, x, y, keys):
        first = self.dxdxi(x)[:, None]
        second = 2 * BETA_X * self.beta
        jets = {}
        for ix, iy in keys:
            parametric = lambda k: self.basis(x, y, ix=k, iy=iy)
            if ix == 0:
                values = parametric(0)
            elif ix == 1:
                values = parametric(1) / first
            elif ix == 2:
                values = parametric(2) / first**2 - second * parametric(1) / first**3
            else:
                values = (parametric(3) / first**3 - 3 * second * parametric(2) / first**4
                          + 3 * second**2 * parametric(1) / first**5)
            jets[ix, iy] = values / self.transverse_scale**iy
        return jets

    def physical_points(self, x, y):
        local = np.stack((CX*x + BETA_X*self.beta*x**2,
                          self.transverse_scale * (y-self.cut),
                          np.zeros_like(x)), axis=-1)
        return origin(self.beta) + local @ self.frame.T

    def control_points(self):
        adjacent = np.array([self.cubic_knots[i+1:i+4] for i in range(self.n + 3)])
        greville = np.mean(adjacent, axis=1)
        t, u, v = adjacent.T
        quadratic = (t*u + t*v + u*v) / 3
        x = CX * greville + BETA_X * self.beta * quadratic
        y = self.transverse_scale * (np.linspace(-1., 1., 4) - self.cut)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        local = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(-1, 3)
        return origin(self.beta) + local @ self.frame.T

    def trace_operators(self, x, active):
        y = np.full(x.shape, self.cut)
        tau = np.tile([1., 0.], (len(x), 1))
        conormal = np.tile([0., 1.], (len(x), 1))
        zero = np.zeros_like(tau)
        traces = self.traces(x, y, tau, zero, conormal, zero, active)
        to_global = block_diag(self.frame, np.ones((1, 1)))
        return tuple(None if v is None else to_global @ v @ self.global_to_local for v in traces)


def origin(beta):
    return np.asarray((.04, .05, -.08)) * beta


def gram_solve(matrix, rhs):
    scale = np.diag(matrix)**-.5
    return scale[:, None] * solve(scale[:, None] * matrix * scale[None, :],
                                  scale[:, None] * rhs, check_finite=False)


def weak_rows(bulk):
    row_norms = np.linalg.norm(bulk, axis=1).reshape(-1, 3)
    return np.max(row_norms / np.max(row_norms, axis=0), axis=1)


def moment_inputs(points, target, donors, centre, scale):
    columns = (points[np.asarray(donors)][:, :2] - centre) / scale
    point = (points[target, :2] - centre) / scale
    return np.vstack((np.ones(len(donors)), columns.T)), np.r_[1., point]


def collect_reduction(carrier, global_bulk):
    rows = weak_rows(global_bulk)
    weak = np.flatnonzero(rows <= WEAK_LIMIT)
    strong = np.flatnonzero(rows > WEAK_LIMIT)
    points = carrier.control_points()
    assert abs(carrier.frame[1, 1]) > .5
    constraints = {}
    for i in weak:
        cost = 1 + (strong // 4 - i // 4)**2 + (strong % 4 - i % 4)**2
        ordered = np.lexsort((strong, cost))
        for size in DONOR_SIZES:
            indices = ordered[:size]
            donors = strong[indices]
            inverse_weight = cost[indices].astype(float)**-2
            centre = np.mean(points[donors, :2], axis=0)
            scale = max(np.max(abs(points[donors, :2] - centre)),
                        np.max(abs(points[i, :2] - centre)))
            matrix, rhs = moment_inputs(points, i, donors, centre, scale)
            gram = (matrix * inverse_weight) @ matrix.T
            condition = np.linalg.cond(gram, 1)
            if condition <= 1e10:
                constraints[int(i)] = (tuple(map(int, donors)), inverse_weight, centre, scale, condition)
                break
            if len(indices) == len(strong):
                raise ValueError("No admissible donor neighbourhood.")
    return strong, constraints, rows


def assemble_reduction(points, strong, constraints, points_dot=None):
    columns = {int(i): k for k, i in enumerate(strong)}
    transform = np.zeros((len(points), len(strong)))
    transform[strong] = np.eye(len(strong))
    transform_dot = np.zeros_like(transform)
    coefficients, coefficients_dot = {}, {}
    for i, (donors, inv_w, centre, scale, _) in constraints.items():
        matrix, rhs = moment_inputs(points, i, donors, centre, scale)
        gram = (matrix * inv_w) @ matrix.T
        multiplier = solve(gram, rhs, check_finite=False)
        theta = inv_w * (matrix.T @ multiplier)
        retained = [columns[j] for j in donors]
        transform[i, retained] = theta
        coefficients[i] = theta
        assert abs(theta.sum() - 1) < 1e-10
        assert np.linalg.norm(theta @ points[list(donors)] - points[i]) < 1e-10
        if points_dot is not None:
            matrix_dot = np.vstack((np.zeros(len(donors)), points_dot[list(donors), :2].T / scale))
            rhs_dot = np.r_[0., points_dot[i, :2] / scale]
            gram_dot = (matrix_dot * inv_w) @ matrix.T + (matrix * inv_w) @ matrix_dot.T
            multiplier_dot = solve(gram, rhs_dot - gram_dot @ multiplier, check_finite=False)
            theta_dot = inv_w * (matrix_dot.T @ multiplier + matrix.T @ multiplier_dot)
            transform_dot[i, retained] = theta_dot
            coefficients_dot[i] = theta_dot
            affine_error = theta_dot @ points[list(donors)] + theta @ points_dot[list(donors)] - points_dot[i]
            assert np.linalg.norm(affine_error) < 8e-10
            assert abs(theta_dot.sum()) < 8e-10
    return np.kron(transform, np.eye(3)), np.kron(transform_dot, np.eye(3)), coefficients, coefficients_dot


def physical_operators(beta, trim):
    carriers = tuple(Carrier(n, beta, i == 0, trim) for i, n in enumerate(shell.N_SPANS))
    breaks = np.unique(np.concatenate([np.linspace(0., 1., n+1) for n in shell.N_SPANS]))
    x, wx = shell.quadrature(breaks)
    w = wx * carriers[0].dxdxi(x)
    h = w.sum()
    x = np.r_[x, 0., 1.]
    n_a, flux, endpoint = carriers[0].trace_operators(x, True)
    n_b, _, _ = carriers[1].trace_operators(x, False)
    # Endpoint traction work is signed; the endpoint jump norm keeps the length weight.
    flux[-2:] = np.array([-1., 1.])[:, None, None] * endpoint[-2:] / h
    w = np.r_[w, h, h]
    jump = np.concatenate((n_a, -n_b), axis=2)
    bulk = tuple(p.global_to_local.T @ p.bulk @ p.global_to_local for p in carriers)
    return carriers, bulk, jump, flux, endpoint, w, h


def matrix(bulks, blocks, eta):
    stiffness = block_diag(*bulks)
    for (skew, penalty), scale in zip(blocks, eta):
        stiffness += -skew + skew.T + scale * penalty
    return stiffness


def recovered(carriers):
    return tuple(p.stresses() @ p.global_to_local for p in carriers)


def stress_response(stress, u, rho=shell.RHO, shift=None, gradient=False):
    stop = stress[0].shape[-1]
    values = np.concatenate((stress[0] @ u[:stop], stress[1] @ u[stop:]))
    sx, sy, shear = values.T
    vm = np.sqrt(sx*sx - sx*sy + sy*sy + 3*shear*shear)
    assert np.min(vm.real) > 1e-5
    shift = float(vm.real.max()) if shift is None else shift
    exponential = np.exp(rho * (vm - shift))
    mean = exponential.sum()
    value = shift + np.log(mean) / rho
    if not gradient:
        return value
    response_gradient = .5 * exponential[:, None] / (mean*vm[:, None]) * np.column_stack(
        (2*sx-sy, 2*sy-sx, 6*shear))
    split = len(stress[0])
    partial = np.r_[np.einsum("pi,pin->n", response_gradient[:split], stress[0]),
                    np.einsum("pi,pin->n", response_gradient[split:], stress[1])]
    return float(value), partial, shift, float(vm.min())


def solve_state(beta, trim, retain=False):
    carriers, bulk, jump, flux, edge, w, length = physical_operators(beta, trim)
    a, b = carriers
    strong, constraints, rows = collect_reduction(a, bulk[0])
    assert np.min(weak_rows(bulk[1])) > WEAK_LIMIT
    assert constraints, "The fixture must admit at least one weak control point."
    pa, _, theta, _ = assemble_reduction(a.control_points(), strong, constraints)
    p = block_diag(pa, np.eye(b.dofs))
    cal = shell.calibration(pa.T @ bulk[0] @ pa, flux @ pa, w)
    eta = 9 * np.array([c["d"] for c in cal])
    k = matrix(bulk, shell.weighted_trace_blocks(jump, flux, w), eta)
    f = np.r_[a.fixed_exterior_load(-1., EDGE_LOADS[0]), b.fixed_exterior_load(1., EDGE_LOADS[1])]
    free = np.ones(p.shape[1], dtype=bool)
    a_tip = int(np.nonzero(strong == (a.n+2)*4)[0].item())
    free[[*CLAMPS, 3*a_tip + 1, 3*a_tip + 2, pa.shape[1] + 11]] = False
    p_free = p[:, free]
    operator = p_free.T @ k @ p_free
    load = p_free.T @ f
    scale = np.diag(operator)**-.5
    scaled = scale[:, None] * operator * scale[None, :]
    symmetric_min = float(eigvalsh(.5*(scaled+scaled.T), subset_by_index=[0, 0])[0])
    assert symmetric_min > 0
    fact = lu_factor(scaled, check_finite=False)
    y = scale * lu_solve(fact, scale * load, check_finite=False)
    u = p_free @ y
    relative = float(np.linalg.norm(operator @ y - load) / np.linalg.norm(load))
    assert relative < 1e-9
    stress = recovered(carriers)
    ks, partial, shift, min_vm = stress_response(stress, u, gradient=True)
    trace_x = np.linspace(0, 1, 101)
    trace_a, trace_b = [s.physical_points(trace_x, np.full(101, s.cut)) for s in carriers]
    interface_error = np.max(np.linalg.norm(trace_a - trace_b, axis=1))
    assert interface_error < 1e-12
    verification_x, verification_y = np.meshgrid(np.linspace(0, 1, 19), np.linspace(-1, 1, 7))
    verification_x, verification_y = verification_x.ravel(), verification_y.ravel()
    geometry_fit = max(np.max(np.linalg.norm(
        pch.basis(verification_x, verification_y) @ pch.control_points()
        - pch.physical_points(verification_x, verification_y), axis=1)) for pch in carriers)
    assert geometry_fit < 1e-12
    gap = [c["gap"] for c in cal]
    signature = {"weak_controls": list(constraints),
                 "donors": [list(x[0]) for x in constraints.values()],
                 "positive_bulk_rank": int(pa.shape[1] - 6),
                 "sample_count": sum(len(x) for x in stress), "free_dofs": int(free.sum()),
                 "trace_breaks": len(np.unique(np.concatenate([np.linspace(0, 1, n+1) for n in shell.N_SPANS])))}
    result = {"beta": beta, "response": [ks, float(f@u)], "interface_length_mm": float(length),
              "dihedral_deg": float(np.degrees(np.arccos(a.frame[:, 2] @ b.frame[:, 2]))),
              "sample_min_vm_MPa": min_vm, "trace_closure_mm": float(interface_error),
              "carrier_control_net_fit_mm": float(geometry_fit),
              "scaled_symmetric_min_eigenvalue": symmetric_min,
              "skew_fraction": float(np.linalg.norm(operator-operator.T) / np.linalg.norm(operator)),
              "signed_cut_coordinates": [float(s.cut) for s in carriers],
              "signature": signature, "spectral_gap": gap, "penalties": eta.tolist(),
              "exterior_force_N": f.reshape(-1, 3).sum(axis=0).tolist(),
              "max_weak_row_ratio": float(np.max(rows[list(constraints)])),
              "min_strong_row_ratio": float(np.min(rows[strong])),
              "equilibrium_residual": relative,
              "theta": [t.tolist() for t in theta.values()]}
    if not retain:
        return result
    rhs = p_free.T @ np.column_stack((partial, f))
    adjoint = scale[:, None] * lu_solve(fact, scale[:, None] * rhs, trans=1, check_finite=False)
    result["adjoint_residual"] = float(np.linalg.norm(operator.T @ adjoint-rhs) / np.linalg.norm(rhs))
    assert result["adjoint_residual"] < 1e-9
    return result, (a, b, strong, constraints, pa, p_free, free, k, f, u, y,
                    adjoint, cal, eta, shift, partial, theta)


def coupled_derivative(beta, trim):
    state, (a, b, strong, constraints, pa, p, free, k, f, u, y,
            adj, cal, eta, shift, partial, theta) = solve_state(beta, trim, retain=True)
    h = shell.DELTA
    carriers, bulk, j, v, edge, w, arc = physical_operators(beta + 1j*h, trim)
    points_dot = carriers[0].control_points().imag / h
    _, pa_dot, _, theta_dot = assemble_reduction(a.control_points(), strong, constraints, points_dot)
    p_dot = block_diag(pa_dot, np.zeros((b.dofs, b.dofs)))[:, free]
    s = recovered(carriers)
    stress_operator_dot = stress_response(s, u, shift=shift).imag / h
    explicit_recovery = np.array([stress_operator_dot, 0.])
    objective_full_gradient = np.column_stack((partial, f))
    plus_blocks = shell.weighted_trace_blocks(j, v, w)
    frozen_cal_dot = shell.eta_direction(cal, pa.T @ bulk[0] @ pa, v @ pa, w)
    plus_pa = pa + 1j*h*pa_dot
    cal_dot = shell.eta_direction(cal, plus_pa.T @ bulk[0] @ plus_pa, v @ plus_pa, w)
    k_dot = matrix(bulk, plus_blocks, eta + 1j*h*cal_dot).imag / h
    k_dot_frozen_cal = matrix(bulk, plus_blocks, eta + 1j*h*frozen_cal_dot).imag / h
    residual = k @ u - f
    full = explicit_recovery + (p_dot @ y) @ objective_full_gradient - adj.T @ (
        p_dot.T @ residual + p.T @ (k_dot @ u + k @ p_dot @ y))
    frozen_direct = explicit_recovery - adj.T @ (p.T @ k_dot @ u)
    frozen_all = explicit_recovery - adj.T @ (p.T @ k_dot_frozen_cal @ u)
    assert np.linalg.norm(pa_dot) > 1e-3
    state.update(adjoint_derivative=full.tolist(), direct_reduction_omitted=frozen_direct.tolist(),
                 full_reduction_omitted=frozen_all.tolist(), reduction_derivative_norm=float(np.linalg.norm(pa_dot)),
                 max_weight_derivative=float(max(np.max(abs(x)) for x in theta_dot.values())),
                 theta_dot=[x.tolist() for x in theta_dot.values()],
                 donor_matrix_condition=[float(v[-1]) for v in constraints.values()],
                 endpoint_force_derivative_norm=float(np.linalg.norm(edge[-2:].imag/h)),
                 carrier_control_motion_norms=[float(np.linalg.norm(s.control_points().imag/h)) for s in carriers],
                 chain_length_derivative=float(arc.imag/h), penalty_derivative=cal_dot.tolist(),
                 frozen_reduction_relative_difference=(abs(full-frozen_all)/abs(full)).tolist(),
                 direct_frozen_reduction_relative_difference=(abs(full-frozen_direct)/abs(full)).tolist())
    return state


def study(trim):
    result = coupled_derivative(BETA, trim)
    gradient = np.asarray(result["adjoint_derivative"])
    reference = []
    for step in (.004, .002, .001, .0005):
        minus = solve_state(BETA-step, trim)
        plus = solve_state(BETA+step, trim)
        assert minus["signature"] == result["signature"] == plus["signature"]
        difference = (np.array(plus["response"])-minus["response"])/(2*step)
        weight_difference = (np.array(plus["theta"])-minus["theta"])/(2*step)
        reference.append({"step":step, "plus":plus, "minus":minus, "finite_difference":difference.tolist(),
                          "relative_error":(abs(difference-gradient)/abs(gradient)).tolist(),
                          "max_weight_error":float(np.max(abs(weight_difference-np.array(result["theta_dot"])))),
                          "relative_penalty_error":(abs((np.array(plus["penalties"])-minus["penalties"])
                                                       /(2*step)-result["penalty_derivative"])
                                                     /abs(np.array(result["penalty_derivative"]))).tolist(),
                          "forward_taylor_remainder":abs(np.array(plus["response"])
                                                         -result["response"]-step*gradient).tolist()})
    remainders = np.array([x["forward_taylor_remainder"] for x in reference])
    orders = np.log2(remainders[:-1] / remainders[1:])
    assert np.all((orders > 1.95) & (orders < 2.05)), orders
    final_error = np.asarray(reference[-1]["relative_error"])
    omission = np.asarray(result["frozen_reduction_relative_difference"])
    assert np.max(final_error) < 2e-6
    assert np.all(omission > 50*final_error)
    assert reference[-1]["max_weight_error"] < 1e-7
    assert max(reference[-1]["relative_penalty_error"]) < 2e-6
    return {"forward_state_assemblies": 1 + 2*len(reference),
            "forward_taylor_orders": orders.tolist(),
            "model":{"young_MPa":shell.YOUNG, "poisson":shell.POISSON, "thickness_mm":shell.THICKNESS,
                     "longitudinal_spans":list(shell.N_SPANS), "transverse_spans":[1,1],
                     "gauss_points_per_interval":len(shell.TR), "weak_row_limit":WEAK_LIMIT,
                     "active_cut_offset":trim, "ks_sharpness_per_MPa":shell.RHO,
                     "load_per_reference_x_N":EDGE_LOADS},
            "response_units":["MPa","N mm"],
            "base":result, "reassembled_centered_differences":reference}


if __name__ == "__main__":
    trim = float(sys.argv[1]) if len(sys.argv) > 1 else -0.85
    result = study(trim)
    out = HERE/"both_carriers_joint.json"
    out.write_text(json.dumps(result, indent=2) + "\n",encoding="utf-8")
    base = result["base"]
    print(f"beta={base['beta']} trim={trim}  weak controls={base['signature']['weak_controls']}")
    print(f"gradient={base['adjoint_derivative']} frozen={base['full_reduction_omitted']}")
    for row in result["reassembled_centered_differences"]:
        print(f"h={row['step']:.4g} error={row['relative_error']} weight={row['max_weight_error']}")
