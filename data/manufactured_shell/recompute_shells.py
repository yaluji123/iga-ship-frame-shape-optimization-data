"""Assemble and solve the twelve 2x2, 10 mm parabolic-cylinder reference cases."""
import csv
from math import factorial
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import solve
from numpy.polynomial.legendre import leggauss


ARMS = ("compatible", "native", "dual", "full_dual")
FIELDS = ("generator", "bending_twist", "membrane_shear")
STRAIN_DEGREES = ((2, 3), (3, 2), (2, 2))
LENGTH, WIDTH, CURVATURE, THICKNESS = 1000., 250., 0.0005, 10.
YOUNG, NU, ALPHA = 210000., .3, 1e-4
C = YOUNG / (1 - NU**2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU)/2]])


def knots(degree):
    return np.r_[np.zeros(degree + 1), .5, np.ones(degree + 1)]


def basis(x, degree, der=0):
    t = knots(degree)
    return BSpline(t, np.eye(t.size - degree - 1), degree)(x, nu=der)


def tensor(a, b):
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)


def action(b, vector):
    return (b[:, :, None] * vector[:, None, :]).reshape(len(b), -1)


def qrule(order):
    x, w = leggauss(order)
    x = np.r_[(1+x)/4, (3+x)/4]
    w = np.tile(w/4, 2)
    u, v = (m.ravel() for m in np.meshgrid(x, x, indexing="ij"))
    return u, v, np.outer(w, w).ravel()


def kinematics(u, v):
    s = LENGTH * u
    r = np.sqrt(1 + (CURVATURE*s)**2)
    x = np.stack((np.ones_like(r), np.zeros_like(r), CURVATURE*s), axis=1)/r[:, None]
    y = np.tile((0., 1., 0.), (len(r), 1))
    normal = np.cross(x, y)
    ru, rv = LENGTH*r, np.full_like(r, WIDTH)
    christoffel = LENGTH * CURVATURE**2 * s / r**2
    bu = [basis(u, 3, j) for j in range(3)]
    bv = [basis(v, 3, j) for j in range(3)]
    n = tensor(bu[0], bv[0])
    du, dv = tensor(bu[1], bv[0]), tensor(bu[0], bv[1])
    duu, duv, dvv = tensor(bu[2], bv[0]), tensor(bu[1], bv[1]), tensor(bu[0], bv[2])
    bm = np.stack((action(du, x/ru[:, None]),
                   action(dv, y/rv[:, None]),
                   action(du, y/ru[:, None])+action(dv, x/rv[:, None])), axis=1)
    bb = np.stack((
        action(duu-christoffel[:, None]*du, normal/ru[:, None]**2),
        action(dvv, normal/rv[:, None]**2),
        action(duv, 2*normal/(ru*rv)[:, None])), axis=1)
    return n, bm, bb, (ru*rv)


def source(field, u, v):
    s = LENGTH*u
    r = np.sqrt(1+(CURVATURE*s)**2)
    disp = np.zeros((len(u), 3))
    membrane = np.zeros_like(disp)
    bend = np.zeros_like(disp)
    if field == "membrane_shear":
        disp[:, 1] = ALPHA*s**4/LENGTH**3
        membrane[:, 2] = 4*ALPHA*s**3/(LENGTH**3*r)
    elif field == "generator":
        disp[:, 0] = -.75*CURVATURE*s**4/LENGTH**3
        disp[:, 2] = s**3/LENGTH**3
        bend[:, 0] = 6*s/(LENGTH**3*r)
    else:
        disp[:, 0] = -.75*CURVATURE*s**4*(1+v)/LENGTH**3
        disp[:, 1] = -CURVATURE*s**5/(20*WIDTH*LENGTH**3)
        disp[:, 2] = s**3*(1+v)/LENGTH**3
        bend[:, 0] = 6*s*(1+v)/(LENGTH**3*r)
        bend[:, 2] = 6*s*s/(LENGTH**3*WIDTH)
    return disp, membrane, bend


def knot_moment(order, internal_knots):
    """Centered knot moments in the one-dimensional approximate-dual recurrence."""
    if order == 0:
        return 1.
    r = len(internal_knots)
    h = internal_knots - np.mean(internal_knots)
    mu2 = np.mean(h**2)
    if order == 1:
        return r**2 * mu2
    mu4 = np.mean(h**4)
    if order == 2:
        return r**2 * ((r*r-3*r+3)*mu2**2 - (r-1)*mu4)/2
    mu3, mu6 = np.mean(h**3), np.mean(h**6)
    return (r**3*(r-2)*(r*r-7*r+15)*mu2**3 - 3*r*r*(r-2)*(r*r-5*r+10)*mu4*mu2
            - 2*r*r*(3*r*r-15*r+20)*mu3**2 + 2*r*r*(r-2)*(r-1)*mu6)/6


def dual_matrix(degree):
    t = knots(degree)
    m = degree+1
    n = len(t)-m
    result = np.zeros((n, n))
    differences = np.eye(n)
    for order in range(m):
        if order:
            stride = m+order-1
            span = t[stride:]-t[:-stride]
            # Only the interior columns of this divided-difference map are needed.
            differences = stride*(differences[:, :-1]/span[:-1] - differences[:, 1:]/span[1:])
        f = factorial(m)*factorial(m-order-1)/(factorial(m+order)*factorial(m+order-1))
        weights = [(m+order)*f*knot_moment(order, t[i+1:i+m+order])/(t[i+m+order]-t[i])
                   for i in range(n-order)]
        result += (differences*weights) @ differences.T
    return result


def strain_basis(u, v):
    trials, tests = [], []
    for a, b in STRAIN_DEGREES:
        x, y = basis(u, a), basis(v, b)
        trials.append(tensor(x, y))
        tests.append(tensor(x@dual_matrix(a), y@dual_matrix(b)))
    sizes = np.cumsum([0]+[x.shape[1] for x in trials])
    trial = np.zeros((len(u), 3, sizes[-1]))
    test = np.zeros_like(trial)
    for j in range(3):
        step = slice(sizes[j], sizes[j+1])
        trial[:, j, step], test[:, j, step] = trials[j], tests[j]
    return trial, test


def material_product(a, b, d, w):
    db = np.einsum("ab,qbi->qai", d, b, optimize=True)
    x = a.reshape(-1, a.shape[-1])
    return x.T @ (w[:, None, None]*db).reshape(-1, b.shape[-1])


def projected(b, d, w, j, trial, test, petrov):
    if petrov:
        # The geometry and material factors cancel in the approximate-dual test integral.
        test_t = test.reshape(-1, test.shape[-1]).T
        rhs = test_t @ (w[:, None, None]*b).reshape(-1, b.shape[-1])
        diag = test_t @ (w[:, None]*trial.sum(axis=2)).ravel()
        assert np.all(diag > 0.0) and np.max(np.abs(diag-1)) < 1e-10
        coupling = material_product(b, trial, d, w*j)
        mapping = rhs/diag[:, None]
        return coupling@mapping, mapping, coupling
    gram = material_product(trial, trial, d, w*j)
    rhs = material_product(trial, b, d, w*j)
    scale = 1/np.sqrt(gram.diagonal())
    mapping = scale[:, None]*solve(scale[:, None]*gram*scale, scale[:, None]*rhs, assume_a="pos")
    return rhs.T@mapping, mapping, rhs.T


def main():
    u, v, w = qrule(8)
    _, bm, bb, j = kinematics(u, v)
    trial, test = strain_basis(u, v)
    ue, ve, we = qrule(16)
    ne, bme, bbe, je = kinematics(ue, ve)
    trial_e, _ = strain_basis(ue, ve)
    dm, db = THICKNESS*C, THICKNESS**3*C/12
    fixed = np.zeros((5, 5, 3), dtype=bool)
    fixed[0, :, :] = True
    fixed[1, :, 2] = True
    free = np.flatnonzero(~fixed.ravel())
    assert len(free) == 55
    compatible_m = material_product(bm, bm, dm, w*j)
    compatible_b = material_product(bb, bb, db, w*j)
    columns = ("displacement_L2_relative", "signed_face_stress_L2_relative", "free_relative_equilibrium_residual",
               "recovered_action_free_relative_residual", "discrete_work_gap_relative", "recovered_energy_work_gap_relative")
    path = Path(__file__).with_name("fresh_shell_sweep.csv")
    expected = {}
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as stream:
            expected = {(r["arm"],r["field"]):r for r in csv.DictReader(stream)
                        if (int(r["Nu"]),int(r["Nv"]),float(r["t_mm"])) == (2,2,10.)}
        assert len(expected) == 12
    print("arm,field,"+",".join(columns))
    for arm in ARMS:
        km, zm, gm = (compatible_m,None,None) if arm == "compatible" else projected(bm,dm,w,j,trial,test,arm!="native")
        kb, zb, gb = projected(bb,db,w,j,trial,test,True) if arm == "full_dual" else (compatible_b,None,None)
        k = km+kb
        for field in FIELDS:
            u_ref, m_ref, b_ref = source(field,ue,ve)
            f = (material_product(bme,m_ref[...,None],dm,we*je)
                 + material_product(bbe,b_ref[...,None],db,we*je)).ravel()
            coef = np.zeros(k.shape[0])
            coef[free] = solve(k[np.ix_(free, free)], f[free])
            internal_m = gm@(zm@coef) if zm is not None else km@coef
            internal_b = gb@(zb@coef) if zb is not None else kb@coef
            f_norm = np.linalg.norm(f[free])
            equilibrium = np.linalg.norm((k@coef-f)[free])/f_norm
            actual = np.linalg.norm((internal_m+internal_b-f)[free])/f_norm
            U = ne@coef.reshape(-1,3)
            membrane = trial_e@(zm@coef) if zm is not None else bme@coef
            bend = trial_e@(zb@coef) if zb is not None else bbe@coef
            displacement_error = np.sqrt(np.sum((we*je)*np.sum((U-u_ref)**2,axis=1))
                                         /np.sum((we*je)*np.sum(u_ref**2,axis=1)))
            num = den = 0.
            for signed in (-.5,.5):
                ref = (m_ref-signed*THICKNESS*b_ref)@C.T
                diff = (membrane-signed*THICKNESS*bend)@C.T - ref
                tensor_shear = np.array([1.,1.,2.])
                num += (we*je) @ (diff**2@tensor_shear)
                den += (we*je) @ (ref**2@tensor_shear)
            stress_error = np.sqrt(num/den)
            dwork = coef@f
            energy = coef@(k@coef)/2
            erec = (we*je)@(np.sum((membrane@dm)*membrane+(bend@db)*bend,axis=1))/2
            values = (displacement_error, stress_error, equilibrium, actual,
                      abs(energy-dwork/2)/energy, abs(erec-dwork/2)/erec)
            assert max(values[2:5]) < 1e-10, (arm,field,values)
            if expected:
                ref = expected[(arm,field)]
                for val,col in zip((values[0],values[1],values[5]),
                                   ("u_L2_relative","signed_two_face_tensor_stress_L2_relative","physical_energy_work_gap_relative_diagnostic")):
                    assert np.isclose(val,float(ref[col]),atol=5e-10,rtol=5e-6), (arm,field,col,val,ref[col])
            print(arm+","+field+","+",".join(f"{x:.16g}" for x in values))
    if expected:
        print("Twelve fresh reference cases match the sweep.")

if __name__=="__main__":
    main()


