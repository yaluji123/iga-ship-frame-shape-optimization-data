"""One Q3 displacement on a moving retained interval. Outputs normalized membrane quantities."""
import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

ORDER = 8
DEGREES = ((2, 3), (3, 2), (2, 2))
# Outward co-normal on the right cut is +x, and rotations map material components to x,y.
FRAME = np.array([[0., -1.], [1., 0.]])
# Exact Bernstein coefficients for u_y=eta*p(xi) on the full unit carrier.
CONTROL_Y = np.outer([-1., 3., -3., 1.], np.arange(4) / 3.)
FD_STEPS = (2.**-12, 2.**-13, 2.**-14)


def basis(n, x):
    return np.stack([comb(n,i)*x**i*(1.-x)**(n-i) for i in range(n+1)], axis=1)


def sample(x, y, frame=FRAME):
    bx = basis(3,x)
    by = basis(3,y)
    deriv_x = 3.*np.einsum("ni,ij,nj->n", basis(2,x), np.diff(CONTROL_Y,axis=0), by)
    deriv_y = 3.*np.einsum("ni,ij,nj->n", bx, np.diff(CONTROL_Y,axis=1), basis(2,y))
    tensor = np.zeros((len(x),2,2))
    tensor[:,1,1] = deriv_y
    tensor[:,0,1] = tensor[:,1,0] = 0.5*deriv_x
    rotated = frame.T @ tensor @ frame
    strain = np.column_stack((rotated[:,0,0], rotated[:,1,1], 2*rotated[:,0,1]))
    phi = np.zeros((len(x),3,33))
    start = 0
    for component,(nx,ny) in enumerate(DEGREES):
        values = (basis(nx,x)[:,:,None]*basis(ny,y)[:,None,:]).reshape(len(x),-1)
        phi[:,component,start:start+values.shape[1]] = values
        start += values.shape[1]
    return phi, strain


def quadrature(a):
    points, weights = leggauss(ORDER)
    half = 0.5*(points+1)
    x, y = np.meshgrid(a*half,half,indexing="ij")
    return x.ravel(), y.ravel(), (0.25*a*np.outer(weights,weights)).ravel()


def projected_coefficients(phi, strain, weight, metric):
    root = np.linalg.cholesky(metric).T
    scales = np.sqrt(weight)
    A = np.einsum("ij,njk,n->nik", root,phi,scales).reshape(-1,33)
    b = np.einsum("ij,nj,n->ni",root,strain,scales).ravel()
    q,r = np.linalg.qr(A,mode="reduced")
    return np.linalg.solve(r,q.T@b)


def solve(a, nu, frozen=None, frame=FRAME):
    x,y,w = quadrature(a)
    phi,strain = sample(x,y,frame)
    D = np.array([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1.-nu)/2.]])
    z = frozen if frozen is not None else [projected_coefficients(phi,strain,w,m)
                                           for m in (D,np.eye(3))]
    eps = [phi @ coef for coef in z]
    energies = np.array([0.5*np.einsum("ni,ij,nj,n",e,D,e,w) for e in eps])
    # Evaluate the traction at one off-quadrature point; its x component is constant on this cut.
    trace,_ = sample(np.array([a]),np.array([0.371]),frame)
    tractions = []
    for coef in z:
        n1,n2,n12 = D @ (trace[0] @ coef)
        stress = frame @ np.array([[n1,n12],[n12,n2]]) @ frame.T
        tractions.append(stress @ np.array([1.,0.]))
    return z,energies,np.array(tractions),eps,strain


def poly(x):
    return ((20*x-30)*x+12)*x-1


def check(a,nu):
    x,y,_ = quadrature(a)
    r = a**3*poly(x/a)
    q = poly(x)-r
    shear = -y*(60*x*x-60*x+12)
    z,energies,tractions,eps,strain = solve(a,nu)
    expected_strains = (np.column_stack((q,nu*r,shear)),np.column_stack((q,0*r,shear)))
    compatible = np.column_stack((poly(x),0*r,shear))
    field_error = max(float(np.max(abs(strain-compatible))),
                      *(float(np.max(abs(e-v))) for e,v in zip(eps,expected_strains)))
    pred_gap = nu**2*a**7/14
    pred_gap_derivative = nu**2*a**6/2
    pred_freeze = (1-nu**2)*(poly(a)-a**3)*a**3
    pred_full = (poly(a)**2+(1-nu)*(60*a*a-60*a+12)**2/6.-(1-nu**2)*a**6)/2.
    # Central differences rebuild quadrature on both sides; frozen keeps the same parent-chart polynomials.
    fd = []
    for h in FD_STEPS:
        full = (solve(a+h,nu)[1]-solve(a-h,nu)[1])/(2*h)
        fixed = (solve(a+h,nu,z)[1][0]-solve(a-h,nu,z)[1][0])/(2*h)
        fd.append(np.array([full[0],full[0]-full[1],full[0]-fixed]))
    extrapolated = (4*fd[-1]-fd[-2])/3.
    actual_gap = float(energies[0]-energies[1])
    traction_delta = tractions[0]-tractions[1]
    # On this square the parameter directions coincide with x,y. Keep the material metric,
    # quadrature and displacement identical and assign the component spaces in that frame.
    axes = np.eye(2)
    _, aligned_energies, aligned_tractions, aligned_fields, aligned_strain = solve(a, nu, frame=axes)
    frame_gap = aligned_energies[0] - energies[0]
    aligned_fd = []
    for h in FD_STEPS[-2:]:
        plus = solve(a + h, nu, frame=axes)[1][0]
        minus = solve(a - h, nu, frame=axes)[1][0]
        aligned_fd.append((plus - minus) / (2*h))
    aligned_derivative = (4*aligned_fd[1] - aligned_fd[0]) / 3
    aligned_exact = pred_full + (1-nu**2)*a**6/2
    errors = dict(field=field_error,energy=abs(actual_gap-pred_gap),
                  traction=float(np.max(abs(traction_delta-np.array([nu*a**3,0.])))),
                  derivative=float(np.max(abs(extrapolated-[pred_full,pred_gap_derivative,pred_freeze]))))
    errors['frame_comparison'] = max(float(np.max(abs(aligned_fields[0]-aligned_strain))),
                                     abs(frame_gap-(1-nu**2)*a**7/14),
                                     float(np.max(abs(aligned_tractions[0]-tractions[0]))))
    errors['aligned_derivative'] = abs(aligned_derivative-aligned_exact)
    if errors['frame_comparison'] > 2e-11 or errors['aligned_derivative'] > 5e-9:
        raise ArithmeticError((a,nu,errors))
    if max(errors[k] for k in ("field","energy","traction")) > 2e-11 or errors["derivative"] > 5e-9:
        raise ArithmeticError((a,nu,errors))
    return dict(a=a,nu=nu,material_energy=float(energies[0]),component_energy=float(energies[1]),
                energy_gap=actual_gap,traction_difference_xy=traction_delta.tolist(),
                aligned_material_energy=float(aligned_energies[0]),
                aligned_minus_transverse_energy=float(frame_gap),
                aligned_energy_derivative_fd=float(aligned_derivative),
                aligned_energy_derivative_exact=float(aligned_exact),
                fd_column_order=["dE_material","d_energy_gap","dE_material_minus_frozen"],
                fd_central_by_step=[v.tolist() for v in fd],fd_richardson=extrapolated.tolist(),
                analytic_derivatives=[pred_full,pred_gap_derivative,pred_freeze],absolute_errors=errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    cases = [check(a,nu) for a in (.5,.8,.95) for nu in (0.,.3,-.2)]
    output = dict(status="VERIFIED_SINGLE_CELL_FIXED_DISPLACEMENT_IDENTITY",
                  scope="Independent dense planar membrane projection; no global equilibrium solve or interface penalty.",
                  units="x=L*xi, y=L*eta, u_y=L*eps0*eta*p(xi); energy/(c*eps0^2*L^2), traction/(c*eps0), derivative with respect to dimensionless a.",
                  material="c=t*E/(1-nu^2); normalized c=1 in each case",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  numpy_version=np.__version__,quadrature_per_direction=ORDER,projection_unknowns=33,
                  coordinate_basis="parent-cell Bernstein in xi,eta; material frame e1=y,e2=-x",
                  fd_steps=FD_STEPS,absolute_tolerances=dict(field_energy_traction=2e-11,richardson_derivatives=5e-9),
                  cases=cases)
    data=json.dumps(output,indent=2,allow_nan=False)
    if args.output:
        with args.output.open("x",encoding="utf8") as stream:
            stream.write(data+"\n")
    else:
        print(data)
