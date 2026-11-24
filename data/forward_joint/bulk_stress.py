"""Evaluate projected membrane strain and compatible bending of the Q3 ship shells without CAD."""
import os
for key in ("MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS"): os.environ[key]="1"
import numpy as np
from scipy import sparse
from scipy.interpolate import NdBSpline
from scipy.sparse.linalg import splu

def dot(a,b):return np.einsum("...i,...i->...",a,b)

def projector(bases,normal_weight,shear_weight,nu):
    # Scale the retained columns only; Poisson coupling remains in the normal block.
    phi=[]
    for b,weight in zip(bases,[normal_weight,normal_weight,shear_weight]):
        d=np.asarray(b.power(2).T@weight).ravel()
        active=d.real>0
        if not active.any():raise ValueError("Empty strain space")
        phi.append(b[:,active].multiply(1/np.sqrt(d[active])).tocsr())
    a,b,s=phi
    wa=a.multiply(normal_weight[:,None]).tocsr()
    wb=b.multiply(normal_weight[:,None]).tocsr()
    ws=s.multiply(shear_weight[:,None]).tocsr()
    m12=wa.T@b
    m=sparse.bmat([[wa.T@a,nu*m12],[nu*m12.T,wb.T@b]],format="csc")
    ns=splu(m);ss=splu((ws.T@s).tocsc()); na=a.shape[1]
    def solve(y):
        c=ns.solve(np.r_[wa.T@(y[:,0]+nu*y[:,1]), wb.T@(nu*y[:,0]+y[:,1])])
        z=ss.solve(ws.T@y[:,2])
        return np.column_stack((a@c[:na],b@c[na:],s@z))
    def transpose(y):
        c=ns.solve(np.r_[a.T@y[:,0],b.T@y[:,1]],trans="T")
        g0=normal_weight*(a@c[:na]);g1=normal_weight*(b@c[na:])
        return np.column_stack((g0+nu*g1,nu*g0+g1,shear_weight*(s@ss.solve(s.T@y[:,2],trans="T"))))
    return solve,transpose

def derivative_matrix(knot,degree):
    n=len(knot)-degree-1
    den=knot[degree+1:-1]-knot[1:n]
    alpha=np.divide(degree,den,out=np.zeros(n-1),where=den>0)
    return sparse.diags((-alpha,alpha),(0,1),shape=(n-1,n),format="csr")

def derivative_basis(knots,uv,diff):
    factors=[]; reduced=[]; degrees=[]
    for t,order in zip(knots,diff):
        t=np.asarray(t);p=3; mat=sparse.eye(len(t)-4,format="csr")
        for _ in range(order):
            mat=derivative_matrix(t,p)@mat;t=t[1:-1];p-=1
        factors.append(mat);reduced.append(t);degrees.append(p)
    basis=NdBSpline.design_matrix(uv,tuple(reduced),tuple(degrees),extrapolate=False)
    return (basis@sparse.kron(*factors,format="csr")).tocsr()

def evaluate(knots,cp,director,material,rule,state,low=None,*,gradient=False,strains=False):
    uv=np.ascontiguousarray(rule[:,:2]);E,nu,t,offset=material
    if offset!=0 or not (E>0 and 0<t and -1<nu<.5):raise ValueError("Unsupported material")
    if state.shape!=cp.shape or not np.isfinite(state).all():raise ValueError("Invalid displacement")
    # Subtract a constant to avoid cancellation from large world coordinates.
    u=NdBSpline(knots,state-state[0,0],(3,3),extrapolate=False)
    geom=NdBSpline(knots,cp-cp[0,0],(3,3),extrapolate=False)
    lo=None if low is None else NdBSpline(knots,low-low[0,0],(3,3),extrapolate=False)
    def displacement(d):
        val=u(uv,nu=d)
        return val if lo is None else val+lo(uv,nu=d)
    xu=geom(uv,nu=(1,0));xv=geom(uv,nu=(0,1))
    uu=displacement((1,0));uv_disp=displacement((0,1))
    nc=np.cross(xu,xv); jac=np.sqrt(dot(nc,nc))
    if min(jac.real)<=0 or not np.isfinite(jac).all(): raise ValueError("Degenerate surface")
    n=nc/jac[:,None];du=np.cross(xv,n)/jac[:,None];dv=np.cross(n,xu)/jac[:,None]
    e1=director-(n@director)[:,None]*n
    e1_length=np.sqrt(dot(e1,e1))
    if min(e1_length.real)<=1.e-12*np.linalg.norm(director):raise ValueError("Invalid material chart")
    e1/=e1_length[:,None];e2=np.cross(n,e1)
    a,b,c,d=dot(e1,du),dot(e1,dv),dot(e2,du),dot(e2,dv)
    def material_components(e,f,h):
        return np.column_stack((a*a*e+b*b*f+a*b*h,c*c*e+d*d*f+c*d*h,2*a*c*e+2*b*d*f+(a*d+b*c)*h))
    eps=material_components(dot(xu,uu),dot(xv,uv_disp),dot(xu,uv_disp)+dot(xv,uu))
    nd=-dot(n,uu)[:,None]*du-dot(n,uv_disp)[:,None]*dv
    def curvature(d):return dot(n,displacement(d))+dot(nd,geom(uv,nu=d))
    kap=material_components(curvature((2,0)),curvature((0,2)),2*curvature((1,1)))
    weights=rule[:,2]*rule[:,3]*jac
    if min(weights.real)<=0:raise ValueError("Nonpositive quadrature")
    powers=[(2,3),(3,2),(2,2)]
    basis=[NdBSpline.design_matrix(uv,tuple(k[1:-1] if p==2 else k for k,p in zip(knots,pair)),pair,extrapolate=False).tocsr() for pair in powers]
    project,transpose=projector(basis,weights*(E*t/(1-nu*nu)),weights*(E*t/(2*(1+nu))),nu)
    eps=project(eps)
    if strains:return eps,kap,weights
    fibre=eps[:,None,:]+kap[:,None,:]*np.array([.5*t,-.5*t])[None,:,None]
    sx=E/(1-nu*nu)*(fibre[:,:,0]+nu*fibre[:,:,1]);sy=E/(1-nu*nu)*(fibre[:,:,1]+nu*fibre[:,:,0])
    tau=E/(2*(1+nu))*fibre[:,:,2]
    vm=np.sqrt(.5*((sx-sy)**2+sx*sx+sy*sy)+3*tau*tau)
    if not np.isfinite(vm).all():raise ValueError("Invalid recovered stress")
    if not gradient: return vm,jac
    # Patch KS uses the physical sharpness. Multiplying by the patch's global weight
    # gives the corresponding contribution to the whole-ship displacement gradient.
    if np.any(vm==0): raise ValueError("Von Mises derivative is undefined at zero")
    z=np.exp(.5*(vm-vm.max()));z/=z.sum()
    dsx=z*(2*sx-sy)/(2*vm);dsy=z*(2*sy-sx)/(2*vm)
    material_gradient=np.stack((E/(1-nu*nu)*(dsx+nu*dsy),E/(1-nu*nu)*(nu*dsx+dsy),
                               E/(2*(1+nu))*(z*3*tau/vm)),axis=-1)
    def pull(y):
        g,h,i=y.T
        return np.column_stack((g*a*a+h*c*c+i*2*a*c,g*b*b+h*d*d+i*2*b*d,g*a*b+h*c*d+i*(a*d+b*c)))
    membrane=pull(transpose(material_gradient.sum(axis=1)))
    bend=pull(.5*t*(material_gradient[:,0]-material_gradient[:,1]))
    g,h,i=membrane.T;r,f,z=bend.T
    shape=r[:,None]*geom(uv,nu=(2,0))+f[:,None]*geom(uv,nu=(0,2))+2*z[:,None]*geom(uv,nu=(1,1))
    at_u=g[:,None]*xu+i[:,None]*xv-dot(du,shape)[:,None]*n
    at_v=h[:,None]*xv+i[:,None]*xu-dot(dv,shape)[:,None]*n
    state_grad=np.zeros((cp.shape[0]*cp.shape[1],3))
    for deriv,field in [((1,0),at_u),((0,1),at_v),((2,0),r[:,None]*n),((0,2),f[:,None]*n),((1,1),2*z[:,None]*n)]:
        state_grad+=derivative_basis(knots,uv,deriv).T@field
    return vm,jac,state_grad.reshape(cp.shape)
