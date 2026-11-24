"""Independent moving-cut derivative test with the skew/Nitsche whole-chain endpoint rule."""

import json
from math import comb, factorial
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BSpline
from scipy.linalg import eigh, lu_factor, lu_solve, solve, svd

HERE = Path(__file__).resolve().parent
YOUNG = 210000.0
POISSON = 0.3
THICKNESS = 0.1
NORMAL = YOUNG / (1 - POISSON**2)
SHEAR = YOUNG / (2 * (1 + POISSON))
BEND = NORMAL * THICKNESS**3 / 12
RHO = 0.5
ALPHA = 0.2
DELTA = 1e-24
N_SPANS = (3, 5)
SPECTRAL_GAP_FLOOR = 1e-6
GAUSS, WEIGHTS = leggauss(10)
TR, TW = (GAUSS + 1) / 2, WEIGHTS / 2


def quadrature(breaks):
    width = np.diff(breaks)
    return (breaks[:-1, None] + width[:, None] * TR).ravel(), (width[:, None] * TW).ravel()


def spread(b, axis):
    R = np.zeros((b.shape[0], 3*b.shape[1]), dtype=b.dtype)
    R[:, axis::3] = b
    return R


def line_input(alpha):
    breaks = np.unique(np.concatenate([np.linspace(0., 1., n+1) for n in N_SPANS]))
    x, wx = quadrature(breaks)
    x = np.r_[x, 0., 1.]
    y = 4*alpha*x*(1-x)
    slope = 4*alpha*(1-2*x)
    speed = np.sqrt(1+slope**2)
    tangent = np.column_stack((1/speed, slope/speed))
    tangent_arc = -8*alpha*np.column_stack((-slope, np.ones(len(x))))/speed[:, None]**4
    normal = np.column_stack((-tangent[:,1], tangent[:,0]))
    normal_arc = np.column_stack((-tangent_arc[:,1], tangent_arc[:,0]))
    w = wx*speed[:-2]
    h = np.sum(w)
    return x, y, tangent, tangent_arc, normal, normal_arc, np.r_[w, h, h], h


class Patch:
    def __init__(self, n, alpha, retained):
        self.n = n
        self.cubic_knots = np.r_[np.zeros(4), np.arange(1,n)/n, np.ones(4)]
        self.dofs = (n+3)*4*3
        x, wx = quadrature(np.linspace(0., 1., n+1))
        y_top = 4*alpha*x*(1-x) if retained else np.ones_like(x)
        thickness = (1+y_top)[:, None]
        self.x = np.repeat(x, len(TR))
        self.y = (-1 + thickness*TR).ravel()
        self.weights = (wx[:, None]*thickness*TW).ravel()
        self.b = self.jets(self.x, self.y, ((1,0),(0,1),(2,0),(0,2),(1,1)))
        nx = spread(self.b[1,0],0)
        ny = spread(self.b[0,1],1)
        gamma = spread(self.b[0,1],0)+spread(self.b[1,0],1)

        a,b,c = self.strain_bases(self.x,self.y)
        wa,wb,wc = (v*self.weights[:,None] for v in (a,b,c))
        ab = POISSON*a.T@wb
        M = np.block([[a.T@wa, ab],[ab.T,b.T@wb]])
        G = np.concatenate((wa.T@(nx+POISSON*ny),wb.T@(ny+POISSON*nx)))
        Z = solve(M,G,check_finite=False)
        m = c.T@wc
        g = wc.T@gamma
        z = solve(m,g,check_finite=False)
        self.maps = Z[:a.shape[1]], Z[a.shape[1]:], z
        membrane = THICKNESS*(NORMAL*G.T@Z + SHEAR*g.T@z)
        xx,yy,xy = (spread(self.b[k],2) for k in ((2,0),(0,2),(1,1)))
        bend = BEND*(xx.T@(self.weights[:,None]*(xx+POISSON*yy))+
                     yy.T@(self.weights[:,None]*(yy+POISSON*xx))+
                     2*(1-POISSON)*xy.T@(self.weights[:,None]*xy))
        self.bulk = membrane + bend

    def basis(self,x,y,px=3,py=3,ix=0,iy=0):
        knots = self.cubic_knots if px==3 else self.cubic_knots[1:-1]
        values = BSpline(knots,np.eye(len(knots)-px-1),px,extrapolate=False)(x,nu=ix)
        v = (1+y)/2
        # Analytic continuation of the one-span transverse Bernstein basis.
        if iy>py:
            bern = np.zeros((len(v),py+1),dtype=v.dtype)
        else:
            order=py-iy
            lower=np.stack([comb(order,j)*v**j*(1-v)**(order-j) for j in range(order+1)],axis=1)
            bern=np.zeros((len(v),py+1),dtype=v.dtype)
            amplitude=factorial(py)/factorial(order)*.5**iy
            for j in range(iy+1):
                bern[:,j:j+order+1]+=amplitude*(-1)**(iy-j)*comb(iy,j)*lower
        return np.einsum("pi,pj->pij",values,bern).reshape(len(v),-1)

    def jets(self,x,y,keys):
        return {k:self.basis(x,y,ix=k[0],iy=k[1]) for k in keys}

    def strain_bases(self,x,y):
        return tuple(self.basis(x,y,px,py) for px,py in ((2,3),(3,2),(2,2)))

    def membrane(self,x,y,bases=None):
        a,b,c=self.strain_bases(x,y) if bases is None else bases
        r,s,t=(v@z for v,z in zip((a,b,c),self.maps))
        return r,s,t

    def stresses(self):
        e0,e1,e2=self.membrane(self.x,self.y)
        b0,b1,b2 = (spread(self.b[k],2) for k in ((2,0),(0,2),(1,1)))
        blocks=[]
        for z in (-THICKNESS/2,THICKNESS/2):
            x=e0-z*b0
            y=e1-z*b1
            xy=e2-2*z*b2
            blocks.append(np.stack((NORMAL*(x+POISSON*y),NORMAL*(y+POISSON*x),SHEAR*xy),axis=1))
        return np.concatenate(blocks)

    def traces(self,x,y,t,t_s,mu,mu_s,active):
        keys=((0,0),(1,0),(0,1))
        if active:
            keys+=((2,0),(0,2),(1,1),(3,0),(2,1),(1,2),(0,3))
        b=self.jets(x,y,keys)
        rot=np.column_stack((-t[:,1],t[:,0]))
        N=np.stack([*(spread(b[0,0],i) for i in range(3)),
                      spread(rot[:,0,None]*b[1,0]+rot[:,1,None]*b[0,1],2)],axis=1)
        if not active:
            return N,None,None
        e0,e1,e2=self.membrane(x,y)
        sx,sy,ss=THICKNESS*NORMAL*(e0+POISSON*e1),THICKNESS*NORMAL*(e1+POISSON*e0),THICKNESS*SHEAR*e2
        Vx=mu[:,0,None]*sx+mu[:,1,None]*ss
        Vy=mu[:,0,None]*ss+mu[:,1,None]*sy

        def tensor(xx,yy,xy):
            q11=BEND*(xx+POISSON*yy)
            q22=BEND*(yy+POISSON*xx)
            q12=BEND*(1-POISSON)*xy
            return np.stack((np.stack((q11,q12),axis=1),np.stack((q12,q22),axis=1)),axis=1)

        mul=lambda Q,u,v:np.einsum("pi,pijd,pj->pd",u,Q,v)
        Q=tensor(b[2,0],b[0,2],b[1,1])
        Qs=tensor(t[:,0,None]*b[3,0]+t[:,1,None]*b[2,1],
                  t[:,0,None]*b[1,2]+t[:,1,None]*b[0,3],
                  t[:,0,None]*b[2,1]+t[:,1,None]*b[1,2])
        twist=mul(Q,t,mu)
        twist_s=mul(Q,t_s,mu)+mul(Qs,t,mu)+mul(Q,t,mu_s)
        div_mu=BEND*(mu[:,0,None]*(b[3,0]+b[1,2])+mu[:,1,None]*(b[2,1]+b[0,3]))
        V=np.stack((Vx,Vy,spread(-div_mu-twist_s,2),spread(mul(Q,rot,mu),2)),axis=1)
        S=np.zeros_like(V)
        S[:,2]=spread(twist,2)
        return N,V,S

    def fixed_exterior_load(self, y, magnitude):
        x,weights=quadrature(np.linspace(0.,1.,self.n+1))
        load=(1.+.3*x+.24*np.sin(2*np.pi*x))[:,None]*np.asarray(magnitude)[None,:]
        coeff=self.basis(x,np.full_like(x,y)).T@(weights[:,None]*load)
        return coeff.ravel()


def rows(alpha):
    a=Patch(N_SPANS[0],alpha,True)
    b=Patch(N_SPANS[1],alpha,False)
    x,y,t,t_s,mu,mu_s,w,h=line_input(alpha)
    N,V,S=a.traces(x,y,t,t_s,mu,mu_s,True)
    Nr,_,_=b.traces(x,y,t,t_s,mu,mu_s,False)
    # Transport the passive kinematics at the same physical material edge.
    jump=np.concatenate((N,-Nr),axis=2)
    V[-2:]=np.array([-1.,1.])[:,None,None]*S[-2:]/h
    return (a,b),jump,V,S,w,h


def weighted_trace_blocks(jump,V,w):
    b=len(jump[0,0])-V.shape[2]
    result=[]
    for sl in (slice(0,3),slice(3,4)):
        J=jump[:,sl]
        f=V[:,sl]
        A=np.concatenate((f,np.zeros((*f.shape[:-1],b),dtype=f.dtype)),axis=2)
        skew=np.einsum("psj,psk,p->jk",J,A,w,optimize=True)
        H=np.einsum("psj,psk,p->jk",J,J,w,optimize=True)
        result.append((skew,H))
    return result


def calibration(K,V,W):
    scale=np.diag(K)**-.5
    A=K*scale[:,None]*scale[None,:]
    lam,Q=eigh(.5*(A+A.T),check_finite=False)
    assert np.sum(lam<lam[-1]*1e-10)==6
    coords=scale[:,None]*Q
    claims=[]
    for sl in (slice(0,3),slice(3,4)):
        weighted = (np.sqrt(W[:,None,None])*V[:,sl]).reshape(-1,len(K))@coords
        rigid=float(np.linalg.norm(weighted[:,:6])/np.linalg.norm(weighted[:,6:]))
        assert rigid < 1e-8, rigid
        _,s,rt = svd(weighted[:,6:]/np.sqrt(lam[6:]),full_matrices=False,check_finite=False)
        d=float(s[0]**2)
        gap=float(1-(s[1]/s[0])**2)
        assert gap >= SPECTRAL_GAP_FLOOR, gap
        witness=coords[:,6:]@(rt[0]/np.sqrt(lam[6:]))
        energy=float(witness@K@witness)
        trace=V[:,sl]@witness
        assert abs(float(np.einsum("ps,ps,p->",trace,trace,W))/energy/d-1)<1e-9
        claims.append(dict(d=d, gap=gap, witness=witness, energy=energy, rigid=rigid))
    return claims


def eta_direction(cal, bulk, V,w):
    result=[]
    Kdot = bulk.imag/DELTA
    for sl,c in zip((slice(0,3),slice(3,4)),cal):
        r=c["witness"]
        flux=V[:,sl]@r
        measured=np.einsum("ps,ps,p->",flux,flux,w).imag/DELTA
        derivative=9*(measured-c["d"]*(r@Kdot@r))/c["energy"]
        result.append(float(derivative))
    return np.array(result)


def matrix(patch, blocks, eta):
    na=patch[0].dofs
    K=np.zeros((na+patch[1].dofs,)*2,dtype=blocks[0][0].dtype)
    K[:na,:na]=patch[0].bulk
    K[na:,na:]=patch[1].bulk
    for (B,H),v in zip(blocks,eta):
        K+=-B+B.T+v*H
    return K


def stress_response(patch,state,g=False,shift=None):
    first=patch[0].dofs
    s=[q.stresses() for q in patch]
    sig=np.concatenate([s[0]@state[:first],s[1]@state[first:]])
    x,y,t=sig.T
    vm=np.sqrt(x*x-x*y+y*y+3*t*t)
    assert np.min(vm.real)>1e-5
    shift=float(vm.real.max()) if shift is None else shift
    wt=np.exp(RHO*(vm-shift))
    value=shift+np.log(wt.sum())/RHO
    if g:
        weights=wt/wt.sum()
        rows=np.column_stack((2*x-y,2*y-x,6*t))*((.5*weights/vm)[:,None])
        count=len(s[0])
        return value,np.concatenate((np.einsum("pi,pin->n",rows[:count],s[0]),
                                     np.einsum("pi,pin->n",rows[count:],s[1]))),float(vm.min()),shift
    return value


def forward(alpha, keep=False):
    patch,J,V,S,W,h=rows(alpha)
    cal=calibration(patch[0].bulk,V,W)
    eta=np.array([9*c["d"] for c in cal])
    blocks=weighted_trace_blocks(J,V,W)
    K=matrix(patch,blocks,eta)
    m=patch[0].dofs
    F=np.r_[patch[0].fixed_exterior_load(-1.,(.025,-.018,.025)),
            patch[1].fixed_exterior_load(1.,(.013,-.058,-.062))]
    free=np.ones(len(F),dtype=bool)
    free[[0,1,2,m-11,m-10,m+11]]=False
    A=K[free][:,free]
    s=np.diag(A)**-.5
    lu=lu_factor(s[:,None]*A*s[None,:],check_finite=False)
    u=np.zeros(len(F))
    u[free]=s*lu_solve(lu,s*F[free],check_finite=False)
    res=float(np.linalg.norm((K@u-F)[free])/np.linalg.norm(F[free]))
    assert res<1e-9
    js,gs,minvm,shift=stress_response(patch,u,g=True)
    value=np.array([js,F@u])
    result=dict(a=alpha,response=value.tolist(),eta=eta.tolist(),spectral_gaps=[x["gap"] for x in cal],
                rigid_traction_relative=[x["rigid"] for x in cal],free_residual=res,h=float(h),min_vm=float(minvm),
                sample_count=sum(len(q.x)*2 for q in patch),free_dofs=int(free.sum()),skew_fraction=float(np.linalg.norm(A-A.T)/np.linalg.norm(A)))
    if keep:
        rhs=np.column_stack((gs,F))
        adj=np.zeros((len(F),2))
        adj[free]=s[:,None]*lu_solve(lu,s[:,None]*rhs[free],trans=1,check_finite=False)
        ar=float(np.linalg.norm((K.T@adj-rhs)[free])/np.linalg.norm(rhs[free]))
        assert ar<1e-9
        result["transpose_adjoint_residual"]=ar
        return result,(patch,J,V,S,W,h,cal,eta,blocks,K,F,u,adj,shift)
    return result


def gradient(alpha):
    truth,b=forward(alpha,keep=True)
    _,j0,_,s0,w0,_,cal,eta,hb,_,_,u,adj,bshift=b
    patch,j,v,s,w,h=rows(alpha+1j*DELTA)
    etad=eta_direction(cal,patch[0].bulk,v,w)
    block=weighted_trace_blocks(j,v,w)
    Kdot=matrix(patch,block,eta+1j*DELTA*etad).imag/DELTA
    explicit=np.array([stress_response(patch,u,shift=bshift).imag/DELTA,0.])
    response=explicit-adj.T@(Kdot@u)
    truth.update(eta_shape_direction=etad.tolist(),chain_length_direction=float(h.imag/DELTA),
                 endpoint_S_direction_norm=float(np.linalg.norm(s[-2:].imag/DELTA)),
                 frozen_state_explicit_part=explicit.tolist(),adjoint_derivative=response.tolist())
    n0=v.shape[2]
    end_s=np.zeros_like(Kdot)
    end_h=np.zeros_like(Kdot)
    penalty=np.zeros_like(Kdot)
    for k,sl in enumerate((slice(0,3),slice(3,4))):
        Js=j0[-2:,sl]
        target=s[-2:,sl].imag/DELTA*np.array([-1.,1.])[:,None,None]
        c=np.einsum("psj,psk->jk",Js,target)
        E=np.zeros_like(Kdot)
        E[:,:n0]=c
        end_s+=-E+E.T
        end_h+=eta[k]*(h.imag/DELTA)*np.einsum("psj,psk->jk",Js,Js)
        penalty+=etad[k]*hb[k][1]
    omit={}
    for key,term in (("direct_endpoint_force",end_s),("explicit_endpoint_length_in_jump_penalty",end_h),
                     ("calibration_penalty_total_derivative",penalty),("all_three",end_s+end_h+penalty)):
        d=adj.T@(term@u)
        omit[key]=dict(omitted_derivative_matrix_frobenius_norm=float(np.linalg.norm(term)),
                       change_from_full_derivative=d.tolist(),omitted_derivative=(response+d).tolist())
    return truth,omit


def run_study():
    base,omit=gradient(ALPHA)
    analytic=np.asarray(base["adjoint_derivative"])
    reference=[]
    for h in (0.004,0.002,0.001,0.0005):
        plus=forward(ALPHA+h)
        minus=forward(ALPHA-h)
        fd=(np.asarray(plus["response"])-minus["response"])/(2*h)
        absolute=abs(fd-analytic)
        taylor=abs(np.asarray(plus["response"])-base["response"]-h*analytic)
        assert plus["sample_count"]==minus["sample_count"]==base["sample_count"]
        eta_fd=(np.asarray(plus["eta"])-minus["eta"])/(2*h)
        arc_fd=(plus["h"]-minus["h"])/(2*h)
        reference.append(dict(h=h,plus=plus,minus=minus,centered_fd=fd.tolist(),absolute_error=absolute.tolist(),
                              relative_error=(absolute/abs(analytic)).tolist(),forward_Taylor_remainder=taylor.tolist(),
                              eta_centered_fd=eta_fd.tolist(),chain_length_centered_fd=arc_fd))
    for a,b in zip(reference,reference[1:]):
        fourth=np.asarray(a["forward_Taylor_remainder"])/b["forward_Taylor_remainder"]
        assert np.all((fourth>3.5)&(fourth<4.5)),fourth
    assert np.allclose(reference[-1]["eta_centered_fd"],base["eta_shape_direction"],rtol=2e-5,atol=1e-9)
    assert np.isclose(reference[-1]["chain_length_centered_fd"],base["chain_length_direction"],rtol=2e-5,atol=1e-9)
    assert max(reference[-1]["relative_error"])<2e-5,reference
    e0=np.array(reference[0]["absolute_error"])
    e1=np.array(reference[1]["absolute_error"])
    assert np.min(e0/e1)>3.5,(e0,e1)
    for label,value in omit.items():
        rel=abs(np.asarray(value["change_from_full_derivative"]))/np.asarray(reference[-1]["absolute_error"])
        value["omission_difference_to_finest_fd_error"]=rel.tolist()
        assert min(rel)>100.,(label,rel)
    return dict(scope=("Two independent coplanar cubic Kirchhoff–Love shells joined only along the shared moving cut; "
                       "retained edge on shell one, interior trace on the uncut shell two. Fixed carrier coordinates. "
                       "This evaluates the equations and is not a production-code derivative replay."),
                geometry=dict(length_mm=1.,carrier_width_mm=2.,trim="y=4*a*x*(1-x)",
                              analysis_spans_longitudinal=list(N_SPANS),transverse_spans=[1,1],
                              thickness_mm=THICKNESS,young_modulus_MPa=YOUNG,poisson_ratio=POISSON),
                method=("Analytic polynomial transverse basis with complex-step directions for the moving quadrature, projected "
                        "bulk matrices, recovered stress and physical trace rows. The leading simple generalized eigenvalue "
                        "direction is computed at its normalized real witness, rather than with a complex eigensolve. "
                        "The state factorization is reused in transpose mode for the two adjoints. Reference differences are "
                        "real, independently assembled state solves with recomputed spectral calibration."),
                endpoint_omission_scope=("Each omission alters only the named derivative contribution. The full forward operator, "
                                         "original state and adjoints, moving sample convention, and all other directional components "
                                         "retain their original values. The endpoint length omission removes the explicit h' in the "
                                         "endpoint jump penalty, preserving its indirect contribution through eta'. The skew "
                                         "endpoint-work product w_e V_e already cancels its h factors exactly."),
                responses=["unweighted KS over both faces (MPa)","compliance (N mm)"],
                load_definition=("Exterior edge load intensity factor 1+.3*x+.24*sin(2*pi*x), where x is the numerical "
                                 "coordinate on [0,1]. At y=-1 on the active side, the intensity multiplier is (.025,-.018,.025) N/mm; "
                                 "at y=1 on the receiving side it is (.013,-.058,-.062) N/mm. Six displacement values "
                                 "at three exterior corners are zero. There are no fixed interface endpoints."),
                shape_units="a is the mid-span trim height in mm; x represents x/(1 mm) in the trim formula.",
                sharpness_per_MPa=RHO,shape_value=ALPHA,complex_step=DELTA,
                spectral_gap_floor=SPECTRAL_GAP_FLOOR,base=base,real_differences=reference,omissions=omit)


if __name__=="__main__":
    data=run_study()
    (HERE/"moving_skew_gradient.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf8")
    print("base",data["base"])
    for x in data["real_differences"]:
        print(x["h"],x["relative_error"],x["forward_Taylor_remainder"])
    for key,x in data["omissions"].items():
        print(key,x["change_from_full_derivative"],x["omission_difference_to_finest_fd_error"])
