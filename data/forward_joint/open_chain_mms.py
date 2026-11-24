"""Manufactured open curved cut with the skew Nitsche sign and whole-chain end weights."""

from pathlib import Path
import os
import sys
import json
import hashlib

for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[v] = "1"

import numpy as np
from scipy.interpolate import BSpline, NdBSpline
from scipy.linalg import eigh, solve, svdvals
from numpy.polynomial.legendre import leggauss

HERE = Path(__file__).resolve().parent
SOURCE = HERE
from bulk_matrix import stiffness

E, nu, thickness = 210000., .3, .10
MATERIAL = [E, nu, thickness, 0.]
DIRECTOR = np.array([1., 0., 0.])
D = E * thickness**3 / (12 * (1. - nu**2))
Dm = E * thickness / (1. - nu**2)
wscale = .002
Ugrad = .001 * np.array([[.4, .3], [.1, -.2]])
SIGMA = Dm * np.array([[Ugrad[0,0] + nu*Ugrad[1,1], (1-nu)*.5*(Ugrad[0,1]+Ugrad[1,0])],
                       [(1-nu)*.5*(Ugrad[0,1]+Ugrad[1,0]), nu*Ugrad[0,0] + Ugrad[1,1]]])
GP, GW = leggauss(12)


def line(a, b):
    return (a+b)/2 + (b-a)/2*GP, (b-a)/2*GW


def intervals(n):
    for i in range(n):
        yield line(i/n,(i+1)/n)


def trace_height(x, amplitude):
    return 4.*amplitude*x*(1.-x)


def exact(xy, quartic):
    x,y = xy.T
    u = np.column_stack((xy@Ugrad.T, wscale*(x*y+quartic*x**4)))
    grad = wscale*np.column_stack((y+4*quartic*x**3,x))
    Q = np.zeros((len(x),2,2))
    Q[:,0,0] = D*wscale*12*quartic*x*x
    Q[:,1,1] = nu*Q[:,0,0]
    Q[:,0,1] = Q[:,1,0] = D*(1-nu)*wscale
    div = np.column_stack((D*wscale*24*quartic*x,np.zeros(len(x))))
    return u,grad,Q,div


def exact_t(xy, quartic):
    x,y = xy.T
    k = .18
    u = np.column_stack((xy@UGRAD_T.T, wscale*((x+quartic*x**4)*y+k*y*y)))
    grad = wscale*np.column_stack(((1.+4.*quartic*x**3)*y, x+quartic*x**4+2.*k*y))
    Q = np.zeros((len(x),2,2))
    Q[:,0,0] = D*wscale*(12.*quartic*x*x*y+2.*nu*k)
    Q[:,1,1] = D*wscale*(12.*nu*quartic*x*x*y+2.*k)
    Q[:,0,1] = Q[:,1,0] = D*(1.-nu)*wscale*(1.+4.*quartic*x**3)
    div = D*wscale*np.column_stack((24.*quartic*x*y,12.*quartic*x*x))
    return u,grad,Q,div


UGRAD_T = Ugrad.copy()
UGRAD_T[1,0] = 0.
SIGMA_T = SIGMA.copy()
SIGMA_T[0,1] = SIGMA_T[1,0] = Dm*(1.-nu)*.5*UGRAD_T[0,1]
RX90 = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])


def spline_basis(knots, xy, powers):
    v = np.column_stack((xy[:,0],(xy[:,1]+1.)*.5))
    values = [BSpline(k, np.eye(len(k)-4), 3, extrapolate=False)(v[:,i], nu=power)
              for i, (k,power) in enumerate(zip(knots,powers))]
    b = np.einsum("pi,pj->pij", *values).reshape(len(v),-1)
    return b * .5**powers[1]


def shape_matrix(value, axis):
    z = np.zeros((*value.shape[:-1],3*value.shape[-1]))
    z[:,axis::3] = value
    return z


class Patch:
    def __init__(self, n, side, amplitude, full=False):
        self.n=n
        self.side=side
        self.full=full
        ku=np.r_[np.zeros(4), np.arange(1,n)/n, np.ones(4)]
        kv=np.r_[np.zeros(4),np.ones(4)]
        self.knots=(ku,kv)
        x = (ku[1:n+4]+ku[2:n+5]+ku[3:n+6])/3.
        y = np.linspace(-1.,1.,4)
        cp = np.zeros((n+3,4,3))
        cp[:,:,0]=x[:,None]
        cp[:,:,1]=y[None,:]
        rows=[]
        for x,w in intervals(n):
            for xi,wi in zip(x,w):
                g=trace_height(xi,amplitude)
                edge=(-1.,1.) if full else ((-1.,g) if side==0 else (g,1.))
                y,wy=line(*edge)
                rows.extend(zip(np.full_like(y,xi),.5*(y+1),np.full_like(y,wi),.5*wy))
        rule=np.array(rows)
        self.bulk,self.fit=stiffness(self.knots,cp,DIRECTOR,MATERIAL,rule,return_fit=True)
        self.points=np.column_stack((rule[:,0],2.*rule[:,1]-1.))
        self.weights=2.*rule[:,2]*rule[:,3]
        self.dofs=len(self.bulk)
        eig,val_scale=self.bulk_matrix_spectrum()
        self.bulk_eigen=eig
        self.bulk_scale=val_scale

    def bulk_matrix_spectrum(self):
        d=np.diag(self.bulk)**-.5
        eig,Q=eigh(self.bulk*d[:,None]*d[None,:],check_finite=False)
        assert np.sum(eig<eig[-1]*1e-10)==6, (self.n,self.side,eig[:10])
        return (eig,Q),d

    def b(self, xy, ix,iy):
        return spline_basis(self.knots,xy,(ix,iy))

    def traces(self, xy, tau, tau_s, mu, mu_s):
        b0=self.b(xy,0,0); bx=self.b(xy,1,0); by=self.b(xy,0,1)
        rot=np.column_stack((-tau[:,1],tau[:,0]))
        N=np.stack([*(shape_matrix(b0,d) for d in range(3)),
                    shape_matrix(rot[:,0,None]*bx+rot[:,1,None]*by,2)],axis=1)
        uv=np.column_stack((xy[:,0],(xy[:,1]+1.)*.5))
        strain=[]
        for z,deg in zip(self.fit,((2,3),(3,2),(2,2))):
            k=tuple(t[1:-1] if p==2 else t for t,p in zip(self.knots,deg))
            strain.append(NdBSpline.design_matrix(uv,k,deg)@z)
        sx,sy,txy=Dm*(strain[0]+nu*strain[1]),Dm*(nu*strain[0]+strain[1]),Dm*(1-nu)/2*strain[2]
        Vx=mu[:,0,None]*sx+mu[:,1,None]*txy
        Vy=mu[:,0,None]*txy+mu[:,1,None]*sy
        bxx=self.b(xy,2,0); byy=self.b(xy,0,2); bxy=self.b(xy,1,1)
        bxxx=self.b(xy,3,0); bxxy=self.b(xy,2,1); bxyy=self.b(xy,1,2); byyy=self.b(xy,0,3)

        def tensor(xx,yy,xy_):
            qxx=D*(xx+nu*yy); qyy=D*(yy+nu*xx); qxy=D*(1-nu)*xy_
            return np.stack((np.stack((qxx,qxy),axis=1),np.stack((qxy,qyy),axis=1)),axis=1)

        def mul(q,x,y):
            return np.einsum("pi,pijd,pj->pd",x,q,y)

        Q=tensor(bxx,byy,bxy)
        Qs=tensor(tau[:,0,None]*bxxx+tau[:,1,None]*bxxy,
                  tau[:,0,None]*bxyy+tau[:,1,None]*byyy,
                  tau[:,0,None]*bxxy+tau[:,1,None]*bxyy)
        twist=mul(Q,tau,mu)
        twist_s=mul(Q,tau_s,mu)+mul(Qs,tau,mu)+mul(Q,tau,mu_s)
        vw=-D*(mu[:,0,None]*(bxxx+bxyy)+mu[:,1,None]*(bxxy+byyy))-twist_s
        vrot=mul(Q,rot,mu)
        V=np.stack((Vx,Vy,shape_matrix(vw,2),shape_matrix(vrot,2)),axis=1)
        S=np.zeros_like(V)
        S[:,2]=shape_matrix(twist,2)
        return N,V,S

    def external(self, amplitude, quartic, orthogonal):
        # The boundary loading is obtained before tangent integration by parts.
        # Only the proposed interface assembly turns the twisting work into end forces.
        F=np.zeros(self.dofs)
        n=self.n
        analytic = exact_t if orthogonal else exact
        sigma = SIGMA_T if orthogonal else SIGMA
        def add_edge(xy,weights,mu):
            _,_,Q,div=analytic(xy,quartic)
            physical_bending_flux=Q@mu
            membrane_flux=sigma@mu
            volume_term=div@mu
            B=self.b(xy,0,0)
            bx=self.b(xy,1,0); by=self.b(xy,0,1)
            work=((bx*physical_bending_flux[:,0,None]+by*physical_bending_flux[:,1,None])-B*volume_term[:,None])
            F[0::3]+=(weights@B)*membrane_flux[0]
            F[1::3]+=(weights@B)*membrane_flux[1]
            F[2::3]+=weights@work

        for y in ((-1.,1.) if self.full else ((-1.,) if self.side==0 else (1.,))):
            for x,wx in intervals(n):
                add_edge(np.column_stack((x,np.full_like(x,y))),wx,np.array([0.,y]))
        for x0,nx in ((0.,-1.),(1.,1.)):
            g=trace_height(x0,amplitude)
            edge=(-1.,1.) if self.full else ((-1.,g) if self.side==0 else (g,1.))
            y,wy=line(*edge)
            add_edge(np.column_stack((np.full_like(y,x0),y)),wy,np.array([nx,0.]))
        body = self.points[:,1] if orthogonal else np.ones(len(self.points))
        F[2::3]+=D*wscale*24*quartic*((self.weights*body)@self.b(self.points,0,0))
        return F

    def spectral_bound(self,V,weights,j):
        eig,Q=self.bulk_eigen
        factor=(Q*self.bulk_scale[:,None])
        traces=(V[:,j]*np.sqrt(weights[:,None,None])).reshape(-1,self.dofs)@factor
        rel=np.linalg.norm(traces[:,:6])/np.linalg.norm(traces[:,6:])
        assert rel < 1e-7, rel
        s=svdvals(traces[:,6:]/np.sqrt(eig[None,6:]),check_finite=False)
        return s[0]**2,float(rel),float(1-(s[1]/s[0])**2)


def curve(x,amp):
    g=trace_height(x,amp)
    gx=4.*amp*(1.-2.*x); gxx=np.full_like(x,-8.*amp)
    jac=np.hypot(1.,gx)
    t=np.column_stack((1/jac,gx/jac))
    t_s=gxx[:,None]*np.column_stack((-gx,np.ones_like(gx)))/jac[:,None]**4
    n=np.column_stack((-t[:,1],t[:,0])); ns=np.column_stack((-t_s[:,1],t_s[:,0]))
    return np.column_stack((x,g)),jac,t,t_s,n,ns



def edge_reference(xy,t,t_s,mu,mu_s,quartic,orthogonal):
    analytic=exact_t if orthogonal else exact
    sigma=SIGMA_T if orthogonal else SIGMA
    _,_,Q,div=analytic(xy,quartic)
    Qs=np.zeros_like(Q)
    x,y=xy.T
    if orthogonal:
        assert np.all(y==0.) and np.all(t[:,0]==1.)
        Qs[:,0,1]=Qs[:,1,0]=D*wscale*(1.-nu)*12.*quartic*x*x
    else:
        Qs[:,0,0]=D*wscale*24.*quartic*x*t[:,0]
        Qs[:,1,1]=nu*Qs[:,0,0]
    mul=lambda u,q,v:np.einsum("pi,pij,pj->p",u,q,v)
    twist_change=mul(t_s,Q,mu)+mul(t,Qs,mu)+mul(t,Q,mu_s)
    force=np.column_stack((mu@sigma.T,-np.sum(mu*div,axis=1)-twist_change))
    rotation=np.column_stack((-t[:,1],t[:,0]))
    moment=mul(rotation,Q,mu)
    return np.column_stack((force,moment)),mul(t,Q,mu)


def system(n,amp,quartic,rc=.1,end_scale=1.,orthogonal=False):
    assert not orthogonal or amp==0.
    analytic=exact_t if orthogonal else exact
    sigma=SIGMA_T if orthogonal else SIGMA
    patch=[Patch(n,0,amp),Patch(2*n,1,amp,full=orthogonal)]
    xx,ww=np.array(list(intervals(2*n))).transpose(1,0,2).reshape(2,-1)
    xy,j,t,t_s,no,no_s=curve(np.r_[xx,0.,1.],amp)
    gw=ww*j[:-2]
    h=gw.sum()
    he=end_scale*h
    weights=np.r_[gw,he,he]
    rows=[]
    endpoint=[]
    for i,p in enumerate(patch):
        sign=(-1.)**i
        N,V,S=p.traces(xy,t,t_s,sign*no,sign*no_s)
        V[-2:]=S[-2:]*np.array([-1.,1.])[:,None,None]/he
        if orthogonal and i==1:
            N[:,:3]=RX90@N[:,:3]
            V[:]=0.
            S[:]=0.
        rows.append((N,V))
        endpoint.append(S[-2:])
    B = np.zeros((sum(p.dofs for p in patch),)*2)
    H = np.zeros_like(B)
    Bombit=np.zeros_like(B)
    delta=np.concatenate((rows[0][0],-rows[1][0]),axis=2)
    calibration=[]
    for j in (slice(0,3),slice(3,4)):
        result=[p.spectral_bound(row[1],weights,j) for p,row in zip(patch,rows) if np.any(row[1][:,j])]
        d1=result[0][0]
        d2=result[1][0] if len(result)==2 else 0.
        gamma=np.array([d2,d1])/(d1+d2) if len(result)==2 else np.array([1.,0.])
        eta=(1./rc-1)*d1*d2/(d1+d2) if len(result)==2 else (1./rc-1)*d1
        A=(delta[:,j]*np.sqrt(weights)[:,None,None]).reshape(-1,len(B))
        V=np.concatenate(tuple((-1)**i*g*row[1][:,j] for i,(g,row) in enumerate(zip(gamma,rows))),axis=2)
        flux=(V*np.sqrt(weights)[:,None,None]).reshape(-1,len(B))
        B+=A.T@flux
        Bombit+=A[:-2*V.shape[1]].T@flux[:-2*V.shape[1]]
        H+=eta*(A.T@A)
        calibration.append(dict(gamma=gamma.tolist(),eta=float(eta),d=[float(d1),float(d2)],
                                kernel_traction_relative=[v[1] for v in result], leading_relative_gap=[v[2] for v in result]))

    Kbulk=np.zeros_like(B)
    m=patch[0].dofs
    Kbulk[:m,:m]=patch[0].bulk
    Kbulk[m:,m:]=patch[1].bulk
    Kplus=Kbulk+H
    K=Kplus-B+B.T
    Kbad=Kplus-Bombit+Bombit.T
    F=np.concatenate([p.external(amp,quartic,orthogonal) for p in patch])
    if orthogonal:
        # Body/exterior loads on each shell come from its exact field. Add the
        # analytically manufactured receiving line load to balance the material-edge
        # flux of the web. Integrate unprocessed bending work, with no endpoint rows.
        _,_,Q,div=analytic(xy[:-2],quartic)
        traction_world=np.column_stack((no[:-2]@sigma.T,-np.sum(no[:-2]*div,axis=1)))
        receiver=patch[1]
        F[m:]+=(gw[:,None]*(receiver.b(xy[:-2],0,0))).T.dot(traction_world@RX90).ravel()
        F[m+1::3]+=(gw*Q[:,0,1])@receiver.b(xy[:-2],1,0)
        F[m+2::3]+=(gw*Q[:,1,1])@receiver.b(xy[:-2],0,1)
    fix=np.array([0,1,2,m-11,m-10,m+11])
    pts=np.array([[0.,-1.],[1.,-1.],[0.,1.]])
    exact_fixed=analytic(pts,quartic)[0]
    prescribed=np.r_[exact_fixed[0],exact_fixed[1,1:],exact_fixed[2,2]]
    free=np.ones(len(F),dtype=bool)
    free[fix]=False
    sf=np.diag(K)[free]**-.5
    out=np.zeros(len(F))
    out[fix]=prescribed
    R=F[free]-K[free][:,fix]@prescribed
    out[free]=sf*solve(sf[:,None]*K[free][:,free]*sf[None,:],sf*R,check_finite=False)
    residual=float(np.linalg.norm((K@out-F)[free])/np.linalg.norm(R))
    assert residual < 1e-9

    uexact,gexact,_,div=analytic(xy,quartic)
    n_theta=np.column_stack((-t[:,1],t[:,0]))
    qexact=np.concatenate((uexact,np.sum(n_theta*gexact,axis=1)[:,None]),axis=1)
    U=[out[:m],out[m:]]
    jumps=delta@out
    jump_error=np.sqrt(gw@np.sum(jumps[:-2,:3]**2,axis=1)/(gw@np.sum(qexact[:-2,:3]**2,axis=1)))
    angle_error=np.sqrt(gw@(jumps[:-2,3]**2)/(gw@(qexact[:-2,3]**2)))
    err=0.
    denom=0.
    resultant=[]
    reference=[]
    end=[]
    edge_records=[]
    for i,p in enumerate(patch):
        field=p.b(p.points,0,0)@U[i].reshape(-1,3)
        ex=analytic(p.points,quartic)[0]
        err+=p.weights@np.sum((field-ex)**2,axis=1)
        denom+=p.weights@np.sum(ex**2,axis=1)
        rr=weights@(rows[i][1]@U[i])[:,:3]
        external_force=np.column_stack((no@sigma.T,-np.sum(no*div,axis=1)))*(-1.)**i
        ref=gw@external_force[:-2]
        resultant.append(rr); reference.append(ref)
        end.append((endpoint[i]@U[i])[:,2])
        if not orthogonal or i==0:
            outward=(-1.)**i
            target,twisting=edge_reference(xy,t,t_s,outward*no,outward*no_s,quartic,orthogonal)
            measured=rows[i][1][:-2]@U[i]
            rms_ref=np.sqrt(gw@(target[:-2]**2)/h)
            rms_difference=np.sqrt(gw@((target[:-2]-measured)**2)/h)
            by_row={key:dict(exact_RMS=float(v),RMS_error=float(e),relative_RMS_error=float(e/v) if v>1e-10 else None)
                    for key,v,e in zip(("Fx_N_per_mm","Fy_N_per_mm","Fz_N_per_mm","M_about_t_N"),rms_ref,rms_difference)}
            edge_records.append(dict(side=i+1,trace_errors_by_row=by_row,exact_endpoint_twisting_N=twisting[-2:].tolist(),
                                     maximum_endpoint_twisting_absolute_error_N=float(max(abs(twisting[-2:]-end[-1])))))
    resultant=np.asarray(resultant)
    reference=np.asarray(reference)
    if orthogonal:
        transfer = (K-Kbulk)@out
        resultant = np.array([-transfer[:m].reshape(-1,3).sum(axis=0),
                              -RX90@transfer[m:].reshape(-1,3).sum(axis=0)])
    per_side=np.linalg.norm(resultant-reference,axis=1)/np.linalg.norm(reference,axis=1)
    outdict=dict(spans=[n,2*n],transverse_spans=[1,1],amplitude=amp,quartic=quartic,h=float(h),
        orthogonal_material_boundary_interior=orthogonal,end_weight_mm=float(he),end_weight_scale=end_scale,
        target_interface_compliance_ratio=rc,calibration=calibration,condition_scope="Each side isolated and diagonally equilibrated before rigid-mode removal.",
        dofs=int(len(F)),free_residual=residual,relative_displacement_L2=float(np.sqrt(err/denom)),
        translation_jump_relative_L2=float(jump_error),rotation_jump_relative_L2=float(angle_error),
        side_force_resultants_N=resultant.tolist(),exact_side_force_resultants_N=reference.tolist(),
        relative_resultant_error_by_side=per_side.tolist(),
        sum_side_forces_relative=float(np.linalg.norm(resultant.sum(axis=0))/np.linalg.norm(reference[0])),
        endpoint_twisting_rows_N=np.asarray(end).tolist(),active_edge_force_and_moment_checks=edge_records)
    if quartic == 0.:
        coeff=[]
        for p in patch:
            abscissae=[(k[1:-3]+k[2:-2]+k[3:-1])/3. for k in p.knots]
            xx, yy=np.meshgrid(abscissae[0],2.*abscissae[1]-1.,indexing="ij")
            grid=np.column_stack((xx.ravel(),yy.ravel()))
            u=solve(p.b(grid,0,0),analytic(grid,0.)[0],check_finite=False)
            assert np.max(abs(p.b(p.points,0,0)@u-analytic(p.points,0.)[0]))<1e-14
            coeff.append(u.ravel())
        ex=np.concatenate(coeff)
        good=(K@ex-F)[free]
        bad=(Kbad@ex-F)[free]
        outdict["drop_endpoint_work_relative_residual"]=float(np.linalg.norm(bad)/np.linalg.norm(F[free]))
        outdict["retained_endpoint_work_relative_residual"]=float(np.linalg.norm(good)/np.linalg.norm(F[free]))
    return outdict


def run_study():
    patch_test=system(2,.2,0.)
    assert patch_test["relative_displacement_L2"] < 1e-8, patch_test
    assert patch_test["drop_endpoint_work_relative_residual"] > 1e-4, patch_test
    ttest=system(2,0.,0.,orthogonal=True)
    assert ttest["relative_displacement_L2"] < 1e-8, ttest
    assert ttest["drop_endpoint_work_relative_residual"] > 1e-4, ttest
    tcurve=[system(n,0.,.3,orthogonal=True) for n in (1,2,4,8,16)]
    assert all(b["relative_displacement_L2"]<a["relative_displacement_L2"] for a,b in zip(tcurve,tcurve[1:])),tcurve
    assert tcurve[-1]["relative_displacement_L2"] < 2e-3
    curved=[system(n,.2,.3) for n in (1,2,4,8,16)]
    assert all(b["relative_displacement_L2"]<a["relative_displacement_L2"] for a,b in zip(curved,curved[1:])),curved
    assert curved[-1]["relative_displacement_L2"] < 2e-3
    auxiliary=[system(4,.2,.3,rc=rc,end_scale=e) for rc,e in ((.05,1.),(.2,1.),(.1,.5),(.1,2.))]
    hashes={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in ("bulk_matrix.py","bulk_stress.py")}
    data=dict(scope=("Independent manufactured Kirchhoff–Love shell check for the skew Nitsche sign, whole-chain endpoint-work rows with "
                     "h weighting, and two-active-side trace quotient in Eqs. (21)–(29). Two coplanar, nonmatching cubic carriers sample "
                     "an interior quadratic cut of a flat rectangle. An orthogonal T pair checks the one-sided material-edge/interior rule "
                     "with a separately manufactured receiving line load. These are local manufactured models."),
        material=MATERIAL,reference_displacement="u_x=.001*(.4*x+.3*y); u_y=.001*(.1*x-.2*y); w=.002*(x*y+a*x^4).",
        reference_loading=("Independent external loading from the unintegrated exact bending boundary work "
                           "(Q mu) dot grad(test) - div(Q) dot mu times test plus div(div(Q)) body term. "
                           "Six exact corner displacements fix the rigid-body gauge; none are at interface endpoints."),
        orthogonal_T_reference_displacement=("u_x=.001*(.4*x+.3*y); u_y=-.0002*y; w=.002*((x+a*x^4)*y+.18*y^2) on each local shell. "
                    "The receiver is uncut and rotates 90 degrees about x. The manufactured line load applies the physical flux of the exact web "
                    "using unintegrated trace work on the receiver, then compares the coupled solution to the same exact displacement."),
        retained_quadrature="12-point Gauss quadrature in each longitudinal knot span and each retained transverse interval.",
        source_code_sha256=hashes,polynomial_patch_test=patch_test,curved_refinement=curved,auxiliary_calibration=auxiliary,
        orthogonal_T_patch_test=ttest,orthogonal_T_refinement=tcurve)
    return data


def main():
    data=run_study()
    patch_test=data["polynomial_patch_test"]
    ttest=data["orthogonal_T_patch_test"]
    curved=data["curved_refinement"]
    tcurve=data["orthogonal_T_refinement"]
    out=HERE/"open_chain_mms.json"
    out.write_text(json.dumps(data,indent=2)+"\n",encoding="utf8")
    print("patch test",patch_test["relative_displacement_L2"],patch_test["drop_endpoint_work_relative_residual"])
    for a in curved:
        print(a["spans"],a["relative_displacement_L2"],a["translation_jump_relative_L2"],
              a["rotation_jump_relative_L2"],a["relative_resultant_error_by_side"],flush=True)
    print("orthogonal T patch test",ttest["relative_displacement_L2"],ttest["drop_endpoint_work_relative_residual"])
    for q in tcurve:
        print(q["spans"],q["relative_displacement_L2"],q["translation_jump_relative_L2"],
              q["rotation_jump_relative_L2"],q["relative_resultant_error_by_side"],flush=True)


if __name__=="__main__":
    main()
