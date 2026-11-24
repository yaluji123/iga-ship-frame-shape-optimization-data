"""Assemble the material-projected membrane and Kirchhoff–Love bending stiffness of a cubic CAD patch."""
from bulk_stress import np,sparse,NdBSpline,splu,derivative_basis,dot

def scatter(terms,n):
    # Basis derivatives act on three interleaved global Cartesian displacement components.
    components=[]
    for axis in range(3):
        pieces=[a.multiply(x[:,axis,None]) for a,x in terms]
        components.append(sum(pieces[1:],pieces[0]).tocsr())
    order=np.arange(3*n).reshape(3,n).T.ravel()
    out=sparse.hstack(components,format="csr")[:,order]
    out.eliminate_zeros();return out

def gram_projection(phi,b0,b1,weight,nu,return_fit=False):
    normalized=[]
    keep=[]
    for p in phi:
        diag=p.power(2).T@weight;active=diag>0
        scale=1/np.sqrt(diag[active])
        normalized.append(p[:,active].multiply(scale).tocsr())
        keep.append((active,scale))
    a=normalized[0];wa=a.multiply(weight[:,None]).tocsr()
    if len(normalized)==1:
        mass=(wa.T@a).tocsc();rhs=(wa.T@b0).tocsr()
    else:
        b=normalized[1];wb=b.multiply(weight[:,None]).tocsr()
        cross=nu*(wa.T@b)
        mass=sparse.bmat([[wa.T@a,cross],[cross.T,wb.T@b]],format="csc")
        rhs=sparse.vstack((wa.T@(b0+nu*b1),wb.T@(nu*b0+b1)),format="csr")
    solver=splu(mass)
    # The dense coefficient map is only local to this patch. Bound simultaneous RHS storage.
    n=rhs.shape[1];out=np.empty((n,n))
    if return_fit:
        fits=[np.zeros((len(sel),n)) for sel,_ in keep]
    for a in range(0,n,96):
        stop=min(a+96,n);z=solver.solve(rhs[:,a:stop].toarray(order="F"))
        out[:,a:stop]=rhs.T@z
        if return_fit:
            first=0
            for fit,(sel,scale) in zip(fits,keep):
                fit[sel,a:stop]=scale[:,None]*z[first:first+len(scale)]
                first+=len(scale)
    return (out,fits) if return_fit else out

def stiffness(knots,cp,director,m,rule,*,return_fit=False):
    E,nu,t,offset=m
    if offset!=0 or not np.isfinite(rule).all() or min(rule[:,2]*rule[:,3])<=0:raise ValueError("Unsupported patch")
    uv=rule[:,:2].copy();surf=NdBSpline(knots,cp-cp[0,0],(3,3));ncp=np.prod(cp.shape[:2])
    xu=surf(uv,nu=(1,0));xv=surf(uv,nu=(0,1))
    nc=np.cross(xu,xv);j=np.sqrt(dot(nc,nc));n=nc/j[:,None]
    du=np.cross(xv,n)/j[:,None];dv=np.cross(n,xu)/j[:,None]
    ax=director-(n@director)[:,None]*n
    if min(j)<=0 or min(np.linalg.norm(ax,axis=1))<1e-12:raise ValueError("Degenerate material chart")
    ax/=np.linalg.norm(ax,axis=1)[:,None];ay=np.cross(n,ax)
    a,b,c,d=dot(ax,du),dot(ax,dv),dot(ay,du),dot(ay,dv)
    nu1,nv,nuu,nuv,nvv=[derivative_basis(knots,uv,q) for q in ((1,0),(0,1),(2,0),(1,1),(0,2))]
    xuu,xuv,xvv=[surf(uv,nu=q) for q in ((2,0),(1,1),(0,2))]
    membrane=[];bend=[]
    for h,k,l in [(a*a,b*b,a*b),(c*c,d*d,c*d),(2*a*c,2*b*d,a*d+b*c)]:
        membrane.append(scatter([(nu1,h[:,None]*xu+l[:,None]*xv),(nv,k[:,None]*xv+l[:,None]*xu)],ncp))
        shape=h[:,None]*xuu+k[:,None]*xvv+2*l[:,None]*xuv
        bend.append(scatter([(nuu,h[:,None]*n),(nvv,k[:,None]*n),(nuv,2*l[:,None]*n),
                             (nu1,-dot(du,shape)[:,None]*n),(nv,-dot(dv,shape)[:,None]*n)],ncp))
    weight=rule[:,2]*rule[:,3]*j
    w=weight*t*E/(1-nu**2);ws=weight*t*E/(2*(1+nu))
    phi=[NdBSpline.design_matrix(uv,tuple(k[1:-1] if p==2 else k for k,p in zip(knots,pair)),pair).tocsr()
         for pair in [(2,3),(3,2),(2,2)]]
    normal=gram_projection(phi[:2],membrane[0],membrane[1],w,nu,return_fit)
    shear=gram_projection(phi[2:],membrane[2],None,ws,nu,return_fit)
    if return_fit:
        K,normal_fit=normal
        Ks,shear_fit=shear
        K+=Ks
    else:
        K=normal
        K+=shear
    b0,b1,b2=bend
    bend_term=b0.T@(b0+nu*b1).multiply(w[:,None])+b1.T@(nu*b0+b1).multiply(w[:,None])+b2.T@b2.multiply(ws[:,None])
    K+=bend_term.toarray()*(t*t/12)
    if not np.isfinite(K).all():raise ValueError("Invalid stiffness")
    return (K,normal_fit+shear_fit) if return_fit else K
