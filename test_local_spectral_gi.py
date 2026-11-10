import sys, numpy as np, scipy.linalg as sla
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import local_spectral_gi as L

def icosphere(level, r=1.0):
    t=(1+5**.5)/2
    V=[[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]]
    F=[[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
       [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]]
    V=[np.array(v,float)/np.linalg.norm(v) for v in V]
    for _ in range(level):
        cache={}; nF=[]
        def mid(a,b):
            k=tuple(sorted((a,b)))
            if k not in cache:
                m=V[a]+V[b]; V.append(m/np.linalg.norm(m)); cache[k]=len(V)-1
            return cache[k]
        for a,b,c in F:
            ab,bc,ca=mid(a,b),mid(b,c),mid(c,a)
            nF+=[[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        F=nF
    return np.array(V)*r, np.array(F)

# 1. Spectrum of the unit sphere: l(l+1) with multiplicity 2l+1
V,F=icosphere(4); K,M=L.fem_matrices(V,F)
ev,_=L.lb_eigenpairs(K,M,25)
print("1. sphere eigenvalues:", np.round(ev[:16],3), " area", round(M.sum(),3), "(4pi=12.566)")

# 2. Parseval check against the explicit windowed spectrogram (full basis, small mesh)
V,F=icosphere(2, r=10.0); n=len(V); K,M=L.fem_matrices(V,F)
lam,Phi=sla.eigh(K.toarray(), M.toarray()); lam=np.clip(lam,0,None); lam[0]=0
rng=np.random.default_rng(0)
f=np.sin(V[:,0]/2)+0.3*np.cos(V[:,1]*0.9)*V[:,2]/10 + 0.1*rng.standard_normal(n)
tau=3.0
A=M.sum(); g=L.heat_window(lam,tau); W=Phi@np.diag(g)@Phi.T          # W[:,i] = w_i
Md=M.toarray()
S=np.empty((n,n))                      # S[i,k] = < f, c*phi_k * c*w_i >_M,  c=sqrt(A)
for i in range(n): S[i]=A*(Phi.T@(Md@(f*W[:,i])))
sgi_ref=(S**2).sum(1)/A; wgi_ref=((lam/lam[1])**2*S**2).sum(1)/A
gi=L.band_groups(lam, lam[1])
bp_ref=np.array([(S[:,lo:hi+1]**2).sum(1)/A if hi>=lo else np.full(n,np.nan) for lo,hi in gi])
res=L.local_spectral_indices(V,M,f,lam,Phi,tau,bands=gi,batch=37,radius_factor=50,verbose=False)
rel=lambda a,b: np.nanmax(np.abs(a-b))/np.nanmax(np.abs(b))
print(f"2. Parseval vs explicit spectrogram (n={n}): sGI rel.err {rel(res['sGI'],sgi_ref):.1e}, "
      f"wGI rel.err {rel(res['wGI'],wgi_ref):.1e} , band rel.err {rel(res['band_power'],bp_ref):.1e}")
print("   bands sum to sGI:", np.allclose(np.nansum(res['band_power'],0), res['sGI']))

# 3. Constant signal: sGI ~ f^2 = 1 everywhere (area mean exactly ~1)
res=L.local_spectral_indices(V,M,np.ones(n),lam,Phi,tau,verbose=False)
m=np.asarray(M.sum(1)).ravel()
print(f"3. constant f=1: sGI area-mean {L.area_mean(res['sGI'],m):.4f}, range [{res['sGI'].min():.3f}, {res['sGI'].max():.3f}]")

# 4. Localisation, local neighbourhoods and resolution independence on a finer sphere
for lev in (4,5):
    V,F=icosphere(lev, r=20.0); K,M=L.fem_matrices(V,F); m=np.asarray(M.sum(1)).ravel()
    tau=L.tau_from_sigma(3.0); need=L.eigenpairs_needed(m.sum(),tau)
    ncut=1200; ev,ph=L.lb_eigenpairs(K,M,max(ncut,need)+20)
    f=np.where(V[:,2]>0, 0.2, 0.0) * (1+0.5*np.sin(V[:,0]))       # 'folding' only on the top half
    r=L.local_spectral_indices(V,M,f,ev,ph,tau,n_cut=ncut,verbose=False)
    top=V[:,2]>8; bot=V[:,2]<-8
    print(f"4. level {lev}: N={len(V)}, {len(ev)} eigenpairs (window needs ~{need}), sums over {ncut}, truncated={r['window_truncated']}; "
          f"sGI top {L.area_mean(r['sGI'],m,top):.4f} vs bottom {L.area_mean(r['sGI'],m,bot):.1e}; "
          f"mean H^2 top {L.area_mean(f**2,m,top):.4f}; wGI top {L.area_mean(r['wGI'],m,top):.3f}")
