"""Private deterministic constructed application instances; NumPy-only evaluation.
Not a production panel; no optimizer, random evaluator, integer decisions or hidden repairs.
ABI: designs and batch-major M/F/G are float32. Physical kernels default FP32.
The sensitive Arrhenius information kernel uses FP64, then returns FP32;
CORT offline Gram accumulation is FP64, persisted Gram/evaluation remain FP32.
All M/F/G returns stay FP32. Arrhenius F2=log(raw variance M2), a Root-approved
strictly monotone objective revision; static bounds are separate, not data-fitted.
"""
from pathlib import Path
import hashlib, json
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'domain-data'
FAMILIES={
 'RESERVOIR':('BASE','DRY','WET'), 
 'LOGISTICS':('BASE','RAIL_CAP','TIME_PRESSURE'), 'RIVER_DO':('BASE','LOW_REAERATION','HIGH_LOAD'),
 'LAKE':('BASE','FAST_SETTLING','HIGH_INFLOW'), 'CORT':('BASE','CORE_LIMIT','NORMAL_TISSUE'),
 'ARRHENIUS':('BASE','WIDE_PRIOR','HIGH_HEAT_LOSS')}
CASES=tuple('DOM_'+f+'_'+v for f,vs in FAMILIES.items() for v in vs)
SOURCES={
 'RESERVOIR':'https://www.hec.usace.army.mil/confluence/cwmsdocs/rsum/power-plants-58631684.html',
 
 'LOGISTICS':'https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_trnsport.html',
 'RIVER_DO':'https://glossary.ametsoc.org/wiki/streeter-phelps-equation/',
 'LAKE':'https://openresearchsoftware.metajnl.com/articles/10.5334/jors.293',
 'CORT':'https://pmc.ncbi.nlm.nih.gov/articles/PMC4326207/',
 'ARRHENIUS':'https://kitchingroup.cheme.cmu.edu/discopt-doe/notebooks/model_based_doe.html'}

def identity(case):
    for f,vs in FAMILIES.items():
        for i,v in enumerate(vs):
            if case=='DOM_'+f+'_'+v: return f,i
    raise KeyError(case)

def _input(f,i):
    t=np.arange(24,dtype=float)
    if f=='RESERVOIR':
        mean,amp,demand,flood=[(8,4,8,60),(6.5,3,8.5,55),(9.5,5,8,55)][i]
        q=mean+amp*np.sin(2*np.pi*(t+.5)/24)
        return dict(q=q,demand=demand+2*np.cos(2*np.pi*t/24),evap=np.full(24,.2),
                    initial=np.array(50.),smin=np.array(20.),smax=np.array(100.),terminal=np.array(45.),
                    flood=np.array(flood),physical_lower=np.full(24,2.5),physical_upper=np.full(24,16.),
                    nominal=q-.2,head_intercept=np.array(30.),head_slope=np.array(.4),eta=np.array(.9))
    if f=='LOGISTICS':
        origin=np.array([[0,0],[0,8],[10,0],[10,8]],float)
        destination=np.array([[2,2],[2,6],[5,4],[8,2],[8,6]],float)
        d=100*np.linalg.norm(origin[:,None,:]-destination[None,:,:],axis=-1); dr=1.15*d+20
        ii=np.arange(4)[:,None]
        c=np.stack([(.1+.02*ii)*d,(.04+.005*ii)*dr+8],axis=-1)
        e=np.stack([.1*d,.025*dr+.5],axis=-1)
        h=np.stack([d/60+2,dr/40+(18 if i!=2 else 30)],axis=-1)
        supply=np.array([40,60,50,50.]); demand=np.array([30,45,40,50,35.])
        base=supply[:,None]*demand[None,:]/200
        basis=np.zeros((12,4,5)); amplitudes=[]
        for k,(row,col) in enumerate(( (row,col) for row in range(3) for col in range(4) )):
            basis[k,row,col]=1; basis[k,3,4]=1; basis[k,row,4]=-1; basis[k,3,col]=-1
            amplitudes.append(.8*min(base[row,col],base[row,4]/4,base[3,col]/3,base[3,4]/12))
        rail_fraction=.2 if i==1 else .4
        return dict(cost=c,emissions=e,hours=h,supply=supply,demand=demand,rail_cap=supply*rail_fraction,
                    base_flow=base,cycle_basis=basis,cycle_amplitudes=np.array(amplitudes),
                    physical_lower=np.r_[np.full(12,-1.),np.zeros(20)],physical_upper=np.r_[np.ones(12),np.full(20,rail_fraction)],
                    nominal=np.zeros(32),equality_tolerance=np.array(1e-4))
    if f=='RIVER_DO':
        n=np.arange(24); x=np.arange(241)*.25; sites=2*(n+1)
        kd=.25; kr=.5 if i==1 else .8; loads=(1000+200*(n%4))*(1.25 if i==2 else 1)
        age=np.maximum(x[:,None]-sites,0)/25.92
        kernel=kd/(kr-kd)*(np.exp(-kd*age)-np.exp(-kr*age))*(x[:,None]>=sites)
        age0=x/25.92
        bg=.5*np.exp(-kr*age0)+2*kd/(kr-kd)*(np.exp(-kd*age0)-np.exp(-kr*age0))
        return dict(x=x,sites=sites,loads=loads,flow=np.array(10.),kd=np.array(kd),kr=np.array(kr),velocity=np.array(25.92),
                    kernel=kernel,bg=bg,price=.4+.1*(n%3),budget=3000+500*(n%5),sat=np.array(9.),hard_do=np.array(4.),target_do=np.array(7.),
                    physical_lower=np.zeros(24),physical_upper=np.full(24,.95),nominal=np.full(24,.95))
    if f=='LAKE':
        b=[.42,.48,.42][i]; y=[.02,.02,.03][i]
        return dict(b=np.array(b),y=np.full(100,y),alpha=np.array(.4),discount=.98**np.arange(100),
                    physical_lower=np.zeros(100),physical_upper=np.full(100,.1),nominal=np.full(100,.01))
    if f=='ARRHENIUS':
        ks=[.25,.5,1.] if i!=1 else [.15,.5,1.5]
        es=[40000,50000,60000.] if i!=1 else [30000,50000,70000.]
        scenarios=np.array([(np.log(k),e/(8.314*315)) for k in ks for e in es])
        pred=np.array([(T,tau) for T in [295,310,325,340] for tau in [.5,1,2,3]])
        nominal=np.column_stack([np.tile([300.,330.],6),.5+3*np.arange(12)/11]).ravel()
        return dict(scenarios=scenarios,prediction=pred,sigma=np.array(.02),prior=np.array(1e-8),tref=np.array(315.),ambient=np.array(290.),
                    cp=np.array(4.18),heat_loss=np.array(4. if i==2 else 2.),
                    physical_lower=np.tile([290.,.1],12),physical_upper=np.tile([340.,4.],12),nominal=nominal)
    if f=='CORT':
        return dict(physical_lower=np.zeros(40),physical_upper=np.full(40,50.),nominal=np.full(40,20.),
                    prescription=np.array(1.),target_interval=np.array([.95,1.05]),
                    core_mean_limit=np.array([0.,1.05,0.][i]),normal_mean_limit=np.array([0.,0.,.14][i]))
    raise KeyError(f)

def prepare_inputs():
    """Explicit one-time deterministic input construction; never called by build/evaluate."""
    DATA.mkdir(exist_ok=True)
    manifest={}
    for case in CASES:
        f,i=identity(case); p=DATA/(case+'.npz')
        arrays={k:np.asarray(v,dtype=np.float32) for k,v in _input(f,i).items()}
        if p.exists():
            with np.load(p,allow_pickle=False) as existing:
                if set(existing.files)!=set(arrays) or any(not np.array_equal(existing[k],v) for k,v in arrays.items()):
                    raise RuntimeError('Refusing to mutate frozen input '+str(p))
        else: np.savez_compressed(p,**arrays)
        manifest[case]={'array_file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
          'family':f,'source':SOURCES[f],'classification':'new_fixed_parameter_construction_on_published_principles' if f!='CORT' else 'new_grouped_triobjective_derivative_of_public_TG119_phantom',
          'parameter_instance_is_not_new_domain':True,'large_domain':'water_environment' if f in ('RIVER_DO','LAKE') else f,
          'arrays_dtype':'float32','evaluation':'unit-box float32 -> FP32 physical/output; Arrhenius sensitive kernel FP64 -> FP32; no clipping'}
    path=DATA/'inputs-manifest.json'
    path.write_text(json.dumps({'schema':'domains-inputs-v1','cases':manifest},indent=2),encoding='utf-8')
    return manifest

class DomainCase:
    n_obj=3
    def __init__(self,case):
        self.name=case; self.family,self.instance=identity(case)
        with np.load(DATA/(case+'.npz'),allow_pickle=False) as a: self.p={k:np.asarray(a[k],dtype=np.float32) for k in a.files}
        self.physical_lower=self.p['physical_lower']; self.physical_upper=self.p['physical_upper']
        self.n_var=len(self.physical_lower); self.lower=np.zeros(self.n_var,np.float32); self.upper=np.ones(self.n_var,np.float32)
        self.nominal=((self.p['nominal']-self.physical_lower)/(self.physical_upper-self.physical_lower)).astype(np.float32)
        self.objective_scale=np.ones(3); self.objective_offset=np.zeros(3)
        f=self.family; p=self.p
        if f=='RESERVOIR':
            self.objective_names=('squared_relative_supply_deficit','negative_hydropower_MWh','flood_space_occupancy')
            extras=tuple('storage_'+str(t+1) for t in range(24))+('mass_balance_error',); self.n_ieq_constr=49
            lo=[0,-24*2.725*float(p['eta'])*(30+.4*100)*16,0]; hi=[1,0,1]
            self.bound_rationale='Feasible storage20..100, release2.5..16, head38..70; all normalized deficits<=1.'
        elif f=='LOGISTICS':
            self.objective_names=('cost_currency','emissions_kgCO2e','tonne_hours')
            extras=tuple('supply_residual_'+str(j) for j in range(4))+tuple('demand_residual_'+str(j) for j in range(5))+tuple('rail_flow_'+str(j) for j in range(4)); self.n_ieq_constr=13
            lo=[0,0,0]; hi=[200*p[k].max() for k in ('cost','emissions','hours')]
            self.bound_rationale='Total feasible shipment=200t; each objective<=200*maximum unit coefficient.'
        elif f=='RIVER_DO':
            self.objective_names=('treatment_cost_currency_per_day','mean_DO_target_shortfall_squared','max_relative_budget_burden')
            extras=tuple('DO_at_'+str(j) for j in range(241)); self.n_ieq_constr=241
            cm=p['price']*p['loads']*(.95+2*.95**2)
            lo=[0,0,0]; hi=[cm.sum(),9,(cm/p['budget']).max()]
            self.bound_rationale='Costs monotone in u<=.95; feasible DO>=4 gives shortfall to7<=3 everywhere on fixed grid.'
        elif f=='LAKE':
            self.objective_names=('peak_phosphorus','negative_discounted_utility','policy_change_squared')
            extras=('terminal_phosphorus',); self.n_ieq_constr=0
            lo=[0,-float(p['alpha'])*.1*p['discount'].sum(),0]; hi=[(1+.1+p['y'].max())/float(p['b']),0,.1**2]
            self.bound_rationale='Recycling X²/(1+X²)<=1 gives invariant X<=(1+.1+maxY)/b; bounded controls give variation<=.1².'
        elif f=='ARRHENIUS':
            self.objective_names=('negative_mean_logdet_FIM','log_mean_prediction_variance','heating_energy_kJ')
            extras=tuple('unregularized_information_determinant_'+str(j) for j in range(9)); self.n_ieq_constr=0
            zmax=max(abs(315/290-1),abs(315/340-1)); c=(1+zmax*zmax)/np.e**2
            lam=float(p['prior']); sig=float(p['sigma']); trace=12*c/sig**2
            # Static FP64 metadata only: fixed prediction sensitivities, no candidate/PF information.
            pred=p['prediction'].astype(np.float64); phi=p['scenarios'].astype(np.float64)
            zp=float(p['tref'])/pred[:,0]-1
            qp=np.exp(phi[:,0,None]-phi[:,1,None]*zp)*pred[:,1]
            hnorm2=(qp*np.exp(-qp))**2*(1+zp*zp)
            Hmean=float(np.mean(hnorm2)); Vlo=Hmean/(lam+trace); Vhi=Hmean/lam
            self.variance_bound_metadata=dict(Hmean=Hmean,prior=lam,trace_bound=trace,Vlo=Vlo,Vhi=Vhi,static_precision='float64',objective_revision='Root-approved-log-variance-v1')
            lo=[-2*np.log(lam+trace),np.log(Vlo),0]; hi=[-2*np.log(lam),np.log(Vhi),12*(float(p['cp'])*50+float(p['heat_loss'])*50*4)]
            self.bound_rationale='q*exp(-q)<=1/e; information eigenvalues in[prior,prior+12*c/sigma²]; fixed Hmean=mean||h||² gives Vlo=Hmean/(prior+tracebound),Vhi=Hmean/prior. F2=log(mean raw prediction variance), strictly increasing; M2 remains raw variance.'
        elif f=='CORT':
            with np.load(DATA/'cort-gram.npz',allow_pickle=False) as a: self.cort={k:np.asarray(a[k],dtype=np.float32) for k in a.files}
            self.objective_names=('target_prescription_MSE_Gy2','core_mean_dose_squared_Gy2','normal_mean_dose_squared_Gy2')
            extras=('target_mean_Gy','core_mean_Gy','normal_mean_Gy'); self.n_ieq_constr=2+int(p['core_mean_limit']>0)+int(p['normal_mean_limit']>0)
            max_target=50*self.cort['ones_max'][0]
            lo=[0,0,0]; hi=[max(1,(max_target-1)**2),2500*self.cort['Q'][1].sum(),2500*self.cort['Q'][2].sum()]
            self.bound_rationale='Nonnegative A,B and fluence<=50MU: organ dose<=50*A*1 voxelwise; target squared residual bounded by endpoints.'
        self.metric_names=((self.objective_names[0],'mean_prediction_variance',self.objective_names[2]) if f=='ARRHENIUS' else self.objective_names)+extras
        self.objective_lower=np.asarray(lo,float); self.objective_upper=np.asarray(hi,float)
        self.affine_lower=self.objective_lower.copy(); self.affine_span=self.objective_upper-self.objective_lower
        self.bounds_scope='feasible_set_only; not an empirical ideal/nadir or PF; do not clip infeasible outputs'
    def canonicalize(self,X):
        x=np.asarray(X)
        if x.ndim==1: x=x[None,:]
        if x.ndim!=2 or x.shape[1]!=self.n_var: raise ValueError('Expected batch x '+str(self.n_var))
        if not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>1): raise ValueError('Nonfinite or out-of-unit-box design; no clipping')
        return np.ascontiguousarray(x,dtype=np.float32)
    def physical(self,X):
        x=self.canonicalize(X)
        return self.physical_lower+x*(self.physical_upper-self.physical_lower)
    def compute(self,X):
        return np.asarray(self._compute(X),dtype=np.float32)
    def _compute(self,X):
        x=self.physical(X); p=self.p; n=len(x); f=self.family
        if f=='RESERVOIR':
            s=float(p['initial'])+np.cumsum(p['q']-p['evap']-x,axis=1)
            before=np.column_stack([np.full(n,float(p['initial']),dtype=np.float32),s[:,:-1]])
            head=p['head_intercept']+p['head_slope']*(s+before)/2
            a=np.mean((np.maximum(p['demand']-x,0)/p['demand'])**2,axis=1)
            b=-np.sum(2.725*p['eta']*head*x,axis=1)
            c=np.mean((np.maximum(s-p['flood'],0)/(p['smax']-p['flood']))**2,axis=1)
            mass=s[:,-1]-p['initial']-np.sum(p['q']-p['evap']-x,axis=1)
            return np.column_stack([a,b,c,s,mass])
        if f=='LOGISTICS':
            total=p['base_flow'][None,:,:]+np.einsum('bk,kij->bij',x[:,:12]*p['cycle_amplitudes'],p['cycle_basis'])
            rail=total*x[:,12:].reshape(n,4,5)
            flow=np.stack([total-rail,rail],axis=-1)
            objs=np.column_stack([np.sum(flow*p[k],axis=(1,2,3)) for k in ('cost','emissions','hours')])
            return np.column_stack([objs,flow.sum(axis=(2,3))-p['supply'],flow.sum(axis=(1,3))-p['demand'],flow[:,:,:,1].sum(axis=2)])
        if f=='RIVER_DO':
            cost=p['price']*p['loads']*(x+2*x*x)
            do=p['sat']-p['bg'][None,:]-(p['kernel']@((p['loads']*(1-x)/(86.4*p['flow'])).T)).T
            deficit=np.maximum(p['target_do']-do,0)**2
            integral=.25*(.5*deficit[:,0]+deficit[:,1:-1].sum(axis=1)+.5*deficit[:,-1])/60
            return np.column_stack([cost.sum(axis=1),integral,(cost/p['budget']).max(axis=1),do])
        if f=='LAKE':
            state=np.zeros(n,np.float32); peak=np.zeros(n,np.float32)
            for t in range(100):
                state=(1-p['b'])*state+state*state/(1+state*state)+x[:,t]+p['y'][t]
                peak=np.maximum(peak,state)
            return np.column_stack([peak,-p['alpha']*np.sum(x*p['discount'],axis=1),np.mean(np.diff(x,axis=1)**2,axis=1),state])
        if f=='ARRHENIUS':
            design=x.astype(np.float64).reshape(n,12,2); T=design[:,:,0]; tau=design[:,:,1]
            phi=p['scenarios'].astype(np.float64); z=p['tref']/T-1
            k=np.exp(phi[None,:,0,None]-phi[None,:,1,None]*z[:,None,:]); q=k*tau[:,None,:]; C=np.exp(-q)
            g=np.stack([-q*C,q*C*z[:,None,:]],axis=-1)/p['sigma']
            a=g[:,:,:,0]; b=g[:,:,:,1]; ia,ib=np.triu_indices(12,1)
            # Cauchy-Binet determinant and adjugate quadratic avoid catastrophic a*c-b*b near rank-one designs.
            det0=np.sum((a[:,:,ia]*b[:,:,ib]-a[:,:,ib]*b[:,:,ia])**2,axis=2)
            lam=float(p['prior']); det=det0+lam*np.sum(a*a+b*b,axis=2)+lam*lam
            pred=p['prediction'].astype(np.float64); zp=p['tref']/pred[:,0]-1
            kp=np.exp(phi[:,0,None]-phi[:,1,None]*zp); qp=kp*pred[:,1]; cp=np.exp(-qp)
            h=np.stack([-qp*cp,qp*cp*zp],axis=-1)
            cross=h[None,:,:,0,None]*b[:,:,None,:]-h[None,:,:,1,None]*a[:,:,None,:]
            numerator=lam*np.sum(h*h,axis=-1)[None,:,:]+np.sum(cross*cross,axis=-1)
            variance=np.mean(numerator/det[:,:,None],axis=(1,2))
            physical32=x.reshape(n,12,2); T32=physical32[:,:,0]; tau32=physical32[:,:,1]
            energy=np.sum(p['cp']*(T32-p['ambient'])+p['heat_loss']*(T32-p['ambient'])*tau32,axis=1)
            return np.column_stack([-np.mean(np.log(det),axis=1),variance,energy,det0])
        if f=='CORT':
            Q=self.cort['Q']; means=x@self.cort['mean'].T
            squares=np.einsum('bi,oij,bj->bo',x,Q,x,optimize=True)
            squares[:,0]+=-2*p['prescription']*means[:,0]+p['prescription']**2
            return np.column_stack([squares,means])
        raise AssertionError(f)
    def objectives_from_metrics(self,M):
        m=np.atleast_2d(np.asarray(M,dtype=np.float32))
        if m.shape[1]!=len(self.metric_names): raise ValueError('Metric shape')
        f=((m[:,:3]-self.objective_offset)/self.objective_scale).astype(np.float32)
        if self.family=='ARRHENIUS': f[:,1]=np.log(m[:,1])
        return f
    def constraints_from_metrics(self,M):
        m=np.atleast_2d(np.asarray(M,dtype=np.float32)); p=self.p; f=self.family
        if f=='RESERVOIR':
            s=m[:,3:27]; return np.column_stack([p['smin']-s,s-p['smax'],p['terminal']-s[:,-1]])
        if f=='LOGISTICS': return np.column_stack([np.abs(m[:,3:12])-p['equality_tolerance'],m[:,12:16]-p['rail_cap']])
        if f=='RIVER_DO': return p['hard_do']-m[:,3:]
        if f=='CORT':
            g=[p['target_interval'][0]-m[:,3],m[:,3]-p['target_interval'][1]]
            if p['core_mean_limit']>0: g.append(m[:,4]-p['core_mean_limit'])
            if p['normal_mean_limit']>0: g.append(m[:,5]-p['normal_mean_limit'])
            return np.column_stack(g)
        return np.empty((len(m),0),np.float32)
    def evaluate(self,X):
        M=self.compute(X); return {'F':self.objectives_from_metrics(M),'G':self.constraints_from_metrics(M),'M':M}
    def affine_objectives(self,F):
        """Optional frozen physical-bound profile, not automatic objective clipping."""
        return (np.asarray(F)-self.affine_lower)/self.affine_span

def build(case): return DomainCase(case)
