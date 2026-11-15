"""Unselected public held-out suites in a common normalized design domain.

LIR-CMOP, MMF and RWMOP formulas follow pinned BIMK/PlatEMO commit
e2d58cb4850acf1af5c48dffb669e86e9a89da9e (author notices retained in
the source manifest). CTP follows the official pymoo multimodal formulation,
with the original finite constraint difference instead of division by zero.
Objective/constraint scales are fixed below before optimizer experiments.
"""
import numpy as np

CASES=[f'H-LIR{i:02}'for i in range(1,15)]+[f'H-CTP{i}'for i in range(1,9)]+[f'H-MMF{i}'for i in range(1,9)]+[f'H-RW{i}'for i in range(1,7)]
RW_BOUNDS={1:([.51,.51,10,10],[99.49,99.49,200,200]),2:([.05,.2,.2,.35,3],[.5,.5,.6,.5,6]),
    3:([1e-5,1e-5,1],[100,100,3]),4:([.125,.1,.1,.125],[5,10,10,5]),5:([55,75,1000,11],[80,110,3000,20]),
    6:([2.6,.7,16.51,7.3,7.3,2.9,5],[3.6,.8,28.49,8.3,8.3,3.9,5.5])}
RW_REF={1:[3.5964885e5,-7.3303829e3],2:[-1.2746083e-3,3.1825489e2],3:[.1,1e5],
    4:[3.6679325e1,1.3066667e-2],5:[5.3067,3.0281682],6:[5.9698221e3,1.3e3]}
RW_GSCALE={1:[4,4],2:[2800,.15,.15,.01,.01],3:[.1,1e5,1e5],4:[13600,30000,5,6000],
    5:[20,.4,1,900],6:[1/27,1/397.5,1/1.93,1/1.93,40,12,5,8.3,8.3,1300,850]}
MMF_BOUNDS={1:([1,-1],[3,1]),2:([0,0],[1,2]),3:([0,0],[1,1.5]),4:([-1,0],[1,2]),
    5:([1,-1],[3,3]),6:([1,-1],[3,2]),7:([1,-1],[3,1]),8:([-np.pi,0],[np.pi,9])}


def lir_constraints(F,index):
    x,y=F[:,0],F[:,1]
    if index<=4:raise ValueError('LIR1-4 constraints require distance components')
    if index>=13:
        s=(F*F).sum(1);g=[(s-9)*(4-s),(s-3.61)*(3.24-s)]
        if index==14:g.append((s-3.0625)*(2.56-s))
        return np.column_stack(g)
    if index in (5,6,7,8):
        if index==5:p=q=[1.6,2.5];a=[2,2];b=[4,8]
        elif index==6:p=q=[1.8,2.8];a=[2,2];b=[8,8]
        else:p=q=[1.2,2.25,3.5];a=[2,2.5,2.5];b=[6,12,10]
    else:
        p,q,a,b={9:([1.4],[1.4],[1.5],[6]),10:([1.1],[1.2],[2],[4]),
            11:([1.2],[1.2],[1.5],[5]),12:([1.6],[1.6],[1.5],[6])}[index]
    theta=-np.pi/4;g=[]
    for pk,qk,ak,bk in zip(p,q,a,b):
        g.append(.1-((x-pk)*np.cos(theta)-(y-qk)*np.sin(theta))**2/ak**2
            -((x-pk)*np.sin(theta)+(y-qk)*np.cos(theta))**2/bk**2)
    if index>=9:
        level={9:2.,10:1.,11:2.1,12:2.5}[index]
        g.append(level-(x+y)/np.sqrt(2)+np.sin(4*np.pi*(x-y)/np.sqrt(2)))
    return np.column_stack(g)


def lir(X,index):
    n,d=X.shape;t=X[:,0];j=np.arange(2,d+1)
    if index<=4:
        if index==1:
            target=np.where((j%2)[None,:]==1,np.sin(.5*np.pi*t[:,None]),np.cos(.5*np.pi*t[:,None]))
        else:target=np.broadcast_to(t[:,None],(n,d-1))
        e=(X[:,1:]-target)**2;g1=e[:,j%2==1].sum(1);g2=e[:,j%2==0].sum(1)
        h=t*t if index in (1,3)else np.sqrt(t)
        F=np.column_stack((t+g1,1-h+g2))
        G=[(.5-g1)*(.51-g1),(.5-g2)*(.51-g2)]
        if index>=3:G.append(.5-np.sin(20*np.pi*t))
        return F,np.column_stack(G)
    if index>=13:
        r=1.7057+10*((X[:,2:]-.5)**2).sum(1)
        F=r[:,None]*np.column_stack((np.cos(np.pi*X[:,0]/2)*np.cos(np.pi*X[:,1]/2),
            np.cos(np.pi*X[:,0]/2)*np.sin(np.pi*X[:,1]/2),np.sin(np.pi*X[:,0]/2)))
        return F,lir_constraints(F,index)
    target=np.where((j%2)[None,:]==1,np.sin(.5*j/d*np.pi*t[:,None]),np.cos(.5*j/d*np.pi*t[:,None]))
    e=(X[:,1:]-target)**2;g1=e[:,j%2==1].sum(1);g2=e[:,j%2==0].sum(1)
    h=t*t if index in (6,8,9,12)else np.sqrt(t)
    F=(np.column_stack((t+10*g1+.7057,1-h+10*g2+.7057))if index<=8 else
       1.7057*np.column_stack((t*(1+10*g1),(1-h)*(1+10*g2))))
    return F,lir_constraints(F,index)


def ctp_constraint(F,theta,a,b,c,d,e):
    f1,f2=F.T
    lhs=(f2-e)*np.cos(theta)-f1*np.sin(theta)
    u=(f2-e)*np.sin(theta)+f1*np.cos(theta)
    return a*abs(np.sin(b*np.pi*u**c))**d-lhs


def ctp(X,index):
    t=X[:,0];g=1+10*(X.shape[1]-1)+(X[:,1:]**2-10*np.cos(2*np.pi*X[:,1:])).sum(1)
    if index==1:
        F=np.column_stack((t,g*np.exp(-t/g)));a=b=1.;alpha=1/3;G=[]
        for _ in range(2):
            beta=a*np.exp(-b*alpha);a=(a+beta)/2;b=-np.log(beta/a)/alpha;alpha+=1/3
            G.append(a*np.exp(-b*t)-F[:,1])
        return F,np.column_stack(G)
    F=np.column_stack((t,g*(1-np.sqrt(t/g))))
    args={2:(-.2*np.pi,.2,10,1,6,1),3:(-.2*np.pi,.1,10,1,.5,1),4:(-.2*np.pi,.75,10,1,.5,1),
        5:(-.2*np.pi,.1,10,2,.5,1),6:(.1*np.pi,40,.5,1,2,-2),7:(-.05*np.pi,40,5,1,6,0)}
    G=([ctp_constraint(F,*args[index])]if index<8 else
       [ctp_constraint(F,.1*np.pi,40,.5,1,2,-2),ctp_constraint(F,-.05*np.pi,40,2,1,6,0)])
    return F,np.column_stack(G)


def mmf(X,index):
    x,y=X.T
    if index in (1,5,6,7):t=abs(x-2)
    elif index==4:t=abs(x)
    elif index==8:t=np.sin(abs(x))
    else:t=x
    if index in (2,3):
        first=(y<=1)if index==2 else((y<=.5)|((y<1)&(x>.25)))
        u=y-np.where(first,0.,1. if index==2 else .5)-np.sqrt(x)
        loss=2*(4*u*u-2*np.cos(20*u*np.pi/np.sqrt(2))+2)
    elif index==4:
        u=y-np.where(y<1,0.,1.)-np.sin(np.pi*abs(x));loss=2*u*u
    elif index==7:
        base=(.3*t*t*np.cos(24*np.pi*t+4*np.pi)+.6*t)*np.sin(6*np.pi*t+np.pi);loss=(y-base)**2
    elif index==8:
        u=y-np.where(y<=4,0.,4.)-np.sin(abs(x))-abs(x);loss=2*u*u
    else:
        if index==1:offset=0.
        elif index==5:offset=np.where(y<=1,0.,2.)
        else:
            first=(y<=0)|((y<=1)&((x<=7/6)|((x>8/6)&(x<=9/6))|((x>10/6)&(x<=11/6))|
                ((x>13/6)&(x<=14/6))|((x>15/6)&(x<=16/6))|(x>17/6)))
            offset=np.where(first,0.,1.)
        loss=2*(y-offset-np.sin(6*np.pi*t+np.pi))**2
    base=1-t*t if index==4 else np.sqrt(np.maximum(0.,1-t*t))if index==8 else 1-np.sqrt(t)
    return np.column_stack((t,base+loss)),np.empty((len(X),0))


def rwmop(X,index):
    x=list(X.T)
    if index==1:
        x1,x2,x3,x4=x;x1=np.floor(x1+.5);x2=np.floor(x2+.5);z1=.0625*x1;z2=.0625*x2
        F=[1.7781*z1*x3*x3+.6224*z1*x2*x4+3.1661*z1*z1*x4+19.84*z1*z1*x3,
           -np.pi*x3*x3*x4-4/3*np.pi*x3**3];G=[.00954*x3-z2,.0193*x3-z1]
    elif index==2:
        d1,d2,d3,b,L=x
        mu=2*b*(100*d1+2770*(d2-d1)+7780*(d3-d2))
        EI=2*b/3*(1.6*d1**3+70*(d2**3-d1**3)+7780*(d3-d2))
        F=[-np.pi/(2*L)**2*np.sqrt(abs(EI/mu)),2*b*L*(500*d1+1500*(d2-d1)+800*(d3-d2))]
        G=[mu*L-2800,d1-d2,d2-d1-.15,d2-d3,d3-d2-.01]
    elif index==3:
        a,b,c=x;F=[a*np.sqrt(16+c*c)+b*np.sqrt(1+c*c),20*np.sqrt(16+c*c)/(c*a)]
        G=[F[0]-.1,F[1]-1e5,80*np.sqrt(1+c*c)/(c*b)-1e5]
    elif index==4:
        a,b,c,d=x;P=6000;L=14;E=30e6;shear=12e6
        pc=4.013*E*np.sqrt((c*c+d**6)/36)/L**2*(1-c/(2*L)*np.sqrt(E/(4*shear)))
        sigma=6*P*L/(d*c*c);J=2*np.sqrt(2)*a*b*(b*b/12+((a+c)/2)**2)
        R=np.sqrt(b*b/4+((a+c)/2)**2);torque=P*(L+b/2);t1=P/(np.sqrt(2)*a*b);t2=torque*R/J
        tau=np.sqrt(t1*t1+2*t1*t2*b/(2*R)+t2*t2)
        F=[1.10471*a*a*b+.04811*c*d*(14+b),4*P*L**3/(E*d*c**3)]
        G=[tau-13600,sigma-30000,a-d,P-pc]
    elif index==5:
        a,b,c,d=x;gap=b*b-a*a;cube=b**3-a**3
        F=[4.9e-5*gap*(d-1),9.82e6*gap/(c*d*cube)]
        G=[20-(b-a),c/(3.14*gap)-.4,2.22e-3*c*cube/gap**2-1,900-2.66e-2*c*d*cube/gap]
    else:
        a,b,c,d,e,f,g=x;c=np.floor(c+.5)
        F=[.7854*a*b*b*(10*c*c/3+14.933*c-43.0934)-1.508*a*(f*f+g*g)+7.477*(f**3+g**3)+.7854*(d*f*f+e*g*g),
           np.sqrt((745*d/(b*c))**2+1.69e7)/(.1*f**3)]
        G=[1/(a*b*b*c)-1/27,1/(a*b*b*c*c)-1/397.5,d**3/(b*c*f**4)-1/1.93,
           e**3/(b*c*g**4)-1/1.93,b*c-40,a/b-12,-a/b+5,1.9-d+1.5*f,1.9-e+1.1*g,
           F[1]-1300,np.sqrt((745*e/(b*c))**2+1.575e8)/(.1*g**3)-850]
    return np.column_stack(F),np.column_stack(G)


class ApplicationProblem:
    def __init__(self,case,data=None):
        assert case in CASES
        self.case=case;self.family=case.split('-')[1].rstrip('0123456789');self.index=int(case.split('-')[1][len(self.family):])
        i=self.index;self.n_obj=3 if self.family=='LIR'and i>=13 else 2
        if self.family=='LIR':
            low=np.zeros(30);high=np.ones(30);self.n_ieq_constr=2 if i in (1,2,5,6,9,10,11,12,13)else 3
            self.objective_scale=np.full(self.n_obj,2.5);self.constraint_scale=np.ones(self.n_ieq_constr)
        elif self.family=='CTP':
            low=np.r_[0.,np.full(9,-5.12)];high=np.r_[1.,np.full(9,20. if i in (6,8)else 5.12)]
            self.n_ieq_constr=2 if i in (1,8)else 1
            self.objective_scale=np.array([1.,20. if i in (6,8)else 2.]);self.constraint_scale=np.ones(self.n_ieq_constr)
        elif self.family=='MMF':
            low,high=MMF_BOUNDS[i];self.n_ieq_constr=0;self.objective_scale=np.ones(2);self.constraint_scale=np.ones(0)
        else:
            low,high=RW_BOUNDS[i];self.n_ieq_constr=len(RW_GSCALE[i]);self.objective_scale=np.abs(RW_REF[i]);self.constraint_scale=np.array(RW_GSCALE[i])
        self.physical_lower=np.asarray(low,float);self.physical_upper=np.asarray(high,float);self.n_var=len(low)
        self.lower=np.zeros(self.n_var);self.upper=np.ones(self.n_var)
        self.metric_names=[f'f{i+1}'for i in range(self.n_obj)]

    def physical(self,Z):return self.physical_lower+np.asarray(Z)*(self.physical_upper-self.physical_lower)

    def canonicalize(self,Z):
        out=np.clip(np.asarray(Z,float),0.,1.).copy()
        if self.family=='RW'and self.index in (1,6):
            columns=[0,1]if self.index==1 else[2]
            for j in columns:
                v=np.floor(self.physical(out)[:,j]+.5)
                out[:,j]=(v-self.physical_lower[j])/(self.physical_upper[j]-self.physical_lower[j])
        return out

    def native(self,Z):
        X=self.physical(self.canonicalize(Z))
        return {'LIR':lir,'CTP':ctp,'MMF':mmf,'RW':rwmop}[self.family](X,self.index)

    def evaluate(self,Z,return_as_dictionary=False):
        F,G=self.native(Z);out=dict(F=F/self.objective_scale,G=G/self.constraint_scale,M=F)
        return out if return_as_dictionary else(out['F'],out['G'])
