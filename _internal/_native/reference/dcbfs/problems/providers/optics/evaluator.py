"""Controlled optical difficulty profiles with common engineering dB scales."""
import importlib.util
from pathlib import Path

import numpy as np

if __package__:
    from .orthogonal_applications import ApplicationProblem as Base
else:
    path=Path(__file__).with_name('base_application_problem.py')
    spec=importlib.util.spec_from_file_location('frozen_optical_base',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Base=module.ApplicationProblem


# Adapted from FreeBSD msun e_log10f.c and k_logf.h (binary32 path).
# ====================================================
# Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
# Developed at SunPro, a Sun Microsystems, Inc. business.
# Permission to use, copy, modify, and distribute this
# software is freely granted, provided that this notice
# is preserved.
# ====================================================
from numba import njit


@njit(cache=(not bool(__import__('os').environ.get('DCBFS_NO_FILES'))),
      nogil=True, fastmath=False)
def _log10_fp32(values):
    """Fixed binary32 operations; no platform log or fused multiply-add."""
    flat=values.ravel()
    result=np.empty(flat.size,np.float32)
    for index in range(flat.size):
        x=flat[index]
        hx=np.float32(x).view(np.int32)
        k=0
        if hx < 0x00800000:
            if (hx & 0x7fffffff) == 0:
                result[index]=np.float32(-np.inf)
                continue
            if hx < 0:
                result[index]=np.float32(np.nan)
                continue
            k-=25
            x=x*np.float32(33554432.)
            hx=np.float32(x).view(np.int32)
        if hx >= 0x7f800000:
            result[index]=x+x
            continue
        if hx == 0x3f800000:
            result[index]=np.float32(0.)
            continue
        k+=(hx >> 23)-127
        hx &= 0x007fffff
        i=(hx+0x4afb0d)&0x800000
        x=np.int32(hx|(i^0x3f800000)).view(np.float32)
        k+=(i >> 23)
        y=np.float32(k)
        f=x-np.float32(1.)
        hfsq=np.float32(.5)*f*f
        s=f/(np.float32(2.)+f)
        z=s*s
        w=z*z
        t1=w*(np.float32(.40000972152)+w*np.float32(.24279078841))
        t2=z*(np.float32(.66666662693)+w*np.float32(.28498786688))
        r=s*(hfsq+(t2+t1))
        hi=f-hfsq
        hi=np.int32(np.float32(hi).view(np.int32)&np.int32(-4096)).view(np.float32)
        lo=((f-hi)-hfsq)+r
        a=y*np.float32(7.9034151668e-7)
        b=(lo+hi)*np.float32(-3.1689971365e-5)
        c=lo*np.float32(.43432617188)
        d=hi*np.float32(.43432617188)
        e=y*np.float32(.30102920532)
        result[index]=(((a+b)+c)+d)+e
    return result.reshape(values.shape)


def _portable_log10(values):
    if values.dtype == np.dtype('float32'):
        return _log10_fp32(values)
    return np.log10(values)


class ApplicationProblem(Base):
    def __init__(self,case,data):
        super().__init__('OPT40',data)
        self.name=case
        self.variant=case
        self.requirements=self.data['requirements']
        self.metric_names=['pass_mean_T','base_stop_mean_T','thickness_nm','max_absorption',
            'worst_angle_pass_mean_T','extension_stop_mean_T']
        self.objective_names=['pass_insertion_loss_dB','stop_attenuation_dB','coating_thickness']

    def compute(self,X):
        thickness=20+200*X
        _,T,absorption=self.spectrum(thickness)
        angular_pass=T[:,self.data['pass_indices']].reshape(len(X),4,-1).mean(axis=2)
        return np.column_stack((angular_pass.mean(axis=1),T[:,self.data['stop_indices']].mean(axis=1),
            thickness.sum(axis=1),absorption.max(axis=1),angular_pass.min(axis=1),
            T[:,self.data['extension_indices']].mean(axis=1)))

    def stop_power(self,M):
        return np.maximum(M[:,1],M[:,5])if bool(self.data['extended_stop'])else M[:,1]

    def objectives_from_metrics(self,M):
        insertion=-10*_portable_log10(np.maximum(M[:,0],np.finfo(float).tiny))
        rejection=-10*_portable_log10(np.maximum(M[:,1],np.finfo(float).tiny))
        utility=np.column_stack((1-insertion/6,rejection/60,1-M[:,2]/8800))
        return -np.clip(utility,0,1)

    def constraints_from_metrics(self,M):
        p,t,h,a=self.requirements
        return np.column_stack((p-M[:,4],self.stop_power(M)-t,M[:,2]-h,M[:,3]-a))/np.array([.3,.6,6500,.05])

    def qualification_metrics(self,X):
        p=.70+.25*np.sin(6.3*X[:,0])
        stop=np.power(10.,-2+1.7*np.cos(6.3*X[:,1]))
        thickness=2800+2000*np.sin(6.3*X[:,2])
        extra=np.power(10.,-2+1.7*np.sin(6.3*X[:,3]))
        return np.column_stack((p,stop,thickness,np.full(len(X),.01),p-.03,extra))
