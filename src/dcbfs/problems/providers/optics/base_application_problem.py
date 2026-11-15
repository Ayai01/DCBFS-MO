"""Fast application models with frozen data and a common unit-box interface."""
from pathlib import Path

import numpy as np


class ApplicationProblem:
    """Models expose only F/G/M; solver settings never depend on the case."""
    def __init__(self, case, data):
        self.case = case
        with np.load(Path(data), allow_pickle=False) as z:
            self.data = {k:z[k].copy() for k in z.files}
        self.n_obj = 3
        self.constraints_depend_on_objectives = False
        self.compact_cache_keys = True
        self.fidelity_levels = ('highest',)
        if case == 'OPT40':
            self.n_var, self.n_ieq_constr = 40, 4
            self.metric_names = ['pass_mean_T','stop_mean_T','thickness_nm','max_absorption','worst_angle_pass_mean_T']
            self.objective_names = ['pass_transmission','stop_rejection','coating_thinness']
        elif case == 'PORT48':
            self.n_var, self.n_ieq_constr = 48, 51
            self.metric_names = ['annual_mean_return','annual_volatility','daily_CVaR95','turnover','effective_assets','max_weight']
            self.metric_names += ['weight_'+n for n in self.data['industry_names'].tolist()]
            self.objective_names = ['return','CVaR95','turnover']
        elif case == 'NET':
            self.n_var = int(self.data['routing_parameter_count'])+len(self.data['capacity_lower'])
            self.n_ieq_constr = len(self.data['capacity_lower'])
            self.metric_names = ['capacity_cost','traffic_weighted_hops','max_utilization']
            self.metric_names += ['utilization_'+n for n in self.data['link_names'].tolist()]
            self.objective_names = ['capacity_cost','forwarding_work','peak_utilization']
        else:
            raise ValueError(case)
        self.lower, self.upper = np.zeros(self.n_var), np.ones(self.n_var)
        self.nominal = np.full(self.n_var, .5)
        self.name = case

    def canonicalize(self, X):
        return np.clip(np.asarray(X, float), 0., 1.).copy()

    def spectrum(self, thickness_nm):
        """Coherent transfer matrices, dispersive complex n, s/p at 0/20deg."""
        d = self.data
        wave = d['wavelength_nm']
        indices = d['layer_indices']
        angle = d['scenario_angle_deg']
        p_pol = d['scenario_p_polarized']
        sin0 = np.sin(np.deg2rad(angle))
        cos0 = np.cos(np.deg2rad(angle))
        adm0 = np.where(p_pol, 1./cos0, cos0)
        n_sub = 1.52
        cos_sub = np.sqrt(1-(sin0/n_sub)**2)
        adm_sub = np.where(p_pol, n_sub/cos_sub, n_sub*cos_sub)
        n = len(thickness_nm)
        a = np.ones((n,len(wave)), complex)
        b, c, e = np.zeros_like(a), np.zeros_like(a), a.copy()
        for layer in range(self.n_var):
            refr = indices[layer % 2]
            cosine = np.sqrt(1-(sin0/refr)**2)
            adm = np.where(p_pol, refr/cosine, refr*cosine)
            delta = 2*np.pi*thickness_nm[:,layer,None]*refr*cosine/wave
            cc, ss = np.cos(delta), -1j*np.sin(delta)
            a, b, c, e = (a*cc+b*adm*ss, a*ss/adm+b*cc,
                          c*cc+e*adm*ss, c*ss/adm+e*cc)
        denominator = adm0*a+adm0*adm_sub*b+c+adm_sub*e
        reflection = (adm0*a+adm0*adm_sub*b-c-adm_sub*e)/denominator
        transmission = 2*adm0/denominator
        R = np.abs(reflection)**2
        T = np.real(adm_sub)/np.real(adm0)*np.abs(transmission)**2
        return R, T, 1-R-T

    def weights(self, X):
        # Forty-eight independent log-ratios for forty-nine simplex weights.
        logits = np.column_stack((8*X-4, np.zeros(len(X))))
        logits -= logits.max(axis=1,keepdims=True)
        w = np.exp(logits)
        return w/w.sum(axis=1,keepdims=True)

    def routing(self, X):
        d = self.data
        n = len(X)
        logits = np.full((n,len(d['demands']),3), -np.inf)
        logits[:,d['valid_flow'],d['valid_slot']] = 0.
        logits[:,d['parameter_flow'],d['parameter_slot']] = 6*X[:,:int(d['routing_parameter_count'])]-3
        logits -= logits.max(axis=2,keepdims=True)
        shares = np.exp(logits)
        shares /= shares.sum(axis=2,keepdims=True)
        path_flows = shares[:,d['valid_flow'],d['valid_slot']]*d['demands'][d['valid_flow']]
        load = path_flows@d['path_link_incidence']
        capacities = d['capacity_lower']+X[:,int(d['routing_parameter_count']):]*(d['capacity_upper']-d['capacity_lower'])
        return path_flows, load, capacities

    def compute(self, X):
        d = self.data
        if self.case == 'OPT40':
            thickness = 20+200*X
            _, T, absorption = self.spectrum(thickness)
            angular_pass = T[:,d['pass_indices']].reshape(len(X),4,-1).mean(axis=2)
            return np.column_stack((angular_pass.mean(axis=1),T[:,d['stop_indices']].mean(axis=1),
                thickness.sum(axis=1),absorption.max(axis=1),angular_pass.min(axis=1)))
        if self.case == 'PORT48':
            w = self.weights(X)
            daily = w@d['train_returns'].T
            losses = -daily
            mass = .05*losses.shape[1]
            count = int(np.floor(mass))
            ordered = np.partition(losses,losses.shape[1]-count-1,axis=1)
            cvar = (ordered[:,-count:].sum(axis=1)+(mass-count)*ordered[:,-count-1])/mass
            return np.column_stack((252*daily.mean(axis=1),np.sqrt(252)*daily.std(axis=1,ddof=1),cvar,
                .5*np.abs(w-1/49).sum(axis=1),1/np.square(w).sum(axis=1),w.max(axis=1),w))
        flows, load, capacities = self.routing(X)
        utilization = load/capacities
        cost = ((capacities-d['capacity_lower'])/d['module_capacity']*d['module_cost']).sum(axis=1)
        hops = (flows*d['path_hops']).sum(axis=1)/d['demands'].sum()
        return np.column_stack((cost,hops,utilization.max(axis=1),utilization))

    def objectives_from_metrics(self, M):
        if self.case == 'OPT40':
            utilities = np.column_stack((M[:,0],1-M[:,1],1-M[:,2]/8800))
        elif self.case == 'PORT48':
            utilities = np.column_stack(((M[:,0]+.1)/.5,1-M[:,2]/.1,1-M[:,3]/.65))
        else:
            d = self.data
            utilities = np.column_stack((1-M[:,0]/float(d['max_capacity_cost']),
                1-(M[:,1]-float(d['minimum_hops']))/float(d['hop_range']),1-M[:,2]))
        return -np.clip(utilities,0.,1.)

    def constraints_from_metrics(self, M):
        if self.case == 'OPT40':
            return np.column_stack(((.30-M[:,4])/.30,(M[:,1]-.60)/.60,(M[:,2]-6500)/6500,(M[:,3]-.05)/.05))
        if self.case == 'PORT48':
            return np.column_stack((M[:,6:]/.15-1,(12-M[:,4])/12,(M[:,3]-.65)/.65))
        return M[:,3:]-.95

    def constraint_utilities(self, G):
        return 1-np.asarray(G)

    def evaluate(self, X, return_as_dictionary=False):
        X = self.canonicalize(X)
        M = self.compute(X)
        result = dict(F=self.objectives_from_metrics(M),G=self.constraints_from_metrics(M),M=M)
        if not return_as_dictionary:
            raise ValueError('Application adapter requires dictionary evaluation')
        return result
