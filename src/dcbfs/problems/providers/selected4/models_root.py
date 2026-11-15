"""Private deterministic application-model instances; NOT production problems.

These are new illustrative parameter instances of published physical relations,
not measured buildings/farms nor previously published benchmark problems.
Runtime dependency: NumPy only. Physics/evaluation/persistence: FP32.
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CASES = ("ENV36", "ENV48", "ENV60", "AGR48-D", "AGR48-N", "AGR48-W")


def prepare_data():
    """Construct frozen, fully deterministic illustrative parameters (no RNG)."""
    dest = HERE / "root-data"
    dest.mkdir(exist_ok=True)
    def save_once(path, data):
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if set(old.files) != set(data) or any(old[key].dtype != data[key].dtype or not np.array_equal(old[key], data[key]) for key in old.files):
                    raise ValueError(f"Refusing to overwrite different scientific input: {path}")
            return
        np.savez_compressed(path, **data)
    for k, zones in enumerate((12, 16, 20)):
        i = np.arange(zones, dtype=np.float32)
        d = dict(kind=np.asarray("envelope"), n_var=np.int32(3*zones),
            area_m2=(25+5*(i%7)).astype(np.float32),
            degree_hours_Kh=((55000+1800*(i%6))*(1+.12*k*(i%3-1))).astype(np.float32),
            base_R_m2K_W=(.35+.07*(i%5)).astype(np.float32),
            conductivity_W_mK=np.asarray([.036,.024,.045],np.float32),
            max_thickness_m=np.asarray([.12,.12,.12],np.float32),
            max_total_thickness_m=np.full(zones,.18,np.float32),
            cost_EUR_m3=np.asarray([120,270,190],np.float32),
            carbon_kgCO2e_m3=np.asarray([65,145,22],np.float32))
        save_once(dest/f"ENV{3*zones}.npz", d)
    for k, (suffix, rain_factor) in enumerate((("D",.55),("N",1.),("W",1.45))):
        i = np.arange(12,dtype=np.float32)
        crop = np.arange(12,dtype=np.int32)%3
        # FAO/IAEA public Table 1 (Tr.0111/1011/1101/1110): bean,
        # groundnut, sugar cane. Product across stages is OUR model choice.
        ky=np.asarray([[.2,1.1,.75,.2],[.2,.8,.6,.2],[.75,.5,.5,.1]],np.float32)[crop]
        area=(.8+.2*(i%5)).astype(np.float32)
        et=np.asarray([70,110,140,70],np.float32)[None,:]*(1+.15*crop[:,None])
        et=et.astype(np.float32)
        rain=(rain_factor*(.8+.1*(i%5))[:,None]*np.asarray([12,10,8,5],np.float32)[None,:]).astype(np.float32)
        eff=(.7+.02*(i%6)).astype(np.float32)
        maximum=(1.15*et/eff[:,None]).astype(np.float32)
        max_volume=(maximum*area[:,None]*10).sum(axis=0,dtype=np.float32)
        d=dict(kind=np.asarray("irrigation"),n_var=np.int32(48),area_ha=area,
            crop_code=crop,ky=ky,et_max_mm=et,rain_mm=rain,
            irrigation_efficiency=eff,max_irrigation_mm=maximum,
            pump_head_m=(12+4*(i%8)+2*k*(i%3)).astype(np.float32),
            pump_efficiency=(.65+.03*(i%5)).astype(np.float32),
            stage_cap_m3=(max_volume*np.asarray([.55,.48,.56,.60],np.float32)).astype(np.float32),
            # Constructed potential crop yields, used only for diagnostic tonnes.
            yield_max_t_ha=np.asarray([4.,3.,80.],np.float32)[crop])
        save_once(dest/f"AGR48-{suffix}.npz", d)


class RootModel:
    constraints_depend_on_objectives = False
    compact_cache_keys = True
    fidelity_levels = ("highest",)

    def __init__(self, case):
        if case not in CASES:
            raise KeyError(case)
        self.case = self.name = case
        with np.load(HERE/"root-data"/(case+".npz"),allow_pickle=False) as z:
            self.data = {k:z[k].copy() for k in z.files}
        d=self.data
        self.kind=str(d["kind"])
        self.n_var=int(d["n_var"])
        self.lower=np.zeros(self.n_var,np.float32)
        self.upper=np.ones(self.n_var,np.float32)
        self.xl=self.lower; self.xu=self.upper
        self.nominal=np.full(self.n_var,.3,np.float32)
        self.variable_names=[f"x{i+1}" for i in range(self.n_var)]
        if self.kind=="envelope":
            self.family="building_thermal"
            self.n_obj=3; self.n_ieq_constr=self.n_var//3
            self.objective_names=["annual_heating_kWh","material_cost_EUR","embodied_carbon_kgCO2e"]
            self.metric_names=self.objective_names+[f"total_layer_thickness_{i+1}_m" for i in range(self.n_ieq_constr)]
            area=d["area_m2"]
            self.objective_scale=np.asarray([
                np.sum(area*d["degree_hours_Kh"]/d["base_R_m2K_W"])/1000,
                np.sum(area)*np.sum(d["max_thickness_m"]*d["cost_EUR_m3"]),
                np.sum(area)*np.sum(d["max_thickness_m"]*d["carbon_kgCO2e_m3"])],np.float32)
        else:
            self.family="agricultural_irrigation"
            self.n_obj=4; self.n_ieq_constr=4
            self.objective_names=["area_weighted_relative_yield_loss","irrigation_m3","pumping_kWh","worst_plot_relative_yield_loss"]
            self.metric_names=self.objective_names+[f"stage_{s+1}_irrigation_m3" for s in range(4)]+["harvest_tonnes"]
            vmax=d["max_irrigation_mm"]*d["area_ha"][:,None]*10
            ecoef=(np.float32(.002725)*d["pump_head_m"]/d["pump_efficiency"])
            self.objective_scale=np.asarray([1,vmax.sum(),(vmax*ecoef[:,None]).sum(),1],np.float32)
        self.n_constr=self.n_ieq_constr
        self.n_eq_constr=0

    def canonicalize(self,X):
        return np.clip(np.asarray(X,dtype=np.float32),0,1).copy()

    def compute(self,X):
        d=self.data
        X=np.asarray(X,dtype=np.float32)
        if self.kind=="envelope":
            thickness=X.reshape(len(X),-1,3)*d["max_thickness_m"]
            resistance=d["base_R_m2K_W"][None,:]+np.sum(thickness/d["conductivity_W_mK"],axis=2,dtype=np.float32)
            heat=np.sum(d["area_m2"]*d["degree_hours_Kh"]/resistance,axis=1,dtype=np.float32)/np.float32(1000)
            volume=thickness*d["area_m2"][None,:,None]
            cost=np.sum(volume*d["cost_EUR_m3"],axis=(1,2),dtype=np.float32)
            carbon=np.sum(volume*d["carbon_kgCO2e_m3"],axis=(1,2),dtype=np.float32)
            return np.column_stack((heat,cost,carbon,thickness.sum(axis=2,dtype=np.float32))).astype(np.float32)
        irrigation=X.reshape(len(X),12,4)*d["max_irrigation_mm"]
        effective=irrigation*d["irrigation_efficiency"][None,:,None]+d["rain_mm"][None,:,:]
        ratio=np.minimum(effective/d["et_max_mm"][None,:,:],np.float32(1))
        stage_factor=np.maximum(np.float32(0),np.float32(1)-d["ky"][None,:,:]*(np.float32(1)-ratio))
        relative_yield=np.prod(stage_factor,axis=2,dtype=np.float32)
        loss=(np.float32(1)-relative_yield)
        weighted_loss=np.sum(loss*d["area_ha"],axis=1,dtype=np.float32)/np.sum(d["area_ha"],dtype=np.float32)
        volume=irrigation*d["area_ha"][None,:,None]*np.float32(10)
        stage_volume=volume.sum(axis=1,dtype=np.float32)
        ecoef=np.float32(.002725)*d["pump_head_m"]/d["pump_efficiency"]
        energy=np.sum(volume*ecoef[None,:,None],axis=(1,2),dtype=np.float32)
        harvest=np.sum(relative_yield*d["area_ha"]*d["yield_max_t_ha"],axis=1,dtype=np.float32)
        return np.column_stack((weighted_loss,stage_volume.sum(axis=1,dtype=np.float32),energy,loss.max(axis=1),stage_volume,harvest)).astype(np.float32)

    def objectives_from_metrics(self,M):
        return (np.asarray(M,dtype=np.float32)[:,:self.n_obj]/self.objective_scale).astype(np.float32)

    def constraints_from_metrics(self,M):
        d=self.data; M=np.asarray(M,dtype=np.float32)
        if self.kind=="envelope":
            return ((M[:,3:]-d["max_total_thickness_m"])/d["max_total_thickness_m"]).astype(np.float32)
        return (M[:,4:8]/d["stage_cap_m3"]-np.float32(1)).astype(np.float32)

    def evaluate(self,X,return_as_dictionary=False):
        M=self.compute(self.canonicalize(X))
        out=dict(M=M,F=self.objectives_from_metrics(M),G=self.constraints_from_metrics(M))
        return out if return_as_dictionary else out["F"]


def build(case):
    return RootModel(case)


if __name__=="__main__":
    prepare_data()
    print("prepared",len(CASES),"private illustrative model instances")
