#!/usr/bin/env python3
"""Assemble unchanged baselines and locked SCARF-v3 outputs without duplicate rows."""
from pathlib import Path
import argparse,json,shutil
import pandas as pd

def main(root,baseline,scarf,ablation,external):
    root=Path(root);baseline=Path(baseline);scarf=Path(scarf);ablation=Path(ablation);external=Path(external) if external else None
    out=root/'results';out.mkdir(exist_ok=True)
    for name in ['monthly_metrics.csv','drift_metrics.csv','group_metrics.csv','selected_samples.csv']:
        b=pd.read_csv(baseline/name);b=b[b.method!='SCARF'];s=pd.read_csv(scarf/name)
        z=pd.concat([b,s],ignore_index=True)
        keys=['seed','budget','method','month']+(['index'] if name=='selected_samples.csv' else [])+(['group'] if name=='group_metrics.csv' else [])
        assert not z.duplicated(keys).any(),(name,'duplicate keys')
        z.to_csv(out/name,index=False)
    shutil.copyfile(baseline/'static_metrics.csv',out/'static_metrics.csv')
    shutil.copyfile(ablation/'ablation_metrics.csv',out/'ablation_metrics.csv')
    if external:
        for name in ['external_metrics.csv','external_static.csv','external_diagnostics.csv','external_training_manifest.json']:
            shutil.copyfile(external/name,out/name)
    protocol={
      'dataset':'BODMAS timestamps parquet mirror','initial_training':'2019-08','calibration':'2019-09',
      'test_months':['2019-10','2019-11','2019-12','2020-01','2020-02','2020-03','2020-04','2020-05','2020-06','2020-07','2020-08','2020-09'],
      'development_months':['2019-10','2019-11','2019-12','2020-01','2020-02','2020-03'],
      'locked_late_months':['2020-04','2020-05','2020-06','2020-07','2020-08','2020-09'],
      'base_features':384,'seeds':[13,42,87],'budgets':[25,50,100,200],'prequential':True,
      'scarf_version':'final','ensemble_members':3,'ensemble_trees':240,'ensemble_learning_rate':0.05,'query_weight':2.0,'pseudo_weight':0.1,
      'pseudo_cap_per_class':500,'reliability_auc':0.995,'reliability_scale':0.05,
      'baseline_source':'Unchanged outputs from the frozen single-model baseline implementation'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--baseline',required=True);p.add_argument('--scarf',required=True);p.add_argument('--ablation',required=True);p.add_argument('--external');a=p.parse_args();main(a.root,a.baseline,a.scarf,a.ablation,a.external)
