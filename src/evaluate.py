"""P3: Standard prediction schema and MAE/RMSE/MAPE evaluation.
Run from repository root: python src/evaluate.py predictions/naive_c6.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PREDICTION_COLUMNS = ['model','config','region','horizon','origin_date','target_date','y_true','y_pred']
METRIC_COLUMNS = ['model','config','region','horizon','n','mae','rmse','mape']
HORIZONS = (7,14,30)
REGIONS = {'Bac','Trung','Nam'}


def evaluate(y_true, y_pred):
    """Return score dictionary; ignore nonfinite pairs, MAPE excludes zero y_true.
    Empty input after filtering is an error. MAPE=NaN when all truths are zero.
    """
    a=np.asarray(y_true,dtype=float).reshape(-1)
    b=np.asarray(y_pred,dtype=float).reshape(-1)
    if len(a)!=len(b):raise ValueError('y_true va y_pred khac so phan tu')
    good=np.isfinite(a)&np.isfinite(b)
    a,b=a[good],b[good]
    if not len(a):raise ValueError('Khong co cap du lieu hop le')
    err=a-b
    nz=(a!=0)
    return {'n':int(len(a)),'mae':float(np.mean(np.abs(err))),
            'rmse':float(np.sqrt(np.mean(err**2))),
            'mape':float(np.mean(np.abs(err[nz]/a[nz]))*100) if nz.any() else float('nan')}


def validate_predictions(df):
    missing=set(PREDICTION_COLUMNS)-set(df.columns)
    if missing:raise ValueError(f'Thieu cot CSV: {sorted(missing)}')
    if df.empty:raise ValueError('CSV trong')
    if df[list(PREDICTION_COLUMNS[:6])].isna().any().any():raise ValueError('Cot khoa co gia tri thieu')
    d=df.copy()
    d['horizon']=pd.to_numeric(d.horizon,errors='raise')
    if not d.horizon.isin(HORIZONS).all():raise ValueError('horizon phai la 7,14,30')
    if not d.region.isin(REGIONS).all():raise ValueError('region phai la Bac,Trung,Nam')
    for col in ['origin_date','target_date']:
        d[col]=pd.to_datetime(d[col],errors='raise')
    if not ((d.target_date-d.origin_date).dt.days==d.horizon).all():
        raise ValueError('target_date phai bang origin_date + horizon ngay')
    keys=['model','config','region','horizon','origin_date']
    if d.duplicated(keys).any():raise ValueError('Trung khoa du bao')
    d['y_true']=pd.to_numeric(d.y_true,errors='coerce')
    d['y_pred']=pd.to_numeric(d.y_pred,errors='coerce')
    return d


def evaluate_predictions(predictions, quality=None):
    """Group by model/config/region/horizon; optionally filter origin Sunday flags.

    The official 8-column prediction CSV does not carry is_sunday_copy.
    Pass quality dataframe with (date,region,is_sunday_copy) to enforce that rule.
    """
    d=validate_predictions(predictions)
    if quality is not None:
        required={'date','region','is_sunday_copy'}
        if not required.issubset(quality.columns):raise ValueError('Quality thieu cot ngay/mien/is_sunday_copy')
        q=quality[['date','region','is_sunday_copy']].copy()
        q['date']=pd.to_datetime(q.date)
        if q.duplicated(['date','region']).any():raise ValueError('Trung ngay/mien quality')
        d=d.merge(q.rename(columns={'date':'origin_date'}),on=['origin_date','region'],how='left',validate='many_to_one')
        if d.is_sunday_copy.isna().any():raise ValueError('Khong ghep duoc co sunday')
        d=d.loc[d.is_sunday_copy.eq(0)].copy()
    d=d.loc[np.isfinite(d.y_true)&np.isfinite(d.y_pred)]
    results=[]
    for keys,g in d.groupby(['model','config','region','horizon'],sort=True):
        results.append({**dict(zip(['model','config','region','horizon'],keys)),**evaluate(g.y_true,g.y_pred)})
    return pd.DataFrame(results,columns=METRIC_COLUMNS)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('predictions',help='Input 8-column prediction CSV')
    parser.add_argument('--output',default='results/tables/p3_naive_metrics.csv')
    parser.add_argument('--quality',default='data/raw/csv/supervised_region_daily.csv')
    args=parser.parse_args()
    d=pd.read_csv(args.predictions)
    quality=pd.read_csv(args.quality) if args.quality else None
    out=evaluate_predictions(d,quality)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output,index=False,float_format='%.6f')
    print(out.to_string(index=False));print('Da luu:',args.output)

if __name__=='__main__':main()
