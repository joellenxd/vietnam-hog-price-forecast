"""P3 stage 0: Naive direct h-step forecasts, using P1 original labels and split.
No ARIMA/SARIMAX in stage 0; that belongs to the next stage.
Run: python src/naive_arima.py
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate import PREDICTION_COLUMNS, evaluate_predictions
from walkforward import load_config, segment_origins
from split import assign_split, is_valid_origin

HORIZONS=(7,14,30)

def run_naive(data, config):
    """Naive yhat(t+h|t)=price_ffill(t); do not use future features.
    Evaluate only TEST origins with horizon landing inside TEST.
    Filter origin Sunday-copy and missing targets; count n per region/horizon.
    """
    required={'date','region','price_ffill','is_sunday_copy','y_h7','y_h14','y_h30'}
    if not required.issubset(data.columns):raise ValueError(f'Thieu cot: {required-set(data.columns)}')
    d=data.copy()
    d['date']=pd.to_datetime(d.date)
    if d.duplicated(['date','region']).any():raise ValueError('Trung ngay/mien')
    d['split']=assign_split(d.date).to_numpy()
    test=d.loc[d.split.eq('test')].copy()
    test['retrain_date']=segment_origins(test.date,config).to_numpy()
    pieces=[]
    for h in HORIZONS:
        subset=test.loc[is_valid_origin(test.date,h).to_numpy()].copy()
        subset=subset.loc[subset.is_sunday_copy.eq(0)].copy()
        subset['y_true']=pd.to_numeric(subset[f'y_h{h}'],errors='coerce')
        subset['y_pred']=pd.to_numeric(subset.price_ffill,errors='coerce')
        subset=subset.loc[np.isfinite(subset.y_true)&np.isfinite(subset.y_pred)].copy()
        piece=pd.DataFrame({'model':'naive','config':'c6','region':subset.region,
                            'horizon':h,'origin_date':subset.date.dt.strftime('%Y-%m-%d'),
                            'target_date':(subset.date+pd.Timedelta(days=h)).dt.strftime('%Y-%m-%d'),
                            'y_true':subset.y_true,'y_pred':subset.y_pred})
        pieces.append(piece)
    return pd.concat(pieces,ignore_index=True)[PREDICTION_COLUMNS].sort_values(['region','horizon','origin_date']).reset_index(drop=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='data/raw/csv/supervised_region_daily.csv')
    parser.add_argument('--config',default='data/split/split_config.json')
    parser.add_argument('--predictions',default='predictions/naive_c6.csv')
    parser.add_argument('--metrics',default='results/tables/p3_naive_mae_by_region_horizon.csv')
    args=parser.parse_args()
    data=pd.read_csv(args.input);config=load_config(args.config)
    preds=run_naive(data,config)
    quality=data[['date','region','is_sunday_copy']]
    result=evaluate_predictions(preds,quality)
    for name,df in [(args.predictions,preds),(args.metrics,result)]:
        Path(name).parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(name,index=False,float_format='%.6f')
    print('Output:',args.predictions,'n=',len(preds))
    print(result.to_string(index=False));print('Metrics:',args.metrics)

if __name__=='__main__': main()
