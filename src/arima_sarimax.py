"""P3 stage 1: expanding-window, monthly-refit ARIMA and SARIMAX.

Run from repository root:
    py -m pip install statsmodels pandas numpy
    py src/arima_sarimax.py

Exogenous SARIMAX features are *known calendar values* (weekly sine/cosine).
No future observed price/weather/feed value enters a forecast. Hyperparameters
are selected only on Validation; Test is only scored after selection.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from evaluate import PREDICTION_COLUMNS, evaluate_predictions

HORIZONS=(7,14,30)
MODELS=('arima','sarimax')
CANDIDATES=((0,1,0),(1,1,0))

def known_calendar(dates):
    d=pd.DatetimeIndex(pd.to_datetime(dates))
    w=d.day_of_week.to_numpy(dtype=float)
    return np.column_stack((np.sin(2*np.pi*w/7),np.cos(2*np.pi*w/7)))

def prepare_region(data, region):
    t=data.loc[data.region.eq(region)].copy().sort_values('date')
    t['date']=pd.to_datetime(t.date)
    t=t.set_index('date')
    if t.index.has_duplicates: raise ValueError('Duplicate region-date')
    full=pd.date_range(t.index.min(),t.index.max(),freq='D')
    if not full.equals(t.index): raise ValueError('Expected continuous daily calendar')
    return t

def fit_model(series, model, order):
    # Missing observations remain NaN: Kalman filtering handles these; do not
    # convert imputed values to ground-truth price observations.
    exog=known_calendar(series.index) if model=='sarimax' else None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=UserWarning)
        spec=SARIMAX(series.to_numpy(dtype=float),order=order,trend='n',
                     exog=exog,enforce_stationarity=False,
                     enforce_invertibility=False)
        return spec.fit(disp=False,maxiter=55)

def forecast_from_state(state, model, origin_date):
    dates=pd.date_range(pd.Timestamp(origin_date)+pd.Timedelta(days=1),periods=max(HORIZONS),freq='D')
    x=known_calendar(dates) if model=='sarimax' else None
    return np.asarray(state.forecast(max(HORIZONS),exog=x),dtype=float)

def run_block(region_data, model, order, start, end, train_start=None):
    """Fit once at block start, update Kalman state with each day's realized price.

    Forecast at origin t uses current t observation and known calendar only.
    Daily state update does not reestimate model parameters. Refit at month start.
    """
    start=pd.Timestamp(start);end=pd.Timestamp(end)
    cutoff=start-pd.Timedelta(days=1)
    s=region_data.loc[(region_data.index<=cutoff),'price'].astype(float)
    if train_start is not None:s=s.loc[s.index>=pd.Timestamp(train_start)]
    if s.notna().sum()<35: raise ValueError('Not enough observed training dates')
    fitted=fit_model(s,model,order)
    state=fitted
    records=[]
    for dt in pd.date_range(start,end,freq='D'):
        obs=region_data.loc[dt,'price'] if dt in region_data.index else np.nan
        x=known_calendar([dt]) if model=='sarimax' else None
        state=state.extend(np.array([float(obs)]),exog=x)
        if dt not in region_data.index: continue
        row=region_data.loc[dt]
        if bool(row['is_sunday_copy']):continue
        forecast=forecast_from_state(state,model,dt)
        for h in HORIZONS:
            if not bool(row.get(f'valid_h{h}',False)):continue
            truth=row[f'y_h{h}']
            value=forecast[h-1]
            if not np.isfinite(truth) or not np.isfinite(value):continue
            records.append({'model':model,'config':'c6','region':row['region'],
              'horizon':h,'origin_date':dt.strftime('%Y-%m-%d'),
              'target_date':(dt+pd.Timedelta(days=h)).strftime('%Y-%m-%d'),
              'y_true':float(truth),'y_pred':float(value)})
    return pd.DataFrame(records,columns=PREDICTION_COLUMNS)

def choose_orders(regions):
    """Use Train-fitted models to score Val only; one order / region / model."""
    rows=[]
    for region,frame in regions.items():
        for model in MODELS:
            options=[]
            for order in CANDIDATES:
                preds=run_block(frame,model,order,'2026-04-01','2026-06-30')
                if preds.empty:score=float('inf');n=0
                else:
                    score=float(np.mean(np.abs(preds.y_true-preds.y_pred)));n=len(preds)
                options.append((score,order,n))
                rows.append({'region':region,'model':model,'order':str(order),
                             'val_mae':score,'val_n':n,'selected':False})
            chosen=min(options,key=lambda t:(t[0],t[1]))[1]
            for entry in rows:
                if entry['region']==region and entry['model']==model and entry['order']==str(chosen):
                    entry['selected']=True
    return pd.DataFrame(rows)

def run_experiment(data):
    required={'date','region','price','is_sunday_copy',
              'y_h7','y_h14','y_h30','valid_h7','valid_h14','valid_h30','split'}
    if missing:=required-set(data.columns):raise ValueError(f'Missing: {sorted(missing)}')
    data=data.copy(); data['date']=pd.to_datetime(data.date)
    regions={r:prepare_region(data,r) for r in ('Bac','Trung','Nam')}
    selection=choose_orders(regions)
    all_preds=[]
    blocks=[('2026-07-01','2026-07-31'),('2026-08-01','2026-08-31'),
            ('2026-09-01','2026-09-30'),('2026-10-01','2026-10-30')]
    for region,frame in regions.items():
        for model in MODELS:
            selected=selection.loc[(selection.region==region)&(selection.model==model)&selection.selected]
            if len(selected)!=1:raise RuntimeError('Invalid model selection')
            import ast
            order=tuple(ast.literal_eval(selected.iloc[0]['order']))
            for start,end in blocks:
                preds=run_block(frame,model,order,start,end)
                all_preds.append(preds)
    result=pd.concat(all_preds,ignore_index=True)
    # Guard: every forecast valid in Test and target also in Test
    if not result.empty:
        dt=pd.to_datetime(result.origin_date)
        target=pd.to_datetime(result.target_date)
        if not ((dt>=pd.Timestamp('2026-07-01')) & (target<=pd.Timestamp('2026-10-30'))).all():
            raise AssertionError('Test boundary leakage')
    metrics=evaluate_predictions(result)
    return result[PREDICTION_COLUMNS].sort_values(['model','region','horizon','origin_date']),metrics,selection

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='data/features/data_v1.csv')
    p.add_argument('--prefix',default='p3_arima_sarimax')
    args=p.parse_args()
    data=pd.read_csv(args.input)
    preds,metrics,selection=run_experiment(data)
    Path('predictions').mkdir(exist_ok=True)
    Path('results/tables').mkdir(parents=True,exist_ok=True)
    output1=f'predictions/{args.prefix}.csv'
    output2=f'results/tables/{args.prefix}_metrics.csv'
    output3=f'results/tables/{args.prefix}_val_selection.csv'
    preds.to_csv(output1,index=False,float_format='%.6f')
    metrics.to_csv(output2,index=False,float_format='%.6f')
    selection.to_csv(output3,index=False,float_format='%.6f')
    print('Forecast records:',len(preds),'->',output1)
    print(metrics.to_string(index=False))
    print('Model selection ->',output3)
if __name__=='__main__':main()
