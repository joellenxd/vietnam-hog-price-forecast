"""P3: expanding window monthly walk-forward utilities; no future-label leakage.
Rule: at retrain start D, a training origin t is allowed only if t+h < D.
This module defines shared scheduling; Naive requires no fitted parameters.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

DEFAULT_CONFIG='data/split/split_config.json'

def load_config(path=DEFAULT_CONFIG):
    with open(path,encoding='utf-8') as f: return json.load(f)

def retrain_dates(config):
    dates=pd.to_datetime(config['retrain_dates_in_test'])
    if not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError('Ngay retrain phai tang va khong trung')
    return dates

def segment_origins(origin_dates, config):
    """Map Test forecast origin to most recent monthly retrain date."""
    dates=pd.to_datetime(pd.Series(origin_dates))
    schedules=retrain_dates(config)
    t0=pd.Timestamp(config['test']['start']);t1=pd.Timestamp(config['test']['end'])
    if ((dates<t0)|(dates>t1)).any():raise ValueError('Origin nam ngoai Test')
    i=schedules.searchsorted(dates,side='right')-1
    if (i<0).any():raise ValueError('Origin truoc moc retrain dau')
    return pd.Series(schedules.take(i),index=dates.index,name='retrain_date')

def available_training_rows(df, retrain_date, horizon, date_col='date'):
    """Training origins with fully revealed labels strictly BEFORE retrain_date.
    This is a safe upper bound; callers can additionally filter split/labels.
    """
    if horizon not in (7,14,30):raise ValueError('horizon khong hop le')
    d=pd.to_datetime(df[date_col]);r=pd.Timestamp(retrain_date)
    return df.loc[(d+pd.Timedelta(days=horizon))<r].copy()

def get_walkforward_windows(config, horizon):
    """List (retrain_date, prediction_start, prediction_end, latest_safe_train_origin)."""
    dates=retrain_dates(config)
    end=pd.Timestamp(config['test']['end'])
    windows=[]
    for i,date in enumerate(dates):
        block_end=(dates[i+1]-pd.Timedelta(days=1)) if i+1<len(dates) else end
        windows.append({'retrain_date':date.date().isoformat(),
                        'prediction_start':date.date().isoformat(),
                        'prediction_end':block_end.date().isoformat(),
                        'latest_safe_train_origin':(date-pd.Timedelta(days=horizon+1)).date().isoformat()})
    return windows

if __name__=='__main__':
    for h in (7,14,30):
        print('Horizon',h)
        for row in get_walkforward_windows(load_config(),h):print(row)
