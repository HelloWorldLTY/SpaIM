from options import Options
import os 
import numpy as np
import pandas as pd
import torch
from utils import CalculateMeteics

if __name__ == '__main__':
    opt = Options().parse()
    root = opt.save_path

    impute_all = []
    raw_all = []
    for k in range(10):
        impute = pd.read_pickle(os.path.join(root, 'impute_result_%d.pkl'%(k)))
        raw = pd.read_pickle(os.path.join(root, 'input_result_%d.pkl'%(k)))
        
        impute_all.append(impute)
        raw_all.append(raw)
    impute_all = pd.concat(impute_all, axis=1)
    raw_all = pd.concat(raw_all, axis=1)

    evaluate = CalculateMeteics(raw_all, impute_all, root, 'None', 'SpaImputation')
    acc = evaluate.compute_all(K='none')
    print(acc.T['PCC'].mean())
