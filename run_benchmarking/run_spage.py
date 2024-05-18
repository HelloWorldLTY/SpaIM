import numpy as np
import pandas as pd
import sys
import pickle
import os
import time as tm
from functools import partial
import scipy.stats as st
from scipy.stats import wasserstein_distance
import scipy.stats
import copy
from sklearn.model_selection import KFold
import pandas as pd
import multiprocessing
import matplotlib as mpl 
import matplotlib.pyplot as plt
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats
import seaborn as sns

from scipy.spatial.distance import cdist
import h5py
from scipy.stats import spearmanr
import argparse

import time
import sys
import tangram as tg
from os.path import join
from IPython.display import display

from utils import CalculateMeteics

def eval(dataroot, saveroot, k=0):
    #raw_count_file, impute_count_file, prefix, metric
    raw_count_file = sc.read(os.path.join(dataroot, 'Insitu_count.h5ad'))
    raw_count_file = pd.DataFrame(raw_count_file.X, index=raw_count_file.obs.index, columns=raw_count_file.var.index)
    raw_count_file.columns = [x.lower() for x in raw_count_file.columns]

    impute_count_file = np.load(os.path.join(saveroot, 'SpaGE_impute_predict_%d.pkl'%(k)), allow_pickle=True)
    impute_count_file.columns = [x.lower() for x in impute_count_file.columns]

    genes = impute_count_file.columns
    raw_count_file = raw_count_file[genes]

    prefix = saveroot
    metric = 'all'
    name = 'spage'

    evaluate = CalculateMeteics(raw_count_file, impute_count_file, prefix, metric, name)
    acc = evaluate.compute_all(K=k)
    print(acc.T['PCC'].mean())


def SpaGE_impute(K, RNA_data, Spatial_data, train_gene, predict_gene):
    print ('We run SpaGE for this data')
    sys.path.append("External/SpaGE/")
    from SpaGE.main import SpaGE
    # global RNA_data, Spatial_data, train_gene, predict_gene
    train = np.array(train_gene[K])
    predict = np.array(predict_gene[K])
    predict = list(set(predict) & set(RNA_data.columns))
    
    # RNA_data = RNA_data[train]
    # RNA_data = RNA_data.loc[(RNA_data.sum(axis=1) != 0)]
    # RNA_data = RNA_data.loc[(RNA_data.var(axis=1) != 0)]
    
    # pv = len(train) // 2
    pv = 30
    Spatial = Spatial_data[train]
    Img_Genes = SpaGE(Spatial, RNA_data, n_pv = pv, genes_to_predict = predict)
    result = Img_Genes[predict]
    return result, Img_Genes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nano5-1', help='dataset name')
    opt = parser.parse_args()

    DataDir = '../dataset/%s/'%(opt.dataset)
    outdir = '../results/%s/spage'%(opt.dataset)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    RNA_file = DataDir + 'scRNA_count.h5ad'
    Spatial_file = DataDir + 'Insitu_count.h5ad'
    location_file = DataDir + 'Locations.txt'

    # RNA_data = pd.read_table(RNA_file, header=0, index_col = 0)
    # Spatial_data = pd.read_table(Spatial_file, sep = '\t',header = 0)
    RNA_data_adata = sc.read(RNA_file)
    sc.pp.filter_genes(RNA_data_adata, min_cells=3)
    sc.pp.filter_cells(RNA_data_adata, min_genes=3)
    sc.pp.log1p(RNA_data_adata)

    Spatial_data_adata = sc.read(Spatial_file)
    sc.pp.filter_genes(Spatial_data_adata, min_cells=3)
    sc.pp.filter_cells(Spatial_data_adata, min_genes=3)
    sc.pp.log1p(Spatial_data_adata)
    # locations = np.loadtxt(location_file, skiprows=1)

    train_gene = np.load(DataDir + 'train_list.npy', allow_pickle=True).tolist()
    predict_gene = np.load(DataDir+'val_list.npy', allow_pickle=True).tolist()

    for k in range(11):
        result, all_genes = SpaGE_impute(k, RNA_data_adata.to_df(), Spatial_data_adata.to_df(), train_gene, predict_gene)
        result.to_pickle(os.path.join(outdir, 'SpaGE_impute_predict_%d.pkl'%(k)))
        all_genes.to_pickle(os.path.join(outdir, 'SpaGE_impute_all_%d.pkl'%(k)))

        eval(DataDir, outdir, k)
