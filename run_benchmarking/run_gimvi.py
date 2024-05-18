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
import argparse
warnings.filterwarnings('ignore')

from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats
import seaborn as sns

from scipy.spatial.distance import cdist
import h5py
from scipy.stats import spearmanr

import time
import sys
import tangram as tg
from os.path import join
from IPython.display import display

import scvi
import scanpy as sc
from scvi.external import GIMVI
import torch
from torch.nn.functional import softmax, cosine_similarity, sigmoid

from utils import CalculateMeteics
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def eval(dataroot, saveroot, k=0):
    #raw_count_file, impute_count_file, prefix, metric
    raw_count_file = sc.read(os.path.join(dataroot, 'Insitu_count.h5ad'))
    raw_count_file = pd.DataFrame(raw_count_file.X, index=raw_count_file.obs.index, columns=raw_count_file.var.index)
    raw_count_file.columns = [x.lower() for x in raw_count_file.columns]

    impute_count_file = np.load(os.path.join(saveroot, 'gimvi_impute_predict_%d.pkl'%(k)), allow_pickle=True)
    impute_count_file.columns = [x.lower() for x in impute_count_file.columns]
    genes = impute_count_file.columns
    raw_count_file = raw_count_file[genes]

    prefix = saveroot
    metric = 'all'
    name = 'gimvi'

    evaluate = CalculateMeteics(raw_count_file, impute_count_file, prefix, metric, name)
    acc = evaluate.compute_all(K=k)
    print(acc.T['PCC'].mean(), acc.T['RMSE'].mean())

def gimVI_impute(K, RNA_data_adata, Spatial_data_adata, train_gene, predict_gene):
    print ('We run gimVI for this data\n')
    test_list = np.array(predict_gene[K])
    train_list = np.array(train_gene[K])
    # make sure the genes are in both data
    Genes  = list(set(Spatial_data_adata.var_names) & set(RNA_data_adata.var.index))
    rand_gene_idx = [Genes.index(x) for x in test_list]
    n_genes = len(Genes)
    rand_train_gene_idx = sorted(set(range(n_genes)) - set(rand_gene_idx))
    rand_train_genes = np.array(Genes)[rand_train_gene_idx]
    rand_genes = np.array(Genes)[rand_gene_idx]
    spatial_data_partial = Spatial_data_adata[:, rand_train_genes]
    sc.pp.filter_cells(spatial_data_partial, min_counts= 0)
    seq_data = copy.deepcopy(RNA_data_adata)
    seq_data = seq_data[:, Genes]
    sc.pp.filter_cells(seq_data, min_counts = 0)
    # scvi.data.setup_anndata(spatial_data_partial)
    # scvi.data.setup_anndata(seq_data)

    spatial_data_partial.obs['labels'] = np.zeros(spatial_data_partial.shape[0])
    seq_data.obs['labels'] = np.zeros(seq_data.shape[0])
    spatial_data_partial.obs['batch'] = [0] * spatial_data_partial.shape[0]

    GIMVI.setup_anndata(seq_data, labels_key='labels')
    GIMVI.setup_anndata(spatial_data_partial, labels_key='labels', batch_key='batch')

    model = GIMVI(seq_data, spatial_data_partial)
    model.train(50, batch_size=1280)
    _, imputation = model.get_imputed_values(normalized = False)

    imputed = imputation[:, rand_gene_idx]
    result = pd.DataFrame(imputed, columns = rand_genes)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nano5-1', help='dataset name')
    opt = parser.parse_args()

    DataDir = '../dataset/%s/'%(opt.dataset)
    outdir = '../results/%s/gimvi'%(opt.dataset)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    RNA_file = DataDir + 'scRNA_count.h5ad'
    Spatial_file = DataDir + 'Insitu_count.h5ad'
    # location_file = DataDir + 'Locations.txt'

    # RNA_data = pd.read_table(RNA_file, header=0, index_col = 0)
    # Spatial_data = pd.read_table(Spatial_file, sep = '\t',header = 0)
    RNA_data_adata = sc.read(RNA_file)
    Spatial_data_adata = sc.read(Spatial_file)
    # locations = np.loadtxt(location_file, skiprows=1)

    train_gene = np.load(DataDir + 'train_list.npy', allow_pickle=True).tolist()
    predict_gene = np.load(DataDir+'val_list.npy', allow_pickle=True).tolist()
    device = 'GPU'

    results = []
    for k in range(0,1):
        result = gimVI_impute(k, RNA_data_adata, Spatial_data_adata, train_gene, predict_gene)
        result.to_pickle(os.path.join(outdir, 'gimvi_impute_predict_%d.pkl'%(k)))

        eval(DataDir, outdir, k)