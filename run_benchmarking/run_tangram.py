import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
from anndata import AnnData
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import skimage
import seaborn as sns
import tangram as tg
import os
import random
import os
import numpy as np
import torch

import pandas as pd
from scipy import stats
import scipy.stats as st
import seaborn as sns
import matplotlib as mpl 
import matplotlib.pyplot as plt
import os
import torch
from torch.nn.functional import softmax, cosine_similarity, sigmoid
import tangram as tg
import argparse
from utils import CalculateMeteics

def eval(dataroot, saveroot, k=0):
    #raw_count_file, impute_count_file, prefix, metric
    raw_count_file = sc.read(os.path.join(dataroot, 'Insitu_count.h5ad')).to_df()
    # print(raw_count_file.var_names)
    # raw_count_file = pd.DataFrame(raw_count_file.X, index=raw_count_file.obs.index, columns=raw_count_file.var.index)

    raw_count_file.columns = [x.lower() for x in raw_count_file.columns]

    impute_count_file = np.load(os.path.join(saveroot, 'tangram_impute_predict_%d.pkl'%(k)), allow_pickle=True)
    
    genes = impute_count_file.columns
    raw_count_file = raw_count_file[genes]

    prefix = saveroot
    metric = 'all'
    name = 'tangram'

    evaluate = CalculateMeteics(raw_count_file, impute_count_file, prefix, metric, name)
    acc = evaluate.compute_all(K=k)
    print(acc.T['PCC'].mean())

def Tangram_impute(K, RNA_data_adata1, Spatial_data_adata1, train_gene, predict_gene,
    annotate = None, modes = 'clusters', density = 'rna_count_based'):

    print ('We run Tangram for data')
    test_list = predict_gene[K]
    test_list = [x.lower() for x in test_list]
    train_list = train_gene[K]
    train_list = list(set(train_list) & set(Spatial_data_adata1.var.index) & set(RNA_data_adata1.var.index))
    spatial_data_partial = Spatial_data_adata1[:, train_list]
    train_list = np.array(train_list)
    # if annotate == None:
    #     RNA_data_adata_label = RNA_data_adata1
    #     sc.pp.normalize_total(RNA_data_adata_label)
    #     sc.pp.log1p(RNA_data_adata_label)
    #     sc.pp.highly_variable_genes(RNA_data_adata_label)
    #     RNA_data_adata_label = RNA_data_adata_label[:, RNA_data_adata_label.var.highly_variable]
    #     sc.pp.scale(RNA_data_adata_label, max_value=10)
    #     sc.tl.pca(RNA_data_adata_label)
    #     sc.pp.neighbors(RNA_data_adata_label)
    #     sc.tl.leiden(RNA_data_adata_label, resolution = 0.5)
    #     RNA_data_adata1.obs['leiden']  = RNA_data_adata_label.obs.leiden
    # else:
    #     global CellTypeAnnotate
    #     RNA_data_adata1.obs['merge_cell_type']  = CellTypeAnnotate
    tg.pp_adatas(RNA_data_adata1, spatial_data_partial, genes=train_list)
    device = torch.device('cuda:0')
    if modes == 'clusters':
        ad_map = tg.map_cells_to_space(RNA_data_adata1, spatial_data_partial, device = device, mode = modes, cluster_label = 'merge_cell_type', density_prior = density)
        ad_ge = tg.project_genes(ad_map, RNA_data_adata1, cluster_label = 'merge_cell_type')
    else:
        print(RNA_data_adata1.shape)
        RNA_data_adata1 = tg.adata_to_cluster_expression(RNA_data_adata1, cluster_label = 'merge_cell_type')
        print(RNA_data_adata1.shape)
        ad_map = tg.map_cells_to_space(RNA_data_adata1.copy(), spatial_data_partial.copy(), device = device, mode=modes, target_count=spatial_data_partial.shape[1], lambda_f_reg=1, lambda_count=1)
        ad_ge = tg.project_genes(ad_map, RNA_data_adata1)
    test_list = list(set(ad_ge.var_names) & set(test_list))
    test_list = np.array(test_list)
    pre_gene = pd.DataFrame(ad_ge[:,test_list].X, index=ad_ge[:,test_list].obs_names, columns=ad_ge[:,test_list].var_names)
    all_gene = pd.DataFrame(ad_ge.X, index=ad_ge.obs_names, columns=ad_ge.var_names)
    return pre_gene, all_gene

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nano5-1', help='dataset name')
    opt = parser.parse_args()

    DataDir = '../dataset/%s/'%(opt.dataset)
    outdir = '../results/%s/tangram_nofilter'%(opt.dataset)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    RNA_file = DataDir + 'scRNA_count.h5ad'
    Spatial_file = DataDir + 'Insitu_count.h5ad'
    location_file = DataDir + 'Locations.txt'

    # RNA_data = pd.read_table(RNA_file, header=0, index_col = 0)
    # Spatial_data = pd.read_table(Spatial_file, sep = '\t',header = 0)
    RNA_data_adata = sc.read(RNA_file)
    Spatial_data_adata = sc.read(Spatial_file)
    # locations = np.loadtxt(location_file, skiprows=1)

    train_gene = np.load(DataDir + 'train_list.npy', allow_pickle=True).tolist()
    predict_gene = np.load(DataDir+'val_list.npy', allow_pickle=True).tolist()
    device = 'GPU'

    for k in range(11):
        result, all_results = Tangram_impute(k, RNA_data_adata.copy(), Spatial_data_adata.copy(), train_gene, predict_gene)
        result.to_pickle(os.path.join(outdir, 'tangram_impute_predict_%d.pkl'%(k)))
        eval(DataDir, outdir, k)