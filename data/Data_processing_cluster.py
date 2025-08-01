import scanpy as sc
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def adata_to_cluster_expression(adata, scale=True):
    # unique_labels = np.unique(cluster_label)
    value_counts = adata.obs['leiden'].value_counts(normalize=True)
    unique_labels = value_counts.index
    new_obs = pd.DataFrame({'leiden': unique_labels})
    print(unique_labels)
    adata_ret = sc.AnnData(obs=new_obs, var=adata.var, uns=adata.uns)
    X_new = np.empty((len(unique_labels), adata.shape[1]))
    for index, l, in enumerate(unique_labels):
        X_new[index] = adata[adata.obs['leiden'] == l].X.sum(axis=0)
    adata_ret.X = X_new
    return adata_ret

root = 'dataset/'
dataset_name = 'Dataset1'

scadata = sc.read(os.path.join(root, dataset_name, 'scRNA_count_cluster.h5ad'))
adata_label = scadata.copy()
sc.pp.normalize_total(adata_label)
sc.pp.log1p(adata_label)
sc.pp.highly_variable_genes(adata_label)
adata_label = adata_label[:, adata_label.var.highly_variable]
sc.pp.scale(adata_label, max_value=10)
sc.tl.pca(adata_label)
sc.pp.neighbors(adata_label)
for res in ['0.05', '0.10', '0.30', '0.50', '0.70', '0.90']:
    print(res)
    # generate annotation
    sc.tl.leiden(adata_label, resolution=float(res), random_state=1234)
    print(len(set(adata_label.obs['leiden'])))
    scadata.obs['leiden_%s'%(res)] = adata_label.obs['leiden']

scadata.write(os.path.join(root, dataset_name, 'scRNA_count_cluster.h5ad'))
