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

import time
import resource
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
import Baseline.sprite as sprite



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

def Tangram_impute(K, RNA_data_adata1, Spatial_adata1, train_gene, predict_gene,
    annotate = None, modes = 'clusters', density = 'rna_count_based'):

    print ('We run Tangram for data')
    test_list = predict_gene[K]
    test_list = [x.lower() for x in test_list]
    train_list = train_gene[K]
    train_list = list(set(train_list) & set(Spatial_adata1.var.index) & set(RNA_data_adata1.var.index))
    spatial_data_partial = Spatial_adata1[:, train_list]
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
    datasets_list = ['nano5-1', 'nano5-2', 'nano5-3', 'nano6', 'nano9-1', 'nano9-2', 'nano12', 'nano13']
    parser.add_argument('--dataset', type=str, default='nano5-1', help='dataset name')
    opt = parser.parse_args()
    for data_k in datasets_list:
        opt.dataset = data_k

        DataDir = '../dataset/%s/'%(opt.dataset)
        outdir = '../results/%s/tangram_sprite_nofilter'%(opt.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        RNA_file = DataDir + 'scRNA_count_cluster.h5ad' # 'scRNA_count.h5ad'
        Spatial_file = DataDir + 'Insitu_count.h5ad'
        location_file = DataDir + 'Locations.txt'

        # RNA_data = pd.read_table(RNA_file, header=0, index_col = 0)
        # Spatial_data = pd.read_table(Spatial_file, sep = '\t',header = 0)
        RNAseq_adata = sc.read(RNA_file)
        Spatial_adata = sc.read(Spatial_file)
        # locations = np.loadtxt(location_file, skiprows=1)

        train_gene = np.load(DataDir + 'train_list.npy', allow_pickle=True).tolist()
        predict_gene = np.load(DataDir+'test_list.npy', allow_pickle=True).tolist()
        device = 'GPU'
        # target_gene = "plp1"
        # target_expn = adata[:, target_gene].X.copy()
        # adata = adata[:, [gene for gene in gene_names if gene != target_gene]].copy()

        for k in range(10):
            
            # result, all_results = Tangram_impute(k, RNA_data_adata.copy(), Spatial_adata.copy(), train_gene, predict_gene)
            # result.to_pickle(os.path.join(outdir, 'tangram_impute_predict_%d.pkl'%(k)))
            # eval(DataDir, outdir, k)

            # make genes lowercase
            Spatial_adata.var_names = [x.lower() for x in Spatial_adata.var_names]
            RNAseq_adata.var_names = [x.lower() for x in RNAseq_adata.var_names]

            # preprocess RNAseq data
            sprite.preprocess_data(RNAseq_adata, standardize=False, normalize=True)

            # subset spatial data into shared genes
            gene_names = np.intersect1d(Spatial_adata.var_names, RNAseq_adata.var_names)
            Spatial_adata = Spatial_adata[:, gene_names].copy()

            # val_gene = np.load(os.path.join(opt.root, opt.dataset_name, 'val_list.npy'), allow_pickle=True).tolist()[opt.kfold]

            # # some of the gene is filtered because of the low expression
            # # print(self.seq_data.var.index, '\n\n', self.spa_data.var.index, '\n\n', train_gene)
            # train_gene = list(set(train_gene) & set(self.seq_data.var.index) & set(self.spa_data.var.index))
            # # test_gene = list(set(test_gene) & set(self.seq_data.var.index) & set(self.spa_data.var.index))
            # val_gene = list(set(val_gene) & set(self.seq_data.var.index) & set(self.spa_data.var.index))

            # hold out target gene
            # target_gene = "plp1"
            # target_expn = Spatial_adata[:, target_gene].X.copy()
            target_gene = predict_gene[k]  #11　　＃　191
            # print('predict_gene:', target_gene, len(target_gene))

            Spatial_adata = Spatial_adata[:, [gene for gene in gene_names if gene != target_gene]].copy()

            sprite.predict_gene_expression(Spatial_adata, RNAseq_adata, [target_gene], method="tangram", n_folds=10, n_pv=10)
            
            # 计时开始
            start_time = time.time()
            # 记录开始时的资源使用情况
            start_resources = resource.getrusage(resource.RUSAGE_SELF)

            sprite.reinforce_gene(Spatial_adata, predicted="tangram_predicted_expression",
                                alpha=0.1, tol=1e-8, cv=5)

            # build spatial neighborhood graph
            sprite.build_spatial_graph(Spatial_adata, method="fixed_radius", n_neighbors=50)

            # calculate cosine-based weights for edges
            sprite.calc_adjacency_weights(Spatial_adata, method="cosine")

            # Smooth
            sprite.smooth(Spatial_adata, predicted="reinforced_gene_joint_tangram_predicted_expression",
                        alpha=0.1, tol=1e-8)
            
            # 计时结束
            end_time = time.time()

            # 记录结束时的资源使用情况
            end_resources = resource.getrusage(resource.RUSAGE_SELF)

            # 计算总训练时间
            training_time = end_time - start_time
            print(f"Total training time: {training_time:.2f} seconds")

            # 计算资源消耗
            cpu_time = end_resources.ru_utime - start_resources.ru_utime
            max_memory_used = end_resources.ru_maxrss / 1024  # 转换为MB
            print(f"CPU time used: {cpu_time:.2f} seconds")
            print(f"Maximum memory used: {max_memory_used:.2f} MB")


            # print(Spatial_adata)

            Spatial_adata.obsm['reinforced_gene_joint_tangram_predicted_expression'].to_pickle(os.path.join(outdir, 'tangram_sprite_impute_predict_%d.pkl'%(k)))
            eval(DataDir, outdir, k)

            result.to_pickle(os.path.join(outdir, 'Sprite_tangram_impute_predict_%d.pkl'%(k)))
            eval(DataDir, outdir, k)

            result, all_results = Tangram_impute(k, RNA_data_adata.copy(), Spatial_adata.copy(), train_gene, predict_gene)
            result.to_pickle(os.path.join(outdir, 'tangram_impute_predict_%d.pkl'%(k)))
            eval(DataDir, outdir, k)

            # for sprite
            print(result.shape, result) # (92614, 191)
            #　处理数据，将其变为　ａｄａｔａ
            adata = pd.DataFrame(result)
            adata.index = Spatial_adata.obs_names
            
            sprite.reinforce_gene(adata, predicted="tangram_predicted_expression",
                                alpha=0.1, tol=1e-8, cv=5)

            # build spatial neighborhood graph
            sprite.build_spatial_graph(adata, method="fixed_radius", n_neighbors=50)

            # calculate cosine-based weights for edges
            sprite.calc_adjacency_weights(adata, method="cosine")

            # Smooth
            sprite.smooth(adata, predicted="reinforced_gene_joint_tangram_predicted_expression",
                        alpha=0.1, tol=1e-8)
            result.to_pickle(os.path.join(outdir, 'Sprite_tangram_impute_predict_%d.pkl'%(k)))
            eval(DataDir, outdir, k)


