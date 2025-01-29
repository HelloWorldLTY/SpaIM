import os
import pandas as pd
import numpy as np
import torch
import scanpy as sc
from torch.utils.data import Dataset

class ImputationDataset(Dataset):
    def __init__(self, opt, istrain='train'):
        self.opt = opt
        self.seq_data = self.load_data(opt.seq_data)  # loading 测序数据
        self.spa_data = self.load_data(opt.spa_data)

        # self.seq_cluster = self.seq_data.obs['leiden'].cat.codes.values
        if 'leiden' in opt.cluster:
            self.seq_cluster = self.seq_data.obs[opt.cluster].cat.codes.values
        elif opt.cluster == 'annotate':
            # print('use annotation')
            # print(seq_data.obs.keys())  # Index(['leiden', 'n_genes'], dtype='object')
            # self.seq_cluster = seq_data.obs['leiden'].cat.codes.values   # Benchmark
            self.seq_cluster = self.seq_data.obs['merge_cell_type'].cat.codes.values  # 对应的是nano的设置，需要改 seq_data 的数据源
        else:
            self.seq_cluster = [1]
        self.style_dim = opt.style_dim
        self.istrain = istrain

        self.seq_data = self.aggreate_cell_types() # 测序数据进行 mean 聚合处理

        train_gene = np.load(os.path.join(opt.root, opt.dataset_name, 'train_list.npy'), allow_pickle=True).tolist()[opt.kfold]
        test_gene = np.load(os.path.join(opt.root, opt.dataset_name, 'test_list.npy'), allow_pickle=True).tolist()[opt.kfold]
        train_gene = set(self.seq_data.var_names) & set(self.spa_data.var_names) & set(train_gene)
        test_gene = set(test_gene) & set(self.seq_data.var_names)   # 模型任务是通过对齐的少量数据学习其他数据的空间基因表达

        val_gene = set(self.spa_data.var_names) - set(train_gene)
        val_gene = val_gene & set(self.seq_data.var_names)

        train_gene = list(train_gene)
        val_gene = list(val_gene)
        test_gene = list(test_gene)
        
        # if opt.annotation:
        self.seq_train = self.seq_data[:, train_gene].copy().T
        self.seq_val = self.seq_data[:, val_gene].copy().T

        self.spa_train = self.spa_data[:, train_gene].copy().T
        self.spa_val = self.spa_data[:, val_gene].copy().T
    
    def get_cluster_dim(self):
        return len(set(self.seq_cluster))

    def run_leiden(self):
        adata_label = self.seq_data.copy()
        sc.pp.highly_variable_genes(adata_label)
        adata_label = adata_label[:, adata_label.var.highly_variable]
        sc.pp.scale(adata_label, max_value=10)
        # sc.pp.scale(adata_label)
        sc.tl.pca(adata_label)
        sc.pp.neighbors(adata_label)
        sc.tl.leiden(adata_label, resolution=0.5)
        # sc.tl.leiden(adata_label)
        self.seq_data.obs['leiden'] = adata_label.obs['leiden']

    def aggreate_cell_types(self):
        x = self.seq_data.X
        num_cls = len(set(self.seq_cluster))
        new_x = np.zeros((num_cls, x.shape[1]))
        for i in range(num_cls): # 11
            new_x[i] = np.mean(x[self.seq_cluster == i], axis=0) # 计算每个基因在当前细胞类型中的均值，存储到new_x数组中。
        # print(new_x.shape)  # (11, 17040) 求11种基因的均值
        df = pd.DataFrame(new_x, columns=self.seq_data.var.index)
        new_adata = sc.AnnData(df)
        return new_adata

    def cal_density(self):
        rna_count_per_cell = self.spa_train.T.X.sum(axis=1)
        rna_density = rna_count_per_cell / np.sum(rna_count_per_cell)
        return rna_density

    def gen_leiden(self, adata):
        adata_label = adata.copy()
        sc.pp.normalize_total(adata_label)
        # sc.pp.log1p(adata_label)
        sc.pp.highly_variable_genes(adata_label)
        adata_label = adata_label[:, adata_label.var.highly_variable]
        sc.pp.scale(adata_label, max_value=10)
        sc.tl.pca(adata_label)
        sc.pp.neighbors(adata_label)
        sc.tl.leiden(adata_label, resolution=0.5, random_state=self.opt.seed)
        return adata_label.obs['leiden'].astype('category').cat.codes.values

    def get_debug_genes(self):
        if self.istrain == 'debug_train':
            return self.seq_train.obs_names
        elif self.istrain == 'debug_val':
            return self.seq_val.obs_names

    def get_eval_names(self):
        return self.spa_val.obs_names, self.spa_val.var_names

    def load_data(self, root): # 数值过滤和取对数归一化操作
        adata = sc.read(root) # adata 是 Scanpy 中的 AnnData 对象，用于存储和处理单细胞数据。
        sc.pp.filter_genes(adata, min_cells=3) 
        # 使用 scanpy 中的 filter_genes 函数，过滤掉在数据集中表达的细胞数量低于3的基因。
        # 这有助于去除在很少的细胞中表达的基因，以减少噪音和提高数据的质量。
        sc.pp.filter_cells(adata, min_genes=3) # 基因数量低于3
        sc.pp.log1p(adata)  # 对数据进行logarithm加一操作，即对数据中的每个元素 x，计算 log(1 + x)。

        # some of the data has negative values
        # adata.X[adata.X <0] = 0
        # sc.pp.normalize_total(adata)
        # if not "log1p" in adata.uns_keys():
        # sc.pp.log1p(adata)

        return adata
    
    def __len__(self):
        if self.istrain == 'train':
            # print(self.seq_train.shape, self.spa_train.shape) # (305, 11) (305, 8425)
            return self.seq_train.shape[0]
        elif self.istrain == 'val':
            return self.seq_val.shape[0]
        elif self.istrain == 'debug_train':
            return self.seq_train.shape[0]
        elif self.istrain == 'debug_val':
            return self.seq_val.shape[0]
        else:
            return self.seq_test.shape[0]
    
    def __getitem__(self, index):
        st_style = torch.ones(self.style_dim)
        sc_style = torch.zeros(self.style_dim) 
        seq_cluster = self.seq_cluster 

        if self.istrain == 'train':
            seq_x = self.seq_train.X[index, ...]
            spa_x = self.spa_train.X[index, ...]
            return torch.FloatTensor(seq_x), torch.FloatTensor(spa_x), sc_style, st_style, seq_cluster
        elif self.istrain == 'val':
            seq_x = self.seq_val.X[index, ...]
            spa_x = self.spa_val.X[index, ...]
            return torch.FloatTensor(seq_x), st_style, torch.FloatTensor(spa_x), seq_cluster
        elif self.istrain == 'debug_train':
            seq_x = self.seq_train.X[index, ...]
            spa_x = self.spa_train.X[index, ...]
            return torch.FloatTensor(seq_x), torch.FloatTensor(spa_x), sc_style, st_style
        elif self.istrain == 'debug_val':
            seq_x = self.seq_val.X[index, ...]
            spa_x = self.spa_val.X[index, ...]
            return torch.FloatTensor(seq_x), torch.FloatTensor(spa_x), sc_style, st_style
        else: # c
            seq_x = self.seq_test.X[index, ...]
            return torch.FloatTensor(seq_x), st_style
