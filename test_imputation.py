import warnings
warnings.filterwarnings("ignore")
import os
import torch
import sys
from options import Options
from dataset import Test_Imputation
from tqdm import tqdm
import numpy as np
import scanpy as sc
import pandas as pd

from utils import *
from SpaIM import ImputeModule

# 预测 SC 中存在但 ST 中缺失的 Genes
opt = Options().parse()
opt.kfold = 0 # 选择使用第几个fold的模型
dataset = 'Dataset1'


# 选择对应数据集
st_path = f'./dataset/{dataset}/Insitu_count.h5ad'
sc_path = f'./dataset/{dataset}/scRNA_count_cluster.h5ad'

ST_adata = sc.read(st_path)
SC_adata = sc.read(sc_path)

print(ST_adata, '\n', SC_adata)

valdataset = Test_Imputation(opt, istrain='val')

gene_names, cell_names = valdataset.get_eval_names()

valdataloader = torch.utils.data.DataLoader(
    valdataset, 
    batch_size=opt.batch_size, 
    shuffle=False, 
    num_workers=0
)
opt.sc_dim = valdataset.get_cluster_dim()

model = ImputeModule(opt)
if opt.parallel:
    model = torch.nn.DataParallel(model).cuda().module
else:
    model = model.to(torch.device('cuda:%d'%(opt.gpu)))
    
model_path = f'./SpaIM_results/{dataset}/last_%d.pth'%(opt.kfold)
print('model_path:',model_path)

model.load(model_path)

with torch.no_grad():
    eval_result = None
    input_result = None
    for i, (seq, st_style, _, _) in enumerate(valdataloader):
        inputs = {
            'scx': seq,
            'st_style': st_style
        }
    # print('SCX:', inputs['scx'].shape, len(st_style))
    model.set_input(inputs, istrain=0)
    out = model.inference()
    impute_result = out['st_fake'].detach().cpu().numpy()
    print("impute_result",impute_result.shape) #  (3482, 67798) torch.Size([3482, 67798])
    eval_result = impute_result if eval_result is None else np.concatenate((eval_result, impute_result), axis=0)
    print("eval_result", eval_result.shape)  # (3482, 77890) (3482, 77890)


eval_result = eval_result.T
# print(eval_result[0][:10])
eval_result[eval_result <0] = 0

# print(gene_names.shape)  # 检查 gene_names 的形状
# print(cell_names.shape)  # 检查 cell_names 的形状
df1 = pd.DataFrame(eval_result, index=cell_names, columns=gene_names)
df1.to_pickle(os.path.join(opt.save_path, 'impute_sc_result_%d.pkl'%(opt.kfold)))


# # 读取数据，进行对比
# adata = sc.read("./Insitu_count.h5ad")
# SpaIM_adata1 = pd.read_pickle('./SpaIM/impute_sc_result_0.pkl')
# Tangram_adata2 = pd.read_pickle('./Tangram/impute_sc_result_0.pkl')
# StDiff_adata2 = pd.read_pickle("./StDiff/impute_sc_result_0.pkl")


# # 提取表达矩阵
# raw = adata.to_df()
# spaim = SpaIM_adata1
# stdiff = StDiff_adata2
# tangram = Tangram_adata2

# # 修改行名
# raw.index = ['cell' + str(i) for i in range(1, len(raw) + 1)]
# spaim.index = ['cell' + str(i) for i in range(1, len(spaim) + 1)]
# stdiff.index = ['cell' + str(i) for i in range(1, len(stdiff) + 1)]
# tangram.index = ['cell' + str(i) for i in range(1, len(tangram) + 1)]

# # 修改列名
# raw.columns = raw.columns.str.upper()
# spaim.columns = spaim.columns.str.upper()
# stdiff.columns = stdiff.columns.str.upper()
# tangram.columns = tangram.columns.str.upper()

# # 计算相关系数
# genes = ['SOX4', 'TYK2', 'GPX1', 'EZH2']
# for gene in genes:
#     print(f"PCC between raw and spaim for gene {gene}:", np.corrcoef(raw[gene], spaim[gene])[0, 1])
#     print(f"PCC between raw and stdiff for gene {gene}:", np.corrcoef(raw[gene], stdiff[gene])[0, 1])
#     print(f"PCC between raw and tangram for gene {gene}:", np.corrcoef(raw[gene], tangram[gene])[0, 1])
#     # print('\n')

# # 读取数据
# adata = sc.read("dataset/nano9-1/Insitu_count.h5ad")
# SpaIM_adata1 = df1

# # 提取表达矩阵
# raw = adata.to_df()
# spaim = SpaIM_adata1
# # stdiff = StDiff_adata2.to_df()
# # tangram = Tangram_adata2.to_df()

# # 修改行名
# raw.index = ['cell' + str(i) for i in range(1, len(raw) + 1)]
# spaim.index = ['cell' + str(i) for i in range(1, len(spaim) + 1)]
# # stdiff.index = ['cell' + str(i) for i in range(1, len(stdiff) + 1)]
# # tangram.index = ['cell' + str(i) for i in range(1, len(tangram) + 1)]

# # 修改列名
# raw.columns = raw.columns.str.upper()
# spaim.columns = spaim.columns.str.upper()
# # stdiff.columns = stdiff.columns.str.upper()
# # tangram.columns = tangram.columns.str.upper()

# # 计算相关系数
# genes = ['SOX4', 'TYK2', 'GPX1', 'EZH2']
# for gene in genes:
#     print(f"PCC between raw and spaim for gene {gene}:", np.corrcoef(raw[gene], spaim[gene])[0, 1])
#     # print(f"PCC between raw and stdiff for gene {gene}:", np.corrcoef(raw[gene], stdiff[gene])[0, 1])
#     # print(f"PCC between raw and tangram for gene {gene}:", np.corrcoef(raw[gene], tangram[gene])[0, 1])
#     print('\n')






