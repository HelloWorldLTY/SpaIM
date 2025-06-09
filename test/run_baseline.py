import warnings
warnings.filterwarnings('ignore')
import pickle
import time as tm
from functools import partial
from scipy.stats import wasserstein_distance
import scipy.stats
import copy
from sklearn.model_selection import KFold
import multiprocessing
import matplotlib as mpl 
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats
import seaborn as sns
from scipy.spatial.distance import cdist
import h5py
from scipy.stats import spearmanr
import time
import tangram as tg
from IPython.display import display
import numpy as np
import sys
import os
import scipy.stats as st
import pandas as pd
from os.path import join
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, homogeneity_score, normalized_mutual_info_score
from stPlus import *
import argparse


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--sc_data", type=str, default="scRNA_count_cluster.h5ad")
parser.add_argument("--sp_data", type=str, default='Insitu_count.h5ad')
parser.add_argument("--document", type=str, default='Dataset32')
parser.add_argument("--rand", type=int, default=0)
args = parser.parse_args()
# ******** preprocess ********

n_splits = 10
adata_spatial = sc.read_h5ad('../dataset/' + args.document + args.sp_data)
adata_seq = sc.read_h5ad('../dataset/' + args.document + args.sc_data)

adata_seq2 = adata_seq.copy()
# tangram
adata_seq3 =  adata_seq2.copy()
sc.pp.normalize_total(adata_seq2, target_sum=1e4)
sc.pp.log1p(adata_seq2)
data_seq_array = adata_seq2.X

adata_spatial2 = adata_spatial.copy()
sc.pp.normalize_total(adata_spatial2, target_sum=1e4)
sc.pp.log1p(adata_spatial2)
data_spatial_array = adata_spatial2.X

sp_genes = np.array(adata_spatial.var_names)
sp_data = pd.DataFrame(data=data_spatial_array, columns=sp_genes)
sc_data = pd.DataFrame(data=data_seq_array, columns=sp_genes)

# ****baseline****

def Tangram_impute(annotate=None, modes='clusters', density='rna_count_based'):
    from torch.nn.functional import softmax, cosine_similarity, sigmoid
    import tangram as tg
    print('We run Tangram for this data\n')
    global adata_seq3, adata_spatial, locations
    from sklearn.model_selection import KFold

    raw_shared_gene = adata_spatial.var_names
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_spatial.X)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = list(raw_shared_gene[train_ind])
        test_gene = list(raw_shared_gene[test_ind])

        adata_seq_tmp = adata_seq2.copy()
        adata_spatial_tmp = adata_spatial2.copy()

        adata_spatial_partial = adata_spatial_tmp[:, train_gene]
        train_gene = np.array(train_gene)
        if annotate == None:
            RNA_data_adata_label = adata_seq3.copy()
            sc.pp.normalize_total(RNA_data_adata_label)
            sc.pp.log1p(RNA_data_adata_label)
            sc.pp.highly_variable_genes(RNA_data_adata_label)
            RNA_data_adata_label = RNA_data_adata_label[:, RNA_data_adata_label.var.highly_variable]
            sc.pp.scale(RNA_data_adata_label, max_value=10)
            sc.tl.pca(RNA_data_adata_label)
            sc.pp.neighbors(RNA_data_adata_label)
            sc.tl.leiden(RNA_data_adata_label, resolution=0.5)
            adata_seq_tmp.obs['leiden'] = RNA_data_adata_label.obs.leiden
        else:
            global CellTypeAnnotate
            adata_seq_tmp.obs['leiden'] = CellTypeAnnotate
        tg.pp_adatas(adata_seq_tmp, adata_spatial_partial, genes=train_gene) 

        device = torch.device('cuda:1')
        if modes == 'clusters':
            ad_map = tg.map_cells_to_space(adata_seq_tmp, adata_spatial_partial, device=device, mode=modes,
                                           cluster_label='leiden', density_prior=density)
            ad_ge = tg.project_genes(ad_map, adata_seq_tmp, cluster_label='leiden')
        else:
            ad_map = tg.map_cells_to_space(adata_seq_tmp, adata_spatial_partial, device=device)
            ad_ge = tg.project_genes(ad_map, adata_seq_tmp)
        test_list = list(set(ad_ge.var_names) & set(test_gene))
        test_list = np.array(test_list)
        all_pred_res[:, test_ind] = ad_ge.X[:, test_ind]

        idx += 1
        
    return all_pred_res


def gimVI_impute():
    print ('We run gimVI for this data\n')
    import scvi
    import scanpy as sc
    from scvi.external import GIMVI
    import torch
    from torch.nn.functional import softmax, cosine_similarity, sigmoid
    global adata_seq2, adata_spatial2

    from sklearn.model_selection import KFold
    raw_shared_gene = adata_spatial2.var_names
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)  # shuffle = false 不设置state，就是按顺序划分
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(data_spatial_array)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = np.array(raw_shared_gene[train_ind])
        test_gene = np.array(raw_shared_gene[test_ind])

        Genes = list(adata_spatial2.var_names)
        rand_gene_idx = test_ind
        n_genes = len(Genes)
        rand_train_gene_idx = train_ind
        rand_train_genes = np.array(Genes)[rand_train_gene_idx] # 不就是train_genes吗
        rand_genes = np.array(Genes)[rand_gene_idx] # test_gene
        adata_spatial_partial = adata_spatial2[:, rand_train_genes]
        sc.pp.filter_cells(adata_spatial_partial, min_counts=0)
        seq_data = copy.deepcopy(adata_seq2)
        seq_data = seq_data[:, Genes]
        sc.pp.filter_cells(seq_data, min_counts=0)
        scvi.data.setup_anndata(adata_spatial_partial)
        scvi.data.setup_anndata(seq_data)
        model = GIMVI(seq_data, adata_spatial_partial)
        model.train(200)
        _, imputation = model.get_imputed_values(normalized=False)
        all_pred_res[:, test_ind] = imputation[:, rand_gene_idx]
        idx += 1

    return all_pred_res


def SpaGE_impute():
    print ('We run SpaGE for this data\n')
    sys.path.append("baseline/SpaGE-master/")
    from SpaGE.main import SpaGE
    global sc_data, sp_data, adata_seq, adata_spatial

    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state= args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_spatial.X)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]
        pv = len(train_gene) / 2
        sp_data_partial = sp_data[train_gene]

        Imp_Genes = SpaGE(sp_data_partial, sc_data, n_pv=int(pv),
                          genes_to_predict=test_gene)

        all_pred_res[:, test_ind] = Imp_Genes
        idx += 1

    return all_pred_res


def stPlus_impute():
    global sc_data, sp_data, outdir, adata_spatial

    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_spatial.X)
    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = np.array(raw_shared_gene[train_ind])
        test_gene = np.array(raw_shared_gene[test_ind])

        save_path_prefix = join(outdir, 'process_file/stPlus-demo')
        if not os.path.exists(join(outdir, "process_file")):
            os.mkdir(join(outdir, "process_file"))
        stPlus_res = stPlus(sp_data[train_gene], sc_data, test_gene, save_path_prefix)
        all_pred_res[:, test_ind] = stPlus_res
        idx += 1

    return all_pred_res


def novoSpaRc_impute():
    print ('We run novoSpaRc for this data\n')
    import novosparc as nc
    global RNA_data, Spatial_data, locations, train_gene, predict_gene
    
    global sc_data, sp_data, adata_seq, adata_spatial

    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state= args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_spatial.X)
    for train_list, test_list in kf.split(raw_shared_gene):
        gene_names = np.array(RNA_data.index.values)
        dge = RNA_data.values
        dge = dge.T
        num_cells = dge.shape[0]
        print ('number of cells and genes in the matrix:', dge.shape)

        hvg = np.argsort(np.divide(np.var(dge, axis = 0),np.mean(dge, axis = 0) + 0.0001))
        dge_hvg = dge[:,hvg[-2000:]]

        num_locations = locations.shape[0]

        p_location, p_expression = nc.rc.create_space_distributions(num_locations, num_cells)
        cost_expression, cost_locations = nc.rc.setup_for_OT_reconstruction(dge_hvg,locations,num_neighbors_source = 5,num_neighbors_target = 5)

        insitu_matrix = np.array(Spatial_data[train_list])
        insitu_genes = np.array(Spatial_data[train_list].columns)
        test_genes = np.array(test_list)
        test_matrix = np.array(Spatial_data[test_list])

        markers_in_sc = np.array([], dtype='int')
        for marker in insitu_genes:
            marker_index = np.where(gene_names == marker)[0]
            if len(marker_index) > 0:
                markers_in_sc = np.append(markers_in_sc, marker_index[0])
        cost_marker_genes = cdist(dge[:, markers_in_sc]/np.amax(dge[:, markers_in_sc]),insitu_matrix/np.amax(insitu_matrix))
        alpha_linear = 0.5
        gw = nc.rc._GWadjusted.gromov_wasserstein_adjusted_norm(cost_marker_genes, cost_expression, cost_locations,alpha_linear, p_expression, p_location,'square_loss', epsilon=5e-3, verbose=True)
        sdge = np.dot(dge.T, gw)
        imputed = pd.DataFrame(sdge,index = RNA_data.index)
        result = imputed.loc[test_genes]
        result = result.T        
        all_pred_res[:, train_list] = result
        idx += 1
    return all_pred_res


def SpaOTsc_impute():
    from spaotsc import SpaOTsc
    print ('We run SpaOTsc for this data\n')
    global sc_data, sp_data, outdir, adata_spatial

    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_spatial.X)
    for train_ind, test_ind in kf.split(raw_shared_gene):
        df_sc = RNA_data.T
        df_IS = Spatial_data
        pts = locations
        is_dmat = distance_matrix(pts, pts)
        df_is = df_IS.loc[:,train_ind]

        gene_is = df_is.columns.tolist()
        gene_sc = df_sc.columns.tolist()
        gene_overloap = list(set(gene_is).intersection(gene_sc))
        a = df_is[gene_overloap]
        b = df_sc[gene_overloap]

        rho, pval = stats.spearmanr(a, b,axis=1)
        rho[np.isnan(rho)]=0
        mcc=rho[-(len(df_sc)):,0:len(df_is)]
        C = np.exp(1 - mcc)
        issc = SpaOTsc.spatial_sc(sc_data = df_sc, is_data = df_is, is_dmat = is_dmat)
        issc.transport_plan(C**2, alpha = 0, rho = 1.0, epsilon = 0.1, cor_matrix = mcc, scaling = False)
        gamma = issc.gamma_mapping
        for j in range(gamma.shape[1]):
            gamma[:,j] = gamma[:,j]/np.sum(gamma[:,j])
        X_pred = np.matmul(gamma.T, np.array(issc.sc_data.values))
        result = pd.DataFrame(data = X_pred, columns = issc.sc_data.columns.values)
        all_pred_res[:, test_ind] = result
        idx += 1

    return all_pred_res


def TISSUE_impute():
    print ('We run TISSUE for this data\n')
    import tissue.main, tissue.downstream
    from utils import CalculateMeteics
    global sc_data, sp_data, adata_seq, adata_spatial
    
    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state= args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_spatial.X)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]
        pv = len(train_gene) / 2
        sp_data_partial = sp_data[train_gene]

        # make genes lowercase
        sp_data.var_names = [x.lower() for x in sp_data.var_names]
        sc_data.var_names = [x.lower() for x in sc_data.var_names]

        # preprocess RNAseq data
        tissue.main.preprocess_data(sc_data, standardize=False, normalize=True)

        # subset spatial data into shared genes
        gene_names = np.intersect1d(sp_data.var_names, sc_data.var_names)
        sp_data = sp_data[:, gene_names].copy()

        # hold out target gene
        target_gene = "plp1"
        # target_expn = sp_data[:, target_gene].X.copy()
        # target_gene = predict_gene[k]  #11　　＃　191
        # print('predict_gene:', target_gene, len(target_gene))

        sp_data = sp_data[:, [gene for gene in gene_names if gene != target_gene]].copy()

        tissue.predict_gene_expression(sp_data, sc_data, [target_gene], method="tangram", n_folds=10, n_pv=10)
        
        Imp_Genes = sp_data.obsm['reinforced_gene_joint_tangram_predicted_expression']

        all_pred_res[:, test_ind] = Imp_Genes
        idx += 1

    return all_pred_res


def SPRITE_impute():
    print ('We run SPRITE for this data\n')
    global sc_data, sp_data, adata_seq, adata_spatial
    import Baseline.sprite as sprite
    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state= args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_spatial.X)

    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = raw_shared_gene[train_ind]
        test_gene = raw_shared_gene[test_ind]
        pv = len(train_gene) / 2
        sp_data_partial = sp_data[train_gene]

        # make genes lowercase
        sp_data.var_names = [x.lower() for x in sp_data.var_names]
        sc_data.var_names = [x.lower() for x in sc_data.var_names]

        # preprocess RNAseq data
        sprite.preprocess_data(sc_data, standardize=False, normalize=True)

        # subset spatial data into shared genes
        gene_names = np.intersect1d(sp_data.var_names, sc_data.var_names)
        sp_data = sp_data[:, gene_names].copy()

        # hold out target gene
        target_gene = "plp1"
        # target_expn = sp_data[:, target_gene].X.copy()
        # target_gene = predict_gene[k]  #11　　＃　191
        # print('predict_gene:', target_gene, len(target_gene))

        sp_data = sp_data[:, [gene for gene in gene_names if gene != target_gene]].copy()

        sprite.predict_gene_expression(sp_data, sc_data, [target_gene], method="tangram", n_folds=10, n_pv=10)
        
        sprite.reinforce_gene(sp_data, predicted="tangram_predicted_expression",
                                    alpha=0.1, tol=1e-8, cv=5)

        # build spatial neighborhood graph
        sprite.build_spatial_graph(sp_data, method="fixed_radius", n_neighbors=50)

        # calculate cosine-based weights for edges
        sprite.calc_adjacency_weights(sp_data, method="cosine")

        # Smooth
        sprite.smooth(sp_data, predicted="reinforced_gene_joint_tangram_predicted_expression",
                    alpha=0.1, tol=1e-8)
        Imp_Genes = sp_data.obsm['reinforced_gene_joint_tangram_predicted_expression']

        all_pred_res[:, test_ind] = Imp_Genes
        idx += 1

    return all_pred_res


def StDiff_impute():
    print ('We run StDiff for this data\n')
    sys.path.append("baseline/StDiff-master/")
    from model.stDiff_model import DiT_stDiff
    from model.stDiff_scheduler import NoiseScheduler
    from model.stDiff_train import normal_train_stDiff
    from model.sample import sample_stDiff
    from process.result_analysis import clustering_metrics
    from process.data import *
    import resource
    
    global sc_data, sp_data, adata_seq, adata_spatial

    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state= args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    all_pred_res = np.zeros_like(adata_spatial.X)
    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))


        doc =args.document # 'Dataset11_std+scale_new'
        save_path = 'stDiff-ckpt/' + doc + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path_prefix = save_path + 'stDiff%d.pt' % (idx)
        
        
        # mask
        cell_num = data_spatial_array.shape[0]
        gene_num = data_spatial_array.shape[1]
        mask = np.ones((gene_num,), dtype='float32')
        gene_ids_test = test_ind
        mask[gene_ids_test] = 0

        seq = data_seq_array
        st = data_spatial_array
        data_seq_masked = seq * mask
        data_spatial_masked = st * mask

        seq = seq * 2 - 1
        data_seq_masked = data_seq_masked * 2 - 1

        data_ary = data_spatial_array
        st = st * 2 - 1
        data_spatial_masked = data_spatial_masked * 2 - 1

        dataloader = get_data_loader(
            seq,
            data_seq_masked,
            batch_size=512,
            is_shuffle=True)

        seed = 1202
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = DiT_stDiff(
            input_size=gene_num,
            hidden_size=512,
            depth=8,
            num_heads=4,
            classes=6,
            mlp_ratio=4.0,
            dit_type='dit'
        )

        device = torch.device('cuda:0')
        model.to(device)

        diffusion_step = diffusion_step

        model.train()

        # 计时开始
        start_time = time.time()
        # 记录开始时的资源使用情况
        start_resources = resource.getrusage(resource.RUSAGE_SELF)

        # if not os.path.isfile(save_path_prefix):

        normal_train_stDiff(model,
                                dataloader=dataloader,
                                lr=0.001,
                                num_epoch=800,
                                diffusion_step=diffusion_step,
                                device=device,
                                pred_type='noise',
                                mask=mask)

        torch.save(model.state_dict(), save_path_prefix)
        # else:
        #     print("No pretrained model")
            # model.load_state_dict(torch.load(save_path_prefix))
        
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
        # print(f"CPU time used: {cpu_time:.2f} seconds")
        print(f"Maximum memory used: {max_memory_used:.2f} MB")


        gt = data_spatial_masked
        noise_scheduler = NoiseScheduler(
            num_timesteps=diffusion_step,
            beta_schedule='cosine'
        )

        dataloader = get_data_loader(
            data_spatial_masked,
            data_spatial_masked,
            batch_size=40,
            is_shuffle=False)

        # sample
        model.eval()
        imputation = sample_stDiff(model,
                                    device=device,
                                    dataloader=dataloader,
                                    noise_scheduler=noise_scheduler,
                                    mask=mask,
                                    gt=gt,
                                    num_step=diffusion_step,
                                    sample_shape=(cell_num, gene_num),
                                    is_condi=True,
                                    sample_intermediate=diffusion_step,
                                    model_pred_type='noise',
                                    is_classifier_guidance=False,
                                    omega=0.2)

        all_pred_res[:, gene_ids_test] = imputation[:, gene_ids_test]
        idx += 1

    impu = (all_pred_res + 1) / 2
    cpy_x = adata_spatial.copy()
    cpy_x.X = impu
    return impu


def spscope_impute():
    from SpatialScope.utils import ConcatCells
    
    raw_shared_gene = np.array(adata_spatial.var_names)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.rand)
    kf.get_n_splits(raw_shared_gene)
    torch.manual_seed(10)
    idx = 1
    
    all_pred_res = np.zeros((len(a), adata_spatial.X.shape[1]))
    
    total_size = adata_spatial.X.shape[0]
    chunk_size = 1000
    arr = [(start, min(start + chunk_size, total_size)) for start in range(0, total_size, chunk_size)]

    
    for train_ind, test_ind in kf.split(raw_shared_gene):
        print("\n===== Fold %d =====\nNumber of train genes: %d, Number of test genes: %d" % (
            idx, len(train_ind), len(test_ind)))
        train_gene = np.array(raw_shared_gene[train_ind])
        test_gene = np.array(raw_shared_gene[test_ind])
        test_gene_str = ','.join(test_gene)

        
        for i in arr:
            print(f"{i[0]},{i[1]}")
            decomposition = ['python', './baseline/SpatialScope/Decomposition.py', '--tissue', dataset, '--out_dir', './spscope_output/' + dataset,
                        '--SC_Data', './ckpt/sc/' + dataset_sp + '_spscope.h5ad', '--cell_class_column', 'subclass_label',  '--ckpt_path', 'ckpt/' + dataset + '/model_05000.pt',
                        '--spot_range', f"{i[0]},{i[1]}", '--replicates', '5', '--gpu', '1', '--leave_out_test', '--test_genes', f'{test_gene_str}']
            subprocess.run(decomposition)
        
        spot_range = np.concatenate((np.arange(0,total_size,chunk_size), np.array([total_size])))
        ad_res = ConcatCells(spot_range,file_path='./spscope_output/' + dataset + '/' + dataset + '/',prefix='generated_cells_spot',suffix='.h5ad')

        all_pred_res[:, test_ind] = ad_res.X[:, test_ind]
        idx += 1
        

    return all_pred_res


Data = args.document
outdir = 'Result/' + Data + '/baseline/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

Tangram_result = Tangram_impute() 
Tangram_result_pd = pd.DataFrame(Tangram_result, columns=sp_genes)
Tangram_result_pd.to_csv(outdir +  '/Tangram_impute.csv',header = 1, index = 1)


SpaGE_result = SpaGE_impute() 
SpaGE_result_pd = pd.DataFrame(SpaGE_result, columns=sp_genes)
SpaGE_result_pd.to_csv(outdir +  '/SpaGE_impute.csv',header = 1, index = 1)


gimVI_result = gimVI_impute() 
gimVI_result_pd = pd.DataFrame(gimVI_result, columns=sp_genes)
gimVI_result_pd.to_csv(outdir +  '/gimVI_impute.csv',header = 1, index = 1)


stPlus_result = stPlus_impute() 
stPlus_result_pd = pd.DataFrame(stPlus_result, columns=sp_genes)
stPlus_result_pd.to_csv(outdir +  '/stPlus_impute.csv',header = 1, index = 1)


novoSpaRc_result = novoSpaRc_impute()
novoSpaRc_result_pd = pd.DataFrame(novoSpaRc_result, columns=sp_genes)
novoSpaRc_result_pd.to_csv(outdir +  '/novoSpaRc_impute.csv',header = 1, index = 1)


SpaOTsc_result = SpaOTsc_impute() 
SpaOTsc_result_pd = pd.DataFrame(SpaOTsc_result, columns=sp_genes)
SpaOTsc_result_pd.to_csv(outdir +  '/SpaOTsc_impute.csv',header = 1, index = 1)


TISSUE_result = TISSUE_impute()
TISSUE_result_pd = pd.DataFrame(TISSUE_result, columns=sp_genes)
TISSUE_result_pd.to_csv(outdir +  '/TISSUE_impute.csv',header = 1, index = 1)


SPRITE_result = SPRITE_impute() 
SPRITE_result_pd = pd.DataFrame(SPRITE_result, columns=sp_genes)
SPRITE_result_pd.to_csv(outdir +  '/SPRITE_impute.csv',header = 1, index = 1)


StDiff_result = StDiff_impute() 
StDiff_result_pd = pd.DataFrame(StDiff_result, columns=sp_genes)
StDiff_result_pd.to_csv(outdir +  '/StDiff_impute.csv',header = 1, index = 1)
    

spscope_result = spscope_impute() 
spscope_result_pd = pd.DataFrame(spscope_result, columns=sp_genes)
spscope_result_pd.to_csv(outdir +  '/spscope_impute.csv',header = 1, index = 1)


#******** metrics ********

def cal_ssim(im1, im2, M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim = l12 * c12 * s12

    return ssim


def scale_max(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.max()
        result = pd.concat([result, content], axis=1)
    return result


def scale_z_score(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = st.zscore(content)
        content = pd.DataFrame(content, columns=[label])
        result = pd.concat([result, content], axis=1)
    return result


def scale_plus(df):
    result = pd.DataFrame()
    for label, content in df.items():
        content = content / content.sum()
        result = pd.concat([result, content], axis=1)
    return result


def logNorm(df):
    df = np.log1p(df)
    df = st.zscore(df)
    return df


def clustering_metrics(adata, target, pred, mode="AMI"):
    if(mode=="AMI"):
        ami = adjusted_mutual_info_score(adata.obs[target], adata.obs[pred])
        print("AMI ",ami)
        return ami
    elif(mode=="ARI"):
        ari = adjusted_rand_score(adata.obs[target], adata.obs[pred])
        print("ARI ",ari)
        return ari
    elif(mode=="Homo"):
        homo = homogeneity_score(adata.obs[target], adata.obs[pred])
        print("Homo ",homo)
        return homo
    elif(mode=="NMI"):
        nmi = normalized_mutual_info_score(adata.obs[target], adata.obs[pred])
        print("NMI ", nmi)
        return nmi


class CalculateMeteics:
    def __init__(self, raw_data, genes_name,impute_count_file, prefix, metric):
        self.impute_count_file = impute_count_file
        self.raw_count = pd.DataFrame(raw_data, columns=genes_name)
        self.raw_count.columns = [x.upper() for x in self.raw_count.columns]
        self.raw_count = self.raw_count.T
        self.raw_count = self.raw_count.loc[~self.raw_count.index.duplicated(keep='first')].T
        self.raw_count = self.raw_count.fillna(1e-20)

        self.impute_count = pd.read_csv(impute_count_file, header=0, index_col=0)
        self.impute_count.columns = [x.upper() for x in self.impute_count.columns]
        self.impute_count = self.impute_count.T
        self.impute_count = self.impute_count.loc[~self.impute_count.index.duplicated(keep='first')].T
        self.impute_count = self.impute_count.fillna(1e-20)
        self.prefix = prefix
        self.metric = metric

    def SSIM(self, raw, impute, scale='scale_max'):
        if scale == 'scale_max':
            raw = scale_max(raw)
            impute = scale_max(impute)
        else:
            print('Please note you do not scale data by scale max')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    ssim = 0
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    M = [raw_col.max(), impute_col.max()][raw_col.max() > impute_col.max()]
                    raw_col_2 = np.array(raw_col)
                    raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0], 1)
                    impute_col_2 = np.array(impute_col)
                    impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0], 1)
                    ssim = cal_ssim(raw_col_2, impute_col_2, M)

                ssim_df = pd.DataFrame(ssim, index=["SSIM"], columns=[label])
                result = pd.concat([result, ssim_df], axis=1)
        else:
            print("columns error")
        return result

    def SPCC(self, raw, impute, scale=None):
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    spearmanr = 0
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    spearmanr, _ = st.spearmanr(raw_col, impute_col)
                spearman_df = pd.DataFrame(spearmanr, index=["SPCC"], columns=[label])
                result = pd.concat([result, spearman_df], axis=1)
        else:
            print("columns error")
        return result

    def JS(self, raw, impute, scale='scale_plus'):
        if scale == 'scale_plus':
            raw = scale_plus(raw)
            impute = scale_plus(impute)
        else:
            print('Please note you do not scale data by plus')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    JS = 1
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    raw_col = raw_col.fillna(1e-20)
                    impute_col = impute_col.fillna(1e-20)
                    M = (raw_col + impute_col) / 2
                    JS = 0.5 * st.entropy(raw_col, M) + 0.5 * st.entropy(impute_col, M)
                JS_df = pd.DataFrame(JS, index=["JS"], columns=[label])
                result = pd.concat([result, JS_df], axis=1)
        else:
            print("columns error")
        return result

    def RMSE(self, raw, impute, scale='zscore'):
        if scale == 'zscore':
            raw = scale_z_score(raw)
            impute = scale_z_score(impute)
        else:
            print('Please note you do not scale data by zscore')
        if raw.shape[0] == impute.shape[0]:
            result = pd.DataFrame()
            for label in raw.columns:
                if label not in impute.columns:
                    RMSE = 1.5
                else:
                    raw_col = raw.loc[:, label]
                    impute_col = impute.loc[:, label]
                    impute_col = impute_col.fillna(1e-20)
                    raw_col = raw_col.fillna(1e-20)
                    RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())

                RMSE_df = pd.DataFrame(RMSE, index=["RMSE"], columns=[label])
                result = pd.concat([result, RMSE_df], axis=1)
        else:
            print("columns error")
        return result


    def cluster(self, raw, impu,scale=None):

        ad_sp = adata_spatial2.copy()
        ad_sp.X = raw

        cpy_x = adata_spatial2.copy()
        cpy_x.X = impu

        sc.tl.pca(ad_sp)
        sc.pp.neighbors(ad_sp, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(ad_sp)
        tmp_adata1 = ad_sp

        sc.tl.pca(cpy_x)
        sc.pp.neighbors(cpy_x, n_pcs=30, n_neighbors=30)
        sc.tl.leiden(cpy_x)
        tmp_adata2 = cpy_x

        tmp_adata2.obs['class'] = tmp_adata1.obs['leiden']
        ari = clustering_metrics(tmp_adata2, 'class', 'leiden', "ARI")
        ami = clustering_metrics(tmp_adata2, 'class', 'leiden', "AMI")
        homo = clustering_metrics(tmp_adata2, 'class', 'leiden', "Homo")
        nmi = clustering_metrics(tmp_adata2, 'class', 'leiden', "NMI")
                
        result = pd.DataFrame([[ari, ami, homo, nmi]], columns=["ARI", "AMI", "Homo", "NMI"])
        return result

    def compute_all(self):
        raw = self.raw_count
        impute = self.impute_count
        prefix = self.prefix
        SSIM_gene = self.SSIM(raw, impute)
        Spearman_gene = self.SPCC(raw, impute)
        JS_gene = self.JS(raw, impute)
        RMSE_gene = self.RMSE(raw, impute)

        cluster_result = self.cluster(raw, impute)

        result_gene = pd.concat([Spearman_gene, SSIM_gene, RMSE_gene, JS_gene], axis=0)
        result_gene.T.to_csv(prefix + "_gene_Metrics.txt", sep='\t', header=1, index=1)

        cluster_result.to_csv(prefix + "_cluster_Metrics.txt", sep='\t', header=1, index=1)

        return result_gene


import seaborn as sns
import os
PATH = 'Result/'
DirFiles = os.listdir(PATH)


def CalDataMetric(Data):
    print ('We are calculating the : ' + Data + '\n')
    metrics = ['SPCC(gene)','SSIM(gene)','RMSE(gene)','JS(gene)']
    metric = ['SPCC','SSIM','RMSE','JS']
    impute_count_dir = PATH + Data
    impute_count = os.listdir(impute_count_dir)
    impute_count = [x for x in impute_count if x [-3:] == 'csv']
    methods = []
    if len(impute_count)!=0:
        medians = pd.DataFrame()
        for impute_count_file in impute_count:
            print(impute_count_file)
            if 'result_Tangram.csv' in impute_count_file:
                os.system('mv ' + impute_count_dir + '/result_Tangram.csv ' + impute_count_dir + '/Tangram_impute.csv')
            prefix = impute_count_file.split('_')[0]
            methods.append(prefix)
            prefix = impute_count_dir + '/' + prefix
            impute_count_file = impute_count_dir + '/' + impute_count_file
            # if not os.path.isfile(prefix + '_Metrics.txt'):
            print (impute_count_file)
            CM = CalculateMeteics(data_spatial_array, sp_genes, impute_count_file = impute_count_file, prefix = prefix, metric = metric)
            CM.compute_all()

            median = []
            for j in ['_gene']:
                tmp = pd.read_csv(prefix + j + '_Metrics.txt', sep='\t', index_col=0)
                for m in metric:
                    median.append(np.median(tmp[m]))
            median = pd.DataFrame([median], columns=metrics)
            clu = pd.read_csv(prefix + '_cluster' + '_Metrics.txt', sep='\t', index_col=0)
            median = pd.concat([median, clu], axis=1)
            medians = pd.concat([medians, median], axis=0)

        metrics += ["ARI", "AMI", "Homo", "NMI"]
        medians.columns = metrics
        medians.index = methods
        medians.to_csv(outdir +  '/final_result.csv',header = 1, index = 1)

CalDataMetric(Data)





