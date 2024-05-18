import warnings
warnings.filterwarnings("ignore")
import os
import torch
from options import Options
from dataset import ImputationDataset
from dataset import ImputationNano
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils import seed_everything, CalculateMeteics

# define model
from model.SpaIM import ImputeModule


def val(opt):
    valdataset = ImputationDataset(opt, istrain='val')
    # valdataset = ImputationNano(opt, istrain='val')

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
        model = model.cuda()
    # model.load(os.path.join(opt.save_path, 'best_pcc_%d.pth'%(opt.kfold)))
    model.load(os.path.join(opt.save_path, 'last_%d.pth'%(opt.kfold)))

    with torch.no_grad():
        eval_result = None
        input_result = None
        for i, (seq, st_style, spa, seq_cls) in enumerate(valdataloader):
            inputs = {
                'scx': seq,
                'st_style': st_style,
                'sc_cls': seq_cls
            }
        model.set_input(inputs, istrain=0)
        out = model.inference()
        impute_result = out['st_fake'].detach().cpu().numpy()
        eval_result = impute_result if eval_result is None else np.concatenate((eval_result, impute_result), axis=0)
        input_result = spa.detach().cpu().numpy() if input_result is None else np.concatenate((input_result, spa.detach().cpu().numpy()), axis=0)

    eval_result = eval_result.T
    eval_result[eval_result <0] = 0
    input_result = input_result.T

    df1 = pd.DataFrame(eval_result, index=cell_names, columns=gene_names)
    df1.to_pickle(os.path.join(opt.save_path, 'impute_result_%d.pkl'%(opt.kfold)))

    df2 = pd.DataFrame(input_result, index=cell_names, columns=gene_names)
    df2.to_pickle(os.path.join(opt.save_path, 'input_result_%d.pkl'%(opt.kfold)))


def train(opt):

    # run benchmark
    dataset = ImputationDataset(opt, istrain='train')
    valdataset = ImputationDataset(opt, istrain='val')

    # run nanostring
    # dataset = ImputationNano(opt, istrain='train')
    # valdataset = ImputationNano(opt, istrain='val')

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=0
    )

    valdataloader = torch.utils.data.DataLoader(
        valdataset, 
        batch_size=opt.batch_size, 
        shuffle=False, 
        num_workers=0
    )

    best_pcc = 0
    model_epoch = 0
    opt.sc_dim = dataset.get_cluster_dim()

    model = ImputeModule(opt)
    if opt.parallel:
        model = torch.nn.DataParallel(model).cuda().module
    else:
        model = model.cuda()    

    # for epoch in range(opt.epochs):
    tqdm_bar = tqdm(range(opt.epochs))
    for epoch in tqdm_bar:
        for i, (seq, spa, sc_style, st_style, seq_cls) in enumerate(dataloader):
            inputs = {
                'scx': seq,
                'stx': spa,
                'sc_style': sc_style,
                'st_style': st_style,
                'sc_cls': seq_cls
            }

            model.set_input(inputs)
            model.update_parameters()
            loss_stat = model.get_current_loss()
            msg = 'Epoch: {}/{}, Iter: {}/{}, '.format(epoch, opt.epochs, i, len(dataloader))
            for k,v in loss_stat.items():
                msg += ' %s: %.4f' % (k, v)
    
    model.save(os.path.join(opt.save_path, 'last_%d.pth'%(opt.kfold)))


if __name__ == '__main__':
    opt = Options().parse()
    seed_everything(opt.seed)
    if opt.val_only == 0:
        train(opt)
        val(opt)
    else:
        val(opt)
    torch.cuda.empty_cache()