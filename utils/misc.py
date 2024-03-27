'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import logging
import time
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter', 'ova_loss', 'compute_roc',
           'roc_id_ood', 'ova_ent', 'softmax_ent', 'exclude_dataset', 'exclude_dataset_ratio',
           'test_ood', 'test', 'select_unknown', 'compute_fmse', 'select_ood']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_roc(unk_all, label_all, num_known):
    Y_test = np.zeros(unk_all.shape[0])
    unk_pos = np.where(label_all >= num_known)[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unk_all)


def roc_id_ood(score_id, score_ood):
    id_all = np.r_[score_id, score_ood]
    Y_test = np.zeros(score_id.shape[0]+score_ood.shape[0])
    Y_test[score_id.shape[0]:] = 1
    return roc_auc_score(Y_test, id_all)


def ova_loss(logits_open, label):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    label_s_sp = torch.zeros((logits_open.size(0),
                              logits_open.size(2))).long().to(label.device)
    label_range = torch.range(0, logits_open.size(0) - 1).long()
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :]
                                                + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :]
                                                    + 1e-8) * label_sp_neg, 1)[0])
    Lo = open_loss_neg + open_loss
    return Lo

def compute_fmse(evi_alp_, labels_, epoch):
    target_concentration = 100
    fisher_c = 0.01
    b2 = int(labels_.size(0))
    num = int(evi_alp_.size(1))
    labels_1hot_ = torch.zeros(int(evi_alp_.size(0)), int(evi_alp_.size(1))).cuda().scatter_(1, labels_.view(-1, 1), 1.0)
    labels_1hot_ += 1/num
    evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)
    mask = torch.ones_like(evi_alp0_).squeeze()
    mask[b2:] = 1.0

    gamma1_alp = torch.polygamma(1, evi_alp_)
    gamma1_alp0 = torch.polygamma(1, evi_alp0_)

    gap = labels_1hot_ - evi_alp_ / evi_alp0_

    loss_mse_ = ((gap.pow(2) * gamma1_alp).sum(-1) * mask).mean()

    loss_var_ = (evi_alp_[:b2] * (evi_alp0_[:b2] - evi_alp_[:b2]) * gamma1_alp[:b2] / (evi_alp0_[:b2] * evi_alp0_[:b2] * (evi_alp0_[:b2] + 1))).sum(-1).mean() / 3.
    #loss_var_2 = -(evi_alp_[b2:] * (evi_alp0_[b2:] - evi_alp_[b2:]) * gamma1_alp[b2:] / (evi_alp0_[b2:] * evi_alp0_[b2:] * (evi_alp0_[b2:] + 1)))
    #loss_var_ = torch.cat([loss_var_1, loss_var_2],dim=0).sum(-1).mean()
    
    loss_det_fisher_ = - ((torch.log(gamma1_alp).sum(-1) + torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1))) * mask).mean()

    #evi_alp_ = (evi_alp_ - target_concentration) * (1 - labels_1hot_) + target_concentration
    loss_kl_ = compute_kl_loss(evi_alp_, labels_, target_concentration, mask)
    regr = np.minimum(1.0, (epoch+1)/10.)

    loss = 40 * loss_mse_ + 50* loss_var_ + fisher_c * loss_det_fisher_ + regr * 0.05 * loss_kl_

    return loss, loss_mse_.item(), loss_var_.item(), loss_det_fisher_.item(), loss_kl_.item()


def compute_kl_loss(alphas, labels, target_concentration, mask, concentration=1.0, epsilon=1e-8):
    # TODO: Need to make sure this actually works right...
    # todo: so that concentration is either fixed, or on a per-example setup
    # Create array of target (desired) concentration parameters

    if target_concentration < 1.0:
        concentration = target_concentration

    target_alphas = torch.ones_like(alphas) * concentration
    target_alphas += torch.zeros_like(alphas).scatter_(-1, labels.unsqueeze(-1), target_concentration - 1)

    alp0 = torch.sum(alphas, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

    alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
    alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
    assert torch.all(torch.isfinite(alp0_term)).item()

    alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                            + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                            torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
    alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
    assert torch.all(torch.isfinite(alphas_term)).item()

    loss = (torch.squeeze(alp0_term + alphas_term) * mask).mean()

    return loss
        
def ova_ent(logits_open):
    logits_open = logits_open.view(logits_open.size(0), 2, -1)
    logits_open = F.softmax(logits_open, 1)
    Le = torch.mean(torch.mean(torch.sum(-logits_open *
                                   torch.log(logits_open + 1e-8), 1), 1))
    return Le

def softmax_ent(logits):
    logits = F.softmax(logits, 1)
    Le = torch.mean(torch.sum(-logits *
                                   torch.log(logits + 1e-8), 1))
    return Le


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def exclude_dataset(args, dataset, model, exclude_known=False):
    data_time = AverageMeter()
    end = time.time()
    dataset.init_index()
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    model.eval()
    
    with torch.no_grad():
        for batch_idx, ((_, _, inputs), targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            outputs, outputs_open = model(inputs)
            outputs = F.softmax(outputs, 1)
            #out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.range(0, outputs_open.size(0) - 1).long().cuda()
            pred_open = outputs_open.data.max(1)[1]
            pred_close = outputs.data.max(1)[1]
            known_score = outputs_open[tmp_range, pred_close]
            outputs_open_sec = outputs_open.topk(2)[0]
            outputs_open_sec = outputs_open_sec[tmp_range, 1]
            known_ind = (known_score > 3.1) #* (outputs_open_sec < 3.5)
            if batch_idx == 0:
                pred_all = pred_close
                tar_all = targets
                known_all = known_ind
            else:
                pred_all = torch.cat([pred_all, pred_close], 0)
                tar_all = torch.cat([tar_all, targets], 0)
                known_all = torch.cat([known_all, known_ind], 0)
        if not args.no_progress:
            test_loader.close()
    known_all = known_all.data.cpu().numpy()
    pred_all = pred_all.data.cpu().numpy()
    tar_all = tar_all.data.cpu().numpy()
    #pred_close = pred_close.data.cpu().numpy()
    if exclude_known:
        ind_selected = np.where(known_all == 0)[0]
    else:
        ind_selected = np.where(known_all != 0)[0]
    correct = np.sum(pred_all[ind_selected] == tar_all[ind_selected])
    print("selected acc%s"%( (correct / len(ind_selected))))
    print("selected ratio %s"%( (len(ind_selected)/ len(known_all))))
    model.train()
    dataset.set_index(ind_selected)
    return len(ind_selected)/ len(known_all)

def exclude_dataset_ratio(args, dataset, model,  epoch=30, init_ratio=0.24,exclude_known=False):
    data_time = AverageMeter()
    end = time.time()
    dataset.init_index()
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    model.eval()
    with torch.no_grad():
        for batch_idx, ((_, _, inputs), targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            outputs, outputs_open = model(inputs)
            outputs = F.softmax(outputs, 1)
            #out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.range(0, outputs_open.size(0) - 1).long().cuda()
            pred_open = outputs_open.data.max(1)[1]
            pred_close = outputs.data.max(1)[1]
            outputs_open_sec = outputs_open.topk(2)[0]
            outputs_open_sec = outputs_open_sec[tmp_range, 1]
            #known_score = outputs_open[tmp_range, pred_close]
            #red_score = outputs_open[tmp_range, pred_close]
            #known_score = outputs_open.topk(3)[0].sum(dim=1)
            known_score = outputs_open[tmp_range, pred_close] #known_score * (pred_score > 4.1) #* (outputs_open_sec < 3.5).float()
            if batch_idx == 0:
                pred_all = pred_close
                tar_all = targets
                known_score_all = known_score
            else:
                pred_all = torch.cat([pred_all, pred_close], 0)
                tar_all = torch.cat([tar_all, targets], 0)
                known_score_all = torch.cat([known_score_all, known_score], 0)
        if not args.no_progress:
            test_loader.close()
    sorted_, indices = torch.sort(known_score_all, descending=True)
    num_all = sorted_.shape[0]
    max_ratio = 0.54
    #ratio = (epoch-30) / 110 * (max_ratio-init_ratio) + init_ratio
    #ratio = np.minimum(ratio, max_ratio)
    thres = sorted_[int(max_ratio * num_all)].item()
    known_score_all = known_score_all.data.cpu().numpy()
    pred_all = pred_all.data.cpu().numpy()
    tar_all = tar_all.data.cpu().numpy()
    if exclude_known:
        ind_selected = np.where(known_score_all <= thres)[0]
    else:
        ind_selected = np.where(known_score_all > thres)[0]
    correct = np.sum(pred_all[ind_selected] == tar_all[ind_selected])
    print("selected acc%s"%( (correct / len(ind_selected))))
    print("selected ratio %s"%( (len(ind_selected)/ len(known_score_all))))
    model.train()
    dataset.set_index(ind_selected)

def select_unknown(args, dataset, model):
    data_time = AverageMeter()
    end = time.time()
    dataset.init_index()
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    model.eval()
    with torch.no_grad():
        for batch_idx, ((_, _, inputs), targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            outputs, outputs_open = model(inputs)
            #outputs = F.softmax(outputs, 1)
            out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.range(0, out_open.size(0) - 1).long().cuda()
            #pred_close = outputs.data.max(1)[1]
            unk_score = out_open[tmp_range, 0].min(1)[0]
            unknown_ind = unk_score > 0.5
            if batch_idx == 0:
                unknown_all = unknown_ind
            else:
                unknown_all = torch.cat([unknown_all, unknown_ind], 0)
        if not args.no_progress:
            test_loader.close()
    unknown_all = unknown_all.data.cpu().numpy()
    ind_selected = np.where(unknown_all != 0)[0]
    print("selected unknown ratio %s"%( (len(ind_selected)/ len(unknown_all))))
    model.train()
    dataset.set_index(ind_selected)

def select_ood(args, dataset, model, epoch):
    data_time = AverageMeter()
    end = time.time()
    dataset.init_index()
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False)
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    model.eval()
    with torch.no_grad():
        for batch_idx, ((_, _, inputs), targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.to(args.device)
            outputs, outputs_open = model(inputs)
            #outputs = F.softmax(outputs, 1)
            #out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            #tmp_range = torch.range(0, outputs_open.size(0) - 1).long().cuda()
            #pred_close = outputs.data.max(1)[1]
            known_score = outputs_open.max(1)[0]
            #unknown_ind = known_score < 3
            if batch_idx == 0:
                known_score_all = known_score
            else:
                known_score_all = torch.cat([known_score_all, known_score], 0)
        if not args.no_progress:
            test_loader.close()
    sorted_, indices = torch.sort(known_score_all)
    num_all = sorted_.shape[0]
    ratio = 0.1 + np.minimum((epoch-10) / 100. * 0.1, 0.1)
    thres = sorted_[int(ratio * num_all)].item()
    known_score_all = known_score_all.data.cpu().numpy()
    ind_selected = np.where(known_score_all < thres)[0]
    print("selected ood ratio %s"%( (len(ind_selected)/ len(known_score_all))))
    model.train()
    dataset.set_index(ind_selected)

def test(args, test_loader, model, epoch, val=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    acc = AverageMeter()
    unk = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs, outputs_open = model(inputs)
            outputs = F.softmax(outputs, 1)
            #out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.range(0, outputs_open.size(0) - 1).long().cuda()
            pred_close = outputs.data.max(1)[1]
            #unk_score = -outputs_open[tmp_range, pred_close]# / outputs_open.sum(dim=1)
            unk_score = -outputs_open.topk(40)[0].sum(dim=1)
            #unk_score = outputs_open.sum(dim=1)
            known_score = outputs.max(1)[0] # output of softmax
            targets_unk = targets >= int(outputs.size(1))
            targets[targets_unk] = int(outputs.size(1)) # convert ood label to num_class
            known_targets = targets < int(outputs.size(1))#[0]
            known_pred = outputs[known_targets]
            known_targets = targets[known_targets]

            if len(known_pred) > 0:
                prec1, prec5 = accuracy(known_pred, known_targets, topk=(1, 5))
                top1.update(prec1.item(), known_pred.shape[0])
                top5.update(prec5.item(), known_pred.shape[0])

            ind_unk = unk_score > -5
            pred_close[ind_unk] = int(outputs.size(1))

            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
                known_all = known_score
                label_all = targets
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
                known_all = torch.cat([known_all, known_score], 0)
                label_all = torch.cat([label_all, targets], 0)

            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. "
                                            "Data: {data:.3f}s."
                                            "Batch: {bt:.3f}s. "
                                            "Loss: {loss:.4f}. "
                                            "Closed t1: {top1:.3f} "
                                            "t5: {top5:.3f} ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()
    ## ROC calculation
    #import pdb
    #pdb.set_trace()
    unk_all = unk_all.data.cpu().numpy()
    known_all = known_all.data.cpu().numpy()
    label_all = label_all.data.cpu().numpy()
    if not val:
        roc = compute_roc(unk_all, label_all,
                          num_known=int(outputs.size(1)))
        roc_soft = compute_roc(-known_all, label_all,
                               num_known=int(outputs.size(1)))
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        logger.info("ROC: {:.3f}".format(roc))
        logger.info("ROC Softmax: {:.3f}".format(roc_soft))
        return losses.avg, top1.avg, roc, roc_soft
    else:
        logger.info("Closed acc: {:.3f}".format(top1.avg))
        return top1.avg


def test_ood(args, test_id, test_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()
            inputs = inputs.to(args.device)
            outputs, outputs_open = model(inputs)
            #out_open = F.softmax(outputs_open.view(outputs_open.size(0), 2, -1), 1)
            tmp_range = torch.range(0, outputs_open.size(0) - 1).long().cuda()
            pred_close = outputs.data.max(1)[1]
            #unk_score = -outputs_open[tmp_range, pred_close]
            unk_score = -outputs_open.topk(3)[0].sum(dim=1)
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx == 0:
                unk_all = unk_score
            else:
                unk_all = torch.cat([unk_all, unk_score], 0)
        if not args.no_progress:
            test_loader.close()
    ## ROC calculation
    unk_all = unk_all.data.cpu().numpy()
    roc = roc_id_ood(test_id, unk_all)

    return roc
