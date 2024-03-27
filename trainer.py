import logging
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from dataset import TransformOpenMatch, cifar10_mean, cifar10_std, \
    cifar100_std, cifar100_mean, normal_mean, \
    normal_std, TransformFixMatch_Imagenet_Weak
from tqdm import tqdm
from utils import AverageMeter, ova_loss,\
    save_checkpoint, ova_ent, compute_fmse,\
    test, test_ood, exclude_dataset, exclude_dataset_ratio
import wandb
wandb.init(project="cifar100_80_50_upload")

logger = logging.getLogger(__name__)
best_acc = 0
best_acc_val = 0

def causal_inference(current_logit, qhat, exp_idx, tau=0.5):
    # de-bias pseudo-labels
    debiased_prob = F.softmax(current_logit - tau*torch.log(qhat), dim=1)
    return debiased_prob

def initial_qhat(class_num=1000):
    # initialize qhat of predictions (probability)
    qhat = (torch.ones([1, class_num], dtype=torch.float)/class_num).cuda()
    print("qhat size: ".format(qhat.size()))
    return qhat

def update_qhat(probs, qhat, momentum, qhat_mask=None):
    if qhat_mask is not None:
        mean_prob = probs.detach()*qhat_mask.detach().unsqueeze(dim=-1)
    else:
        mean_prob = probs.detach().mean(dim=0)
    qhat = momentum * qhat + (1 - momentum) * mean_prob
    return qhat

def train(args, labeled_trainloader, unlabeled_dataset, test_loader, val_loader,
          ood_loaders, model, optimizer, ema_model, scheduler):
    if args.amp:
        from apex import amp

    global best_acc
    global best_acc_val

    test_accs = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_o = AverageMeter()
    #losses_oem = AverageMeter()
    losses_socr = AverageMeter()
    losses_fix = AverageMeter()
    mask_probs = AverageMeter()
    end = time.time()


    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
    labeled_iter = iter(labeled_trainloader)
    default_out = "Epoch: {epoch}/{epochs:4}. " \
                  "LR: {lr:.6f}. " \
                  "Lab: {loss_x:.4f}. " \
                  "Open: {loss_o:.4f}"
    output_args = vars(args)
    #default_out += " OEM  {loss_oem:.4f}"
    default_out += " SOCR  {loss_socr:.4f}"
    default_out += " Fix  {loss_fix:.4f}"

    model.train()
    unlabeled_dataset_all = copy.deepcopy(unlabeled_dataset)
    if args.dataset == 'cifar10':
        mean = cifar10_mean
        std = cifar10_std
        func_trans = TransformOpenMatch
    elif args.dataset == 'cifar100':
        mean = cifar100_mean
        std = cifar100_std
        func_trans = TransformOpenMatch
    elif 'imagenet' in args.dataset:
        mean = normal_mean
        std = normal_std
        func_trans = TransformFixMatch_Imagenet_Weak


    unlabeled_dataset_all.transform = func_trans(mean=mean, std=std)
    labeled_dataset = copy.deepcopy(labeled_trainloader.dataset)
    labeled_dataset.transform = func_trans(mean=mean, std=std)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    qhat = initial_qhat(class_num=args.num_classes)

    for epoch in range(args.start_epoch, args.epochs):
        if epoch >= 200:
            exit()
        output_args["epoch"] = epoch
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        
        if epoch >= args.start_fix:
            ## pick pseudo-inliers
            exclude_dataset_ratio(args, unlabeled_dataset, ema_model.ema)

        unlabeled_trainloader = DataLoader(unlabeled_dataset,
                                           sampler = train_sampler(unlabeled_dataset),
                                           batch_size = args.batch_size * args.mu,
                                           num_workers = args.num_workers,
                                           drop_last = True)
        unlabeled_trainloader_all = DataLoader(unlabeled_dataset_all,
                                           sampler=train_sampler(unlabeled_dataset_all),
                                           batch_size=args.batch_size * args.mu,
                                           num_workers=args.num_workers,
                                           drop_last=True)

        unlabeled_iter = iter(unlabeled_trainloader)
        unlabeled_all_iter = iter(unlabeled_trainloader_all)

        for batch_idx in range(args.eval_step):
            ## Data loading

            try:
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.next()
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                labeled_iter = iter(labeled_trainloader)
                (_, inputs_x_s, inputs_x), targets_x = labeled_iter.next()
            try:
                (inputs_u_w, inputs_u_s, _), _ = unlabeled_iter.next()
            except:
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s, _), _ = unlabeled_iter.next()
            try:
                (inputs_all_w, inputs_all_s, _), _ = unlabeled_all_iter.next()
            except:
                unlabeled_all_iter = iter(unlabeled_trainloader_all)
                (inputs_all_w, inputs_all_s, _), _ = unlabeled_all_iter.next()
            data_time.update(time.time() - end)

            b_size = inputs_x.shape[0]

            inputs_all = torch.cat([inputs_all_w, inputs_all_s], 0)
            inputs = torch.cat([inputs_x, inputs_x_s,
                                inputs_all], 0).to(args.device)
            targets_x = targets_x.to(args.device)
            ## Feed data
            logits, alphas_open, feat = model(inputs, feature=True)
            alphas_open_u1, alphas_open_u2 = alphas_open[2*b_size:].chunk(2)
            logits_open_u1, logits_open_u2 = logits[2*b_size:].chunk(2)
            alpha_value1 = torch.max(alphas_open[:2*b_size], dim=-1)[0].mean().item()
            alpha_value2 = torch.max(alphas_open[2*b_size:], dim=-1)[0].mean().item()
            alpha_value_sec = alphas_open[:2*b_size].topk(2)[0]
            tmp_range = torch.range(0, alpha_value_sec.size(0) - 1).long().cuda()
            alpha_value_sec = alpha_value_sec[tmp_range, 1].mean().item()

            ## Loss for labeled samples
            Lx = F.cross_entropy(logits[:2*b_size],
                                      targets_x.repeat(2), reduction='mean')
            
            targets_x = targets_x.repeat(2)
            flat_alpha = model.fc_open_forward(feat[2*b_size:].detach())
            alphas_fmse = torch.cat([alphas_open[:2*b_size], flat_alpha], 0)
            L_odir, loss_mse_, loss_var_, loss_fisher_, loss_kl_= compute_fmse(alphas_fmse, targets_x, epoch)
            wandb.log({'Train/L_odir': round(L_odir.item(), 3),'Train/loss_mse_': round(loss_mse_, 3),
                               'Train/loss_var_': round(loss_var_, 3), 'Train/loss_kl_': round(loss_kl_, 3),
                               'Train/loss_fisher_': round(loss_fisher_, 3), 'Train/alpha_value1': alpha_value1, 'Train/alpha_value2': alpha_value2, 'Train/alpha_value_sec': alpha_value_sec})
            

            if epoch >= args.start_fix:
                inputs_ws = torch.cat([inputs_u_w, inputs_u_s], 0).to(args.device)
                #inputs_u_w = inputs_u_w.to(args.device)
                #logits_ema, _ = ema_model.ema(inputs_u_w)
                logits, logits_open_fix = model(inputs_ws, fc_cp=True)
                logits_u_w, logits_u_s = logits.chunk(2)
                #pseudo_label = torch.softmax(logits_ema.detach()/args.T, dim=-1)
                pseudo_label = causal_inference(logits_u_w.detach(), qhat, exp_idx=0, tau=args.tau)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(args.threshold).float()

                # update qhat
                qhat_mask = mask
                qhat = update_qhat(torch.softmax(logits_u_w.detach(), dim=-1), qhat, momentum=args.qhat_m, qhat_mask=qhat_mask)

                # adaptive marginal loss
                delta_logits = torch.log(qhat)
                logits_u_s = logits_u_s + args.tau*delta_logits

                L_fix = (F.cross_entropy(logits_u_s,
                                         targets_u,
                                         reduction='none') * mask).mean()
                
                mask_probs.update(mask.mean().item())
                L_socr = 0.1*torch.mean(torch.sum(torch.abs(
                    alphas_open_u1 - alphas_open_u2)**2, 1))

            else:
                L_fix = torch.zeros(1).to(args.device).mean()
                L_socr = torch.zeros(1).to(args.device).mean()
            loss = Lx + L_fix + 0.03*L_odir + 0.3*args.lambda_socr * L_socr 
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_o.update(L_odir.item())
            losses_socr.update(L_socr.item())
            losses_fix.update(L_fix.item())

            output_args["batch"] = batch_idx
            output_args["loss_x"] = losses_x.avg
            output_args["loss_o"] = losses_o.avg
            output_args["loss_socr"] = losses_socr.avg
            output_args["loss_fix"] = losses_fix.avg
            output_args["lr"] = [group["lr"] for group in optimizer.param_groups][0]


            optimizer.step()
            if args.opt != 'adam':
                scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(default_out.format(**output_args))
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        if args.local_rank in [-1, 0]:

            val_acc = test(args, val_loader, test_model, epoch, val=True)
            test_loss, test_acc_close,test_roc, test_roc_softm \
                = test(args, test_loader, test_model, epoch)
            wandb.log({'Test/acc': round(test_acc_close, 3), 'Test/ROC': round(test_roc, 3)})

            args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_o', losses_o.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_socr', losses_socr.avg, epoch)
            args.writer.add_scalar('train/5.train_loss_fix', losses_fix.avg, epoch)
            args.writer.add_scalar('train/6.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc_close, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)

            is_best = val_acc > best_acc_val
            best_acc_val = max(val_acc, best_acc_val)
            if is_best:
                close_valid = test_acc_close
                roc_valid = test_roc
                roc_softm_valid = test_roc_softm
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc close': test_acc_close,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)
            test_accs.append(test_acc_close)
            logger.info('Best val closed acc: {:.3f}'.format(best_acc_val))
            logger.info('Valid closed acc: {:.3f}'.format(close_valid))
            logger.info('Valid roc: {:.3f}'.format(roc_valid))
            logger.info('Valid roc soft: {:.3f}'.format(roc_softm_valid))
            logger.info('Mean top-1 acc: {:.3f}\n'.format(
                np.mean(test_accs[-20:])))
            wandb.log({'Test/Mean acc': round(np.mean(test_accs[-20:]), 3)})
    if args.local_rank in [-1, 0]:
        args.writer.close()
