import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, feature_sparsification
# import wandb
import copy
import random
from reparam_module import ReparamModule
from networks import *
from torchvision.utils import save_image

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = [11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.num_scm

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in range(len(dst_train)):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images' % (c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f' % (
        ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    ''' initialize the synthetic data '''
    num_patch_sqrt = int(args.img_size / args.patch_size)
    num_patch = num_patch_sqrt ** 2
    label_syn = torch.tensor([np.ones(args.num_scm, dtype=np.int_) * i for i in range(num_classes)],
                             dtype=torch.long, requires_grad=False, device=args.device).view(
        -1)  # [0,0,0, 1,1,1, ..., 9,9,9]

    saet_mean = torch.zeros(1, args.num_saet, args.dim_saet)
    saet_std = .2 * torch.ones(1, args.num_saet, args.dim_saet)
    saet = torch.normal(mean=saet_mean, std=saet_std)

    scm_mean = torch.zeros(num_classes * args.num_scm, args.num_head, num_patch, args.num_saet)
    scm_std = .2 * torch.ones(num_classes * args.num_scm, args.num_head, num_patch, args.num_saet)
    scm = torch.normal(mean=scm_mean, std=scm_std)

    freenet = CIFAR10_IPC10_freenet_patch4_dec96d1b3h().to(args.device)

    syn_lr = torch.tensor(args.lr_teacher)

    ''' training '''
    saet = saet.detach().to(args.device).requires_grad_(True)
    
    scm = scm.detach().to(args.device).requires_grad_(True)
    freenet = freenet.to(args.device)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)

    optimizer_saet = torch.optim.SGD([saet], lr=args.lr_saet, momentum=0.95)
    optimizer_scm = torch.optim.SGD([scm], lr=args.lr_scm, momentum=0.95)
    optimizer_freenet = torch.optim.SGD(freenet.parameters(), lr=args.lr_freenet, momentum=0.95)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.95)
    optimizer_saet.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins' % get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    for it in range(0, args.Iteration + 1):
        save_this_it = False

        ''' Evaluate synthetic data '''
        if it in eval_it_pool and it > 0:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
                    args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(
                        args.device)  # get a random model

                    saet_eval = copy.deepcopy(saet.detach())
                    scm_eval = copy.deepcopy(scm.detach())
                    freenet_eval = copy.deepcopy(freenet)
                    label_syn_eval = copy.deepcopy(label_syn.detach())

                    scm_eval = feature_sparsification(scm_eval, top_k=args.top_k, device=args.device)

                    images_syn_eval = freenet_eval.forward(
                        saet_eval.repeat(scm_eval.shape[0], 1, 1), scm_eval)
                    args.lr_net = syn_lr.item()
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, images_syn_eval, label_syn_eval,
                                                             testloader, args)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                print('Top-k evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                len(accs_test), model_eval, acc_test_mean, acc_test_std))

                print()
                if args.zca:
                    images_syn_eval = args.zca_trans.inverse_transform(images_syn_eval)
                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis.pdf')
                image_syn_vis = copy.deepcopy(images_syn_eval.detach().cpu())
                for clip_val in [2.5]:
                    std = torch.std(image_syn_vis)
                    mean = torch.mean(image_syn_vis)
                    upsampled = torch.clip(image_syn_vis, min=mean - clip_val * std, max=mean + clip_val * std)
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                upsampled_ = upsampled.detach()
                save_image(upsampled_, save_name, nrow=10, padding=1, normalize=True, scale_each=True)

        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                saet_save = saet.detach()
                scm_save = scm.detach()
                syn_lr_save = syn_lr.detach()

                scm_save = feature_sparsification(scm_save, top_k=args.top_k, device=args.device)
                scm_save = scm_save.to_sparse()

                if args.dataset == 'ImageNet':
                    save_dir = os.path.join(".", "logged_files", args.dataset, args.subset, args.specification)
                else:
                    save_dir = os.path.join(".", "logged_files", args.dataset, args.specification)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(syn_lr_save.cpu(), os.path.join(save_dir, "syn_lr_{}.pt".format(it)))
                torch.save(saet_save.cpu(), os.path.join(save_dir, "saet_{}.pt".format(it)))
                torch.save(scm_save.cpu(), os.path.join(save_dir, "scm_{}.pt".format(it)))
                torch.save(freenet.state_dict(), os.path.join(save_dir, "freenet_{}.pt".format(it)))

                if save_this_it:
                    torch.save(syn_lr_save.cpu(), os.path.join(save_dir, "syn_lr_best.pt"))
                    torch.save(saet_save.cpu(), os.path.join(save_dir, "saet_best.pt"))
                    torch.save(scm_save.cpu(), os.path.join(save_dir, "scm_best.pt"))
                    torch.save(freenet.state_dict(), os.path.join(save_dir, "freenet_best.pt"))

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)

        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch + args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        momentum_list = [torch.zeros(size=starting_params.size()).to(args.device)]

        syn_saet = saet
        synum_scm = scm
        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(num_classes * args.num_scm, device=args.device)
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()

            p = synum_scm[these_indices]
            x = freenet.forward(syn_saet.repeat(p.shape[0], 1, 1), p)
            this_y = y_hat[these_indices]

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            x = student_net(x, flat_param=forward_params)
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
            momentum = args.momentum * momentum_list[-1] + (1 - args.momentum) * grad

            student_params.append(student_params[-1] - syn_lr * momentum)
            momentum_list.append(momentum)

        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        ''' l1 Penalty '''
        l1_penalty = torch.tensor(0.0).to(args.device)
        l1_penalty = torch.sum(torch.abs(scm))

        l1_penalty = args.l1_weight * l1_penalty
        total_loss = grand_loss + l1_penalty

        optimizer_saet.zero_grad()
        optimizer_scm.zero_grad()
        optimizer_freenet.zero_grad()
        optimizer_lr.zero_grad()

        total_loss.backward()

        optimizer_saet.step()
        optimizer_scm.step()
        optimizer_freenet.step()
        optimizer_lr.step()

        syn_lr.data = syn_lr.data.clip(min=0.001)

        for _ in student_params:
            del _

        if it % 10 == 0:
            print('%s iter = %04d, grand_loss = %.4f, l1_penalty = %.4f, syn_lr = %.6f' % (
            get_time(), it, grand_loss.item(), l1_penalty.item(), syn_lr.item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagewoof', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--eval_mode', type=str, default='M', help='eval_mode, check utils.py for more info')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--eval_it', type=int, default=1000, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data. For faster eval, use --epoch_eval_train=300')
    parser.add_argument('--Iteration', type=int, default=20000, help='how many distillation steps to perform')

    parser.add_argument('--lr_saet', type=float, default=10, help='learning rate for updating saet')
    parser.add_argument('--lr_scm', type=float, default=100, help='learning rate for updating scm')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument('--lr_freenet', type=float, default=0.01, help='learning rate for updating freenet')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=128, help='batch size for synthetic data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"], help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'], help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--pretrained_ckpt', type=str, default='')

    parser.add_argument('--expert_epochs', type=int, default=2, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=60, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=20, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
    parser.add_argument('--force_save', action='store_true', help='')
    parser.add_argument('--kip_zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--save_path', type=str, default='./result/CIFAR10', help='path to save results')
    parser.add_argument('--specification', type=str, default='124_96_1_3_48', help='sparse parameterization specification')

    parser.add_argument('--num_scm', type=int, default=124, help='number of scms')
    parser.add_argument('--num_saet', type=int, default=64, help='number of saets')
    parser.add_argument('--dim_saet', type=int, default=96, help='dimension of saets')
    parser.add_argument('--img_size', type=int, default=32, help='size of images')
    parser.add_argument('--patch_size', type=int, default=4, help='size of patches')
    parser.add_argument('--num_head', type=int, default=3, help='number of heads')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--l1_weight', type=float, default=3e-7, help='weight of l1 penalty')

    parser.add_argument('--top_k', type=int, default=48, help='top-k')

    args = parser.parse_args()

    main(args)


