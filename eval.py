import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
# import wandb
import copy
import random
from reparam_module import ReparamModule
from networks import *
from torchvision.utils import save_image
import kornia.augmentation as K
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args):
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    args.dc_aug_param = None
    args.dsa_param = ParamDiffAug()
    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None
    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    saet = torch.load(args.saet_path, map_location='cpu').to(args.device)
    scm = torch.load(args.scm_path, map_location='cpu').to(args.device)
    label_syn = torch.tensor([np.ones(int(scm.shape[0] / num_classes), dtype=np.int_) * i for i in range(num_classes)],
                             dtype=torch.long, requires_grad=False, device=args.device).view(-1)

    freenet_ckpt = torch.load(args.freenet_path, map_location='cpu')
    freenet = CIFAR10_IPC10_freenet_patch4_dec96d1b3h().to(args.device)
    freenet.load_state_dict(freenet_ckpt, strict=True)

    if args.syn_lr_path is not None:
        syn_lr = torch.load(args.syn_lr_path, map_location='cpu').to(args.device)
    else:
        syn_lr = 0.01

    scm = scm.to_dense()
    print(scm[0][0])

    saet_eval = copy.deepcopy(saet.detach())
    scm_eval = copy.deepcopy(scm.detach())
    freenet_eval = copy.deepcopy(freenet)
    label_syn_eval = copy.deepcopy(label_syn.detach())
    images_syn_eval = freenet_eval(saet_eval.repeat(scm_eval.shape[0], 1, 1), scm_eval)

    for model_eval in model_eval_pool:
        if args.dsa:
            print('DSA augmentation strategy: \n', args.dsa_strategy)
            print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
        else:
            print('DC augmentation parameters: \n', args.dc_aug_param)

        accs_test = []
        accs_train = []

        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
            args.lr_net = syn_lr.item()
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, images_syn_eval, label_syn_eval, testloader, args)
            accs_test.append(acc_test)
            accs_train.append(acc_train)

        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)

        print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (len(accs_test), model_eval, acc_test_mean, acc_test_std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagewoof', help='ImageNet subset. This only does anything when --dataset=ImageNet')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode, check utils.py for more info')
    parser.add_argument('--epoch_eval_train', type=int, default=1000,  help='epochs to train a model with synthetic data.  For faster eval, use --epoch_eval_train=300')
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=256, help='batch size for synthetic data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'], help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',  help='differentiable Siamese augmentation strategy')
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--kip_zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--saet_path', type=str, default='./ckpt/CIFAR10/IPC10/saet.pt', help='path to save results')
    parser.add_argument('--scm_path', type=str, default='./ckpt/CIFAR10/IPC10/scm.pt', help='path to save results')
    parser.add_argument('--freenet_path', type=str, default='./ckpt/CIFAR10/IPC10/freenet.pt', help='path to save results')
    parser.add_argument('--syn_lr_path', type=str, default='./ckpt/CIFAR10/IPC10/syn_lr.pt', help='path to save results')


    args = parser.parse_args()

    main(args)