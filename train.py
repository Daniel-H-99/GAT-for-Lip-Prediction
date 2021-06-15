import os 
import time 
import random 
import argparse
import glob
from tqdm import tqdm
import logging
import time

import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from utils import *
from model import GCN, GAT, SpGCN, SpGAT, FGAT
import cv2

def main(args):
    
    # 0. initial setting
    
    # set environmet
    cudnn.benchmark = True

    if not os.path.isdir(os.path.join(args.path, './ckpt')):
        os.mkdir(os.path.join(args.path,'./ckpt'))
    if not os.path.isdir(os.path.join(args.path,'./results')):
        os.mkdir(os.path.join(args.path,'./results'))    
    if not os.path.isdir(os.path.join(args.path, './ckpt', args.name)):
        os.mkdir(os.path.join(args.path, './ckpt', args.name))
    if not os.path.isdir(os.path.join(args.path, './results', args.name)):
        os.mkdir(os.path.join(args.path, './results', args.name))
    if not os.path.isdir(os.path.join(args.path, './results', args.name, "log")):
        os.mkdir(os.path.join(args.path, './results', args.name, "log"))

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler = logging.FileHandler(os.path.join(args.path, "results/{}/log/{}.log".format(args.name, time.strftime('%c', time.localtime(time.time())))))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    args.logger = logger
    
    # set cuda
    if torch.cuda.is_available():
        args.logger.info("running on cuda")
        args.device = torch.device("cuda")
        args.use_cuda = True
    else:
        args.logger.info("running on cpu")
        args.device = torch.device("cpu")
        args.use_cuda = False
        
    args.logger.info("[{}] starts".format(args.name))
    
    # 1. load data
    
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    
    # 2. setup
    CORA_NODES = 2708
    CORA_FEATURES = 1433
    CORA_CLASSES = 7
    CITESEER_NODES = 3327
    CITESEER_FEATURES = 3703
    CITESEER_CLASSES = 6
    
#     (num_nodes, feature_dim, classes) = (CORA_NODES, CORA_FEATURES, CORA_CLASSES) if args.dataset == 'cora' else (CITESEER_NODES, CITESEER_FEATURES, CITESEER_CLASSES)
    (num_nodes, feature_dim, classes) = adj.shape[1], features.shape[1], labels.shape[1]
    args.logger.info("setting up...")
#     model = GCN(args, feature_dim, args.hidden, classes, args.dropout) if args.model == 'gcn' else SpGAT(args, feature_dim, args.hidden, classes, args.dropout, args.alpha, args.n_heads)
    model = FGAT(args, feature_dim, args.hidden, classes, args.dropout, args.alpha, args.n_heads)
    model.to(args.device)
#     loss_fn = nn.NLLLoss()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.load:
    	loaded_data = load(args, args.ckpt)
    	model.load_state_dict(loaded_data['model'])
#     	optimizer.load_state_dict(loaded_data['optimizer'])

    # 3. train / test
    
    if not args.test:
        # train
        best_acc = 100
        args.logger.info("starting training")
        train_loss_meter = AverageMeter(args, name="Loss", save_all=True, x_label="epoch")
        val_acc_meter = AverageMeter(args, name="Val Acc", save_all=True, x_label="epoch")
        earlystop_listener = val_acc_meter.attach_combo_listener((lambda prev, new : prev.max >= new.max), threshold=args.patience)
        steps = 1
        for epoch in range(1, 1 + args.epochs):
            spent_time = time.time()
            model.train()
            train_loss_tmp_meter = AverageMeter(args)
            
            if args.start_from_step is not None:
                if steps < args.start_from_step:
                    steps += 1
                    continue
            optimizer.zero_grad()
            batch = len(idx_train)
            
            output = model(features.to(args.device), adj.to(args.device))
            loss = loss_fn(output[idx_train], labels[idx_train].to(args.device))
            loss.backward()
            optimizer.step()
            train_loss_tmp_meter.update(loss, weight=batch)
            steps += 1

            train_loss_meter.update(train_loss_tmp_meter.avg)
            spent_time = time.time() - spent_time
            args.logger.info("[{}] train loss: {:.5f} took {:.1f} seconds".format(epoch, train_loss_tmp_meter.avg, spent_time))
            
            model.eval()
            spent_time = time.time()
            if not args.fastmode:
                with torch.no_grad():
                    output = model(features.to(args.device), adj.to(args.device))
#             acc = accuracy(output[idx_val], labels[idx_val]) * 100.0

            acc = loss_fn(output[idx_val], labels[idx_val].to(args.device))
            val_acc_meter.update(acc)
            earlystop = earlystop_listener.listen()
            spent_time = time.time() - spent_time
            torch.cuda.empty_cache()
            args.logger.info("[{}] val loss: {:.5f} took {:.1f} seconds".format(epoch, acc, spent_time))
            if acc < best_acc:
                best_acc = acc
                args.logger.info("best record saved")
                save(args, "best".format(epoch), {'model': model.state_dict()})
            if steps % args.save_period == 0:
                save(args, "epoch{}".format(epoch), {'model': model.state_dict()})
                train_loss_meter.plot(scatter=False)
                val_acc_meter.plot(scatter=False)
                val_acc_meter.save()
            
            if earlystop:
                break
 
    else:
        # test
        args.logger.info("starting test")
        f = open('data/face/PCA.pickle', 'rb')
        pca = pkl.load(f)
        f = open('data/face/rawKp.pickle', 'rb')
        rawKps = pkl.load(f)
        f = open('data/face/config.txt', 'r')
        test_nodes = f.readlines()
        model.eval()
        spent_time = time.time()
        with torch.no_grad():
            output = model(features.to(args.device), adj.to(args.device))
        reduced_kp = output[idx_test]
        for idx in range(len(reduced_kp)):
            rkp = reduced_kp[idx][np.newaxis, :]
            tmp = test_nodes[idx].strip().split('/')
            vid, id = tmp[0], tmp[1]
            _, N, theta, mean, _, all_kp = rawKps[vid][id]
            normed_kp = pca.inverse_transform(rkp.detach().cpu().numpy())[0]
            fake_kp = getOriginalKeypoints(normed_kp, N, theta, mean)
            all_kp[48:68] = fake_kp

#             save_name = os.path.join(result, id + '.txt')
#             np.savetxt(save_name, all_ldmk, fmt='%d', delimiter=',')
            img = cv2.imread(os.path.join('../datasets/test/0_2.mp4/img', id + '.png'))
            img = drawLips(all_kp, img)
            cv2.imwrite(os.path.join('./results', 'test', id + '.png'), img)

        acc = loss_fn(output[idx_test], labels[idx_test].to(args.device))
        spent_time = time.time() - spent_time
        args.logger.info("test loss: {:.5f}  took {:.1f} seconds".format(acc, spent_time))
        

if __name__  == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--save_every', type=int, default=10, help='Save every n epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=100, help='patience')
    parser.add_argument('--dataset', type=str, default='face', choices=['cora','citeseer', 'face'], help='Dataset to train.')
    parser.add_argument('--model', type=str, default='gat', choices=['gcn','gat'], help='Model to use.')

    parser.add_argument(
        '--data_dir',
        type=str,
        default='dataset')
    parser.add_argument(
        '--result_dir',
        type=str,
        default='results')
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default="ckpt")
    parser.add_argument(
        '--path',
        type=str,
        default='.')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0)
    parser.add_argument(
        '--test',
        action='store_true')
    parser.add_argument(
        '--save_period',
        type=int,
        default=10)
    parser.add_argument(
        '--name',
        type=str,
        default="train")
    parser.add_argument(
        '--ckpt',
        type=str,
        default='best')
    parser.add_argument(
        '--load',
        action='store_true')
    parser.add_argument(
        '--start_from_step',
        type=int,
        default=None)
    
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    main(args)
    
    