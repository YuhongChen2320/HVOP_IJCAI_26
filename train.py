
import os
import random
import time
import warnings
from matplotlib import pyplot as plt
from sklearn import manifold
import numpy as np
import scipy.io as scio
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.colors as col
from Dataloader import load_data
from args import parameter_parser
from model import GCNCon
from utils import tab_printer, get_evaluation_results, norm_2


def train(args, device):
    feature_list, adj_list, labels, idx_labeled, idx_unlabeled = load_data(args)
    num_classes = len(np.unique(labels))
    labels = labels.to(device)
    N = feature_list[0].shape[0]
    num_view = len(feature_list)
    hidden_dims = [feature_list[0].shape[1]]

    GCN_model = GCNCon(hidden_dims,args.hidden, num_view,num_classes,args.device).to(device)
    optimizer_GCN = torch.optim.Adam(GCN_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function1 = torch.nn.NLLLoss()
    count_list = []
    Loss_list = []
    ACC_list = []
    MAF1_list = []
    MIF1_list = []
    begin_time = time.time()
    hlop_out_num = [60, 120, 180]
    hlop_out_num_inc = [70, 70, 70]
    z_q = []
    torch.autograd.set_detect_anomaly(True)
    with tqdm(total=args.num_epoch, desc="Training",ncols=160) as pbar:
        for v in range(num_view):
            # draw_plt(feature_list[v].cpu(),labels.cpu())
            if v == 0:
                z_past=feature_list[v]
                GCN_model.add_hlop_subspace(hlop_out_num)
                update_hlop = True
                projection = False
                proj_id_list = [0]
                fix_subspace_id_list = None
            else:
                z_past = z_g
                z_q.append(z_g)
                GCN_model.add_hlop_subspace(hlop_out_num_inc)
                update_hlop = True
                projection = True
                proj_id_list = [0]
                fix_subspace_id_list = [0]
            print('\nnow it comes the '+ str(v) +'-th view...')
            for epoch in range(args.num_epoch):
                GCN_model.train()
                z,w1,x_hat,z_ = GCN_model(v,feature_list[v],adj_list[v],z_past,update_hlop=update_hlop,projection=projection,proj_id_list=proj_id_list,fix_subspace_id_list=fix_subspace_id_list)
                z_g = z_.data
                loss_DMF = 0.
                for mm in range(0,v+1):
                    loss_DMF += norm_2(adj_list[mm], x_hat)
                output = F.log_softmax(z, dim=1)
                optimizer_GCN.zero_grad()
                loss_DMF = loss_DMF / x_hat.shape[0] / (v + 1)
                loss_GCN = loss_function1(output[idx_labeled], labels[idx_labeled])
                loss1 = loss_GCN
                loss_GCN += loss_DMF
                loss_GCN.backward(retain_graph=True)
                optimizer_GCN.step()
                with torch.no_grad():
                    GCN_model.eval()
                    pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                    ACC, _, _, MAF1, MIF1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled],output.cpu().detach().numpy()[idx_unlabeled])
                    pbar.set_postfix({'Loss': '{:.6f}'.format((loss_GCN).item()),'Constructed Loss': '{:.6f}'.format((loss_DMF)),
                                      'ACC': '{:.2f}'.format(ACC * 100), 'MAF1': '{:.2f}'.format(MAF1 * 100)
                                      })
                    pbar.update(1)

                    if epoch == 0:
                        print('\n' + str(v) + '-th view ' + 'acc starts ' + str(ACC * 100))
                        count_list.append(ACC * 100)
                    elif epoch == args.num_epoch - 1:
                        print('\n' + str(v) + '-th view ' + 'acc ends ' + str(ACC * 100))
                        count_list.append(ACC * 100)
                    Loss_list.append(float((loss_GCN).item()))
                    ACC_list.append(ACC)
                    MAF1_list.append(MAF1)
                    MIF1_list.append(MIF1)
            GCN_model.merge_hlop_subspace(feature_list[v], adj_list[v],z)

            for j in range(0,v+1):
                GCN_model.eval()
                print("Evaluating the " + str(j) + " view of  model")
                if j == 0:
                    zview, _, _, _ = GCN_model(1, feature_list[j], adj_list[j], 0)
                else:
                    zview, _, _, _ = GCN_model(j, feature_list[j], adj_list[j], z_q[j-1])
                output = F.log_softmax(zview, dim=1)
                pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
                ACC, P, R, MAF1, MIF1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled], pred_labels[idx_unlabeled],output.cpu().detach().numpy()[idx_unlabeled])
                print("------------------------")
                print( str(j) + "view ACC:   {:.2f}".format(ACC * 100))
                print( str(j) + "view MIF1 :   {:.2f}".format(MIF1 * 100))
                print("------------------------")
        GCN_model.eval()
        z, w1, w2, z_ = GCN_model(v, feature_list[v], adj_list[v], z_past)
        output = F.log_softmax(z, dim=1)
        print("Evaluating the final view of model")
        pred_labels = torch.argmax(output, 1).cpu().detach().numpy()
        ACC, P, R, MAF1, MIF1 = get_evaluation_results(labels.cpu().detach().numpy()[idx_unlabeled],
                                                       pred_labels[idx_unlabeled],
                                                       output.cpu().detach().numpy()[idx_unlabeled])
        print("------------------------")
        print("ACC:   {:.2f}".format(ACC * 100))
        print("MIF1 :   {:.2f}".format(MIF1 * 100))
        print("------------------------")

    return ACC, P, R, MAF1, MIF1, Loss_list, ACC_list,count_list


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    CUDA_LAUNCH_BLOCKING = 1
    args = parameter_parser()
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    args.device = device
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    tab_printer(args)
    dataset_dict = {1: '100leaves', 2: '20newsgroups', 3: '3sources', 4: 'ALOI', 5: 'animals', 6: 'BBC3view',
                    7: 'BBC4view', 8: 'BBCSports', 9: 'BDGP', 10: 'Caltech101-7', 11: 'Caltech10120',
                    12: 'Caltech101all',
                    13: 'Cifar10_10k_batch_1', 14: 'citeseer',
                    15: 'COIL', 16: 'COIL20', 17: 'Cora', 18: 'esp-game', 19: 'flickr30k', 20: 'flower17', 21: 'GRAZ02',
                    22: 'handwritten', 23: 'Hdigit',
                    24: 'HW', 25: 'iaprtc12', 26: 'MFeat', 27: 'MITIndoor', 28: 'MNIST', 29: 'MNIST10k', 30: 'MSRC-v1',
                    31: 'NGs', 32: 'NoisyMNIST-30000', 33: 'NottingHill',
                    34: 'NUS-Wide20k', 35: 'NUSWIDE', 36: 'NUSWIDEOBJ', 37: 'ORL', 38: 'Reuters', 39: 'scene15',
                    40: 'smallReuters', 41: 'UCI', 42: 'WebKB', 43: 'WebKB_cornell', 44: 'WebKB_texas',
                    45: 'WebKB_washington',
                    46: 'WebKB_wisconsin', 47: 'Wikipedia', 48: 'YaleB', 49: 'Youtube'}

    select_dataset = [4]
    # for k in select_dataset:
    #     args.dataset = dataset_dict[k]
    all_ACC = []
    all_P = []
    all_R = []
    all_MAF1 = []
    all_MIF1 = []
    all_AUC = []
    all_TIME = []
    count = []
    for i in range(args.n_repeated):
        torch.cuda.empty_cache()
        ACC, P, R, MAF1, MIF1, Loss_list, ACC_list, count_list = train(args, device)
        all_ACC.append(ACC)
        all_P.append(P)
        all_R.append(R)
        all_MAF1.append(MAF1)
        all_MIF1.append(MIF1)
        count.append(count_list)
        print("-----------------------")
        print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
        print("P  : {:.2f} ({:.2f})".format(np.mean(all_P) * 100, np.std(all_P) * 100))
        print("R  : {:.2f} ({:.2f})".format(np.mean(all_R) * 100, np.std(all_R) * 100))
        print("MAF1 : {:.2f} ({:.2f})".format(np.mean(all_MAF1) * 100, np.std(all_MAF1) * 100))
        print("-----------------------")

    print(count)
