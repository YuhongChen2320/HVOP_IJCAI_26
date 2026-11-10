
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys

from hlop_module import HLOP
from proj_linear import LinearProj



def get_0_1_array(array,rate=0.2):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
    new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0 #将一部分换为0
    np.random.shuffle(new_array)#将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
    return re_array



class FAGC(nn.Module):
    def __init__(self, input_dim, num_view, **kwargs):
        super(FAGC, self).__init__(**kwargs)
        self.num_view = num_view
        self.weight = glorot_init(input_dim[0],input_dim[1])
    def forward(self, inputs, adj):

        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        return x, self.weight

class GCNConv(nn.Module):
    def __init__(self, hidden_dims, num_view,num_class):
        super(GCNConv, self).__init__()
        self.lin = nn.ModuleList()
        self.gc = nn.ModuleList()
        for i in range(num_view):
            self.gc.append(FAGC(hidden_dims[i]))
        self.num_view = num_view
        self.a = nn.Parameter(torch.ones(1))
    def forward(self, v, x, adj, z):
        emb1 = x
        # 预处理到同一纬度
        # emb1 = self.lin[v](emb1)
        # 进行传播
        emb1 = self.gc[v](emb1,adj)
        if v != 0:
            emb1 = emb1 + self.a * z
        if v == 0:
            z = emb1
        return emb1,z

class GCNCon(nn.Module):
    def __init__(self, hidden_dims,hidden, num_view,num_class,device,hlop_with_wfr=False,hlop_spiking=False, hlop_spiking_scale=20., hlop_spiking_timesteps=1000.):
        super(GCNCon, self).__init__()
        self.num_view = num_view
        self.gc = FAGC([hidden_dims[0], hidden],num_view)
        self.lin = LinearProj(hidden, num_class, bias=False)
        self.lin0 = LinearProj(hidden_dims[0], hidden, bias=False)
        self.lin1 = LinearProj(num_class, num_class, bias=False)
        self.hlop_modules = nn.ModuleList([])
        self.hlop_modules.append(HLOP(hidden_dims[0], lr=0.0001, spiking=False,device=device).to(device))
        self.hlop_modules.append(HLOP(hidden, lr=0.0001, spiking=False,device=device).to(device))
        self.rep = []
        self.finalrep = []
        self.hlop_with_wfr = hlop_with_wfr
        self.hlop_spiking = hlop_spiking
        self.hlop_spiking_scale = hlop_spiking_scale
        self.hlop_spiking_timesteps = hlop_spiking_timesteps
        self.timesteps =20
        self.tau, self.delta_t =1.0,0.05
        self.a0 = nn.Parameter(torch.ones(1))
        # self.b = nn.Parameter(torch.tensor([7,5]))

    def forward(self, v, x, adj, z, projection=False, proj_id_list=[0], update_hlop=False, fix_subspace_id_list=None):
        a = F.softmax(self.a0)
        emb1 = torch.mm(adj,x)
        if v == 0:
            emb1 = emb1
            z = emb1
        else:
            emb1 =  emb1 + a * z
            z = emb1
        if projection:
            proj_func = self.hlop_modules[0].get_proj_func(subspace_id_list=proj_id_list)
            emb1_ = self.lin0(emb1, projection=True, proj_func=proj_func)
        else:
            emb1_ = self.lin0(emb1, projection=False)
        if update_hlop:
            with torch.no_grad():
                self.hlop_modules[0].forward_with_update(emb1, fix_subspace_id_list=fix_subspace_id_list)
        emb1 = emb1_
        # emb1 = F.relu(emb1_)

        if projection:
            proj_func = self.hlop_modules[1].get_proj_func(subspace_id_list=proj_id_list)
            emb1_ = self.lin(emb1, projection=True, proj_func=proj_func)
        else:
            emb1_ = self.lin(emb1, projection=False)
        if update_hlop:
            with torch.no_grad():
                self.hlop_modules[1].forward_with_update(emb1, fix_subspace_id_list=fix_subspace_id_list)

        emb1 = emb1_
        output = emb1_
        x_hat = F.softmax(torch.mm(emb1, emb1.T))
        return output,emb1,x_hat,z
        # return emb1a,w1,w1,emb1a
    def merge_hlop_subspace(self,x,adj,z):
        for m in self.hlop_modules:
            m.merge_subspace(x,adj,z)
    def add_hlop_subspace(self, out_numbers):
        if isinstance(out_numbers, list):
            for i in range(len(self.hlop_modules)):
                self.hlop_modules[i].add_subspace(out_numbers[i])
        else:
            for m in self.hlop_modules:
                m.add_subspace(out_numbers)

def weight_rate_spikes(data, timesteps, tau, delta_t):
    chw = data.size()[1:]
    # data_reshape = data.view(timesteps, -1, *chw).permute(list(range(1,len(chw)+2)) + [0])
    weight = torch.tensor([math.exp(-1/tau*(delta_t * timesteps-ii*delta_t)) for ii in range(1, timesteps+1)]).to(data.device)

    return (weight * data).sum(dim=2) / weight.sum()

class GCN(nn.Module):
    def __init__(self, hidden_dims,hidden, num_view,num_class):
        super(GCN, self).__init__()
        self.lin = nn.ModuleList()
        self.num_view = num_view
        self.gc = FAG([hidden_dims[0], hidden, num_class], num_view)

    def forward(self, v, x, adj, z):
        emb1 = x
        emb1, w1, w2 = self.gc(emb1, adj)

        z = emb1

        return emb1, z, w1, w2



class FAG(nn.Module):
    def __init__(self, input_dim,num_view, **kwargs):
        super(FAG, self).__init__(**kwargs)
        self.num_view = num_view
        self.weight = glorot_init(input_dim[0],input_dim[1])
        self.weight1 = glorot_init(input_dim[1],input_dim[2])


    def forward(self, inputs, adj):

        x = inputs
        # x = torch.mm(x, w)
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        x = F.relu(x)
        x1 = torch.mm(x, self.weight1)
        x1 = torch.mm(adj, x1)

        # w1 = self.weight1 + 0.01 * torch.mm(torch.mm(adj, x).T, x1)
        # adj1 = adj  + 0.01 * torch.mm(adj, torch.mm())
        # emb1 = torch.mm(x, w1)
        # x1 = torch.mm(adj, emb1)

        return x1, self.weight, self.weight1

class FAGCN(nn.Module):
    def __init__(self, input_dim,num_view,device, **kwargs):
        super(FAGCN, self).__init__(**kwargs)
        self.num_view = num_view
        self.device = device
        self.weight = glorot_init(input_dim[0],input_dim[1])
        self.weight1 = glorot_init(input_dim[1],input_dim[2])
        self.mask = np.ones(input_dim[0]*input_dim[1])
        self.mask = self.mask.reshape(input_dim[0],input_dim[1])
        # self.mask = np.random.randint(0, 2, (input_dim[0],input_dim[1]))
        new_arr = get_0_1_array(self.mask, rate=1-1/num_view)
        self.mask1 = torch.from_numpy(new_arr).float()
        # self.m = np.random.choice(self.mask,input_dim[0]*input_dim[1]/num_view,False)
        # self.mask = np.where(self.mask<1/num_view,self.mask,1)
        # self.mask1 = torch.rand(input_dim[1],input_dim[2])

    def forward(self, inputs, adj):
        # new_arr = get_0_1_array(self.mask, rate=1 - 1 / (self.num_view+1))
        new_arr = get_0_1_array(self.mask, rate = 1 - 1 / (self.num_view))
        n = np.sum(new_arr == 1)
        self.mask1 = torch.from_numpy(new_arr).float().to(self.device)
        w = torch.mul(self.mask1,self.weight).to(self.device)
        x = inputs
        x = torch.mm(x, w)
        # x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        x = F.relu(x)
        x = torch.mm(x, self.weight1)
        x = torch.mm(adj, x)
        return x,self.weight,self.weight1


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


