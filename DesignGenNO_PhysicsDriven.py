# /*
#  * @Author: yaohua.zang 
#  * @Date: 2025-06-30 09:08:13 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2025-06-30 09:08:13 
#  */
import numpy as np
import h5py
import torch 
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True
random_seed = 3047
setup_seed(random_seed)
device = 'cuda:0'
dtype = torch.float32
tag = 'PhysicsDriven'
######################################
# Load training data
######################################
data_train = h5py.File('./Dataset/data_train_32.mat', 'r')
data_test = h5py.File('./Dataset/data_test_32.mat', 'r')
res_coe, res_sol = 32, 64
mu = [10., 2.] # [phase=1, phase=0]
#####
class Get_High_Resolution_a(object):

    def __init__(self, res=res_coe):
        super(Get_High_Resolution_a, self).__init__()
        self.res = res
        self.delta = 1./(res-1)

    def __call__(self, x_mesh, a):
        ''' 
        Input:
            x_mesh: size(n_batch, n_mesh, 2)
            a: size(n_batch, nx*ny)
        Result:
            a_new: size(n_batch, n_mesh)
        '''
        x_loc = torch.floor(x_mesh[...,0] / self.delta + 0.5).int()
        y_loc = torch.floor(x_mesh[...,1] / self.delta + 0.5).int()
        loc = y_loc * self.res + x_loc
        #
        a_new = a[torch.arange(a.size(0)).unsqueeze(1), loc]
        
        return a_new
########
from Utils.utils import *
n_train, n_test = 10000, 200
#
def get_data(data, ndata, dtype, n0=0):
    a = np2tensor(np.array(data["coe"][n0:n0+ndata,...]), dtype)
    a[a==1.] = mu[0]; a[a==0.] = mu[1];
    ux = np2tensor(np.array(data["ux_fem"][n0:n0+ndata,...]), dtype)
    uy = np2tensor(np.array(data["uy_fem"][n0:n0+ndata,...]), dtype)
    duxx = np2tensor(np.array(data["dux_x_fem"][n0:n0+ndata,...]), dtype)
    duyy = np2tensor(np.array(data["duy_y_fem"][n0:n0+ndata,...]), dtype)
    cd = np2tensor(np.array(data["conductivity"][n0:n0+ndata,...]), dtype)
    # #
    X, Y = np.array(data['X_sol']), np.array(data['Y_sol'])
    mesh = np2tensor(np.vstack([X.ravel(), Y.ravel()]).T, dtype)
    gridx = mesh.reshape(-1, 2)
    #
    a = a.reshape(ndata, -1)
    a_high_res = Get_High_Resolution_a()(gridx, a)
    ux = ux.reshape(ndata, -1, 1)
    uy = uy.reshape(ndata, -1, 1)
    duxx = duxx.reshape(ndata, -1, 1) 
    duyy = duyy.reshape(ndata, -1, 1) 
    #
    feat = a.reshape(ndata, -1)
    return feat, a_high_res, ux, duxx, uy, duyy, gridx, cd
#
feat_train, a_train, ux_train, duxx_train, uy_train, duyy_train, gridx, cd_train = get_data(data_train, n_train, dtype)
feat_test, a_test, ux_test, duxx_test, uy_test, duyy_test, gridx, cd_test = get_data(data_test, n_test, dtype)
#
print('The shape of feat_train:', feat_train.shape)
print('The shape of x_train:', gridx.shape)
print('The shape of a_train:', a_train.shape)
print('The shape of ux_train:', ux_train.shape)
print('The shape of duxx_train:', duxx_train.shape)
print('The shape of uy_train:', uy_train.shape)
print('The shape of duyy_train:', duyy_train.shape)
#
print('The shape of x_test:', gridx.shape)
print('The shape of feat_test:', feat_test.shape)
print('The shape of a_test:', a_test.shape)
print('The shape of ux_test:', ux_test.shape)
print('The shape of duxx_test:', duxx_test.shape)
print('The shape of uy_test:', uy_test.shape)
print('The shape of duyy_test:', duyy_test.shape)

### Generate boundary data
from Utils.GenPoints import Point2D
pointGen = Point2D(x_lb=[0., 0.], x_ub=[1.,1.], dataType=dtype, random_seed=random_seed)
N_bd_each_edge = 250
x_bd = pointGen.boundary_point(num_each_edge=N_bd_each_edge, method='hypercube')
x_lr = x_bd[0:2*N_bd_each_edge]
x_bu = x_bd[2*N_bd_each_edge:]


########################################
# The loss function
#######################################
import torch.nn as nn
from torch.autograd import grad, Variable
from Utils.GenPoints import Point2D
from Utils.TestFun_ParticleWNN import TestFun_ParticleWNN
###############################
# The test function
###############################
pointGen = Point2D(x_lb=[0., 0.], x_ub=[1.,1.], dataType=dtype, random_seed=random_seed)
#
int_grid, v, dv_dr = TestFun_ParticleWNN(
    fun_type='Wendland', dim=2, n_mesh_or_grid=7, dataType=dtype).get_testFun()
print('int_grid shape:', int_grid.shape, 'v shape:', v.shape)

###############################
# Set normalizer
###############################
class UnitGaussianNormalizer():

    def __init__(self, a, eps=1e-8):
        super(UnitGaussianNormalizer, self).__init__()
        '''Apply normaliztion to inputs or outputs
        Input:
            a: size(N, mesh_size)
        Output:
            mean: size(mesh_szie,)
            std: size(mesh_size,)
        '''
        self.mean = torch.mean(a, 0)
        self.std = torch.std(a, 0)
        self.eps = eps
    
    def encode(self, a):
        '''
        Input:
            a: a: size(N, mesh_size)
        '''
        return (a - self.mean) / (self.std + self.eps)
    
    def decode(self, a):
        #
        return a * (self.std + self.eps) + self.mean
#
normalizer_feat = UnitGaussianNormalizer(feat_train.to(device))

###############################
class mollifer_x(object):

    def __inint__(self):
        super(mollifer_x, self).__init_()
        
    def __call__(self, u, x):
        '''
        u: size(n_batch, nx*ny) -> size(n_batch, nx*ny, 1)
        '''
        xx, yy = x[...,0], x[...,1]
        u = u * torch.sin(np.pi * xx) + torch.sin(np.pi/2 * xx)
        return u.unsqueeze(-1)
#
class mollifer_y(object):

    def __inint__(self):
        super(mollifer_y, self).__init_()
        
    def __call__(self, u, x):
        '''
        u: size(n_batch, nx*ny) -> size(n_batch, nx*ny, 1)
        '''
        xx, yy = x[...,0], x[...,1]
        u = u * torch.sin(np.pi * yy) + torch.sin(np.pi/2 * yy)
        return u.unsqueeze(-1)
        
################################
class LossClass(object):

    def __init__(self, solver):
        super(LossClass, self).__init__()
        ''' '''
        self.solver = solver
        self.dtype = solver.dtype
        self.device = solver.device
        #
        self.mollifer_x = mollifer_x()
        self.mollifer_y = mollifer_y()
        #
        self.fun_a = Get_High_Resolution_a(res_sol)
        self.model_enc = solver.model_dict['enc']
        self.model_a = solver.model_dict['a']
        self.model_ux = solver.model_dict['ux']
        self.model_uy = solver.model_dict['uy']
        self.model_sx1 = solver.model_dict['sx1']
        self.model_sx2 = solver.model_dict['sx2']
        self.model_sy1 = solver.model_dict['sy1']
        self.model_sy2 = solver.model_dict['sy2']
        #######################
        self.int_grid = int_grid.to(self.device)
        self.n_grid = int_grid.shape[0]
        self.v = v.to(self.device)
        self.dv_dr = dv_dr.to(self.device)
        self.dx, self.dy = 1/(res_sol-1), 1/(res_sol-1)

    def Loss_pde(self, idx, train_or_test, nc=100):
        ''' The PDE loss'''
        if train_or_test=='train':
            feat = feat_train[idx].to(self.device)
            a = a_train[idx].to(self.device)
        elif train_or_test=='test':
            feat = feat_test[idx].to(self.device)
            a = a_test[idx].to(self.device)
        #
        n_batch = len(idx)
        beta = self.model_enc(normalizer_feat.encode(feat))
        ############### Data points ###############
        # xc:size(nc, 1, 2) R:size(nc, 1, 1)
        xc, R = pointGen.weight_centers(n_center=nc, R_max=1e-4, R_min=1e-4)
        xc, R = xc.to(self.device), R.to(self.device)
        nc = xc.shape[0]
        # size(nc, n_grid, 2)
        x = self.int_grid * R + xc
        # size(nc*n_grid, 2) -> (n_batch, nc*n_grid, 2)
        x = x.reshape(-1, 2).repeat((n_batch,1,1))
        x = Variable(x, requires_grad=True)
        ############### Test functions #############
        v = self.v.repeat((nc,1,1)).reshape(-1, 1) # size(nc*n_grid, 1)
        dv = (self.dv_dr / R).reshape(-1, 2) # size(nc*n_grid, 2)
        ################ model prediction ###########
        # a_detach: size(n_batch, nc*n_grid) -> (n_batch, nc*n_grid, 1)
        a_detach = self.fun_a(x.detach(), a).unsqueeze(-1)
        ############### The x-direction
        # ux: size(n_batch, nc*n_grid) -> size(n_batch, nc*n_grid)
        ux = self.model_ux(x, beta)
        ux = self.mollifer_x(ux, x)
        # sx: size(n_batch, nc*n_grid, 2)
        sx = torch.stack([self.model_sx1(x, beta), self.model_sx2(x, beta)], dim=-1)
        # dux: size(n_batch, nc*n_grid, 2)
        dux = grad(inputs=x, outputs=ux, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
        #
        res1 = torch.sum( (sx - a_detach * dux)**2, dim=-1).reshape(n_batch, nc, self.n_grid)
        res1 = torch.mean(res1, dim=-1) # size(n_batch, nc)
        res2 = torch.sum(sx * dv, dim=-1).reshape(n_batch, nc, self.n_grid)
        res2 = torch.mean(res2, dim=-1)**2 # size(n_batch, nc)
        ############### The y-direction
        # uy: size(n_batch, nc*n_grid) -> size(n_batch, nc*n_grid, 1)
        uy = self.model_uy(x, beta)
        uy = self.mollifer_y(uy, x)
        # sy: size(n_batch, nc*n_grid, 2)
        sy = torch.stack([self.model_sy1(x, beta), self.model_sy2(x, beta)], dim=-1)
        # duy: size(n_batch, nc*n_grid, 2)
        duy = grad(inputs=x, outputs=uy, grad_outputs=torch.ones_like(uy), create_graph=True)[0]
        #
        res3 = torch.sum( (sy - a_detach * duy)**2, dim=-1).reshape(n_batch, nc, self.n_grid)
        res3 = torch.mean(res3, dim=-1) # size(n_batch, nc)
        res4 = torch.sum(sy * dv, dim=-1).reshape(n_batch, nc, self.n_grid)
        res4 = torch.mean(res4, dim=-1)**2 # size(n_batch, nc)

        return (torch.mean(res1) + torch.mean(res3))*1. + (torch.mean(res2) + torch.mean(res4)) * 2. * np.sqrt(nc)
        
    def Loss_data(self, idx, train_or_test):
        '''The data loss'''
        n_batch = len(idx)
        if train_or_test=='train':
            x = gridx.repeat((n_batch, 1, 1)).to(self.device)
            feat = feat_train[idx].to(self.device)
            a = a_train[idx].to(self.device)
            ux, uy = ux_train[idx].to(self.device), uy_train[idx].to(self.device)
        elif train_or_test=='test':
            x = gridx.repeat((n_batch, 1, 1)).to(self.device)
            feat = feat_test[idx].to(self.device)
            a = a_test[idx].to(self.device)
            ux, uy = ux_test[idx].to(self.device), uy_test[idx].to(self.device)
        #
        beta = self.model_enc(normalizer_feat.encode(feat))
        ######################## The recovery loss of a
        a_pred = nn.Sigmoid()(self.model_a(x, beta))
        a_true = (a-mu[1])/(mu[0]-mu[1])
        #
        loss_a = nn.functional.binary_cross_entropy(a_pred, a_true, reduction='mean')
        ######################## The boundary loss
        loss_bd = self.get_bd_loss(idx, beta)

        return 2.*loss_a + 5.*loss_bd 

    def get_bd_loss(self, idx, beta):
        ''' '''
        n_batch = len(idx)
        ######################## The neumann boundary loss of ux
        x_bd = Variable(x_lr.repeat((n_batch, 1, 1)).to(self.device), requires_grad=True)
        #
        ux = self.model_ux(x_bd, beta)
        ux = self.mollifer_x(ux, x_bd)
        dux = grad(inputs=x_bd, outputs=ux, grad_outputs=torch.ones_like(ux), create_graph=True)[0]
        loss_neumann_x = torch.mean(dux[...,1]**2)
        ######################## The neumann boundary loss of uy
        x_bd = Variable(x_bu.repeat((n_batch, 1, 1)).to(self.device), requires_grad=True)
        #
        uy = self.model_uy(x_bd, beta)
        uy = self.mollifer_y(uy, x_bd)
        duy = grad(inputs=x_bd, outputs=uy, grad_outputs=torch.ones_like(uy), create_graph=True)[0]
        loss_neumann_y = torch.mean(duy[...,0]**2)

        return loss_neumann_x + loss_neumann_y

    def Error(self):
        ''' '''
        x = gridx.repeat((n_test, 1, 1)).to(self.device)
        feat = feat_test.to(self.device)
        ux, uy = ux_test.to(self.device), uy_test.to(self.device)
        ############################
        beta = self.model_enc(normalizer_feat.encode(feat))
        #
        ux_pred = self.model_ux(x, beta)
        ux_pred = self.mollifer_x(ux_pred, x)
        #
        uy_pred = self.model_uy(x, beta)
        uy_pred = self.mollifer_y(uy_pred, x)
        #
        err_x = self.solver.getError(ux_pred, ux)
        err_y = self.solver.getError(uy_pred, uy)
        
        return torch.tensor([err_x, err_y])
        
######################################
# Steups of the model
######################################
from Solvers.DGNO import DGNO
solver = DGNO.Solver(device=device, dtype=dtype)
netType = 'MultiONetBatch'
beta_size = 128   
hidden_size_a = 256
hidden_size = 128

####################################### The beta model
from Networks.EncoderNet import EncoderFCNet
class Encoder(nn.Module):
    def __init__(self, layers_list, activation, dtype):
        super(Encoder, self).__init__()
        self.encoder = EncoderFCNet(layers_list, activation, dtype) 
        
    def forward(self, feat):
        '''
        Input:
            feat: size(batch_size, a_size)
        Return:
            output: size(?, beta_size) -> size(? beta_size)
        '''
        beta = self.encoder(feat)
        beta = torch.tanh(beta)
        return beta

model_enc = Encoder([feat_train.shape[-1], 512, 256, beta_size], 'SiLU', dtype).to(device)

###################################### The u model
trunk_layers, branch_layers = [hidden_size_a]*5, [hidden_size_a]*5
model_a = solver.getModel(x_in_size=2, a_in_size=beta_size, 
                          trunk_layers=trunk_layers, branch_layers=branch_layers,
                          activation_trunk='SiLU_Sin', activation_branch='SiLU_Id',
                           netType=netType)
#
trunk_layers, branch_layers = [hidden_size]*5, [hidden_size]*5
model_ux = solver.getModel(x_in_size=2, a_in_size=beta_size, 
                          trunk_layers=trunk_layers, branch_layers=branch_layers,
                          activation_trunk='SiLU_Sin', activation_branch='SiLU_Id',
                           netType=netType)
model_uy = solver.getModel(x_in_size=2, a_in_size=beta_size, 
                          trunk_layers=trunk_layers, branch_layers=branch_layers,
                          activation_trunk='SiLU_Sin', activation_branch='SiLU_Id',
                           netType=netType)
#
trunk_layers, branch_layers = [hidden_size]*5, [hidden_size]*5
model_sx1 = solver.getModel(x_in_size=2, a_in_size=beta_size, 
                          trunk_layers=trunk_layers, branch_layers=branch_layers,
                          activation_trunk='SiLU_Sin', activation_branch='SiLU_Id',
                           netType=netType)
model_sx2 = solver.getModel(x_in_size=2, a_in_size=beta_size, 
                          trunk_layers=trunk_layers, branch_layers=branch_layers,
                          activation_trunk='SiLU_Sin', activation_branch='SiLU_Id',
                           netType=netType)
model_sy1 = solver.getModel(x_in_size=2, a_in_size=beta_size, 
                          trunk_layers=trunk_layers, branch_layers=branch_layers,
                          activation_trunk='SiLU_Sin', activation_branch='SiLU_Id',
                           netType=netType)
model_sy2 = solver.getModel(x_in_size=2, a_in_size=beta_size, 
                          trunk_layers=trunk_layers, branch_layers=branch_layers,
                          activation_trunk='SiLU_Sin', activation_branch='SiLU_Id',
                           netType=netType)
# ###############################
total_trainable_params_enc = sum(p.numel() for p in model_enc.parameters() if p.requires_grad)
print(f'{total_trainable_params_enc:,} training parameters.')
#
total_trainable_params_a = sum(p.numel() for p in model_a.parameters() if p.requires_grad)
print(f'{total_trainable_params_a:,} training parameters.')
#
total_trainable_params_u = sum(p.numel() for p in model_ux.parameters() if p.requires_grad)
print(f'{total_trainable_params_u:,} training parameters.')
#
total_trainable_params_s = sum(p.numel() for p in model_sx1.parameters() if p.requires_grad)
print(f'{total_trainable_params_s:,} training parameters.')
#
print(f'{total_trainable_params_enc + total_trainable_params_a + total_trainable_params_u*2 + total_trainable_params_s*4:,} total parameters')

########################################
# The training part
#######################################
model_dict = {'ux':model_ux, 'uy':model_uy, 'sx1':model_sx1, 'sx2':model_sx2, 
              'sy1':model_sy1, 'sy2':model_sy2, 'a':model_a, 'enc':model_enc}
solver.train_setup(model_dict, lr=5e-4, optimizer='Adam', scheduler_type='StepLR', 
                   gamma=0.6, step_size=np.int32(2000/5))
solver.train_index(LossClass, a_train, a_test, w_pde=0.25, w_data=1., 
                   batch_size=25, epochs=2000, epoch_show=50, 
                   **{'save_path':f'saved_models/DGNO_{tag}/'})