# /*
#  * @Author: yaohua.zang 
#  * @Date: 2025-07-15 14:31:23 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2025-07-15 14:31:23 
#  */
# Note: modified from https://github.com/tonyduan/normalizing-flows

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
#
try:
    from FunActivation import FunActivation
    from NormalizingFlow_utils import unconstrained_RQS
except:
    from .FunActivation import FunActivation
    from .NormalizingFlow_utils import unconstrained_RQS
# The prior distribution
from torch.distributions import MultivariateNormal


######################### The base network
class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim, activation):
        super().__init__()
        # Activation
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        # The network
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)
    
##################################################### RealNVP
class RealNVP(nn.Module):
    """
    Non-volume preserving flow.

    [Dinh et. al. 2017]
    """
    def __init__(self, dim, hidden_dim = 20, activation='Tanh', base_network=FCNN):
        super().__init__()
        self.dim = dim
        self.t1 = base_network(dim // 2, dim // 2, hidden_dim, activation)
        self.s1 = base_network(dim // 2, dim // 2, hidden_dim, activation)
        self.t2 = base_network(dim // 2, dim // 2, hidden_dim, activation)
        self.s2 = base_network(dim // 2, dim // 2, hidden_dim, activation)

    def forward(self, x):
        lower, upper = x[:,:self.dim // 2], x[:,self.dim // 2:]
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = t1_transformed + upper * torch.exp(s1_transformed)
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(s1_transformed, dim=1) + \
                  torch.sum(s2_transformed, dim=1)
        return z, log_det

    def inverse(self, z):
        lower, upper = z[:,:self.dim // 2], z[:,self.dim // 2:]
        t2_transformed = self.t2(upper)
        s2_transformed = self.s2(upper)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        t1_transformed = self.t1(lower)
        s1_transformed = self.s1(lower)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + \
                  torch.sum(-s2_transformed, dim=1)
        return x, log_det

##################################################### NSF_AR
class NSF_AR(nn.Module):
    """
    Neural spline flow, auto-regressive.

    [Durkan et al. 2019]
    """
    def __init__(self, dim, hidden_dim, activation, K = 5, B = 3, base_network = FCNN):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.layers = nn.ModuleList()
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        for i in range(1, dim):
            self.layers += [base_network(i, 3 * K - 1, hidden_dim, activation)]
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.init_param, - 1 / 2, 1 / 2)

    def forward(self, x):
        z = torch.zeros_like(x).to(x)
        log_det = torch.zeros(z.shape[0])
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            z[:, i], ld = unconstrained_RQS(
                x[:, i], W, H, D, inverse=False, tail_bound=self.B)
            log_det += ld
        return z, log_det

    def inverse(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(x.shape[0]).to(z)
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim = 1)
            W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            x[:, i], ld = unconstrained_RQS(
                z[:, i], W, H, D, inverse = True, tail_bound = self.B)
            log_det += ld
        return x, log_det

##################################################### NSF_CL
class NSF_CL(nn.Module):
    """
    Neural spline flow, coupling layer.

    [Durkan et al. 2019]
    """
    def __init__(self, dim, hidden_dim, activation, K = 5, B = 3, base_network = FCNN):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.f1 = base_network(dim // 2, (3 * K - 1) * dim // 2, hidden_dim, activation)
        self.f2 = base_network(dim // 2, (3 * K - 1) * dim // 2, hidden_dim, activation)

    def forward(self, x):
        log_det = torch.zeros(x.shape[0]).to(x)
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
        out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(
            lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower, upper], dim = 1), log_det

    def inverse(self, z):
        log_det = torch.zeros(z.shape[0]).to(z)
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(
            lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim = 1)
        out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim = 2)
        W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(
            upper, W, H, D, inverse = True, tail_bound = self.B)
        log_det += torch.sum(ld, dim = 1)
        return torch.cat([lower, upper], dim = 1), log_det

################################### The NF model
class NFModel(nn.Module):

    def __init__(self, dim, hidden_dim, activation, flow_type:str, 
                 num_flows:int, device='cuda:0', **kwrds):
        super().__init__()
        # The prior distribution (i.e., the latent distribution)
        self.prior = MultivariateNormal(torch.zeros(dim).to(device), 
                                        torch.eye(dim).to(device))
        # The flow model
        self.flow_net = []
        for _ in range(num_flows):
            flow_layer = eval(flow_type)
            self.flow_net.append(flow_layer(dim, hidden_dim, activation, **kwrds))
        self.flow_net = nn.Sequential(*self.flow_net)

    def forward(self, x):
        '''Map x to z
        Input:
            x: size(?, 2)
        '''
        batch_size = x.shape[0]
        log_det = torch.zeros(batch_size).to(x)
        #
        for flow in self.flow_net:
            x, ld = flow.forward(x)
            log_det += ld
        #
        z, prior_logprob = x, self.prior.log_prob(x)
        
        return z, prior_logprob, log_det

    def inverse(self, z):
        '''Map z back to x
        Input:
            z: size(?, ?)
        '''
        batch_size = z.shape[0]
        log_det = torch.zeros(batch_size).to(z)
        #
        for flow in self.flow_net[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        #
        x = z
        
        return x, log_det

    def sample(self, n_samples):
        '''Sample z0 from prior discribution
        '''
        z = self.prior.sample((n_samples,))
        x, _ = self.inverse(z)
        
        return x
    
if __name__=='__main__':
    from torchsummary import summary
    #
    dim = 128
    model = NFModel(dim=dim, hidden_dim=20, activation='SiLU',
                    flow_type='NSF_CL', num_flows=1, device='cpu')
    summary(model, input_size=(dim,), device='cpu')