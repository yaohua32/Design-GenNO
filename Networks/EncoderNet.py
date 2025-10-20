# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-10-15 16:27:27 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-10-15 16:27:27 
#  */
import torch 
import torch.nn as nn
try: 
    from FunActivation import FunActivation
    from FCNet import FCNet
    from CNNet import CNNet1d, CNNet2d
except:
    from .FunActivation import FunActivation
    from .FCNet import FCNet
    from .CNNet import CNNet1d, CNNet2d

############################################ FCNet
class EncoderFCNet(nn.Module):

    def __init__(self, layers_list:list, activation, dtype=None) -> None:
        super(EncoderFCNet, self).__init__()
        # Activation
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        # Network Sequential
        self.net = FCNet(layers_list, activation, dtype)

    def forward(self, x):
        '''
        Input: 
            x: size(n_batch, my*mx, in_size)
        Return:
            beta: size(n_batch, n_latent)
        '''
        # size(n_batch, my*mx, in_size) -> (n_batch, my*mx*in_size)
        x = x.reshape(x.shape[0], -1)
        # size(n_batch, my*mx*in_size) -> (n_batch, n_latent)
        x = self.net(x)

        return x

class EncoderFCNet_VAE(nn.Module):

    def __init__(self, layers_list:list, activation, dtype=None) -> None:
        super(EncoderFCNet_VAE, self).__init__()
        # Activation
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        # Network Sequential
        self.net_mu = FCNet(layers_list, activation, dtype)
        self.net_log_var = FCNet(layers_list, activation, dtype)

    def reparam(self, mu, log_var):
        ''' '''
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(log_var)
        
        return mu + std * eps

    def forward(self, x):
        '''
        Input: 
            x: size(n_batch, my*mx, in_size)
        Return:
            beta: size(n_batch, n_latent)
        '''
        # size(n_batch, my*mx, in_size) -> (n_batch, in_size*my*mx)
        x = x.reshape(x.shape[0], -1)
        # size(n_batch, in_size*my*mx) -> (n_batch, n_latent)
        mu = self.net_mu(x)
        log_var = self.net_log_var(x)

        return self.reparam(mu, log_var), mu, log_var

############################################ CNNet1d
class EncoderCNNet1d(nn.Module):

    def __init__(self, conv_arch:list, fc_arch:list, 
                 activation_conv, activation_fc,
                 dtype=None, kernel_size=5, stride=3) -> None:
        super(EncoderCNNet1d, self).__init__()
        #
        self.in_channel = conv_arch[0]
        self.net = CNNet1d(conv_arch=conv_arch, fc_arch=fc_arch, 
                         activation_conv=activation_conv,
                         activation_fc=activation_fc,
                         kernel_size=kernel_size, 
                         stride=stride,
                         dtype=dtype)
    
    def forward(self, x):
        '''
        Input:
            x: size(n_batch, mesh_size, in_channel)
        Return:
            beta: size(n_batch, n_latent)
        '''
        # size(n_batch, mesh_size, in_channel) -> (n_batch, in_channel, mesh_size)
        x = x.permute(0, 2, 1)
        # size(batch_size, fc_arch[-1])
        x = self.net(x)

        return x

############################################ CNNet2d
class EncoderCNNet2d(nn.Module):

    def __init__(self, conv_arch:list, fc_arch:list, 
                 activation_conv, activation_fc,
                 nx_size:int, ny_size:int, 
                 dtype=None, kernel_size=(5,5), stride=3) -> None:
        super(EncoderCNNet2d, self).__init__()
        #
        self.in_channel = conv_arch[0]
        self.nx_size = nx_size 
        self.ny_size = ny_size
        self.net = CNNet2d(conv_arch=conv_arch, fc_arch=fc_arch, 
                         activation_conv=activation_conv,
                         activation_fc=activation_fc,
                         kernel_size=kernel_size, 
                         stride=stride,
                         dtype=dtype)
    
    def forward(self, x):
        '''
        Input:
            x: size(n_batch, my*mx, in_channel)
        Return:
            beta: size(n_batch, n_latent)
        '''
        # size(n_batch, my*mx, in_channel) -> (n_batch, in_channel, my*mx)
        x = x.permute(0, 2, 1)
        # size(n_batch, in_channel, my*mx) -> (n_batch, in_channel, my, mx)
        x = x.reshape(-1, self.in_channel, self.ny_size, self.nx_size)
        # size(batch_size, fc_arch[-1])
        x = self.net(x)

        return x
    
if __name__=='__main__':
    from torchsummary import summary
    # ################################################### The FCNet
    # x = torch.randn((10, 5*5, 1))
    # #
    # layers, act = [5*5, 256, 128, 64, 8], 'Tanh'
    # # demo = EncoderFCNet(layers_list=layers, activation=act)
    # demo = EncoderFCNet_VAE(layers_list=layers, activation=act)
    # summary(demo, (5*5,1), device='cpu')
    # #
    # beta, _, _ = demo(x)
    # print('The shape of latent_var:', beta.shape)
    ################################################### The CNNet1d
    x = torch.randn((10, 64*64, 1))
    #
    conv_arch, activation_conv = [1, 64, 64, 64], 'Tanh'
    fc_arch, activation_fc = [64*150, 64, 64], 'Tanh'
    demo = EncoderCNNet1d(conv_arch=conv_arch, activation_conv=activation_conv,
                          fc_arch=fc_arch, activation_fc=activation_fc, 
                          kernel_size=5, stride=3)
    summary(demo, (64*64, 1), device='cpu')
    #
    beta = demo(x)
    print('The shape of beta:', beta.shape)
    ################################################### The CNNet2d
    # x = torch.randn((10, 64*64, 1))
    # #
    # conv_arch, activation_conv = [1, 64, 64, 64], 'Tanh'
    # fc_arch, activation_fc = [64*5*5, 64, 64], 'Tanh'
    # demo = EncoderCNNet2d(conv_arch=conv_arch, activation_conv=activation_conv,
    #                    fc_arch=fc_arch, activation_fc=activation_fc,
    #                    nx_size=64, ny_size=64, stride=2)
    # summary(demo, (64*64, 1), device='cpu')
    # #
    # beta = demo(x)
    # print('The shape of beta:', beta.shape)

