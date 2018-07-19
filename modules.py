import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.nn.functional import grid_sample, affine_grid
from matplotlib.pyplot import *

'''
## Attention Window to latent code
## X_att -> z_what
'''
class ObjectEncoder(nn.Module):
    def __init__(self):
        super(ObjectEncoder, self).__init__()
        self.enc = nn.Sequential(nn.Linear(400, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 100))

    def forward(self, data):
        z_what_param = self.enc(data)
        z_what_mu = z_what_param[:, 0:50]
        z_what_sd = nn.Softplus(z_what_param[:, 50:])
        return z_what_mu, z_what_sd


'''
## Reconstruct Attention Window from latent code
## z_what -> Y_att
'''
class ObjectDecoder(nn.Module):
    def __init__(self):
        super(ObjectDecoder, self).__init__()
        self.dec = nn.Sequential(nn.Linear(50, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 400),
                                 nn.Sigmoid())

    def forward(self, z_what):
        return self.dec(z_what)

'''
## RNN hidden state to presence and location
## h -> z_pres, z_where
'''
class Latent_Predictor(nn.Module):
    def __init__(self, ):
        super(Latent_Predictor, self).__init__()
        self.pred = nn.Linear(256, 1+3+3)

    def forward(self, h):
        z_param = self.pred(h)
        z_pres_proba = nn.Sigmoid(z_param[:, 0:1])
        z_where_mu = z_param[:, 1:4]
        z_where_sd = nn.Softplus(z_param[:, 4:])
        return z_pres_proba, z_where_mu, z_where_sd

'''
## Spatial Transformer to shift and scale the reconstructed attention window
## z_where, Y_att -> Y_i
'''
def expand_z_where(z_where):
    # Takes 3-dimensional vectors, and massages them into 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    n = z_where.size(0)
    expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])
    out = torch.cat((torch.zeros([1, 1]).expand(n, 1), z_where), 1)
    return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)

def attentive_stn_decode(z_where, obj):
    n = obj.size(0)
    theta = expand_z_where(z_where)
    grid = affine_grid(theta, torch.Size((n, 1, 50, 50)))
    out = grid_sample(obj.view(n, 1, 20, 20), grid)
    return out.view(n, 50, 50)

'''
## Spatial Transformer to obtain attention window
## z_where, X -> X_att
'''
def z_where_inv(z_where):
    # Take a batch of z_where vectors, and compute their "inverse".
    # That is, for each row compute:
    # [s,x,y] -> [1/s,-x/s,-y/s]
    # These are the parameters required to perform the inverse of the
    # spatial transform performed in the generative model.
    n = z_where.size(0)
    out = torch.cat((torch.ones([1, 1]).type_as(z_where).expand(n, 1), -z_where[:, 1:]), 1)
    out = out / z_where[:, 0:1]
    return out

def attentive_stn_encode(z_where, image):
    n = image.size(0)
    theta_inv = expand_z_where(z_where_inv(z_where))
    grid = affine_grid(theta_inv, torch.Size((n, 1, 20, 20)))
    out = grid_sample(image.view(n, 1, 50, 50), grid)
    return out.view(n, -1)

'''
## Add objects to generate image
## Sum(Y_i) = Y
'''
def lay_obj_in_image(x_prev, y, z_pres):
    return x_prev + y * z_pres.view(-1, 1, 1)

'''
## RNN hidden state from image
## X -> h_i
'''
rnn = nn.LSTMCell(2554, 256)
def compute_hidden_state(data, z_where_prev, z_what_prev, z_pres_prev, h_prev, c_prev):
    rnn_input = torch.cat((data, z_where_prev, z_what_prev, z_pres_prev), 1)
    h, c = rnn(rnn_input, (h_prev, c_prev))
    return h,c
