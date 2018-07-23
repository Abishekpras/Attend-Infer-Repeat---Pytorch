from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import grid_sample, affine_grid
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal

from modules import ObjectEncoder, ObjectDecoder, Latent_Predictor,\
                    attentive_stn_encode, attentive_stn_decode, \
                    lay_obj_in_image, compute_hidden_state

seed = 1
log_interval = 10
epochs = 10
batch_size = 64
use_cuda = False

torch.manual_seed(seed)

# TODO : ADD cuda device compatible code
device = torch.device("cuda" if use_cuda else "cpu") 


class AIR(nn.Module):
    def __init__(self):
        super(AIR, self).__init__()
        self.obj_encode = ObjectEncoder()
        self.obj_decode = ObjectDecoder()
        self.predict = Latent_Predictor()
        self.rnn = nn.LSTMCell((50*50)+1+3+50, 256)
        #self.training = True

    def encode(self, x, z_where_prev, z_what_prev, z_pres_prev, h_prev, c_prev):
        
        kld_loss = 0
        h, c = compute_hidden_state(self.rnn, x, z_where_prev, z_what_prev, z_pres_prev, h_prev, c_prev)
        
        z_pres_proba, z_where_mu, z_where_sd = self.predict(h)
        kld_loss += self.latent_loss(z_where_mu, z_where_sd)
        
        z_pres = Independent( Bernoulli(z_pres_proba * z_pres_prev), 1 ).sample()
        
        z_where = self._reparameterized_sample(z_where_mu, z_where_sd)
        
        x_att = attentive_stn_encode(z_where, x)
        
        z_what_mu, z_what_sd = self.obj_encode(x_att)
        kld_loss += self.latent_loss(z_what_mu, z_what_sd)
        
        z_what = self._reparameterized_sample(z_what_mu, z_what_sd)
        
        return z_where,z_what,z_pres,h,c,kld_loss

    def _reparameterized_sample(self, mean, std):
        if self.training:
            eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(eps).to(device)
            return eps.mul(std).add_(mean)
        else:
            return mean
        
    def latent_loss(self, z_mean, z_sig):
        mean_sq = z_mean * z_mean
        sig_sq = z_sig * z_sig
        return 0.5 * torch.mean(mean_sq + sig_sq - torch.log(sig_sq) - 1)

    def decode(self, x_prev, z_pres, z_where, z_what):

        y_att = self.obj_decode(z_what)

        y = attentive_stn_decode(z_where, y_att)
        
        x = lay_obj_in_image(x_prev, y, z_pres)

        return x

    def forward(self, x):
        kld_loss = 0
        nll_loss = 0

        #h = Variable(torch.zeros(3, x.size(1), 50))
        #z_where, z_what, z_pres = self.latent_priors()
        
        n = x.size(0)
        h=Variable(torch.zeros(n, 256)).to(device)
        c=Variable(torch.zeros(n, 256)).to(device)
        z_pres=Variable(torch.ones(n, 1)).to(device)
        z_where=Variable(torch.zeros(n, 3)).to(device)
        z_what=Variable(torch.zeros(n, 50)).to(device)
        
        y = torch.zeros(x.shape)
        for t in range(3):
            z_where,z_what,z_pres,h,c,loss = self.encode(x, z_where, z_what, z_pres, h, c)
            y = self.decode(y, z_pres, z_where, z_what)
        
            #nll_loss += nn.MSELoss(y, x)
            nll_loss += nn.functional.binary_cross_entropy(y, x, size_average=False)
            kld_loss += loss

        return kld_loss, nll_loss
    
    def latent_priors(self):
        scale_prior_mu, scale_prior_sd = 3.0, 0.1
        pos_prior_mu, pos_prior_sd = 0.0, 1.0
        z_where_mu_prior = nn.Parameter(torch.FloatTensor([scale_prior_mu, pos_prior_mu, pos_prior_mu]).expand(n, -1),
                                             requires_grad=False)
        z_where_sd_prior = nn.Parameter(torch.FloatTensor([scale_prior_sd, pos_prior_sd, pos_prior_sd]).expand(n, -1),
                                             requires_grad=False)
        z_what_mu_prior = nn.Parameter(torch.zeros(50))
        z_what_sd_prior = nn.Parameter(torch.ones(50))
        
        z_where = Independent( Normal(z_where_mu_prior, z_where_sd_prior), 1 ).sample()
        z_what = Independent( Normal(z_what_mu_prior, z_what_sd_prior), 1 ).sample()
        
        z_pres = torch.ones(n, 1)
        
        return z_where, z_what, z_pres
