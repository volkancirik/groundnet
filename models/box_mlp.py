#from __future__ import absolute_import
"""
TODO
- add references & descriptions to models
"""
__author__ = "volkan cirik"

import numpy as np
import torch
import torch.nn as nn

from util.model_util import makevar, outer, zeros, printVec, weight_init, init_forget
EPS=1e-10


class BOX_MLP(nn.Module):
  def __init__(self, wrd_vocab, pos_vocab, non_vocab, config):
    super(BOX_MLP, self).__init__()
    self.config= config
    self.dropout = self.config['dropout']
    self.hdim = self.config['n_hidden']
    self.wdim = self.config['word_dim']
    self.feat_box  = self.config['feat_box']
    self.n_layer  = self.config['n_layer']

    self.use_outer = self.config['use_outer']
    self.fusion    = self.config['fusion']
    self.debug     = self.config['debug']
    self.evaluate  = False

    self.Wwrd = nn.Embedding(len(wrd_vocab), self.wdim)
    self.w2i  = wrd_vocab

    self.SMAX = nn.Softmax()
    self.SIGM = nn.Sigmoid()
    self.RELU = nn.ReLU()
    self.TANH = nn.Tanh()
    self.DROP = nn.Dropout(self.dropout)

    self.WscrSUB = nn.Linear(self.hdim*4, 1)

    self.rnn0 = nn.LSTM(input_size = self.wdim  ,hidden_size = self.hdim, num_layers = 1,bidirectional = True, dropout = self.dropout)
    self.rnn1 = nn.LSTM(input_size = self.hdim*2,hidden_size = self.hdim, num_layers = 1,bidirectional = True, dropout = self.dropout, bias = False)
    init_forget(self.rnn0)
    init_forget(self.rnn1)

    self.Wbox = nn.Linear(self.feat_box, self.wdim)

    mlist = []
    for i in range(self.n_layer):
      if i == 0:
        mlist.append(nn.Linear(self.feat_box + self.wdim, self.hdim))
      elif i == self.n_layer-1:
        mlist.append(nn.Linear(self.hdim, 1))
      else:
        mlist.append(nn.Linear(self.hdim, self.hdim))
    self.Wff = nn.ModuleList(mlist)

  def ENCODE(self, words, orig_tree = None):
    word_rep = self.Wwrd(makevar(words)).squeeze(0)
    word_seq = word_rep.view(word_rep.size(0),1,word_rep.size(1))

    h00 = c00 = makevar(np.zeros((2,1,self.hdim)),numpy_var = True)
    h01 = c01 = makevar(np.zeros((2,1,self.hdim)),numpy_var = True)

    output0, (ht0, ct0) = self.rnn0(word_seq, (h00, c00))
    output1, (ht1, ct1) = self.rnn1(output0, (h01, c01))
    outputs = torch.cat( [output0.view(output0.size(0),-1),output1.view(output1.size(0),-1)],1)

    scores    = self.WscrSUB(self.DROP(outputs)).t()
    att_sub   = self.SMAX(scores).t()
    weighted  = att_sub.expand_as(word_rep) * word_rep
    qsub      = torch.sum(weighted, 0)

    if self.debug:
      print "ENC>att_sub:",printVec(att_sub.t())

    return qsub

  def forward(self, txt, box, orig_tree = None):

    qsub = self.ENCODE(txt)
    layer_in = torch.cat([qsub,box],1)
    for i,ff in enumerate(self.Wff):
      if i == len(self.Wff)-1:
        break
      layer_in = self.RELU(self.DROP(ff(layer_in)))
    prob =  self.SIGM(self.Wff[-1](layer_in))

    return prob

