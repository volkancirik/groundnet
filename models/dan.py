#from __future__ import absolute_import
"""
TODO
- add references & descriptions to models
"""
__author__ = "volkan cirik"

import numpy as np
import torch
import torch.nn as nn

from util.model_util import outer, zeros, printVec, weight_init, makevar, WordDropout, init_forget
EPS=1e-10

class DAN(nn.Module):
  def __init__(self, wrd_vocab, pos_vocab, non_vocab, config):
    super(DAN, self).__init__()
    self.config = config
    self.hdim    = self.config['n_hidden']
    self.wdim     = self.config['word_dim']
    self.dropout = self.config['dropout']
    self.n_layer = self.config['n_layer']
    self.feat_box  = self.config['feat_box']
    self.fusion   = self.config['fusion']

    self.Wwrd = nn.Embedding(len(wrd_vocab), self.wdim)
    self.Wff = nn.ModuleList([nn.Linear(self.wdim, self.hdim) if i == 0 else nn.Linear(self.hdim, self.hdim) for i in range(self.n_layer)])
    self.w2i  = wrd_vocab

    self.RELU = nn.ReLU()
    self.TANH = nn.Tanh()
    self.DROP = nn.Dropout(self.dropout)

    self.Wtxt = nn.Linear(self.hdim, self.hdim)
    self.Wbox = nn.Linear(self.feat_box, self.hdim)

    self.LSMAX= nn.LogSoftmax()

    if self.fusion == "concat":
      self.Wout = nn.Linear(2*self.hdim, 1)
    elif self.fusion in set(["sum","mul"]):
      self.Wout = nn.Linear(self.hdim, 1)
    else:
      raise NotImplementedError()

  def ENCODE(self, words):
    word_rep = torch.cat([self.Wwrd(makevar(self.w2i.get(word,0)),) for word in words],0)
    averaged = torch.mean(word_rep, 0)
    for i,ff in enumerate(self.Wff):
      averaged = self.RELU(self.DROP(ff(averaged)))
      if i == len(self.Wff)-1:
        break
    return self.TANH(self.Wff[-1](averaged))

  def forward(self, txt_input, box, orig_tree = None):
    txt   = self.ENCODE(txt_input)
    t_txt = self.Wtxt(txt)
    t_box = self.Wbox(box)

    if self.fusion == 'mul':
      prod = t_txt.expand_as(t_box) * t_box
      norm  = prod / torch.norm(prod,2,1).expand_as(prod)
    elif self.fusion == 'sum':
      norm = t_txt.expand_as(t_box) + t_box
    elif self.fusion == 'concat':
      norm = torch.cat([t_txt.expand_as(t_box),t_box],1)
    else:
      raise NotImplementedError()

    score = self.Wout(self.RELU(self.DROP(norm)))
    pred  = self.LSMAX(torch.t(score)).add(EPS)
    return pred
