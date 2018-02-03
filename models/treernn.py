#from __future__ import absolute_import
"""
- add references & descriptions to models
"""
__author__ = "volkan cirik"

import numpy as np
import torch
import torch.nn as nn

from util.model_util import makevar, outer, zeros, printVec, weight_init, init_forget
EPS=1e-10

class TreeRNN(nn.Module):
  def __init__(self, wrd_vocab, pos_vocab, non_vocab, config):
    super(TreeRNN, self).__init__()
    self.config= config
    self.dropout = self.config['dropout']
    hdim = self.config['n_hidden']
    self.wdim = self.config['word_dim']
    self.hdim = self.wdim + hdim

    self.Wwrd = nn.Embedding(len(wrd_vocab), self.wdim)
    self.Wpos = nn.Embedding(len(pos_vocab), hdim)
    self.Wnon = nn.Embedding(len(non_vocab), self.hdim)

    self.Wcompose = nn.Linear(2*self.hdim, self.hdim)

    self.TANH = nn.Tanh()
    self.DROP = nn.Dropout(self.dropout)
    self.w2i = wrd_vocab
    self.p2i = pos_vocab
    self.n2i = non_vocab

  def expr_for_txt(self, tree, decorate = False):
    assert(not tree.isleaf())
    if len(tree.children) == 1:
      assert(tree.children[0].isleaf())
      pos_emb = self.Wpos(makevar(self.p2i.get(tree.label, 0)))
      wrd_emb = self.Wwrd(makevar(self.w2i.get(tree.children[0].label, 0)))
      return torch.cat((pos_emb, wrd_emb),1)

    assert(len(tree.children) == 2), tree.children[0]
    e_l = self.expr_for_txt(tree.children[0], decorate)
    e_r = self.expr_for_txt(tree.children[1], decorate)
    non = self.Wnon(makevar(self.n2i.get(tree.label, 0)))
    chd = self.Wcompose(self.DROP(torch.cat((e_l, e_r),1)))
    exp = self.TANH(non + chd)

    if decorate:
      tree._expr = exp
    return exp


class ReferNet(nn.Module):
  def __init__(self, wrd_vocab, pos_vocab, non_vocab, config):
    super(ReferNet, self).__init__()

    self.config = config
    self.txt_model= self.config['model']
    self.hdim     = self.config['n_hidden']
    self.wdim     = self.config['word_dim']
    self.dropout  = self.config['dropout']
    self.feat_box = self.config['feat_box']
    self.fusion   = self.config['fusion']

    if self.txt_model == 'treernn':
      self.txt_net = TreeRNN(wrd_vocab, pos_vocab, non_vocab, config)
    else:
      raise NotImplementedError()

    self.Wtxt = nn.Linear(self.wdim + self.hdim , self.hdim)
    self.Wbox = nn.Linear(self.feat_box, self.hdim)

    self.RELU = nn.ReLU()
    self.SMAX = nn.Softmax()
    self.LSMAX= nn.LogSoftmax()
    self.DROP = nn.Dropout(self.dropout)
    if self.fusion == "concat":
      self.Wout = nn.Linear(2*self.hdim, 1)
    elif self.fusion in set(["sum","mul"]):
      self.Wout = nn.Linear(self.hdim, 1)
    else:
      raise NotImplementedError()

  def forward(self, txt_input, box, orig_tree = None):
    if self.txt_model == 'treelstm':
      txt, _   = self.txt_net.expr_for_txt(txt_input)
    else:
      txt     = self.txt_net.expr_for_txt(txt_input)

    t_txt = self.Wtxt(txt)
    t_box = self.Wbox(box)

    if self.fusion == 'mul':
      t_sum = t_txt.expand_as(t_box) * t_box
      norm  = t_sum / torch.norm(t_sum,2,1).expand_as(t_sum)
    elif self.fusion == 'sum':
      norm = t_txt.expand_as(t_box) + t_box
    elif self.fusion == 'concat':
      norm = torch.cat([t_txt.expand_as(t_box),t_box],1)
    else:
      raise NotImplementedError()

    score = self.Wout(self.RELU(self.DROP(norm)))
    pred  = self.LSMAX(torch.t(score)).add(EPS)
    return pred
