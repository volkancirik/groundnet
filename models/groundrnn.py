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

class GroundRNN(nn.Module):
  def __init__(self, wrd_vocab, pos_vocab, non_vocab, config):
    super(GroundRNN, self).__init__()
    self.config= config
    self.dropout = self.config['dropout']
    self.hdim = self.config['n_hidden']
    self.layer = self.config['n_layer']
    self.wdim = self.config['word_dim']
    self.feat_box  = self.config['feat_box']
    self.use_outer = self.config['use_outer']
    self.fusion    = self.config['fusion']
    self.debug     = self.config['debug']
    self.encoder   = self.config['encoder']
    self.only_spatial = self.config['only_spatial']

    self.Wwrd = nn.Embedding(len(wrd_vocab), self.wdim)
    self.w2i  = wrd_vocab

    self.Wbox = nn.Linear(self.feat_box, self.wdim)

    if self.use_outer:
      if self.only_spatial:
        self.Wrbox= nn.Linear(5*2 + ((5+1)**2), self.wdim)
      else:
        self.Wrbox= nn.Linear(self.feat_box*2 + ((5+1)**2), self.wdim)
    else:
      if self.only_spatial:
        self.Wrbox= nn.Linear(5*2, self.wdim)
      else:
        self.Wrbox= nn.Linear(self.feat_box*2, self.wdim)

    self.SMAX = nn.Softmax()
    self.LSMAX = nn.LogSoftmax()
    self.SIGM = nn.Sigmoid()
    self.RELU = nn.ReLU()
    self.TANH = nn.Tanh()
    self.DROP = nn.Dropout(self.dropout)
    self.WDROP= WordDropout(self.dropout)

    if self.fusion == 'concat':
      out0_dim = self.wdim * 2
    else:
      out0_dim = self.wdim
    if self.layer == 1 and self.fusion == 'concat':
      out1_dim = self.wdim * 2
    else:
      out1_dim = self.wdim

    self.Wrel0 = nn.Linear(out0_dim, self.wdim)
    self.Wrel1 = nn.Linear(out1_dim, 1)

    self.Wout0 = nn.Linear(out0_dim, self.wdim)
    self.Wout1 = nn.Linear(out1_dim, 1)

    self.Wscr = nn.Linear(self.hdim*4, 1)

    if self.encoder == "lstm":
      self.rnn0 = nn.LSTM(input_size = self.wdim  ,hidden_size = self.hdim, num_layers = 1,bidirectional = True, dropout = self.dropout)
      self.rnn1 = nn.LSTM(input_size = self.hdim*2,hidden_size = self.hdim, num_layers = 1,bidirectional = True, dropout = self.dropout, bias = False)
      init_forget(self.rnn0)
      init_forget(self.rnn1)

    elif self.encoder == "gru":
      self.rnn0 = nn.GRU(input_size = self.wdim  ,hidden_size = self.hdim, num_layers = 1,bidirectional = True, dropout = self.dropout)
      self.rnn1 = nn.GRU(input_size = self.hdim*2,hidden_size = self.hdim, num_layers = 1,bidirectional = True, dropout = self.dropout)
    else:
      raise NotImplementedError()
    self.h00 = makevar(np.zeros((2,1,self.hdim)),numpy_var = True)
    self.c00 = makevar(np.zeros((2,1,self.hdim)),numpy_var = True)
    self.h01 = makevar(np.zeros((2,1,self.hdim)),numpy_var = True)
    self.c01 = makevar(np.zeros((2,1,self.hdim)),numpy_var = True)


  def ENCODE(self, words, orig_tree = None):

    word_rep = self.WDROP(self.Wwrd(makevar(words)).squeeze(0))
    word_seq = word_rep.view(word_rep.size(0),1,word_rep.size(1))

    if self.encoder == "lstm":
      output0, (ht0, ct0) = self.rnn0(word_seq, (self.h00, self.c00))
      output1, (ht1, ct1) = self.rnn1(output0, (self.h01, self.c01))
    else:
      output0, ht0 = self.rnn0(word_seq,self.h00)
      output1, ht1 = self.rnn1(output0, self.h01)

    outputs = torch.cat( [output0.view(output0.size(0),-1),output1.view(output1.size(0),-1)],1)
    scores    = self.Wscr(self.DROP(outputs)).t()
    attention = self.SMAX(scores).t()
    weighted = attention.expand_as(word_rep) * word_rep
    waverage = torch.sum(weighted, 0)

    del word_rep

    return waverage, attention

  def LOC(self, t_txt, box):
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

    if self.layer == 2:
      norm = self.Wout0(norm)
    score = self.Wout1(norm).t()

    prob  = self.LSMAX(score).add(EPS)

    if self.debug:
      print "LOC>prob:",printVec(prob,logprob = True)
    return prob

  def REL(self, t_txt, box1, box2):
    if self.only_spatial:
      conc = torch.cat((box1[:,-5:].expand_as(box2[:,-5:]),box2[:,-5:]),1)
    else:
      conc = torch.cat((box1.expand_as(box2),box2),1)

    if self.use_outer:
      qkern0 = outer(box1[:,-5:].view(1,-1),box2[:,-5:])
      conc = torch.cat((conc,qkern0),1)

    spat = self.Wrbox(conc)
    if self.fusion == 'mul':
      prod = t_txt.expand_as(spat) * spat
      norm = prod / torch.norm(prod,2,1).expand_as(prod)
    elif self.fusion == 'sum':
      norm = t_txt.expand_as(spat) + spat
    elif self.fusion == "concat":
      norm = torch.cat([t_txt.expand_as(spat),spat],1)

    if self.layer == 2:
      norm = self.RELU(self.Wrel0(norm))
    score = self.Wrel1(norm)

    return score

  def SHIFT(self,t_txt, box, p_child):

    A = []
    for i in range(box.size(0)):
      A.append(self.REL(t_txt, box[i].view(1,-1), box))
      A[i][i] = 0.0
    A = torch.cat(A, 1).t()

    prob  = self.LSMAX(torch.mm(torch.exp(p_child), A).add(EPS))

    if self.debug:
      print "SHIFT>child",printVec(p_child, logprob = True)
      for i in range(box.size(0)):
        print "SHIFTA[{}]:".format(i),printVec(A[i])
      print "SHIFT>score:",printVec(prob, logprob = True)

    return prob

  def expr_for_txt(self, tree, box, orig_tree = None, decorate = False):
    assert(not tree.isleaf())
    if len(tree.children) == 1:
      if tree.children[0].isleaf():
        wlist = [ self.w2i.get(w,0) for w in filter(lambda a: a!= '', tree.children[0].label.split('_'))]

        txt, attn_weights   = self.ENCODE(wlist, orig_tree)
        prob  = self.LOC(txt, box)
        if decorate:
          tree._expr = prob
          tree._attn = attn_weights
        return prob
      else:
        if tree.label == 'loc':
          wlist = [ self.w2i.get(w,0) for w in filter(lambda a: a!= '', tree.children[0].label.split('_'))]
        else:
          wlist = [ self.w2i.get(w,0) for w in filter(lambda a: a!= '', tree.label.split('_'))]
        txt, attn_weights = self.ENCODE(wlist)

        p_child = self.expr_for_txt(tree.children[0],box, orig_tree, decorate)
        prob    = self.SHIFT(txt,box,p_child)
        if decorate:
          tree._expr = prob
          tree._attn = attn_weights
        return prob

    assert(len(tree.children) == 2)

    p_l = self.expr_for_txt(tree.children[0], box, orig_tree, decorate)
    p_r = self.expr_for_txt(tree.children[1], box, orig_tree, decorate)

    prob = (p_l + p_r)
    norm = torch.exp(prob)
    norm = norm / torch.norm(norm,1,1).expand_as(norm)
    prob = torch.log(norm).add(EPS)

    if decorate:
      tree._attn = None
      tree._expr = prob

    if self.debug:
      print "AND>",printVec(prob, logprob = True)
    return prob

  def forward(self, tree, box, orig_tree = None, decorate = False):
    return self.expr_for_txt(tree, box, orig_tree, decorate)
