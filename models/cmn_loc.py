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


class CMN_LOC(nn.Module):
  def __init__(self, wrd_vocab, pos_vocab, non_vocab, config, cnn_feat = 4096):
    super(CMN_LOC, self).__init__()
    self.config= config
    self.dropout = self.config['dropout']
    self.hdim = self.config['n_hidden']
    self.wdim = self.config['word_dim']
    self.feat_box  = self.config['feat_box']
    self.cnn_feat  = cnn_feat
    self.use_outer = self.config['use_outer']
    self.fusion    = self.config['fusion']
    self.debug     = self.config['debug']
    self.evaluate  = False

    self.Wwrd = nn.Embedding(len(wrd_vocab), self.wdim)
    self.w2i  = wrd_vocab

    self.SMAX = nn.Softmax()
    self.LSMAX = nn.LogSoftmax()
    self.SIGM = nn.Sigmoid()
    self.RELU = nn.ReLU()
    self.TANH = nn.Tanh()
    self.DROP = nn.Dropout(self.dropout)

    self.WscrSUB = nn.Linear(self.hdim*4, 1)

    self.rnn0 = nn.LSTM(input_size = self.wdim  ,hidden_size = self.hdim, num_layers = 1,bidirectional = True, dropout = self.dropout)
    self.rnn1 = nn.LSTM(input_size = self.hdim*2,hidden_size = self.hdim, num_layers = 1,bidirectional = True, dropout = self.dropout, bias = False)
    init_forget(self.rnn0)
    init_forget(self.rnn1)

    if self.use_outer:
      self.Wbox= nn.Linear((self.feat_box - self.cnn_feat + 1)**2 + self.feat_box, self.wdim)
    else:
      self.Wbox = nn.Linear(self.feat_box, self.wdim)

    self.Wout0  = nn.Linear(self.wdim, 1)


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

  def LOC(self, t_txt, box):
    if self.use_outer:
      O = []
      for i in range(box.size(0)):
        qkern0  = outer(box[i,self.cnn_feat:].contiguous().view(1,-1),box[i,self.cnn_feat:].view(1,-1))
        O.append(qkern0)

      box_out = torch.cat((box,torch.cat(O,0)),1)
      t_box   = self.Wbox(box_out)
    else:
      t_box = self.Wbox(box)

    t_sum = t_txt.expand_as(t_box) * t_box
    norm  = t_sum / torch.norm(t_sum,2,1).expand_as(t_sum)

    score = self.Wout0(norm)
    if self.debug:
      print
      print "LOC>norm:",printVec(norm.t())
      print "LOC>score:",printVec(score.t())
    return score

  def forward(self, txt, box, orig_tree = None):

    qsub = self.ENCODE(txt)
    score_sub = self.LOC(qsub,box)
    lprob     = self.LSMAX(score_sub.t().add(EPS)) ###

    if self.debug:
      print
      print "="*20
      print "D>sub",printVec(score_sub.t())
      print "="*20,
      print "D>lprob",printVec(lprob, logprob = True)
      print "*"*20
    # if self.evaluate:
    #   return lprob, [], [], []
    return lprob
