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


class CMN(nn.Module):
  def __init__(self, wrd_vocab, pos_vocab, non_vocab, config):
    super(CMN, self).__init__()
    self.config= config
    self.dropout = self.config['dropout']
    self.hdim = self.config['n_hidden']
    self.wdim = self.config['word_dim']
    self.feat_box  = self.config['feat_box']

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
    self.WscrOBJ = nn.Linear(self.hdim*4, 1)
    self.WscrREL = nn.Linear(self.hdim*4, 1)

    self.rnn0 = nn.LSTM(input_size = self.wdim  ,hidden_size = self.hdim, num_layers = 1,bidirectional = True, dropout = self.dropout)
    self.rnn1 = nn.LSTM(input_size = self.hdim*2,hidden_size = self.hdim, num_layers = 1,bidirectional = True, dropout = self.dropout, bias = False)
    self.h00 = makevar(np.zeros((2,1,self.hdim)),numpy_var = True)
    self.c00 = makevar(np.zeros((2,1,self.hdim)),numpy_var = True)
    self.h01 = makevar(np.zeros((2,1,self.hdim)),numpy_var = True)
    self.c01 = makevar(np.zeros((2,1,self.hdim)),numpy_var = True)

    init_forget(self.rnn0)
    init_forget(self.rnn1)

    self.Wbox = nn.Linear(self.feat_box, self.wdim)

    if self.fusion == 'concat':
#      self.Wout0  = nn.Linear(2*self.wdim, self.wdim)
      self.Wout0  = nn.Linear(2*self.wdim, 1)
#      self.Wrel0 = nn.Linear(2*self.wdim, self.wdim)
    else:
#      self.Wout0  = nn.Linear(self.wdim, self.wdim)
      self.Wout0  = nn.Linear(self.wdim, 1)
#      self.Wrel0 = nn.Linear(self.wdim, self.hdim)
#    self.Wout1 = nn.Linear(self.wdim, 1)

    # if self.use_outer:
    #   self.Wrbox= nn.Linear(self.feat_box*2 + ((5+1)**2), self.wdim)
    # else:
    #   self.Wrbox= nn.Linear(self.feat_box*2, self.wdim)
    self.Wrbox= nn.Linear(5*2, self.wdim)
    self.Wrel1 = nn.Linear(self.wdim, 1)


  def ENCODE(self, words, orig_tree = None):
    word_rep = self.Wwrd(makevar(words)).squeeze(0)
    word_seq = word_rep.view(word_rep.size(0),1,word_rep.size(1))

    output0, (ht0, ct0) = self.rnn0(word_seq, (self.h00, self.c00))
    output1, (ht1, ct1) = self.rnn1(output0, (self.h01, self.c01))
    outputs = torch.cat( [output0.view(output0.size(0),-1),output1.view(output1.size(0),-1)],1)

    scores    = self.WscrSUB(self.DROP(outputs)).t()

    att_sub   = self.SMAX(scores).t()
    weighted  = att_sub.expand_as(word_rep) * word_rep
    qsub      = torch.sum(weighted, 0)

    scores    = self.WscrOBJ(self.DROP(outputs)).t()
    att_obj   = self.SMAX(scores).t()
    weighted  = att_obj.expand_as(word_rep) * word_rep
    qobj      = torch.sum(weighted, 0)

    scores    = self.WscrREL(self.DROP(outputs)).t()
    att_rel   = self.SMAX(scores).t()
    weighted  = att_rel.expand_as(word_rep) * word_rep
    qrel      = torch.sum(weighted, 0)

    if self.debug:
      print "ENC>att_sub:",printVec(att_sub.t())
      print "ENC>att_obj:",printVec(att_obj.t())
      print "ENC>att_rel:",printVec(att_rel.t())

    return qsub,qobj,qrel

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

    score = self.Wout0(norm)
    if self.debug:
      print
      print "LOC>norm:",printVec(norm.t())
      print "LOC>score:",printVec(score.t())
    return score

  def REL(self, t_txt, box1, box2):
    #conc = torch.cat((box1.expand_as(box2),box2),1)
    conc = torch.cat((box1[:,-5:].expand_as(box2[:,-5:]),box2[:,-5:]),1)
    if self.use_outer:
      qkern0 = outer(box1[:,-5:].view(1,-1),box2[:,-5:])
      conc = torch.cat((conc,qkern0),1)

    spat = self.Wrbox(conc)
    if self.fusion == 'mul':
      prod = t_txt.expand_as(spat) * spat
      norm = prod / torch.norm(prod,2,1).expand_as(prod)
    elif self.fusion == 'sum':
      norm = t_txt.expand_as(spat) + spat
    elif self.fusion == 'concat':
      norm  = torch.cat([t_txt.expand_as(spat),spat],1)
    else:
      raise NotImplementedError()

    score = self.Wrel1(norm)
    #    score = self.Wrel1(self.RELU(self.Wrel0(norm)))
    if self.debug:
      print
      print "REL>norm:",printVec(norm.t())
      print "REL>score:",printVec(score.t())

    return score

  def forward(self, txt, box, orig_tree = None):

    qsub, qobj, qrel = self.ENCODE(txt)
    score_sub = self.LOC(qsub,box)
    score_obj = self.LOC(qobj,box)

    score_rel = []
#    score_rel = makevar(np.zeros((box.size(0),box.size(0))),numpy_var=True)
    for i in range(box.size(0)):
#      score_rel[i] = self.REL(qrel, box[i].view(1,-1), box)
      score_rel.append(self.REL(qrel, box[i].view(1,-1), box))

    score_rel = torch.cat(score_rel, 1).t()
    score_tot = score_sub.t().expand_as(score_rel) + score_obj.expand_as(score_rel) + score_rel
    score_fin = score_tot.max(1)[0].view(1,-1)
    lprob     = self.LSMAX(score_fin).add(EPS)

    if self.debug:
      print 
      print "="*20
      print "D>rel",score_rel
      print "="*20
      print "D>obj",printVec(score_obj.t())
      print "="*20
      print "D>sub",printVec(score_sub.t())
      print "="*20
      print "D>tot",score_tot
      print "="*20
      print "D>max",score_tot.max(1)[0].max(0)[0],score_tot.max(1)[0].max(0)[1]
      print "="*20,
      print "D>fin",printVec(score_fin)
      print "="*20,
      print "D>lprob",printVec(lprob, logprob = True)
      print "*"*20

    # if self.evaluate:
    #   return lprob, score_sub, score_obj, score_rel
    return lprob
