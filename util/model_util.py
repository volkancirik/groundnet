#from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.init import xavier_uniform
from torch.nn import Dropout
from torch.nn._functions.dropout import Dropout

def init_forget(l):
  for names in l._all_weights:
    for name in filter(lambda n: "bias" in n,  names):
      bias = getattr(l, name)
      n = bias.size(0)
      start, end = n//4, n//2
      bias.data[start:end].fill_(1.)

class WordDropout(nn.Module):
  """
  Drop individual words instead of embedding units
  """
  def __init__(self, p = 0.5):
    super(WordDropout, self).__init__()
    if p < 0 or p > 1:
      raise ValueError("dropout probability has to be between 0 and 1, "
                       "but got {}".format(p))
    self.p = p
  def _make_noise(self, input):
    noise = makevar(np.ones((1,input.size(0))), numpy_var = True)
    return noise

  def forward(self, input):
    output = input
    if self.p > 0 and self.training and input.size(0) > 1:
      self.noise = self._make_noise(input) * (1 - self.p)
      self.noise = self.noise.bernoulli().t()
      bern_sum   = self.noise.sum()
      self.noise = self.noise.expand_as(input)
      if bern_sum.data[0] == 0:
        self.noise = makevar(np.ones((input.size(0),input.size(1))), numpy_var = True)
      output = input.mul(self.noise)
    return output

    def backward(self, grad_output):
      if self.p > 0 and self.training:
        return grad_output.mul(self.noise)
      else:
        return grad_output

def lamm(pred,ygold):
  '''
  loss augmented max margin
  '''
  s,l  = pred
  l    = makevar(np.array(l),numpy_var = True).view(1,s.size(1))
  mask = np.ones((1,l.size(1)))
  np.put(mask,ygold,0.0)
  m    = makevar(mask,numpy_var = True)
  loss = 0
  for gold in ygold:
    loss += torch.max(torch.clamp(((l + s) - (s[0,gold]).expand_as(l))*m, 0))
  return loss

def mbr(pred,ygold):
  '''
  minimum bayes risk
  '''
  s,l  = pred
  l    = makevar(np.array(l),numpy_var = True).view(1,s.size(1))
  loss = (torch.exp(s) * l).sum()
  return loss

def get_n_params(model, verbose = False):
  n = 0
  for param in model.parameters():
    if param.requires_grad:
      if verbose:
        print(param.size())
      n += reduce(lambda x,y:x*y,param.size())
  return n

def extract_order_feats(spat_feats, cuda = True):

  ### 0 x1 
  ### 1 y1
  ### 2 x2
  ### 3 y2
  ### 4 box_size
  order_feats = []
  for dim in xrange(5):
    order_feats.append(np.argsort(-(spat_feats.cpu().data.numpy())[:,dim]))
  order_feats = np.transpose(np.stack(tuple(order_feats)))
  order_feats = makevar(order_feats, numpy_var = True, cuda = cuda)
  order_spat = torch.cat((order_feats,spat_feats),1)
  return order_spat


def get_box_feats(boxes,CNN,
                  cuda = True, bname = 'rcnn', sname = 'spat',
                  convert_box = True, convert_spat = True, volatile = False):
  box_feats  = []
  spat_feats = []
  for box in boxes:
    box_feats.append(CNN[bname][box])
    spat_feats.append(CNN[sname][box])
  if convert_box:
    box_feats  = makevar(np.stack(tuple(box_feats)), numpy_var = True, cuda = cuda, volatile = volatile)
  else:
    box_feats  = np.stack(tuple(box_feats))
  if convert_spat:
    spat_feats = makevar(np.stack(tuple(spat_feats)), numpy_var = True, cuda = cuda, volatile = volatile)
  else:
    spat_feats = np.stack(tuple(spat_feats))
  return box_feats, spat_feats

def printVec(vec, logprob = False, width = 3):
  r = ""
  try:
    if logprob == True:
      vec = torch.exp(vec)
    r = " ".join(["{:.{width}f}".format(float(v), width = width) for v in vec.data[0]])
  except:
    print "ERR:",vec
    pass
  return r

def makevar(x, numpy_var = False, cuda = True, volatile = False):
  '''
  makes a variable from x. use numpy_var=True if x is already a numpy var
  '''
  if numpy_var:
    v = torch.from_numpy(x).float()
  else:
    v = torch.from_numpy(np.array([x]))
  if cuda:
    return Variable(v.cuda(), volatile = volatile)
  return Variable(v, volatile = volatile)

def zeros(dim, cuda = True):
  '''
  zero variable of dimension dim
  '''
  v = torch.zeros(dim)
  if cuda:
    return Variable(v.cuda())
  return Variable(v)

def outer(v1,v2):
  '''
  takes an outer product of vector v1 and a *matrix* v2
  returns a matrix of size v2.shape[0], (v1.shape[1]+1*(v2.shape[1]+1)
  '''
  v1 = torch.cat([makevar(np.ones((1,1)),numpy_var=True),v1],1)
  v2 = torch.cat([makevar(np.ones((v2.size(0),1)),numpy_var=True),v2],1)

  return torch.ger(v2.view(v2.size(1)*v2.size(0),),v1.view(v1.size(1),)).view(v2.size(0),-1)

def weight_init(m):
  if len(m.size()) == 2 and m.requires_grad:
    xavier_uniform(m)

