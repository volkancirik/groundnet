#from __future__ import absolute_import
import torch

from models import groundnet, cmn, cmn_loc, groundnet_flex_all, groundnet_flex_rel, dan, cmn_bow, cmn_noatt, treernn, box_filter, box_mlp, cmn_loc_deep
from util.model_util import *

def get_model(vocabs, config):
  vectors, w2i, p2i, n2i, i2w, i2p, i2n = vocabs
  config['word_dim'] = vectors.shape[1]
  if config['model'] == 'groundnet':
    net = groundnet.GroundNET(w2i, p2i, n2i, config)
  elif config['model'] == 'groundnetflexall':
    net = groundnet_flex_all.GroundNETflexALL(w2i, p2i, n2i, config)
  elif config['model'] == 'groundnetflexrel':
    net = groundnet_flex_rel.GroundNETflexREL(w2i, p2i, n2i, config)
  elif config['model'] == 'cmn':
    net = cmn.CMN(w2i, p2i, n2i, config)
  elif config['model'] == 'cmn_loc':
    net = cmn_loc.CMN_LOC(w2i, p2i, n2i, config)
  elif config['model'] == 'cmn_loc_deep':
    net = cmn_loc_deep.CMN_LOC_DEEP(w2i, p2i, n2i, config)
  elif config['model'] == 'cmn_bow':
    net = cmn_bow.CMN_BOW(w2i, p2i, n2i, config)
  elif config['model'] == 'cmn_noatt':
    net = cmn_noatt.CMN_NOATT(w2i, p2i, n2i, config)
  elif config['model'] == 'box_filter':
    net = box_filter.BOX_FILTER(w2i, p2i, n2i, config)
  elif config['model'] == 'box_mlp':
    net = box_mlp.BOX_MLP(w2i, p2i, n2i, config)
  elif config['model'] == 'dan':
    net = dan.DAN(w2i, p2i, n2i, config)
  elif config['model'] == 'treernn':
    net = treernn.ReferNet(w2i, p2i, n2i, config)
  else:
    raise NotImplementedError()
  if config['model'] == 'treernn':
    net.txt_net.Wwrd.weight.data.copy_(torch.from_numpy(vectors).cuda())
  else:
    net.Wwrd.weight.data.copy_(torch.from_numpy(vectors).cuda())
  if not config['finetune']:
    if config['model'] == 'maxgroundnet':
      net.Gnet.Wwrd.weight.requires_grad=False
    elif config['model'] == 'treernn':
      net.txt_net.Wwrd.weight.requires_grad=False
    else:
      net.Wwrd.weight.requires_grad=False
  net.cuda()
  for param in net.parameters():
    weight_init(param)
  return net
