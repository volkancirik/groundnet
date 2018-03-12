#from __future__ import absolute_import,print_function
"""
Test a model
"""
__author__ = "volkan cirik"

import time
import math
start = time.time()

import random, sys, os, h5py, json

import numpy as np
import cPickle as pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from util.model_util import makevar, printVec, get_n_params, get_box_feats, lamm, mbr
from models.get_model import get_model
from util.arguments import get_main_args
from collections import OrderedDict, defaultdict

def evaluate(net, split, CNN, config, experiment_log, box_usage = 0, verbose = False, tst_json = [], out_file = '', precision_k = 10):

  box_usage = config['box_usage']
  model     = config['model']

  eval_start = time.time()
  n = correct = 0.0
  trees, boxes, iou, gold = split

  indexes = range(len(trees))
  if verbose:
    pbar = tqdm(indexes)
  else:
    pbar = indexes

  precision_count = [0]*precision_k
  precision_hit = [0]*precision_k

  preds = []
  net.eval()
  net.evaluate = True
  stats = { 'hit' : defaultdict(int), 'cnt' : defaultdict(int) }
  all_supporting = []
  for j in pbar:
    tree = trees[j]
    box_feats,spat_feats = get_box_feats(boxes[j], CNN)

    if box_usage == 0:
      box_rep = torch.cat((box_feats,spat_feats),1)
    elif box_usage == 1:
      box_rep = box_feats
    elif box_usage == 2:
      box_rep = spat_feats
    else:
      raise NotImplementedError()
    if "ground" in config['model'].lower():
      prediction = net(tree, box_rep, tree, decorate = True)
      supporting = []
      loc_count = 0
      for ii,node in enumerate(tree.nonterms()):
        _,phrase_pred = torch.max(node._expr.data,1)
        del node._expr, node._attn
        if node.label == "loc":
          loc_count += 1
        if node.label == "loc" and loc_count>=2:
          supporting += [phrase_pred[0][0]]
    else:
      prediction, sub, obj, rel = net([w2i.get(node.label,0) for node in tree.leaves()], box_rep, tree)
      try:
        _,obj_pred = torch.max(obj.data,0)
        supporting = [obj_pred[0][0]]
      except:
        supporting = []

    _,pred = torch.max(prediction.data,1)

    pred_np = prediction.cpu().data.numpy()
    for idx,k in enumerate(np.argsort(-pred_np)[0][:min(precision_k, pred_np.shape[1])]):
      p_hit = (1.0 if k in set(gold[j]) else 0.0)

      if p_hit == 1.0:
        for jj in xrange(idx,min(precision_k, pred_np.shape[1])):
          precision_hit[jj]   += p_hit
          precision_count[jj] += 1.0
        break
      precision_hit[idx]   += p_hit
      precision_count[idx] += 1.0

    hit        = (1.0 if pred[0][0] in set(gold[j]) else 0.0)
    correct   += hit
    n += 1
    preds.append(int(pred[0][0]))
    all_supporting.append(supporting)
  eval_time = time.time() - eval_start

  print "_"*10
  for k in xrange(precision_k):
    print "precision@{} for {} instances: {}".format(k+1,precision_count[k], precision_hit[k]/precision_count[k])
  if tst_json != []:
    for ii,inst in enumerate(tst_json):
      tst_json[ii]['predicted_bounding_boxes'] = [tst_json[ii]['box_names'][preds[ii]]]
      tst_json[ii]['context_box'] = all_supporting[ii]
    json.dump(tst_json,open(out_file,'w'))
  return correct/n , len(trees)/eval_time

args = get_main_args()
CNN  = h5py.File(args.cnn, 'r')
data = pickle.load(open(args.dump))

dev     = data['dev']
tst     = data['tst']
vocabs  = data['vocabs']
tst_json= data['tst_json']

vectors, w2i, p2i, n2i, i2w, i2p, i2n = vocabs

if args.resume != '':
  net = torch.load(args.resume, map_location=lambda storage, location: storage.cuda(0))
  config = net.config
  net.debug = args.debug_mode
else:
  raise NotImplementedError()

if not os.path.exists(args.save_path):
  os.makedirs(args.save_path)
snapshot_pfx   = 'snapshot.' + ".".join([key.upper()+str(config[key]) for key in config.keys() if key[0] != 'f'])
if args.out_file == '':
  out_file = os.path.join(args.save_path, snapshot_pfx + '.tst-eval.json')
else:
  print("will dump prediction to {}".format(args.out_file))
  out_file = args.out_file

print("="*20)
print("Starting testing {} model".format(config['model']))
print("Snapshots {}.*\nDetails:".format(os.path.join(args.save_path, snapshot_pfx)))
for key in config:
  print("{} : {}".format(key,config[key]))
print("="*20)


start_time = time.time()
print("startup time for {} model: {:5.3f} for {} parameters".format(config['model'],start_time - start, get_n_params(net)))

best_val , val_rate = evaluate(net, dev, CNN, config, '', verbose = True)
tst_score, tst_rate = evaluate(net, tst, CNN, config, '', verbose = True,
                               tst_json = tst_json,
                               out_file = out_file)
print("model scores based on best validation accuracy\nval_acc:{:5.3f} test_acc: {:5.3f} test speed {:5.1f} inst/sec\n".format(best_val,tst_score,tst_rate))
