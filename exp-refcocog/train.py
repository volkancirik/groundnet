#from __future__ import absolute_import,print_function
"""
Train a net
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
def evaluate(net, split, CNN, config, experiment_log, box_usage = 0, verbose = False, tst_json = [], out_file = ''):

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

  preds = []
  net.eval()
  stats = { 'hit' : defaultdict(int), 'cnt' : defaultdict(int) }
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
    if config['model'] in set(["groundnet", "groundnetflexall", "groundnetflexrel","treernn"]):
      prediction = net(tree, box_rep, tree).data
    else:
      prediction = net([w2i.get(node.label,0) for node in tree.leaves()], box_rep, tree).data
    _,pred = torch.max(prediction,1)

    hit        = (1.0 if pred[0][0] in set(gold[j]) else 0.0)
    correct   += hit
    complexity = len(tree.nonterms())
    stats['hit'][complexity] += hit
    stats['cnt'][complexity] += 1
    n += 1
    preds.append(int(pred[0][0]))
  eval_time = time.time() - eval_start

  log = []
  log.append("")
  log.append("="*20)
  log.append("Stats")
  for complexity in stats['hit'].keys():
    log_line = "Complexity {:3d} acc {:5.3f} %{:5.3f}".format(complexity,
                                                         stats['hit'][complexity]*1.0 / stats['cnt'][complexity],
                                                         stats['cnt'][complexity]*1.0 / n)
    log.append(log_line)
  log_line = "Total correct {}/{}".format(correct,n)
  log.append(log_line)
  log.append("="*20)

  for line in log:
    print(line)
    experiment_log.write(line)

  if tst_json != []:
    for ii,inst in enumerate(tst_json):
      tst_json[ii]['predicted_bounding_boxes'] = [tst_json[ii]['box_names'][preds[ii]]]
    json.dump(tst_json,open(out_file,'w'))
  return correct/n , len(trees)/eval_time

args = get_main_args()
CNN  = h5py.File(args.cnn, 'r')
data = pickle.load(open(args.dump))

dev    = data['dev']
tst    = data['tst']
trn    = data['trn']
vocabs  = data['vocabs']
tst_json= data['tst_json']

vectors, w2i, p2i, n2i, i2w, i2p, i2n = vocabs
Xtrn_tree, Xtrn_box, Xtrn_iou, Ytrn   = trn

indexes = range(len(Xtrn_tree))

if args.box_usage == 0:
  feat_box = 4096 + 5
elif args.box_usage == 1:
  feat_box = 4096
elif args.box_usage == 2:
  feat_box = 5
else:
  raise NotImplementedError()

if args.resume != '':
  net = torch.load(args.resume)#, map_location=lambda storage, location: storage.cuda(0))
  config = net.config
  net.debug = args.debug_mode
else:
  config = OrderedDict([('model', args.model),
            ('n_hidden' , args.n_hidden),
            ('n_layer' , args.n_layer),
            ('dropout' , args.dropout),
            ('fusion' , args.fusion),
            ('finetune' , args.finetune),
            ('use_outer' , args.use_outer),
            ('box_usage', args.box_usage),
            ('feat_box' , feat_box),
            ('loss' , args.loss),
            ('optim' , args.optim),
            ('lr' , args.lr),
            ('lr_min' , args.lr_min),
            ('lr_decay' , args.lr_decay),
            ('weight_decay' , args.weight_decay),
            ('clip' , args.clip),
            ('encoder' , args.encoder),
            ('only_spatial' , args.only_spatial),
            ('phrase_context' , args.phrase_context),
            ('debug'  , args.debug_mode)])
  net = get_model(vocabs, config)

if not os.path.exists(args.save_path):
  os.makedirs(args.save_path)
snapshot_pfx   = 'snapshot.' + ".".join([key.upper()+str(config[key]) for key in config.keys() if key[0] != 'f'])
snapshot_model = os.path.join(args.save_path, snapshot_pfx + '.model')
experiment_log = open(os.path.join(args.save_path, snapshot_pfx + '.log'),'w')
out_file = os.path.join(args.save_path, snapshot_pfx + '.tst-eval.json')
print("="*20)
print("Starting training {} model".format(config['model']))
print("Snapshots {}.*\nDetails:".format(os.path.join(args.save_path, snapshot_pfx)))
for key in config:
  print("{} : {}".format(key,config[key]))
print("="*20)


if config['optim'] == 'sgd':
  optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = config['lr'], momentum = 0.95)
else:
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr = config['lr'], weight_decay = config['weight_decay'])
optimizer.zero_grad()

if config['loss'] == 'nll':
  criterion = nn.NLLLoss()
elif config['loss'] == 'smargin':
  criterion = nn.MultiLabelSoftMarginLoss()
elif config['loss'] == 'lamm':
  criterion = lamm
elif config['loss'] == 'mbr':
  criterion = mbr
else:
  raise NotImplementedError()

start_time = time.time()
if args.verbose:
  print("startup time for {} model: {:5.3f} for {} instances for {} parameters".format(config['model'],start_time - start, len(indexes), get_n_params(net)))

best_val = 0
timeout = False
for ITER in range(args.epochs):
  net.train()
  random.shuffle(indexes)
  closs = 0.0
  cinst = 0
  correct = 0.0
  trn_start = time.time()

  if args.verbose and not args.debug_mode:
    pbar = tqdm(indexes, desc='trn_loss')
  else:
    pbar = indexes

  done = 1
  for ii in pbar:
    tree = Xtrn_tree[ii]

    # get cnn feat for boxes
    box_feats, spat_feats = get_box_feats(Xtrn_box[ii], CNN)

    if config['box_usage'] == 0:
      box_rep = torch.cat((box_feats,spat_feats),1)
    elif config['box_usage'] == 1:
      box_rep = box_feats
    elif config['box_usage'] == 2:
      box_rep = spat_feats
    else:
      raise NotImplementedError()

    if args.debug_mode:
      raise NotImplementedError()
    else:
      if config['model'] in set(["groundnet", "groundnetflexall", "groundnetflexrel","treernn"]):
        prediction = net(tree, box_rep, tree)
      else:
        prediction = net([w2i.get(n.label,0) for n in tree.leaves()], box_rep, tree)

      _,pred = torch.max(prediction.data,1)
      correct += (1.0 if pred[0][0] in set(Ytrn[ii]) else 0.0)

      if config['loss'] == 'nll':
        gold = makevar(Ytrn[ii][0])
      elif config['loss'] == 'smargin':
        gold = np.zeros((1,box_rep.size(0)))
        np.put(gold,Ytrn[ii],1.0)
        gold = makevar(gold, numpy_var = True).view(1,box_rep.size(0))
      elif config['loss'] == 'lamm':
        prediction = (prediction,Xtrn_iou[ii])
        gold       = Ytrn[ii]
      elif config['loss'] == 'mbr':
        prediction = (prediction,Xtrn_iou[ii])
        gold       = Ytrn[ii]
      else:
        raise NotImplementedError()

      loss  = criterion(prediction, gold)
      if math.isnan(float(loss.data[0])):
        print("")
        print("="*20)
        print("problem with instance: {}, filename {}".format(ii,tree.fname))
        print("Nonterms:"," ".join([n.label for n in tree.nonterms()]))
        print("Span:",tree.getSpan())
        print("Gold:",Ytrn[ii])
        print("*"*20)
        net.debug = True
        if config['model'] in set(["groundnet", "groundnetflexall", "groundnetflexrel","treernn"]):
          pred = net(tree, box_rep, tree)
        else:
          pred = net([w2i.get(n.label,0) for n in tree.leaves()], box_rep, tree)
        print("Prediction:",printVec(pred, logprob = False))
        print("="*60)
        quit()
      closs += float(loss.data[0])
      cinst += 1
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm(net.parameters(), config['clip'])
      optimizer.step()

      if args.verbose:
        pbar.set_description("trn_loss {:5.3f} trn_acc {:5.3f}".format(closs/cinst,correct/cinst))
      if time.time() - start_time > args.timeout:
        timeout = True
        break

    if timeout:
      break
    if args.val_freq != 0 and done % args.val_freq == 0:
      print("")
      val_score, val_rate = evaluate(net, dev, CNN, config, experiment_log, verbose = False)
      print("epoch {:3d}/{:3d} inst#{:3d} val_acc: {:5.3f}".format(ITER+1,args.epochs,ii,val_score))
    done += 1
  trn_loss = closs / cinst
  trn_acc  = correct/cinst
  trn_rate = len(indexes)/(time.time() - trn_start)
  val_score, val_rate = evaluate(net, dev, CNN, config, experiment_log, verbose = args.verbose)

  log = "epoch {:3d}/{:3d} trn_loss: {:5.3f} trn_acc: {:5.3f} trn speed {:5.1f} inst/sec \n\t\tbest_val: {:5.3f} val_acc: {:5.3f} val speed {:5.1f} inst/sec\n".format(ITER+1,args.epochs,trn_loss,trn_acc,trn_rate,best_val,val_score,val_rate)
  config['lr'] = max(config['lr']*config['lr_decay'],config['lr_min'])

  if config['optim'] == 'sgd':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr = config['lr'], momentum = 0.95)
  else:
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr = config['lr'], weight_decay = config['weight_decay'])

  if args.verbose:
    print(log)
    print("lr is updated to {}".format(config['lr']))
  experiment_log.write(log)
  torch.save(net, snapshot_model + '.EPOCH' + str(ITER+1))

  if best_val < val_score:
    if args.verbose:
      print("Best model is updated at epoch {}".format(ITER+1))
    torch.save(net, snapshot_model + '.best')
    best_val = val_score
  experiment_log.flush()

best_net = torch.load(snapshot_model + '.best', map_location=lambda storage, location: storage.cuda(0))
tst_score, tst_rate = evaluate(best_net, tst, CNN, config, experiment_log,
                               verbose = args.verbose, tst_json = tst_json,
                               out_file = out_file)
log = "model scores based on best validation accuracy\nval_acc:{:5.3f} test_acc: {:5.3f} test speed {:5.1f} inst/sec\n".format(best_val,tst_score,tst_rate)
if args.verbose:
  print(log)
experiment_log.write(log)
experiment_log.close()
