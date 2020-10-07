#!/usr/bin/env/python
from copy import deepcopy
from util.spellchecker import Spellchecker
from util.arguments import get_cmn_imbd
from util.tree import *

import h5py
import random
import codecs
from collections import Counter, defaultdict
import cPickle as pickle
import numpy as np
import os

import sys
usage = '''
usage:
      python 
      $> python process_refcocog_imdb.py --data-root ./data --save stanford_cmn_refcocog_iou05.refcocog.NOtriplets.pkl
      $> python process_refcocog_imdb.py --data-root ./data --save stanford_cmn_refcocog_iou05.refcocog.NOtriplets.pkl --use-triplers
'''
__author__ = "volkan cirik"


def read_raw(filename, tree_type="stanford"):
  if tree_type == "stanford":
    try:
      raw = [line.strip().replace("@", "")
             for line in codecs.open(filename, "r")][0]
    except:
      print >> sys.stderr, "EMPTY:", filename
      raw = ""
      pass
    return [], [raw]
  filename = filename.replace(".np", ".berkeley")

  probs, raws = [], []

  for line in codecs.open(filename.strip(), "r"):
    line = line.strip()
    if line == "":
      continue
    l = line.strip().split('\t')
    prob = l[0]
    raw = l[1][1:-1]
    if prob == "Infinity" or prob == "-Infinity":
      prob = "-0.01"
    prob = np.exp(float(prob))
    probs.append(prob)
    raws.append(raw)
  total = np.sum(np.array(probs))
  probs = [p/total for p in probs]
  if raws == []:
    print >> sys.stderr, "EMPTY:", filename
    raws = [""]
    probs = []
  return probs, raws


def read_trees(filenames, blocklist, mapping, wordvec,
               use_triplets=False, mode=1, verbose=False, UNK="<unk>",
               tree_type="stanford"):
  trees = []
  tree_dict = {}
  stats = defaultdict(int)
  spellcheck = Spellchecker(wordvec.keys())
  done = False
  spell_hit = 0
  n_unk = n_tot = 0.0
  for ii, filename in enumerate(filenames):
    probs, raws = read_raw(filename, tree_type=tree_type)
    if raws[0] == "":
      continue
    toriginal = Tree.from_sexpr(raws[0])
    if toriginal.getSpan() == "there is":
      continue
    original_text = [line.strip() for line in codecs.open(
        filename.replace('.np', '.txt').strip(), "r")][0].replace("there is ", "")
    original_span = toriginal.getSpan()
    toriginal.original_text = original_text
    toriginal.original_raw = raws[0]
    toriginal.fname = filename
    context = " ".join([n.label for n in toriginal.leaves()])

    change = False
    for n in toriginal.leaves():
      n_tot += 1
      w = n.label
      w = mapping.get(w.lower(), w.lower())
      if w not in wordvec and wordvec != {}:
        if done:
          w = UNK
        else:
          change = True
          options = spellcheck.correct(w)
          if len(options) == 1 and options[0] != w:
            w = options[0]
            spell_hit += 1
          else:
            w = UNK
      if n.label != w:
        n.label = w
        if w == UNK:
          n_unk += 1
    if use_triplets:
      t = deepcopy(toriginal)
      t.preClean()
      t.adopt()
      t.adopt()
      t.preTriplet(blocklist)
      to3 = t.children[0] if len(t.children) == 1 else t
      d, ct_raw = to3.findTriplet(blocklist, mode=mode)
      if mode == 2:
        depth = d[0]
      else:
        depth = d
      stats[depth] += 1

      if mode == 2:
        t_candidates = []
        for raw in ct_raw:
          ct = Tree.from_sexpr(raw)
          ct.collapsed_span = ct.getSpan()
          ct.original_span = original_span
          ct.original_text = original_text
          ct.fname = filename
          t_candidates.append(ct)
        trees.append(t_candidates)
      else:
        try:
          ct = Tree.from_sexpr(ct_raw)
        except:
          print >> sys.stderr, "ERROR:", filename
          pass
        ct.collapsed_span = ct.getSpan()
        ct.original_span = original_span
        ct.original_text = original_text
        ct.fname = filename
        trees.append(ct)
    else:
      trees.append(toriginal)

    tree_dict["_".join(filename.split('/')[-2:]
                       ).replace(".np", "")] = trees[-1]

  if use_triplets:
    print >> sys.stderr, "="*20
    for dept in stats:
      print >> sys.stderr, dept, "-->{:5.3f}".format(
          stats[dept]*1.0/len(trees))
    print >> sys.stderr, "="*20
  print >> sys.stderr, "{:5.3f} unknown words and fixed {} typos".format(
      n_unk/n_tot, spell_hit)
  return trees, tree_dict


def get_vocabs(trees, wordvec, n_occur=1, use_triplets=False):
  non_vocab = Counter()
  pos_vocab = Counter()

  if use_triplets:
    wrd = wordvec.keys()
    pos = []
    non = []

    w2i = {w: i for i, w in enumerate(wrd)}
    p2i = {}
    n2i = {}
  else:
    for tree in trees:
      non_vocab.update(
          [n.label for n in tree.nonterms() if len(n.children) == 2])
      pos_vocab.update(
          [n.label for n in tree.nonterms() if len(n.children) == 1])

    wrd = wordvec.keys()
    non = ["_UNK_NON_"] + [x for x, c in non_vocab.iteritems()]
    pos = ["_UNK_POS_"] + [x for x, c in pos_vocab.iteritems()]

    w2i = {w: i for i, w in enumerate(wrd)}
    p2i = {w: i for i, w in enumerate(pos)}
    n2i = {w: i for i, w in enumerate(non)}

  vectors = []

  if wordvec != {}:
    for w in wrd:
      vectors.append(wordvec[w])
    vectors = np.stack(tuple(vectors))
  return vectors, w2i, p2i, n2i, wrd, pos, non


def compute_iou(box1, box2):
  inter_x1 = max(box1[0], box2[0])
  inter_y1 = max(box1[1], box2[1])
  inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
  inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

  if inter_x1 < inter_x2 and inter_y1 < inter_y2:
    inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
  else:
    inter = 0
  union = box1[2]*box1[3] + box2[2]*box2[3] - inter
  if union == 0:
    return 0.0
  return float(inter)/union


def get_golds(boxes, gold_idx, gold_iou):
  gold_list = [gold_idx]
  ious = []
  for i, box in enumerate(boxes):
    iou = compute_iou(box, boxes[gold_idx])
    ious.append(1.0 - iou)  # we want iou to be low when optimizing
    if iou >= gold_iou and gold_idx != i:
      gold_list += [i]
  return gold_list, ious


def convert_cmn_refcocog(tokens_path, box_out, anns_trn, anns_val, blocklist, mapping, wordvec,
                         use_triplets=False, gold_iou=1.0, mode=1, val_prob=0.025, tree_type="stanford"):

  filenames = [os.path.join(tokens_path+'/trn', f) for f in os.listdir(tokens_path+'/trn')
               if os.path.isfile(os.path.join(tokens_path+'/trn', f)) and f.split('.')[-1] == 'np']
  filenames += [os.path.join(tokens_path+'/val', f) for f in os.listdir(tokens_path+'/val')
                if os.path.isfile(os.path.join(tokens_path+'/val', f)) and f.split('.')[-1] == 'np']

  print >> sys.stderr, "loading filenames DONE!"

  # Read Trees
  trees, tree_dict = read_trees(filenames, blocklist, mapping, wordvec,
                                use_triplets=use_triplets, mode=mode, tree_type=tree_type)
  print >> sys.stderr, "loading parse trees DONE!"
  # get vocab
  vectors, w2i, p2i, n2i, i2w, i2p, i2n = get_vocabs(
      trees, wordvec, use_triplets=use_triplets)

  print >> sys.stderr, "loading vocabularies DONE!"

  Xtrn_tree, Xtrn_box, Xtrn_iou, Ytrn = [], [], [], []
  Xdev_tree, Xdev_box, Xdev_iou, Ydev = [], [], [], []
  Xtst_tree, Xtst_box, Xtst_iou, Ytst = [], [], [], []

  split2tree = {'val': Xtst_tree, 'train': Xtrn_tree, 'dev': Xdev_tree}
  split2box = {'val': Xtst_box, 'train': Xtrn_box,  'dev': Xdev_box}
  split2iou = {'val': Xtst_iou, 'train': Xtrn_iou,  'dev': Xdev_iou}
  split2gld = {'val': Ytst,     'train': Ytrn,      'dev': Ydev}

  tst_json = []

  n_boxes = 0
  box_feats = []
  spat_feats = []

  total = multiple = 0.0
  for split, anns in zip(['train', 'val'], [anns_trn, anns_val]):
    print >> sys.stderr, "processing {} ...".format(split)
    for kk, ann in enumerate(anns):
      dsplit = split
      if random.random() <= val_prob and split == 'train':
        dsplit = 'dev'
      box_feats.append(ann['box_feats'])
      spat_feats.append(ann['spatial_batch'])
      for jj, ann_id in enumerate(ann['coco_ann_ids']):
        fname = "_".join(
            [{'train': 'trn', 'val': 'val', 'dev': 'trn'}[dsplit], str(kk), str(jj)])
        if fname not in tree_dict:
          continue
        tree = tree_dict[fname]

        if dsplit == 'val':
          tst_json.append({'annotation_id': ann_id, 'box_names': ann['coco_bboxes'].tolist(), 'gold': int(
              ann['label_batch'][jj]), 'parse_file': fname})  # 'refexp' : tree.original_text

        split2tree[dsplit].append(tree)
        split2box[dsplit].append(
            [n_boxes + idx for idx in range(ann['box_feats'].shape[0])])
        gold_labels, ious = get_golds(
            ann['coco_bboxes'].tolist(), int(ann['label_batch'][jj]), gold_iou)
        split2iou[dsplit].append(ious)
        if len(gold_labels) >= 2:
          multiple += 1
        total += 1
        split2gld[dsplit].append(gold_labels)

      n_boxes += ann['box_feats'].shape[0]
      if kk % 1000 == 99:
        print >> sys.stderr, ".",
      if kk % 5000 == 999:
        print >> sys.stderr, "#"
        print >> sys.stderr, "%d | %d done" % (kk, len(anns))

    print >> sys.stderr, "processing {} DONE!".format(split)
  print >> sys.stderr, "% of multiple boxes {}".format(multiple/total)
  box_stack = np.concatenate(tuple(box_feats))
  spat_stack = np.concatenate(tuple(spat_feats))
  assert(box_stack.shape[0] == spat_stack.shape[0])

  h5file = h5py.File(box_out, 'w')
  h5file.create_dataset("rcnn", data=box_stack)
  h5file.create_dataset("spat", data=spat_stack)
  h5file.close()
  print >> sys.stderr, "box features are dumped to {}".format(box_out)

  # package dataset
  train = Xtrn_tree, Xtrn_box, Xtrn_iou, Ytrn
  devel = Xdev_tree, Xdev_box, Xdev_iou, Ydev
  test = Xtst_tree, Xtst_box, Xtst_iou, Ytst
  vocabs = vectors, w2i, p2i, n2i, i2w, i2p, i2n

  print >> sys.stderr, "total trn|dev|tst instances %d|%d|%d" % (
      len(Xtrn_tree), len(Xdev_tree), len(Xtst_tree))
  print >> sys.stderr, "generating refcocog imdb DONE!"

  return train, devel, test, vocabs, tst_json


def get_dataset(args, blocklist, mapping, wordvec):
  if args.task == 'refcocog':
    anns_trn = np.load(os.path.join(
        args.path, 'refcocog.trn.npy'), allow_pickle=True)
    anns_val = np.load(os.path.join(
        args.path, 'refcocog.val.npy'), allow_pickle=True)
    print >> sys.stderr, "loading annotation data DONE!"

    return convert_cmn_refcocog(os.path.join(args.path, 'refcocog_sentence'), args.dump.split('.')[0] + '.box.h5', anns_trn, anns_val, blocklist, mapping, wordvec, use_triplets=args.use_triplets, gold_iou=args.gold_iou, mode=args.mode, val_prob=args.val_prob, tree_type=args.tree_type)
  else:
    raise NotImplementedError()


if __name__ == '__main__':
  args = get_cmn_imbd()
  random.seed(args.seed)
  blocklist = set([line.strip().split()[0] for line in open(args.blocklist)])
  mapping = {line.strip().split()[0].lower(): line.strip().split()[
      1].lower() for line in open(args.mapping)}
  wordvec = {line.strip().split()[0].lower(): [float(
      n) for n in line.strip().split()[1:]] for line in open(args.wordvec)}

  print >> sys.stderr, "loading blocklist|mapping|wordvec DONE!"
  trn, dev, tst, vocabs, tst_json = get_dataset(
      args, blocklist, mapping, wordvec)
  print >> sys.stderr, "dataset dumping.."
  pickle.dump({'trn': trn, 'dev': dev, 'tst': tst, 'vocabs': vocabs,
               'tst_json': tst_json}, open(args.dump, 'w'))
  print >> sys.stderr, "dataset dumping DONE!"
  print >> sys.stderr, "ALL DONE!bye!"
