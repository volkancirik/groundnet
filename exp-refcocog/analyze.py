#!/usr/bin/env/python
'''
align & filter predictions of refer_context based on parse tree non-literals.

python prediction2parse.py --help
for command line options.

TODO:
- print in latex table format
'''
__author__ = "volkan cirik"

import sys, json, argparse, h5py
import numpy as np
import cPickle as pickle
from pprint import pprint
import os
from collections import defaultdict, Counter
from util.model_util import get_box_feats

parser = argparse.ArgumentParser()
parser.add_argument('--refcocog', dest='refcocog', help='path to refcocog data, default=../data/stanford_cmn_refcocog_iou05.NOtriplets.pkl',default = '../data/stanford_cmn_refcocog_iou05.NOtriplets.pkl')
parser.add_argument('--cnn', dest='cnn', help='path to cnn data, default=../data/stanford_cmn_refcocog_iou05.box+meta+smax.h5',default = '../data/stanford_cmn_refcocog_iou05.box+meta+smax.h5')
parser.add_argument('--eval-path', dest='eval_path', help='path to Google_Refexp_toolbox, default=../Google_Refexp_toolbox/google_refexp_py_lib/', default = '../Google_Refexp_toolbox/google_refexp_py_lib/')
parser.add_argument('--refexp', dest='refexp', help='referring expression dataset file of format: google_refexp_val_coco_aligned.json, default=../data/google_refexp_val_201511_coco_aligned.json', default = '../data/google_refexp_val_201511_coco_aligned.json')
parser.add_argument('--coco-instances', dest='coco_inst', help='coco instances file of format: coco/annotations/instances_*.json, default=../data/instances_val2014.json', default = '../data/instances_val2014.json')
parser.add_argument('--prediction', dest='prediction', help='prediction file of format: json')
parser.add_argument('--parse', dest='parse', help='root path to parsing folder')
parser.add_argument('--latex', dest='latex', help='print to latex file if present, default=""', default = '')
parser.add_argument('--category', dest='category', help='category for analysis loc|words|depth|boxes, default = loc',default = 'loc')
parser.add_argument('--n-supporting', dest='n_supporting', help='number of supporting annotations, default = 2407',default = 2407)
args = parser.parse_args()

sys.path.append(args.eval_path)
from refexp_eval import RefexpEvalComprehension

def getAnnDict(ann):
  w2box = {}
  for box in ann.split('|')[1:]:
    if len(box.split(' ')) >= 2:
      phrase = "_".join(box.split(' ')[1:])
      box = box.split(' ')[0]
      w2box[phrase] = box

  for box in ann.split('|')[0].split(',')[1:]:
    if box.strip() != '':
      w2box[box.strip()] = 'b-1'

  return w2box

def supportingCheck(idx, sup_pred, gold, root = '../notebooks/supp_ann/annotations/', mode = 'n_sup'):

  if os.path.isfile(root + "{}.ann.txt".format(idx)):
    ann = [line.strip() for line in open(root + "{}.ann.txt".format(idx))][0]
    w2box = getAnnDict(ann)
    if len([w for w in w2box if w2box[w] != 'b-1']) >= 1:
      box_set = set([int(w2box[w][1:]) for w in w2box if w2box[w] != 'b-1'])
      sup_set = box_set.difference(set(gold))

      if mode == 'n_sup':
        return len(sup_set)
      elif mode == 'sup_acc':
        if sup_set == set([]):
          return 0 ### no sup object
        if len(sup_set.intersection(set(sup_pred))) == len(sup_set):
          return 1 ### all sup are correct
        elif len(sup_set.intersection(set(sup_pred))) >= 1:
          return 2 ### at least one sup obj is correct
        else:
          return 3
      elif mode == 'missing':
        return len([w for w in w2box if w2box[w] == 'b-1'])
      elif mode == 'missing_all':
        if len([w for w in w2box if w2box[w] == 'b-1']) > 0 and len(sup_set) > 0:
          return 0
        else:
          return 1
      else:
        raise NotImplementedError()
    return 0
  return 0

def getCategoryDicts(refexp_instances, coco_instances):
  refexp = json.load(open(refexp_instances))
  annid2imgid = {}
  imgid2catid2count = {}
  imgid2catid2bsize = {}
  annid2catid = {}
  annid2bsize = {}

  for ann_id in refexp['annotations']:
    inst = refexp['annotations'][ann_id]
    annid2imgid[str(inst['annotation_id'])] = str(inst['image_id'])
    annid2catid[str(inst['annotation_id'])] = str(inst['category_id'])
    annid2bsize[str(inst['annotation_id'])] = int(inst['bbox'][-2]*inst['bbox'][-1])

  coco_val = json.load(open(coco_instances))
  coco_trn = json.load(open(coco_instances.replace('val','train')))
  for coco in [coco_trn,coco_val]:
    for inst in coco['annotations']:
      imgid = str(inst['image_id'])
      catid = str(inst['category_id'])
      if imgid not in imgid2catid2count:
        imgid2catid2count[imgid] = defaultdict(int)
        imgid2catid2bsize[imgid]  = defaultdict(list)
      imgid2catid2count[imgid][catid] += 1
      imgid2catid2bsize[imgid][catid].append(int(inst['bbox'][-2]*inst['bbox'][-1]))

  return annid2imgid, imgid2catid2count, annid2catid, imgid2catid2bsize, annid2bsize

def filterCategories(pred, categories):
  '''
  filter all categories, returns a dictionary of a list of predictions
  '''
  filtered = defaultdict(list)
  for cat in categories:
    filter_set = set(categories[cat])
    for ii,ins in enumerate(pred):
      if ii in filter_set:
        filtered[cat].append(ins)
  print >> sys.stderr, "DONE: filterCategories "
  return filtered

def printLatexTable(f,n_column, rows, caption="My Caption",label="my-label"):
  prefix = ["%"*20,'\\begin{table}[]','\\centering','\\caption{%s}' % caption,'\\label{%s}' % label,
        '\\begin{tabular}{%s}' % ("".join(['l' for i in xrange(n_column)])) ,'\hline']
  suffix = ['\end{tabular}','\end{table}',"%"*20]

  contents = prefix
  for i in xrange(len(rows)):
    assert(n_column == len(rows[i]))
    content = "\t&\t".join([str(tok) for tok in rows[i]]) + " \\\\"
    if i == 0 or i == len(rows)-1:
      content += " \hline"
    contents += [content]
  contents += suffix
  print >> f, "\n".join(contents)

def getCategories(category, trees, boxes, gold_labels, CNN, prediction, cut_off = {'loc' : 5,'boxes' : 20, 'words' : 18, 'depth' : 3}, annid2catid = {}, annid2imgid = {}, imgid2catid2count = {}, vocabulary = {}, imgid2catid2bsize = {}, annid2bsize = {}, n_supporting = 0):
  categories = {'loc' : defaultdict(list), 'boxes' : defaultdict(list), 'words' : defaultdict(list), 'depth' : defaultdict(list), 'obj_cat' : defaultdict(list), 'obj_box_count' : defaultdict(list), 'oov' : defaultdict(list), 'box_size' : defaultdict(list), 'box_order' : defaultdict(list), 'obj_box_order' : defaultdict(list), 'box_distance' : defaultdict(list), 'n_sup' : defaultdict(list), 'sup_acc'  : defaultdict(list), 'missing' : defaultdict(list) , 'box2d' : defaultdict(list), 'missing_all' : defaultdict(list)}

  if category == 'loc':
    for i,tree in enumerate(trees):
      n_loc = Counter([n.label for n in tree.nonterms()])['loc']
      categories[category][min(n_loc,cut_off[category])].append(i)

  elif category == 'box_order':
    for i,box in enumerate(boxes):
      box_feats,spat_feats = get_box_feats(boxes[i], CNN, convert_spat = False, convert_box = False)
      for j in gold_labels[i][:1]: ###
        bracket = min(list(np.argsort(spat_feats[:,-1])[::-1]).index(j),9)
        categories[category][bracket].append(i)

  elif category == 'box_size':
    for i,box in enumerate(boxes):
      box_feats,spat_feats = get_box_feats(boxes[i], CNN, convert_spat = False, convert_box = False)
      for j in gold_labels[i][:1]: ###
        bracket = int(min(99,spat_feats[j,-1]*100))/40
        categories[category][bracket].append(i)

  elif category == 'boxes':
    for i,box in enumerate(boxes):
      n_box = len(box)
      categories[category][min(n_box,cut_off[category])].append(i)

  elif category == 'words':
    for i,tree in enumerate(trees):
      n_words =  len(tree.original_text.split(" "))
      categories[category][min(n_words,cut_off[category])].append(i)

  elif category == 'depth':
    for i,tree in enumerate(trees):
      depth = Counter(tree.getRaw().split(" ")[-1])[')']
      categories[category][min(depth,cut_off[category])].append(i)

  elif category == 'obj_cat':
    for i,pred in enumerate(prediction):
      obj_cat = annid2catid[str(pred['annotation_id'])]
      categories[category][obj_cat].append(i)

  elif category == 'obj_box_count':
    for i,pred in enumerate(prediction):
      imgid = annid2imgid[str(pred['annotation_id'])] 
      catid = annid2catid[str(pred['annotation_id'])]
      count = imgid2catid2count[imgid][catid]
      categories[category][count].append(i)

  elif category == 'obj_box_order':
    for i,pred in enumerate(prediction):
      imgid = annid2imgid[str(pred['annotation_id'])] 
      catid = annid2catid[str(pred['annotation_id'])]
      count = imgid2catid2count[imgid][catid]

      idx = imgid2catid2bsize[imgid][catid].index(annid2bsize[str(pred['annotation_id'])])
      bracket = list(np.argsort(imgid2catid2bsize[imgid][catid])[::-1]).index(idx)
      categories[category][bracket].append(i)

  elif category == 'box_distance':
    for i,box in enumerate(boxes):
      box_feats,spat_feats = get_box_feats(boxes[i], CNN, convert_spat = False, convert_box = False)
      distances = ((spat_feats[:,2] + spat_feats[:,0])/2.0)**2 + ((spat_feats[:,3] + spat_feats[:,1])/2.0)**2
      for j in gold_labels[i][:1]: ###
        bracket = int(min(15000,distances[j]*100))/33
        categories[category][bracket].append(i)

  elif category == 'box2d':
    for i,box in enumerate(boxes):
      box_feats,spat_feats = get_box_feats(boxes[i], CNN, convert_spat = False, convert_box = False)

      idx = gold_labels[i][0]
      density = 25
      x1 = int(min(spat_feats[idx,0]*100 + 100,199)/2)/density
      x2 = int(min(spat_feats[idx,2]*100 + 100,199)/2)/density
      y1 = int(min(spat_feats[idx,1]*100 + 100,199)/2)/density
      y2 = int(min(spat_feats[idx,3]*100 + 100,199)/2)/density

      for xx in xrange(x1,x2+1):
        for yy in xrange(y1,y2+1):
          categories[category]['x'+str(xx)+'y'+str(yy)].append(i)

  elif category == 'oov':
    for i,tree in enumerate(trees):
      oov = 0
      for n in tree.leaves():
        if n.label not in vocabulary:
          oov += 1
      categories[category][oov].append(i)

  elif category == 'n_sup':
    n_total = 1.0
    n_same = 0.0

    n_total_all = 1.0
    n_same_all = 1.0

    for i in xrange(n_supporting):
      n_sup = supportingCheck(i,prediction[i]['context_box'],Ytst[i], mode=category)
      categories[category][n_sup].append(i)

      pred_box = "_".join([str(v) for v in prediction[i]['predicted_bounding_boxes'][0]])
      boxes = ["_".join([str(v) for v in box]) for box in prediction[i]['box_names']]

      target_box = boxes.index(pred_box)

      if n_sup > 0:
        if len(prediction[i]['context_box']) > 0 and prediction[i]['context_box'][0] == target_box:
          n_same += 1
        n_total += 1
        n_total_all+= 1

    for key in categories[category]:
      print key, len(categories[category][key])

    print "="*20
    print "total {} the same {} percentage {}".format(n_total,n_same,n_same/n_total)
    print "all {} the same {} percentage {}".format(n_total_all,n_same_all,n_same_all/n_total_all)
    print "="*20

  elif category == 'sup_acc':
    for i in xrange(n_supporting):
      sup = supportingCheck(i,prediction[i]['context_box'],Ytst[i], mode=category)
      categories[category][sup].append(i)
    print "# of instances with no supporting objects:",len(categories[category][0])
    total_sup = (len(categories[category][1])+len(categories[category][2])+len(categories[category][3]))*1.0
    print "# of instances with supporting objects:",total_sup
    print "# of instances all supporting objects are correct:",len(categories[category][1])/total_sup
    print "# of instances at least one supporting objects is correct:",(len(categories[category][1])+len(categories[category][2]))/total_sup

  elif category == 'missing':
    for i in xrange(n_supporting):
      missing = supportingCheck(i,prediction[i]['context_box'],Ytst[i], mode=category)
      categories[category][missing > 0].append(i)
    for key in categories[category]:
      print key, len(categories[category][key])

  elif category == 'missing_all':
    for i in xrange(n_supporting):
      missing_all = supportingCheck(i,prediction[i]['context_box'],Ytst[i], mode=category)
      categories[category][missing_all].append(i)
    for key in categories[category]:
      print key, len(categories[category][key])

  else:
    raise NotImplementedError()

  return categories

evaluator = RefexpEvalComprehension(args.refexp, args.coco_inst)
(prec_all, eval_results) = evaluator.evaluate(args.prediction, flag_ignore_non_existed_object=True,flag_ignore_non_existed_gt_refexp=True)
print "refcocog dataset is loading"
data = pickle.load(open(args.refcocog))
print "LOADED!"
if 'obj' in args.category:
  print "category dicts are loading"
  annid2imgid, imgid2catid2count, annid2catid, imgid2catid2bsize, annid2bsize = getCategoryDicts(args.refexp, args.coco_inst)
  print "LOADED!"
else:
  annid2imgid, imgid2catid2count, annid2catid, imgid2catid2bsize, annid2bsize = {}, {}, {}, {}, {}



CNN  = h5py.File(args.cnn, 'r')
tst    = data['tst']
Xtst_tree, Xtst_box, Xtst_iou, Ytst   = tst
vectors, w2i, p2i, n2i, i2w, i2p, i2n = data['vocabs']
pred = json.load(open(args.prediction))

categories = getCategories(args.category,Xtst_tree,Xtst_box,Ytst,CNN,pred, annid2catid = annid2catid, annid2imgid = annid2imgid, imgid2catid2count = imgid2catid2count, imgid2catid2bsize = imgid2catid2bsize, annid2bsize = annid2bsize, vocabulary = w2i, n_supporting = args.n_supporting)
print "all categories LOADED!"

filtered = filterCategories(pred, categories[args.category])
print "filtered {} category!".format(args.category)
results = {}
for c in filtered:
  res = {}
  temp_prediction= 'refexp.pred.{}.{}.json'.format(args.category,c)
  f = open(temp_prediction,'w')

  print "D>printing {} instances".format(len(filtered[c]))
  json.dump(filtered[c], f)
  f.close()

  (prec, eval_results) = evaluator.evaluate(temp_prediction, flag_ignore_non_existed_object=True,flag_ignore_non_existed_gt_refexp=True)
  os.system('rm {}'.format(temp_prediction))
  res = {'prec' : round(prec,3), 'frac' : round(len(filtered[c])*1.0/len(pred),3)}
  results[c] = res

print "Category: All Percentage:1.0 Precision@1: {}".format(round(prec_all,3))
for c in results:
  print "Category: {} Percentage:{} Precision@1: {}".format(c,results[c]['frac'],results[c]['prec'])

if args.latex != '':
  f = open(args.latex,'w')
  rows = [["Category","Percentage","Precision"]]
  rows.append(["All","1.0",round(prec_all,3)])
  for c in results:
    rows.append([c,results[c]['frac'],results[c]['prec']])
  printLatexTable(f,len(rows[0]), rows)
  f.close()
