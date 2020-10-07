#from __future__ import absolute_import
import argparse


def get_main_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--load', dest='dump', help='load a dataset dump',
                      default='../data/stanford_cmn_refcocog_iou05.triplets.pkl')
  parser.add_argument('--cnn', dest='cnn', default='../data/stanford_cmn_refcocog_iou05.box+meta+smax.h5',
                      help='cnn features filetype : h5')

  parser.add_argument('--n-hidden', dest='n_hidden', type=int,
                      help='# of hidden units, default = 1000', default=1000)
  parser.add_argument('--n-layers', dest='n_layer', type=int,
                      help='# of layers for REL and LOC projections 1|2, default = 1', default=1)
  parser.add_argument('--model', dest='model',
                      help='model type cmn|groundnet, default:groundnet', default='groundnet')
  parser.add_argument('--dropout', dest='dropout',
                      help='dropout rate, default=0', type=float, default=0.)
  parser.add_argument('--clip', dest='clip',
                      help='gradient clipping, default=10.0', type=float, default=10.0)

  parser.add_argument('--timeout', dest='timeout', type=int,
                      help='timeout in seconds, default = 300000', default=300000)
  parser.add_argument('--seed', dest='seed', type=int,
                      help='seed, default = 0', default=0)

  parser.add_argument('--epochs', type=int, default=6,
                      help='# of epochs, default = 6')
  parser.add_argument('--val-freq', dest='val_freq', type=int, default=0,
                      help='validate every n instances, 0 is for full pass over trn data, default = 0')
  parser.add_argument('--save-path', dest='save_path', type=str,
                      default='exp', help='folder to save experiment')
  parser.add_argument('--resume', dest='resume', type=str,
                      default='', help='resume from this model snapshot')
  parser.add_argument('--box-usage', dest='box_usage', type=int, default=0,
                      help="box features 0:cnn+spatial  1:cnn 2:spatial, default: 0")

  parser.add_argument('--verbose', dest='verbose',
                      action='store_true', help='print to stdout')
  parser.add_argument('--use-outer', dest='use_outer',
                      action='store_true', help='use outer product for features')
  parser.add_argument('--fusion', dest='fusion',
                      help='fusion type mul|sum|concat default=mul', default='mul')
  parser.add_argument('--debug', dest='debug_mode',
                      action='store_true', help='debug mode')
  parser.add_argument('--no-finetune', dest='finetune',
                      action='store_false', help='do not finetune word embeddings')
  parser.add_argument('--optim', dest='optim',
                      help='optimization method adam|sgd, default:sgd', default='sgd')
  parser.add_argument('--loss', dest='loss',
                      help='loss for training nll|smargin|lamm|mbr, default:nll', default='nll')
  parser.add_argument(
      '--lr', dest='lr', help='initial learning rate, default = 0.01', default=0.01, type=float)
  parser.add_argument('--lr-min', dest='lr_min',
                      help='minimum lr, default = 0.00001', default=0.00001, type=float)
  parser.add_argument('--lr-decay', dest='lr_decay',
                      help='learning rate decay, default = 0.5', default=0.5, type=float)
  parser.add_argument('--w-decay', dest='weight_decay',
                      help='weight decay, default = 0.0005', default=0.0005, type=float)
  parser.add_argument('--encoder', dest='encoder',
                      help='rnn encoder  lstm|gru, default:lstm', default='lstm')
  parser.add_argument('--phrase-context', dest='phrase_context',
                      action='store_true', help='use phrase context for FLEX models')
  parser.add_argument('--only-spatial', dest='only_spatial',
                      action='store_true', help='use only spatial features for REL() module')

  parser.add_argument('--out-file', dest='out_file',
                      default='', help='output path for json file')
  args = parser.parse_args()
  return args


def get_cmn_imbd():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', dest='seed', type=int,
                      help='seed, default = 1', default=1)
  parser.add_argument('--save', dest='dump', help='dump the dataset')
  parser.add_argument('--data-root', dest='path',
                      help='path to data folder containing cmn_imbd')
  parser.add_argument('--task', dest='task', default='refcocog',
                      help='task name refcocog|visual7w, default: refcocog')
  parser.add_argument('--blocklist', dest='blocklist', default='../data/blocklist.txt',
                      help="the list of blocklisted words, default=../data/blocklist.txt")
  parser.add_argument('--mapping', dest='mapping', default='../data/mapping.txt',
                      help="mapping of typos, default=../data/mapping.txt")
  parser.add_argument('--wordvec', dest='wordvec', default='../data/wordvec.glove',
                      help="word vectors, default=../data/wordvec.glove")
  parser.add_argument('--val-prob', dest='val_prob',
                      help='probability of trn instance become val instance, default = 0.025', default=0.025, type=float)
  parser.add_argument('--use-triplets', dest='use_triplets',
                      action='store_true', help='use collapsed triplets')
  parser.add_argument('--triplet-mode', dest='mode', type=int, default=1,
                      help='triplet mode 0=depth 1, 1= depth n, 2= depth n with candidate fixes, default = 1')
  parser.add_argument('--gold-iou', dest='gold_iou', type=float,
                      default=0.5, help='0.5 < = gold label iou <= 1.0, default = 0.5')
  parser.add_argument('--tree-type', dest='tree_type',
                      help="tree type berkeley|standord default=stanford", default='stanford')
  parser.add_argument(
      '--ary', dest='ary', help="binary or nary tree .bp|.np default=.np", default='.np')
  parser.add_argument('--split', dest='split_map',
                      help="split map pickle file ../data/umd_split_map.pkl|'' default=''", default='')
  parser.add_argument('--imdbs', dest='imdbs',
                      help="imdbs for headp perturbation '' or '{PATH TO IMDB}/imdb_trn.npy {PATH TO IMDB}/imdb_val.npy' default=''", default='')

  args = parser.parse_args()
  return args
