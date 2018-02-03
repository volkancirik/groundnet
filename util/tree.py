#from __future__ import absolute_import
from copy import deepcopy
import re

def _tokenize_sexpr(s):
  tokker = re.compile(r" +|[()]|[^ ()]+")
  toks = [t for t in [match.group(0) for match in tokker.finditer(s)] if t[0] != " "]
  return toks

def _within_bracket(toks):
  label = next(toks)
  children = []
  for tok in toks:
    if tok == "(":
      children.append(_within_bracket(toks))
    elif tok == ")":
      return Tree(label, children)
    else: children.append(Tree(tok, None))
  assert(False),list(toks)

def returnList(t,mode):
  if mode == 2:
    return [t[0]],[t[1]]
  else:
    return t

class Tree(object):
  def __init__(self, label, children=None):
    self.label = label
    self.children = children

  @staticmethod
  def from_sexpr(string):
    toks = iter(_tokenize_sexpr(string))
    assert next(toks) == "("
    return _within_bracket(toks)

  def __str__(self):
    if self.children is None: return self.label
    return "[%s %s]" % (self.label, " ".join([str(c) for c in self.children]))

  def isleaf(self): return self.children==None

  def leaves_iter(self):
    if self.isleaf():
      yield self
    else:
      for c in self.children:
        for l in c.leaves_iter(): yield l

  def leaves(self): return list(self.leaves_iter())

  def nonterms_iter(self):
    if not self.isleaf():
      yield self
      for c in self.children:
        for n in c.nonterms_iter(): yield n

  def nonterms(self): return list(self.nonterms_iter())

  def getNonSpan(self, sep = ' '):
    try:
      r = sep.join([n.label for n in self.nonterms()])
    except:
      r = "NONE!"
      pass
    return r

  def getSpan(self, sep = ' ', filtered = False):
    if filtered:
      r = '_'.join([n.label for n in self.leaves()])
      return "_".join(filter(lambda a: a!= '', r.split('_')))
    try:
      r = sep.join([n.label for n in self.leaves()])
    except:
      r = "NONE!"
      pass
    return r

  def getSpin(self):
    if self.isleaf():
      return [self.label],["S"]

    words,parse = [],[]
    for ch in self.children:
      w,p = ch.getSpin()
      words += w
      parse += p
    parse += ["R"]
    return words,parse

  def getRaw(self):
    if self.isleaf():
      return self.label
    children = []
    for ch in self.children:
      children.append( ch.getRaw())
    return '(' + self.label  + " " + " ".join(children) + ')'

  def printTree(self):
    print self.getSpan()

  def preClean(self):
    if self == None or self.isleaf() or self.children == None:
      return

    for ch in self.children:
      if ch != None:
        ch.preClean()

    to_remove = []
    for ch in self.children:
      if ch == None or ch.children == None:
        continue
      for grch in ch.children:
        if grch == None:
          continue

        if grch.isleaf() and grch.label in set(['there','is','.','and','that','its','her','his','my','mine','their','hers','us','it']):
          to_remove.append(ch)
    for ch in to_remove:
      self.children.remove(ch)

  def adopt(self):
    if self.isleaf() or self.children == None:
      return
    for ch in self.children:
      ch.adopt()

    to_remove = []

    for i,ch in enumerate(self.children):
      if ch.label in set(['NP']) and ch.getSpan() == '':
        to_remove.append(ch)
      if ch.children == None:
        continue
      for j,grch in enumerate(ch.children):
        if ch.isleaf() == False and grch.isleaf() == False and len(ch.children) == 1:
          ### adoption of NN --> change self.label = np
          self.children[i] = grch
    for ch in to_remove:
      try:
        self.children.remove(ch)
      except:
        pass

  def preTriplet(self,blacklist):
    if self.isleaf() or self.children == None:
      return

    for ch in self.children:
      if ch == None:
        continue
      ch.preTriplet(blacklist)

    n_child = False
    for ch in self.children:
      if ch == None or ch.children == None:
        continue
      if ch.label in set(['DT']) and ch.children != None and ch.children[0] != None and ch.children[0].isleaf():
        ch.children[0].label = ''
      for grch in ch.children:
        if grch == None:
          continue
        if ch.label in set(['NN','NP']) and grch.isleaf() and grch.label in blacklist:
          ch.label = 'PP'
        if ch.label in set(['NN','NP','NNS']) and ((grch.isleaf() and grch.label not in blacklist) or (grch.label in set(['NN','NP','NNS']))) :
          n_child = True
    if n_child == False and self.label in set(['NP']):
      self.label = 'XP'

  def deleteNode(self, child, target_path, path, direction):
    if self.isleaf() and self.children == None:
      return 1

    to_delete = []
    for i,ch in enumerate(self.children):
      if ch == None:
        continue
      if ch.getSpan(sep='') == child.getSpan(sep='') and str(i) + '_' + path == target_path:
        if direction == 'left':
          to_delete = [ j for j in xrange(i+1)]
        elif direction == 'right':
          to_delete = [ j for j in xrange(i,len(self.children))]
    notdeleted = 1

    for ch in to_delete:
      self.children[ch] = Tree('')
      notdeleted = 0
    if notdeleted == 0:
      return notdeleted

    index = -1
    for i,ch in enumerate(self.children):
      if ch == None:
        continue
      chnotdeleted = ch.deleteNode(child,target_path,str(i) + '_' + path,direction)
      if chnotdeleted == 0:
        index = i
        break

    if index != -1:
      if direction == 'left':
        to_delete = [ j for j in xrange(index)]
      if direction == 'right':
        to_delete = [ j for j in xrange(index+1,len(self.children))]
      for ch in to_delete:
        self.children[ch] = Tree('')
        notdeleted = 0
    return notdeleted

  def clean4rel(self,blacklist):
    if self.isleaf() or self.children == None:
      return
    for ch in self.children:
      if ch != None:
        ch.clean4rel(blacklist)
      if (ch.isleaf() and self.label not in set(['RB','JJ','JJS','PP','VBG','IN','VBZ','VB','ADJP','XP']) and ch.label not in blacklist) or ch.label in set(['being','is','show','was','are','that','s','very','be','like','who','whom','whose']):
        ch.label = ''

  def firstNP(self,path,blacklist):
    if self.isleaf() or self.children == None:
      return None,''

    if self.label in set(['NN','NNS','NP']) and not(self.children[0] != None and self.children[0].label in blacklist):
#    if self.label in set(['NP']) and not(self.children[0] != None and self.children[0].label in blacklist):
      return self,path

    best_f, best_p = None,''
    for i,ch in enumerate(self.children):
      if ch == None:
        continue
      found,fpath = ch.firstNP(path+'_'+str(i), blacklist)
      if best_p == '' or (len(fpath)<len(best_p) and fpath != ''):
        best_f = found
        best_p = fpath
    return best_f,best_p

  def checkTree(self):
    out = 1
    if self.getSpan(filtered = True) == '':
      out = 0

    if self.children == None:
      return out

    if len(self.children) == 2 and self.label != 'and':
      out = 0

    for ch in self.children:
      if ch == None:
        continue
      out *= ch.checkTree()
    return out

  def findTriplet(self,blacklist, mode = 1, verbose = False, depth = 0):
    '''
    mode 0 : depth-1 triplet
    mode 1 : depth-n triplet
    mode 2 : depth-n triplet with candidate fix for parse tree
    '''
    if self.children == None or len(self.children) == 1 or len(self.leaves()) <= 2 or self.children[0] == None or self.children[1] == None:
      return returnList((1,"(loc "+ "_".join(self.getSpan(filtered = True).split('_')) +")"),mode)
    if self.children != None:
      ch_labels = set()
      for ch in self.children:
        if ch == None:
          continue
        ch_labels.add(ch.label)
      if len(ch_labels.difference(set(['NN','NNS','JJ','JJR','ADJP','DT','CD','VBN','VBG']))) == 0:
        return returnList((1,"(loc "+ "_".join(self.getSpan(filtered = True).split('_')) +")"), mode)

    ii = -1
    l, lpath, lidx = None, None, None
    r, rpath, ridx = None, None, None
    for ch in self.children:
      ii += 1
      if ch == None:
        continue
      l, lpath = ch.firstNP('_'+str(ii),blacklist)
      if l != None:
        lidx = ii
        break


    for ch in self.children[ii+1:]:
      ii += 1
      if ch == None:
        continue
      r, rpath = ch.firstNP('_'+str(ii),blacklist)
      if r != None:
        ridx = ii
        break


#    l,lpath = self.children[0].firstNP('_0',blacklist)
#    r,rpath = self.children[1].firstNP('_1',blacklist)

    l = deepcopy(l)
    r = deepcopy(r)

    if l == None:
      return returnList((1,"(loc "+self.getSpan(filtered = True)+")"),mode)
    if r == None and l != None:
      dl,l3 = deepcopy(l).findTriplet(blacklist, mode, verbose, depth = depth+1)
      # if len(lpath) == 2:
      #   dl,l3 = deepcopy(self.children[lidx]).findTriplet(blacklist, mode, verbose, depth = depth+1)
      # else:
      #   dl,l3 = returnList((1,"(loc "+ "_".join(l.getSpan(filtered = True).split('_')) +")"),mode)

      if lidx == 0:
        self.deleteNode(l,lpath[::-1],'','')
      else:
        self.deleteNode(l,lpath[::-1],'','right')
      self.clean4rel(blacklist)
      rel = self.getSpan(filtered = True)
      if rel == '':
        return dl,l3
      else:
        if mode == 2:
          tot_d, out = [],[]
          for depth_l, sub_left in zip(dl,l3):
            tot_d.append(depth_l+1)
            out.append(" ".join(['(' + rel, sub_left+')']))
          return tot_d,out
        else:
          return dl+1, "(" + rel + ' ' + l3+')'

    self.deleteNode(l,lpath[::-1],'','left')
    self.deleteNode(r,rpath[::-1],'','right')

    if verbose:
      print "="*20
      print "root:",self.getSpan(filtered = True)
      print "l:",lpath, "|", l.getSpan(filtered = True)
      print "r:",rpath, "|", r.getSpan(filtered = True)
      print "*"*10

    if mode == 0:
      dl,l3 = 1,l.getSpan(filtered = True)
      dr,r3 = 1,r.getSpan(filtered = True)
    else:
      dl,l3  = deepcopy(l).findTriplet(blacklist,mode,verbose, depth = depth+1)
      dr,r3  = deepcopy(r).findTriplet(blacklist,mode,verbose, depth = depth+1)
    self.clean4rel(blacklist)
    rel = self.getSpan(filtered = True)

    if mode == 2:
      tot_d, out = [],[]
      for ii,(depth_l, sub_left) in enumerate(zip(dl,l3)):
        for jj,(depth_r, sub_right) in enumerate(zip(dr,r3)):
          if ii == 0 and depth == 0:
            if rel == '':
              tot_d.append(depth_l+depth_r)
              out.append(" ".join(['(and' + sub_left , sub_right +')']))
            else:
              tot_d.append(depth_l+depth_r)
              out.append(" ".join(['(and', sub_left, "(" + rel , sub_right+'))']))
            continue
          if rel == '':
            tot_d.append(depth_l+depth_r)
            out.append(" ".join(['(and' + sub_left , sub_right +')']))
          else:
            tot_d.append(depth_l+depth_r)
            tot_d.append(depth_l+depth_r)
            out.append(" ".join(['(and', sub_left, "(" + rel , sub_right+'))']))
            out.append(" ".join(['(and', sub_right, "(" + rel, sub_left+'))']))
    else:
      tot_d, out = 0,'()'
      if rel == '':
       tot_d,out = dl+dr," ".join(['(and' + l3 , r3 +')'])
      else:
       tot_d,out = dl+dr," ".join(['(and', l3, "(" + rel , r3 +'))'])
    if verbose:
      if mode == 2:
        for d,o in zip(tot_d,out):
          print d,o
          print "="*20
      else:
          print tot_d,out
    return tot_d, out
