#from __future__ import absolute_import
import re, collections

def words(text):
  return re.findall('[a-z]+', text.lower())


class Spellchecker(object):
  def __init__(self,vocab):
    self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
    self.model = self.train(vocab)

  def train(self,features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
      model[f] += 1
    return model

  def edits1(self,word):
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in self.alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in self.alphabet]
    return set(deletes + transposes + replaces + inserts)

  def known_edits2(self,word):
    return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.model)

  def known(self,words):
    return set(w for w in words if w in self.model)

  def correct(self,word):
    candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
    return list(candidates)
