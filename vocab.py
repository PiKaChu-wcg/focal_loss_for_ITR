r'''
Author       : PiKaChu_wcg
Date         : 2022-01-14 13:48:19
LastEditors  : PiKachu_wcg
LastEditTime : 2022-01-14 13:57:01
FilePath     : /school/VSE_Pytorch/vocab.py
'''
"""
Constructing and loading dictionaries
"""
import numpy as np
from collections import OrderedDict

def build_dictionary(text):
    """
    Build a dictionary
    text: list of sentences (pre-tokenized)
    """
    wordcount = OrderedDict()
    for cc in text:
        words = cc.split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 0
            wordcount[w] += 1
    words = list(wordcount.keys())
    freqs = list(wordcount.values())
    sorted_idx = np.argsort(freqs)
    worddict = OrderedDict()
    for idx, sidx in enumerate(sorted_idx):
        worddict[words[sidx]] = idx + 3   # 0: <eos>, 1: <unk>
    word_idict = dict()


    for kk, vv in worddict.items():
        word_idict[vv] = kk
    word_idict[0]='<PAD>'
    word_idict[1] = '<EOS>'
    word_idict[2] = '<UNK>'
    return worddict,word_idict, wordcount