#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapted from:
CS224N 2022-23: Homework 4
utils.py: Utility Functions
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Vera Lin <veralin@stanford.edu>
Siyan Li <siyanli@stanford.edu>

Edited by Achille Sowa
"""

import math
import os
from typing import List, Tuple, Dict, Set, Union
import torch
import sys
import numpy as np
import nltk
import sentencepiece as spm
import sacrebleu
from collections import namedtuple
import numpy as np
import torch.utils
import torch.utils.data

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents (list[list[str]] or list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str or int): padding token
    @returns sents_padded (list[list[str]] or list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)
    maxlength = max([len(sent) for sent in sents])
    for sent in sents:
        pad = [pad_token for _ in range(maxlength-len(sent))]
        sents_padded.append(sent + pad)
    
    ### END YOUR CODE

    return sents_padded


def read_corpus(file_path, source, vocab_size=2500):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    @param vocab_size (int): number of unique subwords in
        vocabulary when reading and tokenizing
    """
    data = []
    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(source))

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            # only append <s> and </s> to the target sentence
            #if source == 'tgt':
            #    subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)

    return data


def read_target_corpus(lines: List[str], text_model_path: str)-> List[List[str]]:
    """Read lines and tokenize them
    @param lines(List[str]) the corresponding lines
    """
    data = []
    sp = spm.SentencePieceProcessor()
    sp.load(text_model_path)

    for line in lines:
        subword_tokens = sp.encode_as_pieces(line)
        # only append <s> and </s> to the target sentence
        subword_tokens = ['<s>'] + subword_tokens + ['</s>']
        data.append(subword_tokens)

    return data
    


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    # remove the start and end tokens
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    
    # detokenize the subword pieces to get full sentences
    detokened_refs = [''.join(pieces).replace('▁', ' ') for pieces in references]
    detokened_hyps = [''.join(hyp.value).replace('▁', ' ') for hyp in hypotheses]

    # sacreBLEU can take multiple references (golden example per sentence) but we only feed it one
    bleu = sacrebleu.corpus_bleu(detokened_hyps, [detokened_refs])

    return bleu.score


def decode(model, ecg_embeddings, labels, epoch, output_file=None):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    # print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    # test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src', vocab_size=3000)
    # if args['TEST_TARGET_FILE']:
    #     print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
    #     test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt', vocab_size=2000)

    hypotheses = beam_search(model, ecg_embeddings,
                            #  beam_size=int(args['--beam-size']),                      EDIT: BEAM SIZE USED TO BE 5
                             beam_size=10,
                             max_decoding_time_step=70)

    top_hypotheses = [hyps[0] for hyps in hypotheses]
    bleu_score = compute_corpus_level_bleu_score(labels, top_hypotheses)

    if output_file:
        fname, ext = os.path.splitext(output_file)
        output_file = f'{fname}-ep.{epoch}{ext}'
        with open(output_file, 'a') as f:
            for src_sent, hyps in zip(range(len(ecg_embeddings)), hypotheses):
                top_hyp = hyps[0]
                hyp_sent = ''.join(top_hyp.value).replace('▁', ' ')
                f.write(hyp_sent + '\n')

    return bleu_score



def beam_search(model, test_data: List[torch.Tensor], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data (List[torch.Tensor]): List of ecgs, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_ecg in test_data:
            example_hyps = model.beam_search(src_ecg, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: 
        model.train(was_training)

    return hypotheses





def sanity_check():
    "quickly test my pad_sents function"
    sents = [
        ["A", ":", " ", "ok",  "I'm",  "done", "."],
        ["Q", ":", " ", "why", " ", "?", "\n"], 
        ["A", ":", " ", "because"],
        ["Q", ":", "!"]
    ]
    pad_token = "<pad>"
    exp_padded = [
        ["A", ":", " ", "ok",  "I'm",  "done", "."],
        ["Q", ":", " ", "why", " ", "?", "\n"], 
        ["A", ":", " ", "because", pad_token, pad_token, pad_token, ],
        ["Q", ":", "!", pad_token, pad_token, pad_token, pad_token]
    ]
    padded = pad_sents(sents, pad_token)
    assert padded == exp_padded, \
        "expected padded: {}, returned padded: {}".format(padded, exp_padded)
    print("[OK]: pad_sents")

if __name__ == "__main__":
    sanity_check()