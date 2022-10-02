import contextvars
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from collections import Counter

import xml.etree.ElementTree as ElemTree


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    # https://docs.python.org/3/library/xml.etree.elementtree.html
    with open(filename, 'r') as fin:
        str_ = fin.read()
    # ampersand-problem  solution
    str_ = str_.replace('&', '&amp;')
    root = ElemTree.fromstring(str_, parser=ElemTree.XMLParser(encoding="utf-8"))

    sentence_pairs = []
    alignments = []
    for child in root:
        # <english>
        source = child[0].text.split(' ')
        # <czech>
        target = child[1].text.split(' ')
        # <sure>
        sure = []
        if child[2].text is not None:
            pairs = child[2].text.split(' ')
            for p in pairs:
                p1, p2 = p.split('-')
                sure += [(int(p1), int(p2))]
        # <possible>
        possible = []
        if child[3].text is not None:
            pairs = child[3].text.split(' ')
            for p in pairs:
                p1, p2 = p.split('-')
                possible += [(int(p1), int(p2))]
        sentence_pairs += [SentencePair(source, target)]
        alignments += [LabeledAlignment(sure, possible)]

    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff -- natural number -- most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language
        
    Tip: 
        Use cutting by freq_cutoff independently in src and target. Moreover in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary

    """
    cnt_source, cnt_target = [], []
    for pair in sentence_pairs:
        cnt_source += pair.source
        cnt_target += pair.target

    cnt_source, cnt_target = (Counter(cnt_source).most_common(freq_cutoff),
                              Counter(cnt_target).most_common(freq_cutoff))

    dicts_ = [dict(), dict()]

    for d_, c_ in zip(dicts_, [cnt_source, cnt_target]):
        for i, cnt in enumerate(c_):
            d_[cnt[0]] = i

    return dicts_


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    tokenized_pairs = []
    for pair in sentence_pairs:
        tokenized = [[], []]
        sentences = [pair.source, pair.target]
        dicts = [source_dict, target_dict]

        for s_, d_, t_ in zip(sentences, dicts, tokenized):
            for word in s_:
                if word in d_:
                    t_.append(d_[word])

        tokenized = list(map(np.array, tokenized))
        if tokenized[0].size and tokenized[1].size:
            tokenized_pairs += [TokenizedSentencePair(tokenized[0], tokenized[1])]
    return tokenized_pairs
