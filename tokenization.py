# encoding=utf-8
# partly modified from DAGN

from dataclasses import dataclass, field
import argparse
from transformers import AutoTokenizer
import gensim
import re
import numpy as np
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from itertools import groupby
from operator import itemgetter
import copy
stemmer = SnowballStemmer("english")

def has_same_logical_component(set1, set2):
    has_same = False
    overlap = -1
    if len(set1) > 1 and len(set2) > 1:
        overlap = len(set1 & set2)/max(min(len(set1), len(set2)), 1)
        if overlap > 0.5:  # hyper-parameter:0.5
            has_same = 1
    return has_same, overlap


def token_stem(token):
    return stemmer.stem(token)

def get_node_tag(bpe_tokens):
    i = 0
    mask_tag, tag_now = 0, 0
    cond_tag, res_tag = 1, 2
    node_tag = []
    while i < len(bpe_tokens):
        if bpe_tokens[i] == "<cond>" or bpe_tokens[i] == "<mask>" or bpe_tokens[i] == "<unk>":
            tag_now += 1
            # node_tag.append(tag_now)
            if bpe_tokens[i] == "<mask>":
                node_tag.append(cond_tag)
            else:
                node_tag.append(res_tag)
            bpe_tokens.pop(i)
            i += 1
        elif bpe_tokens[i] == "</cond>" or bpe_tokens[i] == "</s>":
            bpe_tokens.pop(i)
        else:
            node_tag.append(mask_tag)
            i += 1
    return bpe_tokens, node_tag

def our_tokenizer(text_a, text_b, text_c, tokenizer, stopwords, relations:dict, punctuations:list,
                  max_gram:int, max_length:int, do_lower_case:bool=False):
    '''
    :param text_a: str. (context in a sample.)
    :param text_b: str. ([#1] option in a sample. [#2] question + option in a sample.)
    :param text_c: str. (question in a sample)
    :param tokenizer: RoBERTa tokenizer.
    :param relations: dict. {argument words: pattern}
    :return:
        - input_ids: list[int]
        - attention_mask: list[int]
        - segment_ids: list[int]
        - argument_bpe_ids: list[int]. value={ -1: padding,
                                                0: non_arg_non_dom,
                                                1: (relation, head, tail)  关键词在句首
                                                2: (head, relation, tail)  关键词在句中，先因后果
                                                3: (tail, relation, head)  关键词在句中，先果后因
                                                }
        - domain_bpe_ids: list[int]. value={ -1: padding,
                                              0:non_arg_non_dom,
                                           D_id: domain word ids.}
        - punctuation_bpe_ids: list[int]. value={ -1: padding,
                                                   0: non_punctuation,
                                                   1: punctuation}
    '''

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _is_stopwords(word, stopwords):
        if word in stopwords:
            is_stopwords_flag = True
        else:
            is_stopwords_flag = False
        return is_stopwords_flag

    def _head_tail_is_stopwords(span, stopwords):
        if span[0] in stopwords or span[-1] in stopwords:
            return True
        else:
            return False

    def _with_septoken(ngram, tokenizer):
        if tokenizer.bos_token in ngram or tokenizer.sep_token in ngram or tokenizer.eos_token in ngram:
            flag = True
        else: flag = False
        return flag

    def _is_argument_words(seq, argument_words):
        pattern = None
        arg_words = list(argument_words.keys())
        if seq.strip() in arg_words:
            pattern = argument_words[seq.strip()]
        return pattern

    def _is_exist(exists:list, start:int, end:int):
        flag = False
        for estart, eend in exists:
            if estart <= start and eend >= end:
                flag = True
                break
        return flag

    def _find_punct(tokens, punctuations):
        punct_ids = [0] * len(tokens)
        for i, token in enumerate(tokens):
            if token in punctuations:
                punct_ids[i] = 1
        return punct_ids

    def _find_arg_ngrams(tokens, max_gram):
        n_tokens = len(tokens)
        global_arg_start_end = []
        argument_words = {}
        argument_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.
                window_start, window_end = i, i + n
                ngram = " ".join(tokens[window_start:window_end])
                pattern = _is_argument_words(ngram, relations)
                if pattern:
                    if not _is_exist(global_arg_start_end, window_start, window_end):
                        global_arg_start_end.append((window_start, window_end))
                        argument_ids[window_start:window_end] = [pattern] * (window_end - window_start)
                        argument_words[ngram] = (window_start, window_end)

        return argument_words, argument_ids

    def _find_dom_ngrams_2(tokens, max_gram):
        '''
        1. 判断 stopwords 和 sep token
        2. 先遍历一遍，记录 n-gram 的重复次数和出现位置
        3. 遍历记录的 n-gram, 过滤掉 n-gram 子序列（直接比较 str）
        4. 赋值 domain_ids.

        '''

        stemmed_tokens = [token_stem(token) for token in tokens]

        ''' 1 & 2'''
        n_tokens = len(tokens)
        d_ngram = {}
        domain_words_stemmed = {}
        domain_words_orin = {}
        domain_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.

                window_start, window_end = i, i+n
                stemmed_span = stemmed_tokens[window_start:window_end]
                stemmed_ngram = " ".join(stemmed_span)
                orin_span = tokens[window_start:window_end]
                orin_ngram = " ".join(orin_span)

                if _is_stopwords(orin_ngram, stopwords): continue
                if _head_tail_is_stopwords(orin_span, stopwords): continue
                if _with_septoken(orin_ngram, tokenizer): continue

                if not stemmed_ngram in d_ngram:
                    d_ngram[stemmed_ngram] = []
                d_ngram[stemmed_ngram].append((window_start, window_end))

        ''' 3 '''
        d_ngram = dict(filter(lambda e: len(e[1]) > 1, d_ngram.items()))
        raw_domain_words = list(d_ngram.keys())
        raw_domain_words.sort(key=lambda s: len(s), reverse=True)  # sort by len(str).
        domain_words_to_remove = []
        for i in range(0, len(d_ngram)):
            for j in range(i+1, len(d_ngram)):
                if raw_domain_words[i] in raw_domain_words[j]:
                    domain_words_to_remove.append(raw_domain_words[i])
                if raw_domain_words[j] in raw_domain_words[i]:
                    domain_words_to_remove.append(raw_domain_words[j])
        for r in domain_words_to_remove:
            try:
                del d_ngram[r]
            except:
                pass

        ''' 4 '''
        d_id = 0
        for stemmed_ngram, start_end_list in d_ngram.items():
            d_id += 1
            for start, end in start_end_list:
                domain_ids[start:end] = [d_id] * (end - start)
                rebuilt_orin_ngram = " ".join(tokens[start: end])
                if not stemmed_ngram in domain_words_stemmed:
                    domain_words_stemmed[stemmed_ngram] = []
                if not rebuilt_orin_ngram in domain_words_orin:
                    domain_words_orin[rebuilt_orin_ngram] = []
                domain_words_stemmed[stemmed_ngram] +=  [(start, end)]
                domain_words_orin[rebuilt_orin_ngram] += [(start, end)]


        return domain_words_stemmed, domain_words_orin, domain_ids

    ''' start '''
    bpe_tokens_a = tokenizer.tokenize(text_a)
    bpe_tokens_b = tokenizer.tokenize(text_b)
    bpe_tokens_c = tokenizer.tokenize(text_c)

    bpe_tokens = [tokenizer.bos_token] + bpe_tokens_a + [tokenizer.sep_token] + \
                    bpe_tokens_b + [tokenizer.sep_token] + \
                    bpe_tokens_c + [tokenizer.eos_token]

    a_mask = [1] * (len(bpe_tokens_a) + 2) + [0] * (max_length - (len(bpe_tokens_a) + 2))
    b_mask = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1) + [0] * (max_length - (len(bpe_tokens_a) + 2 + len(bpe_tokens_b) + 1))
    c_mask = [0] * (len(bpe_tokens_a) + 2) + [0] * (len(bpe_tokens_b) + 1) + [1] * (len(bpe_tokens_c) + 1) + [0] * (max_length - len(bpe_tokens))

    a_mask = a_mask[:max_length]
    b_mask = b_mask[:max_length]
    c_mask = c_mask[:max_length]
    assert len(a_mask) == max_length, 'len_a_mask={}, max_len={}'.format(len(a_mask), max_length)
    assert len(b_mask) == max_length, 'len_b_mask={}, max_len={}'.format(len(b_mask), max_length)
    assert len(c_mask) == max_length, 'len_c_mask={}, max_len={}'.format(len(c_mask), max_length)

    # adapting Ġ.
    assert isinstance(bpe_tokens, list)
    bare_tokens = [token[1:] if "Ġ" in token else token for token in bpe_tokens]
    argument_words, argument_space_ids = _find_arg_ngrams(bare_tokens, max_gram=max_gram)
    domain_words_stemmed, domain_words_orin, domain_space_ids = _find_dom_ngrams_2(bare_tokens, max_gram=max_gram)
    punct_space_ids = _find_punct(bare_tokens, punctuations)
    argument_bpe_ids = argument_space_ids
    domain_bpe_ids = domain_space_ids
    punct_bpe_ids = punct_space_ids

    ''' output items '''
    input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1)

    padding = [0] * (max_length - len(input_ids))
    padding_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids))
    arg_dom_padding_ids = [-1] * (max_length - len(input_ids))
    input_ids += padding_ids
    argument_bpe_ids += arg_dom_padding_ids
    domain_bpe_ids += arg_dom_padding_ids
    punct_bpe_ids += arg_dom_padding_ids
    input_mask += padding
    segment_ids += padding

    input_ids = input_ids[:max_length]
    input_mask = input_mask[:max_length]
    segment_ids = segment_ids[:max_length]
    argument_bpe_ids = argument_bpe_ids[:max_length]
    domain_bpe_ids = domain_bpe_ids[:max_length]
    punct_bpe_ids = punct_bpe_ids[:max_length]

    assert len(input_ids) <= max_length, 'len_input_ids={}, max_length={}'.format(len(input_ids), max_length)
    assert len(input_mask) <= max_length, 'len_input_mask={}, max_length={}'.format(len(input_mask), max_length)
    assert len(segment_ids) <= max_length, 'len_segment_ids={}, max_length={}'.format(len(segment_ids), max_length)
    assert len(argument_bpe_ids) <= max_length, 'len_argument_bpe_ids={}, max_length={}'.format(
        len(argument_bpe_ids), max_length)
    assert len(domain_bpe_ids) <= max_length, 'len_domain_bpe_ids={}, max_length={}'.format(
        len(domain_bpe_ids), max_length)
    assert len(punct_bpe_ids) <= max_length, 'len_punct_bpe_ids={}, max_length={}'.format(
        len(punct_bpe_ids), max_length)

    ''' get the co-reference relation '''
    pattern = r',|\.|;|:'
    sent_list_origin = re.split(pattern, text_a + ' ' + text_b)
    sent_list = [l.strip() for l in sent_list_origin if l!=""]    # spilt the sentence into unit by punctuation
    bare_tokens_ab = [token[1:] if "Ġ" in token else token for token in bpe_tokens_a+bpe_tokens_b]   # context + option
    punct_space_ids_ab = _find_punct(bare_tokens_ab, punctuations)
    assert len(sent_list) == punct_space_ids_ab.count(1), 'len_sent_ids={}, sum_punct_space_ids_ab={}'.format(
        len(sent_list), punct_space_ids_ab.count(1))
    sent_word_set = [set(sent.split())-set(stopwords) for sent in sent_list]  # split each sentence into word set & delete the stopwords
    # print(sent_word_set)
    # print(sent_list)
    coocc_mat = np.zeros((len(sent_word_set), len(sent_word_set)), dtype=int)    # initialize the coocc matrix
    for i in range(len(sent_word_set)):
        for j in range(i+1, len(sent_word_set)):
            if has_same_logical_component(sent_word_set[i], sent_word_set[j]):
                coocc_mat[i, j] = 1
                coocc_mat[j, i] = 1

    output = {}
    output["input_tokens"] = bpe_tokens
    output["input_ids"] = input_ids
    output["attention_mask"] = input_mask
    output["token_type_ids"] = segment_ids
    output["argument_bpe_ids"] = argument_bpe_ids
    output["domain_bpe_ids"] = domain_bpe_ids
    output["punct_bpe_ids"] = punct_bpe_ids
    output["a_mask"] = a_mask
    output["b_mask"] = b_mask
    output["c_mask"] = c_mask
    output["coocc"] = coocc_mat.tolist()

    return output


def arg_tokenizer(text_a, text_b, text_c, tokenizer, stopwords, relations:dict, punctuations:list,
                  max_gram:int, max_length:int, do_lower_case:bool=False):
    '''
    :param text_a: str. (context in a sample.)
    :param text_b: str. ([#1] option in a sample. [#2] question + option in a sample.)
    :param text_c: str. (question in a sample)
    :param tokenizer: RoBERTa tokenizer.
    :param relations: dict. {argument words: pattern}
    :return:
        - input_ids: list[int]
        - attention_mask: list[int]
        - segment_ids: list[int]
        - argument_bpe_ids: list[int]. value={ -1: padding,
                                                0: non_arg_non_dom,
                                                1: (relation, head, tail)  关键词在句首
                                                2: (head, relation, tail)  关键词在句中，先因后果
                                                3: (tail, relation, head)  关键词在句中，先果后因
                                                }
        - domain_bpe_ids: list[int]. value={ -1: padding,
                                              0:non_arg_non_dom,
                                           D_id: domain word ids.}
        - punctuation_bpe_ids: list[int]. value={ -1: padding,
                                                   0: non_punctuation,
                                                   1: punctuation}
    '''

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _is_stopwords(word, stopwords):
        if word in stopwords:
            is_stopwords_flag = True
        else:
            is_stopwords_flag = False
        return is_stopwords_flag

    def _head_tail_is_stopwords(span, stopwords):
        if span[0] in stopwords or span[-1] in stopwords:
            return True
        else:
            return False

    def _with_septoken(ngram, tokenizer):
        if tokenizer.bos_token in ngram or tokenizer.sep_token in ngram or tokenizer.eos_token in ngram:
            flag = True
        else: flag = False
        return flag

    def _is_argument_words(seq, argument_words):
        pattern = None
        arg_words = list(argument_words.keys())
        if seq.strip() in arg_words:
            pattern = argument_words[seq.strip()]
        return pattern

    def _is_exist(exists:list, start:int, end:int):
        flag = False
        for estart, eend in exists:
            if estart <= start and eend >= end:
                flag = True
                break
        return flag

    def _find_punct(tokens, punctuations):
        punct_ids = [0] * len(tokens)
        for i, token in enumerate(tokens):
            if token in punctuations:
                punct_ids[i] = 1
        return punct_ids

    def _find_arg_ngrams(tokens, max_gram):
        n_tokens = len(tokens)
        global_arg_start_end = []
        argument_words = {}
        argument_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.
                window_start, window_end = i, i + n
                ngram = " ".join(tokens[window_start:window_end])
                pattern = _is_argument_words(ngram, relations)
                if pattern:
                    if not _is_exist(global_arg_start_end, window_start, window_end):
                        global_arg_start_end.append((window_start, window_end))
                        argument_ids[window_start:window_end] = [pattern] * (window_end - window_start)
                        argument_words[ngram] = (window_start, window_end)

        return argument_words, argument_ids

    def _find_dom_ngrams_2(tokens, max_gram):
        '''
        1. 判断 stopwords 和 sep token
        2. 先遍历一遍，记录 n-gram 的重复次数和出现位置
        3. 遍历记录的 n-gram, 过滤掉 n-gram 子序列（直接比较 str）
        4. 赋值 domain_ids.

        '''

        stemmed_tokens = [token_stem(token) for token in tokens]

        ''' 1 & 2'''
        n_tokens = len(tokens)
        d_ngram = {}
        domain_words_stemmed = {}
        domain_words_orin = {}
        domain_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.

                window_start, window_end = i, i+n
                stemmed_span = stemmed_tokens[window_start:window_end]
                stemmed_ngram = " ".join(stemmed_span)
                orin_span = tokens[window_start:window_end]
                orin_ngram = " ".join(orin_span)

                if _is_stopwords(orin_ngram, stopwords): continue
                if _head_tail_is_stopwords(orin_span, stopwords): continue
                if _with_septoken(orin_ngram, tokenizer): continue

                if not stemmed_ngram in d_ngram:
                    d_ngram[stemmed_ngram] = []
                d_ngram[stemmed_ngram].append((window_start, window_end))

        ''' 3 '''
        d_ngram = dict(filter(lambda e: len(e[1]) > 1, d_ngram.items()))
        raw_domain_words = list(d_ngram.keys())
        raw_domain_words.sort(key=lambda s: len(s), reverse=True)  # sort by len(str).
        domain_words_to_remove = []
        for i in range(0, len(d_ngram)):
            for j in range(i+1, len(d_ngram)):
                if raw_domain_words[i] in raw_domain_words[j]:
                    domain_words_to_remove.append(raw_domain_words[i])
                if raw_domain_words[j] in raw_domain_words[i]:
                    domain_words_to_remove.append(raw_domain_words[j])
        for r in domain_words_to_remove:
            try:
                del d_ngram[r]
            except:
                pass

        ''' 4 '''
        d_id = 0
        for stemmed_ngram, start_end_list in d_ngram.items():
            d_id += 1
            for start, end in start_end_list:
                domain_ids[start:end] = [d_id] * (end - start)
                rebuilt_orin_ngram = " ".join(tokens[start: end])
                if not stemmed_ngram in domain_words_stemmed:
                    domain_words_stemmed[stemmed_ngram] = []
                if not rebuilt_orin_ngram in domain_words_orin:
                    domain_words_orin[rebuilt_orin_ngram] = []
                domain_words_stemmed[stemmed_ngram] +=  [(start, end)]
                domain_words_orin[rebuilt_orin_ngram] += [(start, end)]


        return domain_words_stemmed, domain_words_orin, domain_ids



    ''' start '''
    bpe_tokens_a = tokenizer.tokenize(text_a)
    bpe_tokens_b = tokenizer.tokenize(text_b)
    bpe_tokens_c = tokenizer.tokenize(text_c)

    bpe_tokens = [tokenizer.bos_token] + bpe_tokens_a + [tokenizer.sep_token] + \
                    bpe_tokens_b + [tokenizer.sep_token] + \
                    bpe_tokens_c + [tokenizer.eos_token]

    a_mask = [1] * (len(bpe_tokens_a) + 2) + [0] * (max_length - (len(bpe_tokens_a) + 2))
    b_mask = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1) + [0] * (max_length - (len(bpe_tokens_a) + 2 + len(bpe_tokens_b) + 1))
    c_mask = [0] * (len(bpe_tokens_a) + 2) + [0] * (len(bpe_tokens_b) + 1) + [1] * (len(bpe_tokens_c) + 1) + [0] * (max_length - len(bpe_tokens))

    a_mask = a_mask[:max_length]
    b_mask = b_mask[:max_length]
    c_mask = c_mask[:max_length]
    assert len(a_mask) == max_length, 'len_a_mask={}, max_len={}'.format(len(a_mask), max_length)
    assert len(b_mask) == max_length, 'len_b_mask={}, max_len={}'.format(len(b_mask), max_length)
    assert len(c_mask) == max_length, 'len_c_mask={}, max_len={}'.format(len(c_mask), max_length)

    # adapting Ġ.
    assert isinstance(bpe_tokens, list)
    bare_tokens = [token[1:] if "Ġ" in token else token for token in bpe_tokens]
    argument_words, argument_space_ids = _find_arg_ngrams(bare_tokens, max_gram=max_gram)
    domain_words_stemmed, domain_words_orin, domain_space_ids = _find_dom_ngrams_2(bare_tokens, max_gram=max_gram)
    punct_space_ids = _find_punct(bare_tokens, punctuations)

    argument_bpe_ids = argument_space_ids
    domain_bpe_ids = domain_space_ids
    punct_bpe_ids = punct_space_ids

    ''' output items '''
    input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1)
    padding = [0] * (max_length - len(input_ids))
    padding_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids))
    arg_dom_padding_ids = [-1] * (max_length - len(input_ids))
    input_ids += padding_ids
    argument_bpe_ids += arg_dom_padding_ids
    domain_bpe_ids += arg_dom_padding_ids
    punct_bpe_ids += arg_dom_padding_ids
    input_mask += padding
    segment_ids += padding

    input_ids = input_ids[:max_length]
    input_mask = input_mask[:max_length]
    segment_ids = segment_ids[:max_length]
    argument_bpe_ids = argument_bpe_ids[:max_length]
    domain_bpe_ids = domain_bpe_ids[:max_length]
    punct_bpe_ids = punct_bpe_ids[:max_length]

    assert len(input_ids) <= max_length, 'len_input_ids={}, max_length={}'.format(len(input_ids), max_length)
    assert len(input_mask) <= max_length, 'len_input_mask={}, max_length={}'.format(len(input_mask), max_length)
    assert len(segment_ids) <= max_length, 'len_segment_ids={}, max_length={}'.format(len(segment_ids), max_length)
    assert len(argument_bpe_ids) <= max_length, 'len_argument_bpe_ids={}, max_length={}'.format(
        len(argument_bpe_ids), max_length)
    assert len(domain_bpe_ids) <= max_length, 'len_domain_bpe_ids={}, max_length={}'.format(
        len(domain_bpe_ids), max_length)
    assert len(punct_bpe_ids) <= max_length, 'len_punct_bpe_ids={}, max_length={}'.format(
        len(punct_bpe_ids), max_length)


    ''' get the co-reference relation original version '''
    '''
    pattern = r',|\.|;|:'
    sent_list_origin = re.split(pattern, text_a + ' ' + text_b)
    sent_list = [l.strip() for l in sent_list_origin if l.strip()!=""]    # spilt the sentence into unit by punctuation

    bare_tokens_ab = [token[1:] if "Ġ" in token else token for token in bpe_tokens_a+bpe_tokens_b]   # context + option
    punct_space_ids_ab = _find_punct(bare_tokens_ab, punctuations)
    if punct_space_ids_ab[-1] != 1:
        punct_space_ids_ab.append(1)
    if len(sent_list) != punct_space_ids_ab.count(1):
        print(text_a)
        print(sent_list)
        print(bare_tokens_ab)

    assert len(sent_list) == punct_space_ids_ab.count(1), 'len_sent_list={}, sum_punct_space_ids_ab={}'.format(
        len(sent_list), punct_space_ids_ab.count(1))
    sent_word_set = [set(sent.split())-set(stopwords) for sent in sent_list]  # split each sentence into word set & delete the stopwords
    # print(sent_word_set)
    # print(sent_list)

    max_nodes = 32
    coocc = []    # initialize the coocc matrix
    for i in range(len(sent_word_set)):
        for j in range(i+1, len(sent_word_set)):
            if has_same_logical_component(sent_word_set[i], sent_word_set[j]):
                coocc.append((i, j))
    coocc += [(-1,-1)] * (max_nodes-len(coocc))
    '''

    ''' get the co-reference relation new version '''
    def _consecutive(seq: list, vals: np.array):
        groups_seq = []
        output_vals = copy.deepcopy(vals)
        for k, g in groupby(enumerate(seq), lambda x: x[0] - x[1]):
            groups_seq.append(list(map(itemgetter(1), g)))
        output_seq = []
        for i, ids in enumerate(groups_seq):
            output_seq.append(ids[0])
            if len(ids) > 1:
                output_vals[ids[0]:ids[-1] + 1] = min(output_vals[ids[0]:ids[-1] + 1])
        return groups_seq, output_seq, output_vals

    max_rel_id = 4
    new_punct_id = max_rel_id + 1  # new_punct_id:5
    new_punct_bpe_ids = [i*new_punct_id for i in punct_bpe_ids]  # punct_id: 1 -> 5. for incorporating with argument_bpe_ids.
    _flat_all_bpe_ids = list(map(lambda x,y:x+y, argument_bpe_ids, new_punct_bpe_ids))  # -1:padding, 0:non, 1-4: arg, 5:punct.
    overlapped_punct_argument_mask = [1 if bpe_id > new_punct_id else 0 for bpe_id in _flat_all_bpe_ids]
    flat_all_bpe_ids = list(map(lambda x,y:x*y, _flat_all_bpe_ids, [1-i for i in overlapped_punct_argument_mask])) \
                        + list(map(lambda x,y:x*y, argument_bpe_ids, overlapped_punct_argument_mask))
    assert max(argument_bpe_ids) <= new_punct_id

    item_seq_len = sum(a_mask) + sum(b_mask)
    item_split_ids = np.array(flat_all_bpe_ids[:item_seq_len])  # type:numpy.array,
    # split_ids_indices = np.where(item_split_ids > 4)[0].tolist()  # select id==5(punctuation)
    split_ids_indices = np.where(item_split_ids > 0)[0].tolist()  # select id==5(punctuation)
    grouped_split_ids_indices, split_ids_indices, item_split_ids = _consecutive(
        split_ids_indices, item_split_ids)
    # print(split_ids_indices)  # [0, 16, 20, 26, 30, 44, 53, 63, 71, 85, 93, 112]
    n_split_ids = len(split_ids_indices)    # the number of split_ids
    item_node_in_seq_indices = []
    sent_list = []
    for i in range(n_split_ids):
        if i != n_split_ids - 1:
            item_node_in_seq_indices.append([i for i in range(grouped_split_ids_indices[i][-1] + 1,
                                                              grouped_split_ids_indices[i + 1][0])])
            sent_list.append(bare_tokens[item_node_in_seq_indices[-1][0]:item_node_in_seq_indices[-1][-1]+1])
    sent_token_set = [set(sent) - set(stopwords+["<", ">", "b", "i", "e", "g", "</", "."]) for sent in sent_list]  # delete the stopwords and convert to set
    coocc = []    # initialize the coocc matrix

    max_nodes = 512
    for i in range(len(sent_token_set)):
        for j in range(i+1, len(sent_token_set)):
            has_same, overlap = has_same_logical_component(sent_token_set[i], sent_token_set[j])
            if has_same:    # judge has_same
                coocc.append((i, j, overlap))
    coocc += [(-1,-1,-1)] * (max_nodes-len(coocc))
    assert len(coocc) <= max_nodes, 'len_coocc={}, max_nodes={}'.format(
        len(coocc), max_nodes)


    '''
    bare_tokens_ab = [token[1:] if "Ġ" in token else token for token in bpe_tokens_a + [tokenizer.sep_token] + bpe_tokens_b]   # context + option
    punct_space_ids_ab = _find_punct(bare_tokens_ab, punctuations)
    # print(punct_space_ids_ab)
    if punct_space_ids_ab[-1] != 1:
        punct_space_ids_ab.append(1)
    punct_indices_list = [i for i in range(len(punct_space_ids_ab)) if punct_space_ids_ab[i]==1]
    ab_length = len(punct_space_ids_ab)
    if punct_indices_list[-1] == ab_length-1:
        punct_indices_list.pop()
    sent_list = []
    for i in range(len(punct_indices_list)):
        if i == 0 and i == len(punct_indices_list)-1:
            sent_list.append(bare_tokens_ab[0:punct_indices_list[i]])
            sent_list.append(bare_tokens_ab[punct_indices_list[i]+1:ab_length])
        elif i == 0:
            sent_list.append(bare_tokens_ab[0:punct_indices_list[i]])
        elif i == len(punct_indices_list)-1:
            sent_list.append(bare_tokens_ab[punct_indices_list[i - 1] + 1:punct_indices_list[i]])
            sent_list.append(bare_tokens_ab[punct_indices_list[i]+1:ab_length])
        else:
            sent_list.append(bare_tokens_ab[punct_indices_list[i-1]+1:punct_indices_list[i]])

    sent_token_set = [set(sent) - set(stopwords+["<", ">", "b", "i", "e", "g", "</", "."]) for sent in sent_list]  # delete the stopwords and convert to set

    max_nodes = 140
    coocc = []    # initialize the coocc matrix
    for i in range(len(sent_token_set)):
        for j in range(i+1, len(sent_token_set)):
            if has_same_logical_component(sent_token_set[i], sent_token_set[j]):
                coocc.append((i, j))
    # print(ab_length)
    # print(punct_indices_list)
    # print(sent_list)
    if len(coocc) > 140:
        print(text_a)
    coocc += [(-1,-1)] * (max_nodes-len(coocc))

    if len(sent_list) != punct_space_ids_ab.count(1):
        print(text_a)
        print(sent_list)
        print(bare_tokens_ab)
    # print(coocc)
    '''

    output = {}
    output["input_tokens"] = bpe_tokens
    output["input_ids"] = input_ids
    output["attention_mask"] = input_mask
    output["token_type_ids"] = segment_ids
    output["argument_bpe_ids"] = argument_bpe_ids
    output["domain_bpe_ids"] = domain_bpe_ids
    output["punct_bpe_ids"] = punct_bpe_ids
    output["a_mask"] = a_mask
    output["b_mask"] = b_mask
    output["c_mask"] = c_mask
    output["coocc"] = coocc

    return output

def main(text, option, question, logic, punctuations):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    stopwords = list(gensim.parsing.preprocessing.STOPWORDS) + punctuations

    inputs = arg_tokenizer(text, option, question, tokenizer, stopwords, relations, punctuations, 5, 256)

    ''' print '''
    # p = []
    # for token, arg, dom, pun in zip(inputs["input_tokens"], inputs["argument_bpe_ids"], inputs["domain_bpe_ids"],
    #                                 inputs["punct_bpe_ids"]):
    #     p.append((token, arg, dom, pun))
    # print(p)
    # print('input_tokens\n{}'.format(inputs["input_tokens"]))
    # print('input_ids\n{}, size={}'.format(inputs["input_ids"], len(inputs["input_ids"])))
    # print('attention_mask\n{}'.format(inputs["attention_mask"]))
    # print('token_type_ids\n{}'.format(inputs["token_type_ids"]))
    # print('argument_bpe_ids\n{}'.format(inputs["argument_bpe_ids"]))
    # print('domain_bpe_ids\n{}, size={}'.format(inputs["domain_bpe_ids"], len(inputs["domain_bpe_ids"])))
    # print('punct_bpe_ids\n{}'.format(inputs["punct_bpe_ids"]))


if __name__ == '__main__':

    import json
    from graph_building_blocks.argument_set_punctuation_v4 import punctuations
    with open('./graph_building_blocks/explicit_arg_set_v4.json', 'r') as f:
        relations = json.load(f)  # key: relations, value: ignore

    context = "There will be three sprinting projects in an institution's track and field sports, namely 60M, 100M and 200M. Lao Zhang, Lao Wang and Lao Li each participated in one of them, and the three people participated in different projects. Lao Li did not participate in 100M, Lao Wang participated in 60M.Xiao Li? Lao Zhang did not participate in 60M, Lao Wang participated in 200M."

    option = "I must be stupid because all intelligent people are nearsighted and I have perfect eyesight."
    question = "The pattern of reasoning displayed above most closely parallels which of the following?"

    logic = [[0,1], [2,0], [3,2], [4,1]]

    main(context, option, question, logic, punctuations)