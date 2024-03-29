from typing import List, Dict, Tuple, Any
from collections import defaultdict
from copy import deepcopy
import logging
import re
import random
import os
import unicodedata
from logging import getLogger, StreamHandler

import click
import requests
from tqdm import tqdm
from sklearn.model_selection import KFold


logger = getLogger('make_dataset')
logger.setLevel(logging.INFO)
stream_handler = StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

HEADERS = {'Content-Type': 'application/json'}

random.seed(0)


def return_ne_label(now_index: int, places: Dict[str, Any],
                    word_len: int, bioul: bool) -> Tuple[str, bool]:
    """
    decide head label and ne_label
    :param now_index: [current index of define position in sentence]
    :type now_index: int
    :param places: [annotation indexes]
    :type places: Dict[str, int]
    :param word_len: [character length of current defining word]
    :type word_len: int
    :param bioul: [if True, BIOUL tagging scheme specification]
    :type bioul: bool
    :return: [tuple(ne_label, Whether it is the end of named entity)]
    :rtype: Tuple[str, bool]
    """

    label = places['ne_label']
    if (now_index >= places['start'] and
            places['start'] + word_len == places['end']):
        if bioul:
            return 'S-' + label, True
        else:
            return 'B-' + label, True
    elif now_index + word_len >= places['end']:
        if bioul:
            return 'E-' + label, True
        else:
            return 'I-' + label, True
    elif places['start'] < now_index < places['end']:
        return 'I-' + label, False
    elif now_index >= places['start']:
        return 'B-' + label, False
    return 'O', False


def request_morph_analysis_api(sentence: str, algo: str, mode: str) \
        -> Tuple[List[str], List[str]]:
    """
    request morphological analysis API and return result
    :param sentence: [a sentence string]
    :type sentence: str
    :param algo: ['mecab', 'mecab_neologd', 'jumanpp', or 'sudachi']
    :type algo: str
    :param mode: [morphological analysis mode of sudachi]
    :type mode: str
    :return: [result of morphological analysis]
    :rtype: Tuple[List[str], List[str]]
    """

    obj = {'sentence': sentence, 'mode': mode}
    res = requests.post('http://localhost:5000/' + algo,
                        json=obj, headers=HEADERS)
    result = res.json()
    words = result['words']
    info = result['info']
    return words, info


def get_annotated_label_info(sentence: str) -> List[Tuple[str, int, int]]:
    """
    get annotation label classes and indexes from annotated marks in a sentence
    :param sentence: [sentence annotated labels]
    :type sentence: str
    :return: [tuple(annotation_label, start_index, end_index)]
    :rtype: List[Tuple[str, int, int]]
    """

    sentence = sentence.replace(' ', '')  # remove white spaces
    only_insert_mark = re.sub(r'\[/l-.+?\]', '', sentence)
    only_close_mark = re.sub(r'\[l-.+?\]', '', sentence)
    insert_idx = get_mark_indexes_and_label(
        only_insert_mark, r'\[l-.+?\]'
    )
    close_idx = get_mark_indexes_and_label(
        only_close_mark, r'\[/l-.+?\]'
    )

    stack_places = []
    for ik, iv in insert_idx.items():
        for i, idx in enumerate(iv):
            stack_places.append((ik, idx, close_idx[ik][i] - 1))
    stack_places.sort(key=lambda x: (x[2], x[1]))
    return stack_places


def get_mark_indexes_and_label(string: str, mark: str) -> Dict[str, List[int]]:
    """
    support `get_annotated_label_info` method
    :param string: [a sentence string]
    :type string: str
    :param mark: [a string using regexp]
    :type mark: str
    :return: [{class: [index1, index2, ...]}]
    :rtype: Dict[List[int]]
    """

    indexes = re.finditer(mark, string)
    mark_indexes = defaultdict(list)
    minus_len = 0
    for idx in indexes:
        ne_label = re.sub(r'\[/*l-', '', idx.group()).replace(']', '')
        mark_indexes[ne_label] += [idx.start() - minus_len]
        minus_len += len(idx.group())

    return mark_indexes


def get_current_labeling_pos(stack_places: List[Tuple[str, int, int]],
                             sentence: str) -> Tuple[List[Any], List[Any]]:
    """
    get current labeling positions from stack_places
    There is duplications of labeling in Sinra dataset.
    Call this method in many times, annotate leaving duplicate information
    :param stack_places: [stacking annotated places]
    :type stack_places: List[Tuple[str, int, int]]
    :param sentence: [a sentence string ignored annotated marks]
    :type sentence: str
    :return: [tuple(list(current labeling positions), list(unlabeled places))]
    :rtype: Tuple[List[Any], List[Any]]
    """

    places = []
    tmp_labels = ['O'] * len(sentence)
    for i in range(len(stack_places)):
        insertflag = True
        kind, start, end = stack_places[i]
        for k in range(start, end):
            if tmp_labels[k] != 'O':
                insertflag = False
                break
        if insertflag:
            tmp_labels[start: end + 1] = [kind] * (end + 1 - start)
            places.append(
                {'ne_label': kind, 'start': start, 'end': end + 1}
            )
            stack_places[i] = ('', -1, -1)  # temporary tuole value
    while stack_places.count(('', -1, -1)) > 0:
        stack_places.remove(('', -1, -1))
    return places, stack_places


def annotation(sentences: List[str], algo: str,
               mode: str, bioul: bool = False,
               char_level: bool = False) -> List[str]:
    """
    annotating sentences
    :param sentences: [list included sentences]
    :type sentences: List[str]
    :param algo: [morphological analysis algorithm]
    :type algo: str
    :param mode: [morphological analysis mode of sudachi]
    :type mode: str
    :param bioul: [if True, using BIOUL labeling scheme], defaults to False
    :type bioul: bool, optional
    :param char_level: [if True, annotating by character], defaults to False
    :type char_level: bool, optional
    :return: [annotated sentences]
    :rtype: List[str]
    """

    tab_num = 1
    tab_counts = []
    annotated_sentences = []
    logger.info('morph analysing ...')
    for sentence in tqdm(sentences):
        sentence = re.sub(r'\s{2,}', '',
                          unicodedata.normalize('NFKC', sentence)).strip()
        if sentence == '':
            continue
        stack_places = get_annotated_label_info(sentence)
        sentence = re.sub(r'\[/*l-.+?\]', '', sentence)
        if char_level:
            morphs = [c for c in sentence.replace(' ', '')]
            words = deepcopy(morphs)
        else:
            words, info = request_morph_analysis_api(sentence, algo, mode)
            morphs = [w + '\t' + i for w,
                      i in zip(words, info) if w != ['', ' ']]

        if len(stack_places) == 0:
            annotated_sentence = [morph + '\t' + label for morph,
                                  label in zip(morphs, ['O'] * len(morphs))]
            if len(annotated_sentence) == 0:
                continue
            tab_counts.append(1)
            annotated_sentences.append(annotated_sentence)
            continue

        tab_cnt = 0
        while len(stack_places) > 0:
            places, stack_places = get_current_labeling_pos(
                stack_places, sentence
            )
            sentence_len = 0
            cnt = 0
            annotation_labels = ['O'] * len(morphs)
            for i, word in enumerate(words):
                if cnt == len(places):
                    break
                annotation_labels[i], flag = return_ne_label(
                    sentence_len, places[cnt], len(word), bioul
                )
                sentence_len += len(word)
                cnt += 1 if flag else 0
            for k, label in enumerate(annotation_labels):
                morphs[k] += '\t' + label
            tab_cnt += 1
            tab_num = tab_cnt if tab_cnt > tab_num else tab_num
        tab_counts.append(tab_cnt)
        annotated_sentences.append(morphs)

    # adjust the number of annotation
    for i, ansentence in enumerate(annotated_sentences):
        for j, morph in enumerate(ansentence):
            annotated_sentences[i][j] += ''.join(
                ['\tO'] * (tab_num - tab_counts[i])
            )
    return ['\n'.join(morphs) for morphs in annotated_sentences]


@click.command()
@click.option('-cp', '--corpus_path', type=str,
              default='data/JP5/annotated_data/City_sentences.txt')
@click.option('-o', '--out_dir', type=str, default='data/JP5/dataset/')
@click.option('-k', '--ksplit_num', type=int, default=1)
@click.option('-b', '--bioul', is_flag=True)
@click.option('-c', '--char_level', is_flag=True)
@click.option('--MECAB', 'morph_analysis', flag_value='mecab', default=True)
@click.option('--NEOLOGD', 'morph_analysis', flag_value='mecab_neologd')
@click.option('--JUMANPP', 'morph_analysis', flag_value='jumanpp')
@click.option('--SUDACHI', 'morph_analysis', flag_value='sudachi')
@click.option('--sudachim', type=click.Choice(['A', 'B', 'C']), default='C')
def main(corpus_path: str, out_dir: str, ksplit_num: int, bioul: bool,
         char_level: bool, morph_analysis: str, sudachim: str):
    """
    :param corpus_path: [a path of annotated corpus]
    :type corpus_path: str
    :param out_dir: [a storage directory of dataset]
    :type out_dir: str
    :param ksplit_num: [dataset split number. default=1 (no split) ]
    :type ksplit_num: int
    :param bioul: [NE label is BIOUL format (IOB2 -> BIOUL).]
    :type bioul: bool
    :param char_level: [make char level annotation.]
    :type char_level: bool
    :param morph_analysis: [morph analysis algorithm]
    :type morph_analysis: str
    :param sudachim: [sudachi morph analyser mode]
    :type sudachim: str
    """

    fname_extract = re.compile(r'([^/]+?)?$')
    out_dir += '/' if out_dir[-1] != '/' else ''
    logger.info('show setting parameters\n\
                 corpus_path: {0}\n\
                 out_dir: {1}\n\
                 ksplit_num: {2}\n\
                 bioul: {3}\n\
                 char_level: {4}\n\
                 morph_analysis: {5}\n\
                 sudachim: {6}'.format(corpus_path, out_dir,
                                       ksplit_num, bioul, char_level,
                                       morph_analysis, sudachim)
                )
    fname = re.sub(r'_.+', '', fname_extract.search(corpus_path).group())
    dirname = out_dir + fname + '/'
    if char_level:
        fname = out_dir + fname + '_char.txt'
    else:
        fname = out_dir + fname + '.txt'
    with open(corpus_path, 'r') as f:
        sentences = f.read().split('\n')
    dataset = annotation(sentences, morph_analysis, sudachim,
                         bioul, char_level)
    num = 1
    logger.info('file saving ...')
    if ksplit_num > 2:
        os.makedirs(dirname, exist_ok=True)
        kf = KFold(n_splits=ksplit_num, shuffle=True, random_state=0)
        for train_idx, test_idx in tqdm(kf.split(dataset)):
            train = [dataset[i] for i in train_idx]
            test = [dataset[i] for i in test_idx]
            dev = random.sample(train, len(test))
            for d in dev:
                train.remove(d)
            if char_level:
                base_name = '_{}_{}_char.txt'.format(morph_analysis, num)
            else:
                base_name = '_{}_{}.txt'.format(morph_analysis, num)
            with open('{}train{}'.format(dirname, base_name), 'w') as trf, \
                    open('{}test{}'.format(dirname, base_name), 'w') as tef, \
                    open('{}dev{}'.format(dirname, base_name), 'w') as df:
                trf.write('\n\n'.join(train))
                tef.write('\n\n'.join(test))
                df.write('\n\n'.join(dev))
            num += 1
    else:
        with open(fname, 'w') as f:
            f.write('\n\n'.join(dataset))


if __name__ == '__main__':
    main()
