from typing import (
    List, Dict, DefaultDict, Any
)
from collections import defaultdict
from glob import glob
import json
import logging
from logging import getLogger, StreamHandler
import random
import re
import requests
from tqdm import tqdm

import bs4
import click


logger = getLogger('make_dataset')
logger.setLevel(logging.INFO)
stream_handler = StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
random.seed(0)


def json_reformatting(json_dir: str, json_class: str) \
        -> DefaultDict[str, List[Dict[str, Any]]]:
    """
    reformat jsonfile
    :param json_dir: [directory including json files path]
    :type json_dir: str
    :param json_class: [class of making dataset]
    :type json_class: str
    :return: [reformatted json contents]
    :rtype: str
    """

    reformated_contents = defaultdict(list)
    json_file = '{}{}_dist.json'.format(json_dir, json_class)
    content = '['
    with open(json_file, 'r') as f:
        content += ''.join(
            [re.sub(r'}\n', '},\n', line)
                for line in f.readlines()]
        )[:-2] + ']'
    json_contents = json.loads(content)
    for json_content in json_contents:
        if json_content['html_offset']['start']['line_id'] ==\
                json_content['html_offset']['end']['line_id']:
            reformated_contents[json_content['page_id']].append(
                {
                    'html_offset': json_content['html_offset'],
                    'attribute': json_content['attribute']
                }
            )
    # sorting
    for k in reformated_contents.keys():
        reformated_contents[k].sort(
            key=lambda x: (x['html_offset']['start']['line_id'],
                           x['html_offset']['start']['offset'])
        )

    return reformated_contents


def annotation_to_line(line: str, start_idx: List[int], end_idx: List[int],
                       labels: List[str]) -> str:
    """
    annotate labels to a sentence
    :param line: a sentence
    :param start_idx: head of named entity indexes
    :param end_idx: tail of named entity indexes
    :param labels: named entity classes
    :return a sentence annotated labels
    """

    plus_len = 0
    for s, e, a in zip(start_idx, end_idx, labels):
        line = line[:e + plus_len] + '[/l-{}]'.format(a) + line[e + plus_len:]
        line = line[:s + plus_len] + '[l-{}]'.format(a) + line[s + plus_len:]
        plus_len += 9 + len(a)
    return line


def annotation_to_lines(answers: DefaultDict[str, List[Dict[str, Any]]],
                        html_files: Dict[str, List[str]]) -> None:
    """
    annotate labels to sentences
    :param answers: reformatted json file of answer
    :param html_dir: plain wikipedia html text
    :return None
    """

    for k, v in tqdm(answers.items()):
        line_ids = [answer['html_offset']['start']['line_id']
                    for answer in answers[k]]
        html_file = html_files[k]
        for i, line in enumerate(html_file):
            if i in line_ids:
                starts = [answer['html_offset']['start']['offset']
                          for answer in answers[k]
                          if answer['html_offset']['start']['line_id'] == i]
                ends = [answer['html_offset']['end']['offset']
                        for answer in answers[k]
                        if answer['html_offset']['start']['line_id'] == i]
                attributes = [answer['attribute']
                              for answer in answers[k]
                              if answer['html_offset']['start']['line_id']
                              == i]
                html_files[k][i] = annotation_to_line(
                    line, starts, ends, attributes
                )


def separate_sentences_and_others(html_content: str,
                                  separate_others: bool) -> Dict[str, str]:
    """

    :param html_content: [a content of html file]
    :type html_content: str
    :param separate_others: [if True, separate sentences, infobox, and others]
    :type separate_others: bool
    :return: [separated a html content]
    :rtype: Dict[str, str]
    """

    soup = bs4.BeautifulSoup(html_content, 'html.parser')

    # pickup infobox
    infobox = pickup_content_from_html('infobox', soup, True)

    if separate_others:
        # pickup sentences
        # "p" tag and "d" tag
        sentences = pickup_content_from_html('p', soup)
        sentences.extend(pickup_content_from_html('dd', soup))
        soup = bs4.BeautifulSoup(html_content, 'html.parser')
        # class
        matching_class_tags = [
            'rellink',
            'thumbcaption',
            'reference-text',
            'thumbimage'
        ]
        for class_tag in matching_class_tags:
            sentences.extend(pickup_content_from_html(class_tag, soup, True))

        # others
        others = soup.get_text()
        return_obj = {
            'sentences': '\n'.join(sentences),
            'infobox': ''.join(infobox),
            'others': others,
        }
    else:
        sentences = soup.get_text()
        return_obj = {
            'sentences': sentences,
            'infobox': ''.join(infobox),
        }

    return return_obj


def pickup_content_from_html(html_tag: str, soup: bs4.BeautifulSoup,
                             cls=False) -> List[str]:
    """
    extract sentences matched html_tag from html
    Also, remove matched content from html
    :param html_tag: html used in matching
    :param soup: beautifulsoup object based on html content string
    :param cls: if True, set `class_=html_tag`
    :return sentences matched html_tag
    """

    elems = soup.find_all(class_=html_tag) if cls else soup.find_all(html_tag)
    return [elem.extract().text for elem in elems]


def saving_annotated_corpus(annotated_contents: List[Dict[str, str]],
                            out_dir: str, cls_name: str) -> None:
    """

    :param annotated_contents: [description]
    :type annotated_contents: List[Dict[str, str]]
    :param out_dir: [description]
    :type out_dir: str
    :param cls_name: [description]
    :type cls_name: str
    :return: [description]
    :rtype: None
    """

    base_name = '{}{}'.format(out_dir, cls_name)
    if 'others' in annotated_contents[0].keys():
        with open('{}_{}.txt'.format(base_name, 'sentences'), 'w') as sentf,\
                open('{}_{}.txt'.format(base_name, 'infobox'), 'w') as infof,\
                open('{}_{}.txt'.format(base_name, 'others'), 'w') as otherf:
            sentences = []
            infoboxes = []
            others = []
            for content in annotated_contents:
                sentences.append(content['sentences'])
                infoboxes.append(content['infobox'])
                others.append(content['others'])
            sentf.write('\n'.join(sentences))
            infof.write('\n'.join(infoboxes))
            otherf.write('\n'.join(others))
    else:
        with open('{}_{}.txt'.format(base_name, 'sentences'), 'w') as sentf,\
                open('{}_{}.txt'.format(base_name, 'infobox'), 'w') as infof:
            sentences = []
            infoboxes = []
            for content in annotated_contents:
                sentences.append(content['sentences'])
                infoboxes.append(content['infobox'])
            sentf.write('\n'.join(sentences))
            infof.write('\n'.join(infoboxes))


def morph_analysing(sentences: List[str], algo: str,
                    mode: str, bioul: bool = False):
    annotated_sentences = []
    headers = {'Content-Type': 'application/json'}
    for sentence in tqdm(sentences):
        label_insert_idx = re.finditer(r'\[l-', sentence)
        label_close_idx = re.finditer(r'\[/l-', sentence)
        sentence = re.sub(r'\[/*l-.+\]', '', sentence)

        obj = {'sentence': sentence, 'mode': mode}
        res = requests.post('http://localhost:5000/' + algo,
                            json=obj, headers=headers)
        result = res.json()
        words = result['words']
        info = result['info']
        morphs = [w + '\t' + i for w, i in zip(words, info)]
        annotation_labels = ['O'] * len(morphs)
        ne_labels = []
        places = []
        for i, c in zip(label_insert_idx, label_close_idx):
            ne_labels.append(i.group().replace('[l-', '').replace(']', ''))
            places.append({'insert': i.start(), 'close': c.start()})
        if len(places) == 0:
            continue
        sentence_len = 0
        cnt = 0
        for i, word in enumerate(words):
            if sentence_len >= places[cnt]['insert']:
                label = ne_labels[cnt]
                start_idx = i
            sentence_len += len(word)
            if sentence_len <= places[cnt]['end']:
                cnt += 1
                if bioul:
                    if start_idx == i:
                        label = 'S-' + label
                    else:
                        label = 'E-' + label
                else:
                    if start_idx == i:
                        label = 'B-' + label
                    else:
                        label = 'I-' + label
            else:
                if start_idx == i:
                    label = 'B-' + label
                else:
                    label = 'I-' + label
            annotation_labels[i] = label
        annotated_sentences.append(
            '\n'.join(
                [
                    morph + '\t' + label
                    for morph, label in zip(morphs, annotation_labels)
                ]
            )
        )
    return annotated_sentences


@click.command()
@click.option('-hd', '--html_dir', type=str,
              default='../data/JP5/HTML/')
@click.option('-ad', '--annotation_dir', type=str,
              default='../data/JP5/annotation/')
@click.option('-cls', '--class_name', type=str,
              default='City')
@click.option('-o', '--out', type=str, default='data/JP5/annotated_data/')
@click.option('-k', '--ksplit_num', type=int, default=1)
@click.option('-b', '--bioul', is_flag=True)
@click.option('-c', '--char_level', is_flag=True)
@click.option('--MECAB', 'morph_analysis', flag_value='mecab', default=True)
@click.option('--NEOLOGD', 'morph_analysis', flag_value='mecab_neologd')
@click.option('--JUMANPP', 'morph_analysis', flag_value='jumanpp')
@click.option('--SUDACHI', 'morph_analysis', flag_value='sudachi')
@click.option('--sudachim', type=click.Choice(['A', 'B', 'C']), default='C')
def main(html_dir: str, annotation_dir: str, class_name: str,
         out: str, ksplit_num: int, bioul: bool, char_level: bool,
         morph_analysis: str, sudachim: str):
    """
    make IE dataset from html and annotation files
    :param html_dir: [location of html files]
    :type html_dir: str
    :param annotation_dir: [location of annotation files]
    :type annotation_dir: str
    :param class_name: [class of making dataset]
    :type class_name: str
    :param out: [location of annotated files]
    :type out: str
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

    html_dir += '/' if html_dir[-1] != '/' else ''
    annotation_dir += '/' if annotation_dir[-1] != '/' else ''
    out += '/' if out[-1] != '/' else ''
    fname_extract = re.compile(r'([^/]+?)?$')

    logger.info('show setting parameters\n\
            html_dir: {0}\n\
            annotation_dir: {1}\n\
            class_name: {2}\n\
            out: {3}\n\
            ksplit_num: {4}\n\
            bioul: {5}\n\
            char_level: {6}\n\
            morph_analysis: {7}\n\
            sudachim: {8}'.format(html_dir, annotation_dir, class_name, out,
                                  ksplit_num, bioul, char_level,
                                  morph_analysis, sudachim)
                )
    contents = {}
    logger.info('loading html files ...')
    for html in tqdm(glob(html_dir + class_name + '/*.html')):
        with open(html, 'r') as f:
            key_name = fname_extract.search(html).group().replace('.html', '')
            contents[key_name] = f.read().split('\n')
    # reformat annotation file
    logger.info('loading annotation files ...')
    answers = json_reformatting(annotation_dir, class_name)
    logger.info('annotating html contents ...')
    annotation_to_lines(answers, contents)
    logger.info('separating sentences, infobox, and others ...')
    annotated_contents = [
        separate_sentences_and_others('\n'.join(content), False)
        for content in tqdm(contents.values())
    ]
    logger.info('saving annotated corpus ...')
    saving_annotated_corpus(annotated_contents, out, class_name)
    annotated_sentences: List[str] = []
    for content in annotated_contents:
        annotated_sentences.extend(content['sentences'])
    logger.info('morphological analysing ...')
    analysed = morph_analysing(annotated_sentences, morph_analysis,
                               sudachim, bioul)


if __name__ == '__main__':
    main()
