from typing import (
    List, Dict, Tuple, DefaultDict, Any
)
from collections import defaultdict
from glob import glob
import json
import logging
from logging import getLogger, StreamHandler
import re

import bs4
import click
from tqdm import tqdm


logger = getLogger('make_dataset')
logger.setLevel(logging.INFO)
stream_handler = StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)


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
        html_offset = json_content['html_offset']
        if html_offset['start']['line_id'] == html_offset['end']['line_id']:
            reformated_contents[json_content['page_id']].append(
                {
                    'html_offset': html_offset,
                    'attribute': json_content['attribute']
                }
            )
    # sorting
    for k in reformated_contents.keys():
        reformated_contents[k].sort(
            key=lambda x: (x['html_offset']['start']['line_id'],
                           x['html_offset']['start']['offset'],
                           x['html_offset']['end']['offset'])
        )

    return reformated_contents


def slide_annotate_idx(plus_lens: List[Tuple[int, int]], i: int) -> int:
    """
    slide annotation index. support `annotation_to_line` method
    :param plus_lens: [(index, label_len), (index, label_len)]
    :type plus_lens: List[Tuple[int, int]]
    :param i: [annotation index]
    :type i: int
    :return: [slide index value]
    :rtype: int
    """

    if len(plus_lens) == 0:
        return 0
    return sum([len_tuple[1] for len_tuple in plus_lens if len_tuple[0] < i])


def annotation_to_line(line: str, start_idx: List[int], end_idx: List[int],
                       labels: List[str]) -> str:
    """
    annotate labels to a sentence
    :param line: [a sentence]
    :type line: str
    :param start_idx: [head of named entity indexes]
    :type start_idx: List[int]
    :param end_idx: [tail of named entity indexes]
    :type end_idx: List[int]
    :param labels: [named entity classes]
    :type labels: List[str]
    :return: [a sentence annotated labels]
    :rtype: [str]
    """

    plus_lens = []
    for s, e, a in zip(start_idx, end_idx, labels):
        line = line[:e + slide_annotate_idx(plus_lens, e)] + \
            '[/l-{}]'.format(a) + line[e + slide_annotate_idx(plus_lens, e):]
        line = line[:s + slide_annotate_idx(plus_lens, s)] + \
            '[l-{}]'.format(a) + line[s + slide_annotate_idx(plus_lens, s):]
        plus_lens.append((s, 4 + len(a)))
        plus_lens.append((e, 5 + len(a)))
    return line


def annotation_to_lines(answers: DefaultDict[str, List[Dict[str, Any]]],
                        html_contents: Dict[str, List[str]]) -> None:
    """
    annotate labels to sentences
    :param answers: [reformatted json file of answer]
    :type answers: DefaultDict[str, List[Dict[str, Any]]]
    :param html_contents: [plain wikipedia html text]
    :type html_contents: Dict[str, List[str]]
    :return: [description]
    :rtype: None
    """

    for k, v in tqdm(answers.items()):
        line_ids = [answer['html_offset']['start']['line_id']
                    for answer in answers[k]]
        html_content = html_contents[k]
        for i, line in enumerate(html_content):
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
                html_contents[k][i] = annotation_to_line(
                    line, starts, ends, attributes
                )


def separate_sentences_and_others(html_content: str,
                                  separate_others: bool) -> Dict[str, str]:
    """
    separate sentences, infobox, and others from a html content
    :param html_content: [a content of html file]
    :type html_content: str
    :param separate_others: [if True, separate sentences, infobox, and others]
    :type separate_others: bool
    :return: [separated a html content]
    :rtype: Dict[str, str]
    """

    sentences = []
    soup = bs4.BeautifulSoup(html_content, 'html.parser')

    # pickup infobox
    infobox = pickup_content_from_html('infobox', soup, True)
    infobox = ''.join(infobox)
    infobox = re.sub(r'\n\s+', '\n', infobox)
    infobox = re.sub(r'\n+', '\n', infobox)

    # 以下を削除する
    # sitesub
    # jump-to-nav
    # plainlinks ambox ambox-move
    # 目次
    # リンク集
    # フッター
    # 案内メニュー
    # 題目(h2)
    # script
    # 参照リスト
    # no print(dablink noprintを除く)
    decompose_list = [soup.find(id='siteSub'),
                      soup.find(id='jump-to-nav'),
                      soup.find(class_='plainlinks ambox ambox-move'),
                      soup.find(class_='toc'),
                      soup.find(class_='catlinks'),
                      soup.find(id='footer'),
                      soup.find(class_='printfooter'),
                      soup.find(id='mw-navigation'),
                      ]
    decompose_list += soup.find_all('h2')
    decompose_list += soup.find_all('script')
    decompose_list += soup.find_all(class_='reflist')
    # 'mw-references-wrap'
    sentences += [
        elem.extract().text
        for elem in soup.find_all(class_='dablink noprint')
    ]
    decompose_list += soup.find_all(class_='noprint')
    [elem.decompose() for elem in decompose_list if elem is not None]

    if separate_others:
        # pickup sentences
        # "p" tag and "d" tag
        sentences += pickup_content_from_html('p', soup)
        sentences += pickup_content_from_html('dd', soup)
        # class
        matching_class_tags = [
            'rellink',
            'thumbcaption',
            'thumbimage'
        ]
        for class_tag in matching_class_tags:
            sentences.extend(pickup_content_from_html(class_tag, soup, True))
        sentences = '\n'.join(sentences)
        sentences = re.sub(r'\n\s+', '\n', sentences)
        sentences = re.sub('\n+', '\n', sentences)

        # others
        others = soup.get_text()
        others = re.sub(r'ファイル:.*\n', '', others)
        others = re.sub(r'\n\s+', '\n', others)
        others = re.sub(r'\n+', '\n', others)
        return_obj = {
            'sentences': sentences,
            'infobox': infobox,
            'others': others,
        }
    else:
        sentences = soup.get_text()
        sentences = re.sub(r'\n\s+', '\n', sentences)
        sentences = re.sub(r'\n+', '\n', sentences)
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
    saving annotated corpus
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


@click.command()
@click.option('-hd', '--html_dir', type=str,
              default='../data/JP5/HTML/')
@click.option('-ad', '--annotation_dir', type=str,
              default='../data/JP5/annotation/')
@click.option('-cls', '--class_name', type=str,
              default='City')
@click.option('-o', '--out', type=str, default='data/JP5/annotated_data/')
def main(html_dir: str, annotation_dir: str, class_name: str, out: str):
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
    """

    html_dir += '/' if html_dir[-1] != '/' else ''
    annotation_dir += '/' if annotation_dir[-1] != '/' else ''
    out += '/' if out[-1] != '/' else ''
    fname_extract = re.compile(r'([^/]+?)?$')

    logger.info('show setting parameters\n\
                html_dir: {0}\n\
                annotation_dir: {1}\n\
                class_name: {2}\n\
                out: {3}\n' .format(
        html_dir, annotation_dir, class_name, out)
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
        separate_sentences_and_others('\n'.join(content), True)
        for content in tqdm(contents.values())
    ]
    logger.info('saving annotated corpus ...')
    saving_annotated_corpus(annotated_contents, out, class_name)
    annotated_sentences: List[str] = []
    for content in annotated_contents:
        annotated_sentences.extend(content['sentences'])


if __name__ == '__main__':
    main()
