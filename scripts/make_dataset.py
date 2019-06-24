from typing import (
    List, Dict, DefaultDict, Any
)
from collections import defaultdict
from glob import glob
import json
import logging
from logging import Logger
import MeCab
import random
import re

import bs4
import click


logger = Logger('make_dataset', logging.INFO)


def json_reformatting(json_file: str) \
        -> DefaultDict[str, List[Dict[str, Any]]]:
    """
    reformat jsonfile
    :param json_file: [json file path]
    :type json_file: str
    :return: [reformatted json content]
    :rtype: str
    """

    content = '['
    with open(json_file, 'r') as f:
        content += re.sub('\n}\n', '\n},\n', f.read())[:-2] + ']'
    json_contents = json.loads(content)
    del content
    reformated_contents = defaultdict(list)
    for json_content in json_contents:
        if json_content['html_offset']['start']['line_id'] ==\
                json_content['html_offset']['end']['line_id']:
            reformated_contents[json_content['page_id']].append(
                {
                    'html_offset': json_content['html_offset'],
                    'attribute': json_content['attribute']
                }
            )
    del json_contents
    for k in reformated_contents.keys():
        reformated_contents[k].sort(
            key=lambda x: (x['start']['line_id'], x['start']['offset'])
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
        line = line[:e + plus_len] + '</{}>'.format(a) + line[e + plus_len:]
        line = line[:s + plus_len] + '<{}>'.format(a) + line[s + plus_len:]
        plus_len += 5 + len(a)
    return line


def annotation_to_lines(answers: DefaultDict[str, List[Dict[str, Any]]],
                        html_file: str) -> List[str]:
    """
    annotate labels to sentences
    :param answers: reformatted json file of answer
    :param html_dir: plain wikipedia html text
    :return sentences annotated labels
    """

    lines = []
    for k, v in answers.items():
        line_ids = [answer['html_offset']['start']['line_id']
                    for answer in answers[k]]
        with open('{}{}.html'.format(html_file, k), 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i in line_ids:
                    starts = [answer['html_offset']['start']['offset']
                              for answer in answers[k]
                              if answer['start']['line_id'] == i]
                    ends = [answer['html_offset']['end']['offset']
                            for answer in answers[k]
                            if answer['start']['line_id'] == i]
                    attributes = [answer['attribute']
                                  for answer in answers[k]
                                  if answer['start']['line_id'] == i]
                    lines.append(
                        annotation_to_line(line, starts, ends, attributes)
                    )
                else:
                    lines.append(line)
    return lines


@click.command()
@click.option('-hd', '--html_dir', type=str, default='../data/JP5/HTML/')
@click.option('-ad', '--annotation_dir', type=str,
              default='../data/JP5/annotation/')
@click.option('-o', '--out', type=str, default='../data/JP5/experiment_data/')
@click.option('-k', '--ksplit_num', type=int, default=1)
@click.option('-b', '--bioul', is_flag=True)
@click.option('-c', '--char_level', is_flag=True)
def main(html_dir: str, annotation_dir: str, out: str,
         ksplit_num: int, bioul: bool, char_level: bool):
    """
    make IE dataset from html and annotation files
    :param html_dir: [location of html files]
    :type html_dir: str
    :param annotation_dir: [location of annotation files]
    :type annotation_dir: str
    :param out: [location of annotated files]
    :type out: str
    :param ksplit_num: [dataset split number. default=1 (no split) ]
    :type ksplit_num: int
    :param bioul: [NE label is BIOUL format (IOB2 -> BIOUL).]
    :type bioul: bool
    :param char_level: [make char level annotation.]
    :type char_level: bool
    """

    html_dir += '/' if html_dir[-1] != '/' else ''
    annotation_dir += '/' if annotation_dir[-1] != '/' else ''

    logger.info('show setting parameters\n\
            html_dir: {0}\n\
            annotation_dir: {1}\n\
            out: {2}\n\
            ksplit_num: {3}\n\
            bioul: {4}\n\
            char_level: {5}\n' .format(html_dir, annotation_dir, out,
                                       ksplit_num, bioul, char_level)
                )
    for json_file in glob(annotation_dir + '*'):
        # reformat annotation file
        answers = json_reformatting(json_file)
        answers = [annotation_to_lines(answers)
                   for html in glob(html_dir + '*')]


if __name__ == '__main__':
    main()
