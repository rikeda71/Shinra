from typing import List, Dict
import glob
import json
import logging
from logging import Logger
import re

import bs4
import click


logger = Logger('make_dataset', logging.INFO)


def json_reformatting(json_file: str) -> str:
    """
    reformatting jsonfile
    :param json_file: [json file path]
    :type json_file: str
    :return: [reformatted json content]
    :rtype: str
    """

    content = '['
    with open(json_file, 'r') as f:
        content += re.sub('\n}\n', '\n},\n', f.read())[:-2] + ']'
    print(content[:500])
    return content


@click.command()
@click.option('-hd', '--html_dir', type=str, default='../data/JP5/HTML/')
@click.option('-ad', '--annotation_dir', type=str, default='../data/JP5/annotation/')
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
    for json_file in glob.glob(annotation_dir + '*'):
        # reformat annotation file
        answers = json.loads(json_reformatting(json_file))
        print(answers)


if __name__ == '__main__':
    main()
