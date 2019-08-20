import argparse


parser = argparse.ArgumentParser()
parser.add_argument('parser', help='set mode, `mecab`, `juman` or `sudachi`',
                    type=str, choices=['mecab', 'juman', 'sudachi'])
parser.add_argument('label_num', type=int, help='labels with a word')
parser.add_argument('--in_file', type=str, help='want to parse file path')
parser.add_argument('--out_file', type=str, default='out.txt')
args = parser.parse_args()


def shape(line: str, parser: str, num: int):
    s = line.split('\t')
    if s == ['\n']:
        return '\n'
    if parser == 'mecab':
        return '{}\t{}\t{}\t{}'.format(
            s[0], s[1], s[2], '\t'.join(
                [s[-i] for i in reversed(range(1, num + 1))])
        )
    elif parser == 'juman':
        return '{}\t{}\t{}\t{}'.format(
            s[0], s[1], s[2], '\t'.join(
                [s[-i] for i in reversed(range(num))])
        )
    elif parser == 'sudachi':
        return '{}\t{}\t{}\t{}'.format(
            s[0], s[1], s[2], '\t'.join(
                [s[-i] for i in reversed(range(num))])
        )


with open(args.in_file, 'r') as f:
    lines = [shape(line, args.parser, args.label_num)
             for line in f.readlines()]

with open(args.out_file, 'w') as f:
    f.write(''.join(lines))
