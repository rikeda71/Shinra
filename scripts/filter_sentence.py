# 学習データとして使うデータを文長によりフィルタリングする

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='file path',
                        default='data/JP5/dataset/train.txt')
    parser.add_argument('--out', type=str, help='out path',
                        default='data/JP5/dataset/train.txt')
    parser.add_argument('--max_seq_len', type=int, default=47)
    parser.add_argument('--min_seq_len', type=int, default=27)
    args = parser.parse_args()

    def filter_rule(morphs: str):
        return args.min_seq_len <= len(morphs.split('\n')) <= args.max_seq_len

    with open(args.file, 'r') as f:
        contents = f.read().split('\n\n')

    bool_list = map(filter_rule, contents)
    filterd_contents = [contents[i]
                        for i, result in enumerate(bool_list) if result]
    del contents
    with open(args.out, 'w') as f:
        f.write('\n\n'.join(filterd_contents))
