import json
import os

from flask import Flask, request, jsonify
from dotenv import load_dotenv
import MeCab
import sudachipy
from sudachipy import dictionary, tokenizer
from pyknp import Juman


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

load_dotenv('/home/.env')

sudachi_tokenizer = dictionary.Dictionary().create()
SPLIT_MODE = tokenizer.Tokenizer.SplitMode


@app.route('/mecab', methods=['POST'])
def mecab():
    if request.method in ['POST'] and \
            request.headers['Content-Type'] == 'application/json':
        sentence = request.get_json()['sentence']
        m = MeCab.Tagger('')
        parsed = m.parse(sentence).split('\n')[:-2]
        words = [morph.split('\t')[0] for morph in parsed]
        info = [morph.split('\t')[1].replace(',', '\t') for morph in parsed]
        response = {'words': words, 'info': info}
        return jsonify(response)
    return jsonify({})


@app.route('/mecab-neologd', methods=['POST'])
def mecab_neologd():
    if request.method in ['POST'] and \
            request.headers['Content-Type'] == 'application/json':
        sentence = request.get_json()['sentence']
        m = MeCab.Tagger('-d ' + os.environ['NEOLOGD_PATH'])
        parsed = m.parse(sentence).split('\n')[:-2]
        words = [morph.split('\t')[0] for morph in parsed]
        info = [morph.split('\t')[1].replace(',', '\t') for morph in parsed]
        response = {'words': words, 'info': info}
        return jsonify(response)
    return jsonify({})


@app.route('/jumanpp', methods=['POST'])
def jumanpp():
    if request.method in ['POST'] and \
            request.headers['Content-Type'] == 'application/json':
        sentence = request.get_json()['sentence']
        juman = Juman(jumanpp=True)
        result = juman.analysis(sentence)
        words = []
        info = []
        for morph in result.mrph_list():
            words.append(morph.midasi)
            info.append('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                morph.hinsi, morph.bunrui, morph.katuyou1, morph.katuyou2,
                morph.yomi, morph.genkei, morph.repname, morph.imis))
        response = {'words': words, 'info': info}
        return jsonify(response)
    return jsonify({})


@app.route('/sudachi', methods=['POST'])
def sudachi():
    if request.method in ['POST'] and \
            request.headers['Content-Type'] == 'application/json':
        sentence = request.get_json()['sentence']
        if 'mode' in request.get_json().keys():
            mode = request.get_json().keys()['mode']
            if mode == 'A':
                mode = SPLIT_MODE.A
            elif mode == 'B':
                mode = SPLIT_MODE.B
            elif mode == 'C':
                mode = SPLIT_MODE.C
        else:
            mode = SPLIT_MODE.C
        words = []
        info = []
        for m in sudachi_tokenizer.tokenize(mode, sentence):
            words.append(m.surface())
            info.append('\t'.join(m.part_of_speech()) +
                        '\t{}\t{}\t{}'.format(
                m.surface(),
                m.dictionary_form(),
                m.reading_form())
            )
        response = {'words': words, 'info': info}
        return jsonify(response)
    return jsonify({})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
