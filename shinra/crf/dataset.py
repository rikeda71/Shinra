from typing import List, Tuple, Dict
import re
import logging

from miner import Miner

from .transformer import Transformer


logging.basicConfig(level=logging.ERROR)


class Dataset:

    def __init__(self, path: str,
                 target_class: str,
                 is_inbracket: bool = False,
                 word_length: bool = False,
                 contain_last_char: bool = False,
                 ws: int = 2):
        """
        get experiment data from processed dataset
        :param path: dataset path
        :param target_class: target class name
        :param is_inbracket: if True, add inbracket feature
        :param word_length: if True, add word length feature
        :param contain_last_char: if True, add a last character in a word
        :param ws: window size
        """

        self._p = path
        self._target_class = target_class
        self._is_inbracket = is_inbracket
        self._train = self._preprocess_dataset('train')
        self._test = self._preprocess_dataset('test')
        self._dev = self._preprocess_dataset('dev')
        self._detect_label_num()
        self._prepare_conv_label_pair()
        self.t = Transformer(ws, self._is_inbracket,
                             word_length, contain_last_char,
                             self._label_num)

    def load(self)\
            -> Dict[str, Tuple[List[List[Dict[str, str]]], List[List[str]]]]:
        """
        return dataset features
        :return dataset using experiment
        """

        return {'train': (self._get_features('train'),
                          self.get_labels('train')),
                'test': (self._get_features('test'),
                         self.get_labels('test')),
                'develop': (self._get_features('develop'),
                            self.get_labels('develop'))}

    def known_NEs(self):
        """
        get Named Entities in training data
        :return:
        """

        sentences = self.get_sentences('train')
        labels = self.get_labels('train')
        m = Miner(labels, labels, sentences)
        return m.return_answer_named_entities()['unknown']

    def get_sentences(self, mode: str) -> List[List[str]]:
        """
        get sententences in train, test, or development set
        :param mode: train or test or develop
        :return: preprocessed dataset
        """

        if mode == 'train':
            target = self._train
        elif mode == 'test':
            target = self._test
        elif mode == 'develop':
            target = self._dev
        else:
            logging.error('Please select "mode" from train, test or develop')
            exit()

        sentences = []
        for morphs in target:
            sentences.append([morph[0] for morph in morphs])
        return sentences

    def get_labels(self, mode: str) -> List[List[str]]:
        """
        get sententences in train, test, or development set
        :param mode: train or test or develop
        :return: preprocessed dataset
        """

        if mode == 'train':
            target = self._train
        elif mode == 'test':
            target = self._test
        elif mode == 'develop':
            target = self._dev
        else:
            logging.error('Please select "mode" from train, test or develop')
            exit()

        sentences = []
        for morphs in target:
            sentences.append([self._japlabel_to_englabel(morph[-self._label_num])
                              for morph in morphs])
        return sentences

    def _japlabel_to_englabel(self, label: str) -> str:
        """

        Args:
            label (str): a NE label in Japanese

        Returns:
            str: a NE label in English
        """

        if label == 'O':
            return 'O'
        head = label[:2]
        nelabel = label[2:]
        return head + self._conv_label_pair[self._target_class][nelabel]

    def _englabel_to_japlabel(self, label: str) -> str:
        """

        Args:
            label (str): a NE label in English

        Returns:
            str: a NE label in Japanese
        """

        if label == 'O':
            return 'O'
        head = label[:2]
        nelabel = label[2:]
        return head + self._conv_label_pair_reversed[self._target_class][nelabel]

    def _preprocess_dataset(self, mode: str) -> List[List[List[str]]]:
        """
        preprocess dataset
        :param mode: train or test or develop
        :return: preprocessed dataset
        """

        with open('{0}{1}.txt'.format(self._p, mode), 'r') as f:
            articles = f.read()
        articles = re.sub(r'\n{3,}', '\n\n', articles)
        sentences = articles.split('\n\n')

        preprocessed = []
        for sentence in sentences:
            preprocessed.append(
                [morph.split('\t')
                 for morph in sentence.split('\n')
                 if morph != ''])
        return preprocessed

    def _get_features(self, mode: str)\
            -> List[List[Dict[str, str]]]:
        """
        sentences -> features list
        :param mode: train or test or develop
        :return:
        """

        if mode == 'train':
            return [self.t.sentence2features(morphs)
                    for morphs in self._train]
        elif mode == 'test':
            return [self.t.sentence2features(morphs)
                    for morphs in self._test]
        elif mode == 'develop':
            return [self.t.sentence2features(morphs)
                    for morphs in self._dev]
        else:
            logging.error('Please select "mode" from train, test or develop')
            exit()

    def _detect_label_num(self):
        """
        detect number of labels from dataset file
        """

        with open(self._p + 'test.txt', 'r') as f:
            one_line = f.readline()
        self._label_num = len(one_line.split('\t')) - 3

    def _prepare_conv_label_pair(self):

        self._conv_label_pair = {}
        self._conv_label_pair['Airport'] = {
            '滑走路数': 'NOR',  # number of runways
            'IATA（空港コード）': 'IATA',
            'ICAO（空港コード）': 'ICAO',
            '名称由来人地の地位職業名': 'ORIGINNAME',
            '年間発着回数データの年': 'YOA',  # year of arrival
            '滑走路の長さ': 'LOR',  # length of runways
            '標高': 'ELEVATION',
            '年間利用者数データの年': 'YUD',  # year of user data
            '国': 'COUNTRY',
            '母都市': 'MCI',  # mother's city
            '年間発着回数': 'ART',  # arrival times
            '開港年': 'YOP',  # year of opening port
            '総面積': 'AREA',
            '年間利用客数': 'NUY',  # number of users in a year
            '近隣空港': 'NEAR',
            '旧称': 'OLDNAME',
            '所在地': 'LOC',
            '運用時間': 'OTIME',  # operation time
            '座標・緯度': 'LATITUDE',
            '名前の謂れ': 'REASON',
            '座標・経度': 'LONGITUDE',
            '運営者': 'OWNER',
            'ふりがな': 'PHONETIC',
            '別名': 'ALIAS'
        }

        self._conv_label_pair['City'] = {
            '種類': 'TYPE',
            '産業': 'INDUSTRY',
            '国内位置': 'PLACE',
            '地形': 'TERRAIN',
            '特産品': 'SPROD',  # special product
            '国': 'COUNTRY',
            '合併市区町村': 'MTOWN',  # merger towns,
            '温泉・鉱泉': 'SPRING',
            '旧称': 'OLDNAME',
            '友好市区町村': 'FTOWN',  # friendly towns,
            '鉄道会社': 'RCOMPANY',  # railway companies
            'ふりがな': 'PHONETIC',
            '人口': 'POPULATION',
            '観光地': 'TOURISTSPOT',
            '所在地': 'LOCATION',
            '施設': 'FACILITY',
            '人口データの年': 'YPD',  # year of population data
            '面積': 'AREA',
            '別名': 'ALIAS',
            '恒例行事': 'AEVENT',  # annual event
            '首長': 'EMILY',
            '人口密度': 'PDENSITY',  # population density
            '座標・緯度': 'LATITUDE',
            '座標・経度': 'LONGITUDE',
            '成立年': 'YESTABLISH',  # year of establishment
            '地名の謂れ': 'REASON'
        }

        self._conv_label_pair['Company'] = {
            '本拠地国': 'HCOUNTRY',  # home country
            '創業国': 'FCOUNTRY',  # founding country
            '種類': 'TYPE',
            '業界': 'INDUSTRY',
            '従業員数（単体）': 'NEMPLOY',  # number of employ
            '別名': 'ALIAS',
            '代表者': 'REPRESENTATIVE',
            '従業員数（連結）': 'NCEMPLOY',  # number of concating employees
            '取扱商品': 'PRODUCT',
            '創業者': 'FOUNDER',
            '創業地': 'FPLACE',  # founding place
            '資本金': 'CAPITAL',
            '売上高データの年': 'YSD',  # year of sales data
            '主要株主': 'MHOLDERS',  # major shareholders
            '売上高（連結）データの年': 'NSCD',  # year of sales concating data
            '売上高（単体）': 'SALES',
            '解散年': 'YDISRUPTION',  # year of disruption
            '従業員数（単体）データの年': 'YNED',  # year of the number of employees data
            '社名使用開始年': 'NAMESTART',
            '資本金データの年': 'YCD',  # year of capital data
            '設立年': 'YESTABLISH',  # year of establish
            '従業員数（連結）データの年': 'YNCED',  # year of the number of concating employees data
            '売上高（連結）': 'SALES',  # concating sales
            '商品名': 'PRODUCTNAME',
            '買収・合併した会社': 'MERGEDCOM',  # merged company
            'ふりがな': 'PHONETIC',
            '過去の社名': 'PASTNAME',
            '創業時の事業': 'FBUSINESS',  # when founding bisiness
            '起源': 'ORIGIN',
            '子会社・合弁会社': 'SUBCOMPANY',
            '事業内容': 'CONTENT',  # (business content)
            '本拠地': 'HOMEBASE',
            '正式名称': 'FORMALNAME',
            'コーポレートスローガン': 'SLOGAN',
            '業界内地位・規模': 'STATUS'
        }

        self._conv_label_pair['Compound'] = {
            '沸点': 'BPOINT',  # boiling point
            '融点': 'MPOINT',  # melting point
            '商標名': 'TRADENAME',
            '原材料': 'METERIALS',
            '化学式': 'FORMULA',
            '生成化合物': 'PCOMPOUNDS',  # product compounds
            'CAS番号': 'CASNUM',
            '種類': 'TYPE',
            '読み': 'READING',
            '示性式': 'EQUATION',
            '密度': 'DENSITY',
            '用途': 'USAGE',
            '別称': 'ALIAS',
            '特性': 'CHARACTERISTIC',
            '製造方法': 'PROMETHOD'  # production method
        }

        self._conv_label_pair['Person'] = {
            '死因': 'DEATHCAUSE',
            '国籍': 'CITIZENSHIP',
            '家族': 'FAMILY',
            '両親': 'PARENTS',
            '時代': 'ERA',
            '師匠': 'MASTER',
            '職業': 'PROFESSION',
            '居住地': 'RESIDENCE',
            '別名': 'ALIAS',
            'ふりがな': 'PHONETIC',
            '所属組織': 'AFFILIATION',
            '生誕地': 'BIRTHPLACE',
            '学歴': 'EDUCATIONAL',
            '没地': 'MATH',
            '没年月日': 'DEATHDAY',
            '生年月日': 'BIRTHDAY',
            '本名': 'REALNAME',
            '称号': 'TITLE',
            '作品': 'ARTWORK',
            '異表記': 'VARIATION',
            '参加イベント': 'JOINEVENT',
            '受賞歴': 'AWARDS'
        }

        self._conv_label_pair_reversed = {class_name: {
            v: k for k, v in self._conv_label_pair[class_name].items()
        } for class_name in self._conv_label_pair.keys()}
