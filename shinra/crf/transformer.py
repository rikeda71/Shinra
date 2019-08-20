from typing import List, Dict, Any
import re


class Transformer:

    def __init__(self, ws: int = 2,
                 is_inbracket: bool = False,
                 word_length: bool = False,
                 contain_last_char: bool = False,
                 label_num: int = 1):
        """

        :param ws: window size
        :param is_inbracket: if True, add in bracket feature
        :param word_length: if True, add word length feature
        :param contain_last_char: if True, add a last character in a word
        :param label_num: number of labels
        """

        self._ws = ws
        self._is_inbracket = is_inbracket
        self._word_length = word_length
        self._last_char = contain_last_char
        self._label_num = label_num

    def sentence2features(self, morphs: List[List[str]])\
            -> List[Dict[str, Any]]:
        """
        sentence -> features list and NE label
        :param morphs: morphs list
        :return:
        """

        sentence_features = []
        word_list = [morph[0] for morph in morphs]
        for k in range(len(morphs)):
            features = {}
            near_morphs = self._get_near_morphs(morphs, k)
            # store is in bracket feature
            # if self._is_inbracket:
            #     features.update(
            #         {'inBracket': self._isInBracket(word_list, k)})
            features.update(
                {'{0}:word'.format(-self._ws + i): morph[0]
                    for i, morph in enumerate(near_morphs)})
            features.update(
                {'{0}:word_type'.format(-self._ws + i):
                    self._get_char_types(morph[0])
                    for i, morph in enumerate(near_morphs)})
            # store part of speech
            features.update(
                {'{0}:pos'.format(-self._ws + i): morph[-2]
                 for i, morph in enumerate(near_morphs)}
            )
            features.update(
                {'{0}:subpos'.format(-self._ws + i): morph[-1]
                 for i, morph in enumerate(near_morphs)}
            )
            if self._last_char:
                features.update(
                    {'{0}:lastword'.format(-self._ws + i): morph[0][-1]
                        for i, morph in enumerate(near_morphs)
                        if morph[0] not in ['BOS', 'EOS']})
            if self._word_length:
                features.update(
                    {'{0}:word_length'.format(-self._ws + i): len(morph[0])
                        for i, morph in enumerate(near_morphs)})
            sentence_features.append(features)
        del features
        del near_morphs
        del word_list
        return sentence_features

    def _get_near_morphs(self, morphs: List[List[str]], i: int)\
            -> List[List[str]]:
        """
        get nearby morphs
        :param morphs: morphs in a sentence
        :param i: current index
        :return: nearby morphs
        """

        near_morphs = []
        for idx in range(i - self._ws, i + self._ws + 1):
            if idx < 0:
                morph = ['BOS'] * (len(morphs[0]) - 1)
            elif idx >= len(morphs):
                morph = ['EOS'] * (len(morphs[0]) - 1)
            else:
                morph = morphs[idx][:-self._label_num]
            near_morphs.append(morph)
        return near_morphs

    def _get_char_types(self, token: str) -> str:
        """
        get char types of a token
        :param token: a word or a character
        :return: all char types in a token
        """

        if token in ['BOS', 'EOS']:
            return 'TOKEN'
        char_types = map(self._char_type, token)
        char_types_str = "-".join(sorted(set(char_types)))
        return char_types_str

    @staticmethod
    def _get_pos_info(morph: List[str]) -> str:
        """
        get information of part of speech
        :param morph: morphological information
        :return: information of part of speech
        """

        return "-".join(morph[-2:])

    @staticmethod
    def _isInBracket(sentence: List[str], idx: int) -> bool:
        """
        return whether token (sentence[idx]) is between open and close brackets
        :param sentence: a sentece (divided by each word)
        :param idx: current index
        :return: True or False
        """

        open_brackets = ["「", "『", "“", "（", "＜", "【"]
        close_brackets = ["」", "』", "”", "）", "＞", "】"]
        open_dis = -1
        close_dis = -1

        # search open and close brackets
        for i in range(len(sentence)):
            word = sentence[i][0]
            if word in open_brackets and open_dis < 0:
                open_dis = i
            elif word in close_brackets:
                close_dis = i

        # token is between open and close brackets -> True
        if open_dis < idx < close_dis:
            return True
        return False

    def _char_type(self, char: str) -> str:
        """
        return char type
        :param char: a character
        :return: char type
        """

        if char.isdigit():
            return 'ZDIGIT'
        elif char.islower():
            return 'ZLLET'
        elif char.isupper():
            return 'ZULET'
        elif self.__hiragana(char):
            return 'HIRAG'
        elif self.__katakana(char):
            return 'KATAK'
        elif self.__kanji(char):
            return 'KANJI'
        elif self.__double_byte_symbol(char):
            return 'KIGO'
        elif char == "␣":
            return 'ZSPACE'
        else:
            return 'OTHER'

    @staticmethod
    def __hiragana(char: str) -> bool:
        """
        Whether it is hiragana
        :param char: a character
        :return: True or False
        """

        return 0x3040 <= ord(char) <= 0x309F

    @staticmethod
    def __katakana(char: str) -> bool:
        """
        Whether it is katakana
        :param char: a character
        :return: True or False
        """

        return 0x30A0 <= ord(char) <= 0x30FF

    @staticmethod
    def __kanji(char: str) -> bool:
        """
        Whether it is kanji
        :param char: a character
        :return: True or False
        """

        return 0x4E00 <= ord(char) <= 0x9FFF

    @staticmethod
    def __double_byte_symbol(char: str) -> bool:
        """
        Whether it is double byte symbol
        :param char: a character
        :return: True or False
        """

        if "" == re.sub(r'[^ -~｡-ﾟ]', '', char):
            return True
        return False
