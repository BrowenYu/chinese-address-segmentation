import numpy as np
import random
import pickle

from utils.utils import deleteRandom, shuffle_first, del_first
from processing.PKLmodel import Pkl


class SequenceProducter(object):
    """
    序列工厂类
    生成序列，创建对象调用generateSequence方法
    处理序列，查看handleSequence与process_first方法
    """
    def __init__(self, Pkl):
        self.Pkl = Pkl

    def __getPklRandomWord(self, tag):
        """
        根据tag PKL类型数据随机取一个pkl
        """
        tag_pkls = self.Pkl.tag_pkl_map[tag]
        one_tag_pkl = random.sample(tag_pkls, 1)[0]

        if not isinstance(one_tag_pkl, (list, tuple)):
            raise ValueError("pkl数据错误")
        word = one_tag_pkl[0]
        tag_ = one_tag_pkl[1]
        return word, tag_

    def __getAssignWord(self, dict_tag):
        """
        根据dict_tag取dict中规则的pkl
        """
        dict_items = list(dict_tag.items())
        temp_item = random.sample(dict_items, 1)[0]
        tag_ = temp_item[0].lower()
        if isinstance(temp_item[1], list):
            word = random.sample(temp_item[1], 1)[0]
        elif isinstance(temp_item[1], str):
            word = temp_item[1]

        # 指定开头词语，或者指定结尾词语
        if word.startswith("*") or word.endswith("*"):
            word = self.__getAssignStartOrEndWrod(word, tag_)
        return word, tag_
    
    def __getAssignStartOrEndWrod(self, word:str, tag_):
        # 匹配前后位置，true为前面 开发区*，false为后面 *开发区
        matching_location = ""
        if word.startswith("*"):
            matching_word = word[1:]
            matching_location = "end"
        else:
            matching_word = word[:1]
            matching_location = "start"
        # 查看之前是否缓存过这种规则数据
        new_pkl_type_name = tag_ + matching_location + matching_word
        if new_pkl_type_name in self.Pkl.tag_pkl_map:
            word = random.sample(self.Pkl.tag_pkl_map[new_pkl_type_name], 1)[0]
        else:
            # 做缓存
            self.Pkl.tag_pkl_map[new_pkl_type_name] = []
            tag_pkls = self.Pkl.tag_pkl_map[tag_]
            for pkl in tag_pkls:
                if isinstance(pkl, list) or isinstance(pkl, tuple):
                    pkl_word = pkl[0]
                else:
                    pkl_word = pkl
                # 一个pkl是一个list，多个list中的词组合成一个词
                if isinstance(pkl_word, list):
                    if isinstance(pkl_word[0], list):
                        pkl_word = "".join(pkl_word[0])
                    else:
                        pkl_word = "".join(pkl_word)

                if matching_location == "end":
                    if pkl_word.endswith(matching_word):
                        self.Pkl.tag_pkl_map[new_pkl_type_name].append(pkl_word)
                elif matching_location == "start":
                    if pkl_word.startswith(matching_word):
                        self.Pkl.tag_pkl_map[new_pkl_type_name].append(pkl_word)
            word = random.sample(self.Pkl.tag_pkl_map[new_pkl_type_name], 1)[0]
        return word

    def __getComposeWord(self, tags):
        """
        根据tags，取tags中tag，所有tag生成的词语组合一个pkl
        """
        compose_words = []
        for rule in tags:
            word, tag_ = self.generateOneWord(rule)
            compose_words.extend(word)
        word = "".join(compose_words)
        return word, tag_

    def __getRandomAssignWord(self, tag):
        """
        根据tag是一个list 随机取一个规则生成一个pkl
        """
        random_pkl_rule = random.sample(tag, 1)[0]
        word, tag_ = self.generateOneWord(random_pkl_rule)
        return word, tag_

    def generateOneWord(self, tag):
        if isinstance(tag, str) and tag.lower() in self.Pkl.tag_pkl_map:
            # 从指定pkl文件中随机取一个pkl
            word, tag_ = self.__getPklRandomWord(tag.lower())
        elif isinstance(tag, dict):
            # 指定pkl
            word, tag_ = self.__getAssignWord(tag)
        elif isinstance(tag, tuple):
            # 组合(r1, r30)
            word, tag_ = self.__getComposeWord(tag)
        elif isinstance(tag, list):
            # 随机list中规则取一个
            word, tag_ = self.__getRandomAssignWord(tag)
        else:
            raise ValueError("pattern有错误.")

        # 数据格式统一, 全部转换为list
        if isinstance(tag_, list):
            tag_ = [temp.upper() for temp in tag_]
        else:
            tag_ = [tag_.upper()]
        tag_ = [temp.split("_")[0] for temp in tag_]
        if isinstance(word, str):
            word = [word]
        return word, tag_

    @staticmethod
    def __get_word_part_by_rebuild_index(word, rebuild_str):
        """
            word = "明天的天气真好"
            rebuild_str = "---1-"
            取到1的部分进行返回，上例取到 ‘天气真’
        """
        start = 0
        end = 0
        for s in rebuild_str:
            if s=="-":
                start += 1
            else:
                break
        for s in rebuild_str[::-1]:
            if s=="-":
                end += 1
            else:
                break
        end = -end
        if end == 0:
            return word[start:]
        else:
            return word[start:end]
    
    @staticmethod
    def __rebuild_word(word, rebuild_str, part_of_word):
        """
            暂时未使用该功能
            根据rebuild_str规则重构这个词语
            word: 此位置的原词语
            rebuild_str: "###tag1#"
            part_of_word: 标号为1的词语
        """
        start = 0
        end = 0
        for s in rebuild_str:
            if s == "#":
                start += 1
            else:
                break
        for s in rebuild_str[::-1]:
            if s == "#":
                end += 1
            else:
                break
        start_str = ""
        end_str = ""
        end = -end
        start_str = word[:start]
        if end != 0:
            end_str = word[end:]
        return start_str + part_of_word + end_str

    @staticmethod
    def __rebuild_by_list_rule(word_rule, words, part_of_word):
        re_word = ""
        if not isinstance(word_rule, list):
            word_rule = [word_rule]
        for r in word_rule:
            if isinstance(r, int):
                re_word += words[r-1]
            elif isinstance(r, str):
                re_word += part_of_word[r]
            else:
                raise ValueError("rebuild_by_list_rule中规则有误" + str(rule))
        return re_word

    @staticmethod
    def __rebuild_sequence(rebuild_rule, words, tags):
        # if len(words) != len(rebuild_rule):
        #     return words, tags
        # [{"tag1": "-1--"}, [0, 1, {"r2": "tag1"}, {"r1": [1, "tag1"]}, 4]]
        rebuild_words = []
        rebuild_tags = []
        part_of_word = {}
        # 根据字典规则提取后面重组序列需要的词语部分
        for key, rule in rebuild_rule[0].items():
            for char_rule in rule:
                if char_rule.isdigit():
                    words_index = int(char_rule) - 1
                    if key not in part_of_word:
                        part_word = SequenceProducter.__get_word_part_by_rebuild_index(words[words_index], 
                                                                        rule)
                        part_of_word[key] = part_word
                        break
        # 根据数组规则重组序列
        for _, rule in enumerate(rebuild_rule[1]):
            if isinstance(rule, dict):
                r = list(rule.items())[0]
                re_tag = r[0]
                word_rule = r[1]
                re_word = SequenceProducter.__rebuild_by_list_rule(word_rule, words, part_of_word)
            elif isinstance(rule, int):
                re_word = words[rule - 1]
                re_tag = tags[rule - 1]
            else:
                raise ValueError("rebuild_sequence规则有误" + str(rebuild_rule[1]))

            if re_word:
                rebuild_words.append(re_word)
                rebuild_tags.append(re_tag)
        return rebuild_words, rebuild_tags

    def generateOneSequence(self, pattern, handle=True, rebuild_rule=None):
        """
        根据pattern生成一个pkl序列
        """
        words = []
        tags = []
        for tag in pattern:
            word, tag_ = self.generateOneWord(tag)
            words.extend(word)
            tags.extend(tag_)

        if handle and words:
            words, tags = SequenceProducter.handleSequence(words, tags)
        
        if rebuild_rule and handle == False:
            # [0, ["1--", "r1"], 2, ["1--", "r2"], 4]
            words, tags = SequenceProducter.__rebuild_sequence(rebuild_rule, words, tags)
        # tag转大写
        tags = [tag.upper() for tag in tags]
        return words, tags

    def generateSequence(self, pattern_hand_with_rebuild, pattern_RX_with_rebuild, nums, handle):
        """
        生成nums个pkl序列

        pattern_hand: 一级模型生成pkl序列规则, 一个list里面是生成pkl的规则 
        如: ["r1", "r2", {"R3": "开发区"}]
        """
        sequences = []
        sequences_hand = []
        sequences_RX = []
        hand_rebuild_rule = None
        RX_rebuild_rule = None
        # 获取规则 判断是否有序列重组规则，获取重组规则
        if pattern_hand_with_rebuild and isinstance(pattern_hand_with_rebuild[-1], bool) and pattern_hand_with_rebuild[-1]:
            hand_rebuild_rule = pattern_hand_with_rebuild[1]
            pattern_hand = pattern_hand_with_rebuild[0]
        else:
            pattern_hand = pattern_hand_with_rebuild
        
        if pattern_RX_with_rebuild and isinstance(pattern_RX_with_rebuild[-1], bool) and pattern_RX_with_rebuild[-1]:
            RX_rebuild_rule = pattern_RX_with_rebuild[1]
            pattern_RX = pattern_RX_with_rebuild[0]
        else:
            pattern_RX = pattern_RX_with_rebuild

        for _ in range(nums):
            hand_words, hand_tags = self.generateOneSequence(pattern_hand, handle=handle, rebuild_rule=hand_rebuild_rule)
            RX_words, RX_tags = self.generateOneSequence(pattern_RX, handle=False, rebuild_rule=RX_rebuild_rule)
            # 校验tag
            tag_list = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R20", "R21", "R22", "R23", "R24", "R25", "R30", "R31", "R90", "R99"]
            tag_error = 0
            for t in hand_tags:
                if t not in tag_list:
                    tag_error = 1
            for t in RX_tags:
                if t not in tag_list:
                    tag_error = 1
            if tag_error:
                print('tag error')
                continue

            if len(hand_words) != len(hand_tags) or len(RX_words) != len(RX_tags):
                print('length error')
                print((RX_words, RX_tags))
                continue
            # 整合hand和Rx部分
            if hand_words or RX_words:
                sequence_words = hand_words + RX_words
                sequence_tags = hand_tags + RX_tags
                sequences.append(SequenceProducter.wordTagSequenceToStr(sequence_words, sequence_tags))
            # 整合一级模型数据, 保存二级模型数据
            if RX_words:
                hand_words.append("".join(RX_words))
                hand_tags.append("RX")
                sequences_RX.append(SequenceProducter.wordTagSequenceToStr(RX_words, RX_tags))
            # 保存一级模型数据
            if hand_words:
                sequences_hand.append(SequenceProducter.wordTagSequenceToStr(hand_words, hand_tags))
        return sequences, sequences_hand, sequences_RX
    
    @staticmethod
    def wordTagSequenceToStr(words_fins, tags_fins):
        return ','.join(words_fins) + '|' + ','.join(tags_fins)

    @staticmethod
    def process_first(words,tags,p1,p2):
        b = random.uniform(0, 1)
        if b <= p1:
            words, tags = del_first(words, tags)
        if b <= p2:
            words, tags = shuffle_first(words, tags)
        return words,tags

    @staticmethod
    def handleSequence(words, tags):
        repeat_num = random.randint(1, 2)
        words = words * repeat_num
        tags = tags * repeat_num
        if repeat_num == 1:
            words,tags = SequenceProducter.process_first(words,tags,0.5,0.7)
        else:
            words,tags = SequenceProducter.process_first(words,tags,0.6,0.7)
        return words, tags