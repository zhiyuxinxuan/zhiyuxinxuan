import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba._compat")

import unittest
import jieba
import numpy as np
from difflib import SequenceMatcher

JACCARD_THRESHOLD = 0.2  # Jaccard初筛阈值
COSINE_WEIGHT = 0.5  # 余弦相似度权重
LCS_WEIGHT = 0.5  # LCS权重


# 读取文件内容
def read_file(file_path):
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径.")
    try:
        content = ""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                content += line.strip()
        return content
    except Exception as e:
        print(f"文件读取失败: {e}")
        return None


# 文本预处理
def preprocess(text, use_stopwords=True):
    # 停用词表（可根据需求扩展）
    stopwords = {'，', '。', '的', '了', '是', '我', '要', '在', '于', '和',
                 '而且', '或者', '着', '得', '他', '她', '它', '他们',
                 '一个', '一些', '这个', '那个', '啊', '呀', '呢', '吧'} if use_stopwords else set()
    words = jieba.lcut(text)
    return [word for word in words if word not in stopwords and len(word) > 1]


# 相似度计算
def jaccard_similar(set1, set2):
    """Jaccard相似度计算"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0


def cosine_similar(vec1, vec2):
    """余弦相似度计算"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 * norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2 + 1e-8)  # 防止除以0


def lcs_similar(seq1, seq2):
    """最长公共子序列相似度计算"""
    matcher = SequenceMatcher(None, seq1, seq2)
    return matcher.ratio()


# 综合相似度计算流程
def hybrid_similarity(orig_text, copy_text):
    orig_preprocessed = preprocess(orig_text, use_stopwords=True)
    copy_preprocessed = preprocess(copy_text, use_stopwords=True)

    # 特殊情况处理：如果原始文本和抄袭文本都为空
    if not orig_preprocessed and not copy_preprocessed:
        return 0.0

    # Jaccard快速初筛
    orig_set = set(orig_preprocessed)
    copy_set = set(copy_preprocessed)
    jaccard = jaccard_similar(orig_set, copy_set)
    if jaccard < JACCARD_THRESHOLD:
        return 0.0

    # 构建余弦相似度向量（不使用停用词过滤）
    orig_tokens = preprocess(orig_text, use_stopwords=False)
    copy_tokens = preprocess(copy_text, use_stopwords=False)

    vocabulary = list(set(orig_tokens + copy_tokens))
    vec_orig = np.array([orig_tokens.count(word) for word in vocabulary])
    vec_copy = np.array([copy_tokens.count(word) for word in vocabulary])

    # 计算各项相似度
    cosine = cosine_similar(vec_orig, vec_copy)
    lcs = lcs_similar(orig_tokens, copy_tokens)

    # 加权综合结果
    return (cosine * COSINE_WEIGHT) + (lcs * LCS_WEIGHT)


class TestSimilarity(unittest.TestCase):
    def test_identical_text(self):
        """测试完全相同的文本"""
        orig_text = "这是一个测试文本，用于验证查重功能是否正常工作。"
        copy_text = "这是一个测试文本，用于验证查重功能是否正常工作。"
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertAlmostEqual(round(similarity, 2), 1.00)

    def test_empty_texts(self):
        """测试两个空文本"""
        orig_text = ""
        copy_text = ""
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertEqual(round(similarity, 2), 0.00)

    def test_one_empty_text(self):
        """测试一个为空的文本"""
        orig_text = "这是一个测试文本"
        copy_text = ""
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertEqual(round(similarity, 2), 0.00)

    def test_only_stopwords(self):
        """测试只包含停用词的文本"""
        orig_text = "的是了，。和"
        copy_text = "的是了，。和"
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertEqual(round(similarity, 2), 0.00)

    def test_jaccard_below_threshold(self):
        """测试Jaccard相似度低于阈值的情况"""
        orig_text = "今天天气很好，适合去公园散步"
        copy_text = "明天会下雨，建议待在家里"
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertEqual(round(similarity, 2), 0.00)

    def test_partial_similarity(self):
        """测试部分相似的文本"""
        orig_text = "人工智能是计算机科学的一个分支，研究如何使机器具有智能。"
        copy_text = "人工智能属于计算机科学领域，主要探讨如何让机器拥有智能。"
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertTrue(0.3 < similarity < 0.8)  # 预期有一定相似度但不是很高

    def test_highly_similar(self):
        """测试高度相似但不完全相同的文本"""
        orig_text = "数据结构是计算机存储、组织数据的方式。"
        copy_text = "数据结构是计算机中存储和组织数据的方式方法。"
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertTrue(similarity > 0.8)  # 预期有较高相似度

    def test_special_characters(self):
        """测试包含特殊字符的文本"""
        orig_text = "#### 这是标题 #### 正文内容！！！"
        copy_text = "#### 这是标题 #### 正文信息？？？"
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertTrue(similarity > 0.5)  # 预期有一定相似度

    def test_long_texts(self):
        """测试较长文本的相似度"""
        orig_text = """
        机器学习是人工智能的一个分支，它使计算机系统能够自动学习和改进，
        而无需明确编程。它专注于开发可以访问数据并使用数据自行学习的程序。
        机器学习算法使用历史数据作为输入来预测新的输出值。
        """
        copy_text = """
        机器学习作为人工智能的重要分支，能够让计算机系统自动学习并改进，
        不需要进行明确编程。其核心是开发可访问数据并利用数据自主学习的程序。
        这些算法通常将历史数据作为输入，用来预测新的输出结果。
        """
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertTrue(similarity > 0.7)  # 预期有较高相似度

    def test_file_not_found(self):
        """测试文件不存在的情况"""
        non_existent_path = "non_existent_file.txt"
        with self.assertRaises(FileNotFoundError):
            read_file(non_existent_path)

if __name__ == '__main__':
    unittest.main()
