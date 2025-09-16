"""
论文查重单元测试
覆盖各种文本场景，确保相似度计算正确
"""
import unittest
import os
from difflib import SequenceMatcher
import jieba  # type: ignore[import-untyped]
import numpy as np

JACCARD_THRESHOLD = 0.2
COSINE_WEIGHT = 0.5
LCS_WEIGHT = 0.5


def read_file(file_path):
    """读取文件内容"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径.")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except IOError as e:
        print(f"文件读取失败: {e}")
        return None


def preprocess(text, use_stopwords=True):
    """文本预处理"""
    stopwords = {
        '，', '。', '的', '了', '是', '我', '要', '在', '于', '和',
        '而且', '或者', '着', '得', '他', '她', '它', '他们',
        '一个', '一些', '这个', '那个', '啊', '呀', '呢', '吧'
    } if use_stopwords else set()
    words = jieba.lcut(text)
    return [word for word in words if word not in stopwords and len(word) > 1]


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
    return dot_product / (norm1 * norm2 + 1e-8)


def lcs_similar(seq1, seq2):
    """LCS相似度计算"""
    matcher = SequenceMatcher(None, seq1, seq2)
    return matcher.ratio()


def hybrid_similarity(orig_text, copy_text):
    """综合相似度计算"""
    orig_preprocessed = preprocess(orig_text, use_stopwords=True)
    copy_preprocessed = preprocess(copy_text, use_stopwords=True)

    if not orig_preprocessed and not copy_preprocessed:
        return 0.0

    orig_set = set(orig_preprocessed)
    copy_set = set(copy_preprocessed)
    jaccard = jaccard_similar(orig_set, copy_set)
    if jaccard < JACCARD_THRESHOLD:
        return 0.0

    orig_tokens = preprocess(orig_text, use_stopwords=False)
    copy_tokens = preprocess(copy_text, use_stopwords=False)

    vocabulary = list(set(orig_tokens + copy_tokens))
    vec_orig = np.array([orig_tokens.count(word) for word in vocabulary])
    vec_copy = np.array([copy_tokens.count(word) for word in vocabulary])

    cosine = cosine_similar(vec_orig, vec_copy)
    lcs = lcs_similar(orig_tokens, copy_tokens)

    return (cosine * COSINE_WEIGHT) + (lcs * LCS_WEIGHT)


class TestPaperCheck(unittest.TestCase):
    """论文查重测试类"""

    def test_identical_text(self):
        """测试完全相同的文本"""
        orig = "今天是星期天，天气晴，今天晚上我要去看电影。"
        copy = "今天是星期天，天气晴，今天晚上我要去看电影。"
        self.assertAlmostEqual(round(hybrid_similarity(orig, copy), 2), 1.00)

    def test_similar_text(self):
        """测试样例中的相似文本"""
        orig = "今天是星期天，天气晴，今天晚上我要去看电影。"
        copy = "今天是周天，天气晴朗，我晚上要去看电影。"
        # 预期有较高相似度（0.7-0.9之间）
        self.assertTrue(0.7 < hybrid_similarity(orig, copy) < 0.9)

    def test_empty_texts(self):
        """测试两个空文本"""
        self.assertEqual(round(hybrid_similarity("", ""), 2), 0.00)

    def test_one_empty_text(self):
        """测试一个为空的文本"""
        orig = "测试文本"
        copy = ""
        self.assertEqual(round(hybrid_similarity(orig, copy), 2), 0.00)

    def test_no_similarity(self):
        """测试完全不相似的文本"""
        orig = "人工智能是计算机科学的一个分支"
        copy = "天气晴朗适合户外活动"
        self.assertEqual(round(hybrid_similarity(orig, copy), 2), 0.00)

    def test_stopwords_only(self):
        """测试只包含停用词的文本"""
        orig = "的是了，。和"
        copy = "的是了，。和"
        self.assertEqual(round(hybrid_similarity(orig, copy), 2), 0.00)

    def test_partial_copy(self):
        """测试部分抄袭的文本"""
        orig = "机器学习是人工智能的核心技术，包括监督学习、无监督学习和强化学习。"
        copy = "人工智能的核心技术是机器学习，包含监督学习和无监督学习。"
        # 预期相似度在0.5-0.8之间
        self.assertTrue(0.5 < hybrid_similarity(orig, copy) < 0.8)

    def test_long_texts(self):
        """测试长文本相似度"""
        orig = """
        自然语言处理是人工智能的一个重要分支，它研究计算机与人类语言的交互。
        主要任务包括机器翻译、情感分析、文本摘要等。近年来，基于Transformer的模型
        在自然语言处理领域取得了突破性进展。
        """
        copy = """
        自然语言处理作为人工智能的重要领域，专注于计算机与人类语言的交互方式。
        其主要任务有机器翻译、情感分析和文本摘要等。最近几年，基于Transformer的
        模型在该领域获得了显著进步。
        """
        self.assertTrue(0.7 < hybrid_similarity(orig, copy) < 0.9)

    def test_special_characters(self):
        """测试含特殊字符的文本"""
        orig = "【论文标题】：Python在数据分析中的应用\n关键词：Python、数据分析、 Pandas"
        copy = "【论文标题】：Python在数据分析中的应用\n关键词：Python、数据分析、NumPy"
        self.assertTrue(0.8 < hybrid_similarity(orig, copy) < 1.0)

    def test_file_not_found(self):
        """测试文件不存在的情况"""
        with self.assertRaises(FileNotFoundError):
            read_file("nonexistent_file.txt")


if __name__ == '__main__':
    unittest.main()
