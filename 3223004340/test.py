"""
论文查重单元测试模块
包含对论文查重主程序各功能的测试用例
"""

import unittest
import os
from unittest.mock import patch
import numpy as np

from main import (
    read_file, preprocess, jaccard_similar,
    cosine_similar, lcs_similar, hybrid_similarity, main
)


class TestPaperCheck(unittest.TestCase):
    """论文查重功能的单元测试类"""

    def test_preprocess_with_stopwords(self):
        """测试带停用词的文本预处理"""
        raw_text = "今天是星期天，天气晴，今天晚上我要去看电影。"
        expected = ["星期天", "天气", "晴", "晚上", "电影"]
        result = preprocess(raw_text, use_stopwords=True)
        self.assertEqual(result, expected)

    def test_preprocess_without_stopwords(self):
        """测试不带停用词的文本预处理"""
        raw_text = "今天是星期天，天气晴。"
        expected = ["今天", "是", "星期天", "，", "天气", "晴", "。"]
        result = preprocess(raw_text, use_stopwords=False)
        self.assertEqual(result, expected)

    def test_hybrid_stopwords_only(self):
        """测试全停用词文本的混合相似度计算"""
        orig = "的是了，。和我要今天"
        copy = "的是了，。和我要今天"
        orig_pre = preprocess(orig, use_stopwords=True)
        copy_pre = preprocess(copy, use_stopwords=True)
        self.assertTrue(not orig_pre and not copy_pre)
        result = hybrid_similarity(orig, copy)
        self.assertEqual(round(result, 2), np.float64(0.00))

    @patch('sys.argv', ['main.py', 'orig.txt', 'copy.txt', 'result.txt'])
    @patch('main.read_file', return_value="测试文本")
    @patch('main.hybrid_similarity', return_value=0.85)
    def test_main_normal_execution(self, _, __):
        """测试主函数正常执行流程"""
        main()
        with open('result.txt', 'r', encoding='utf-8') as f:
            self.assertEqual(f.read(), "0.85")
        os.remove('result.txt')

    def test_cosine_similar_zero_vector(self):
        """测试余弦相似度（零向量场景）"""
        vec1 = np.array([0, 0, 0])
        vec2 = np.array([1, 2, 3])
        self.assertEqual(cosine_similar(vec1, vec2), 0.0)

    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', side_effect=IOError("权限不足"))
    def test_read_file_io_error(self, _, __):
        """测试文件读取IO错误场景"""
        with self.assertRaises(SystemExit) as cm:
            read_file("test.txt")
        self.assertEqual(cm.exception.code, 1)

    def test_cosine_similar_identical(self):
        """测试余弦相似度（相同向量场景）"""
        vec1 = np.array([3, 1, 2])
        vec2 = np.array([3, 1, 2])
        self.assertAlmostEqual(cosine_similar(vec1, vec2), 1.0, delta=1e-4)

    def test_hybrid_empty_texts(self):
        """测试空文本的混合相似度计算"""
        self.assertEqual(round(hybrid_similarity("", ""), 2), 0.00)

    def test_hybrid_identical_text(self):
        """测试完全相同文本的混合相似度计算"""
        orig = "今天是星期天，天气晴，今天晚上我要去看电影。"
        copy = "今天是星期天，天气晴，今天晚上我要去看电影。"
        self.assertAlmostEqual(round(hybrid_similarity(orig, copy), 2), 1.00)

    def test_hybrid_long_texts(self):
        """测试长文本的混合相似度计算"""
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
        self.assertTrue(0.6 < hybrid_similarity(orig, copy) < 0.9)

    def test_hybrid_no_similarity(self):
        """测试完全不相似文本的混合相似度计算"""
        orig = "人工智能是计算机科学的一个分支"
        copy = "天气晴朗适合户外活动"
        self.assertEqual(round(hybrid_similarity(orig, copy), 2), 0.00)

    def test_hybrid_one_empty_text(self):
        """测试一个文本为空时的混合相似度计算"""
        orig = "测试文本"
        copy = ""
        self.assertEqual(round(hybrid_similarity(orig, copy), 2), 0.00)

    def test_hybrid_partial_copy(self):
        """测试部分抄袭文本的混合相似度计算"""
        orig = "机器学习是人工智能的核心技术，包括监督学习、无监督学习和强化学习。"
        copy = "人工智能的核心技术是机器学习，包含监督学习和无监督学习。"
        self.assertTrue(0.5 < hybrid_similarity(orig, copy) < 0.8)

    def test_hybrid_similar_text(self):
        """测试相似文本的混合相似度计算"""
        orig = "今天是星期天，天气晴，今天晚上我要去看电影。"
        copy = "今天是周天，天气晴朗，我晚上要去看电影。"
        self.assertTrue(0.6 < hybrid_similarity(orig, copy) < 0.9)

    def test_hybrid_special_characters(self):
        """测试含特殊字符文本的混合相似度计算"""
        orig = "【论文标题】：Python在数据分析中的应用\n关键词：Python、数据分析、 Pandas"
        copy = "【论文标题】：Python在数据分析中的应用\n关键词：Python、数据分析、NumPy"
        self.assertTrue(0.8 < hybrid_similarity(orig, copy) < 1.0)

    def test_jaccard_similar_empty_union(self):
        """测试Jaccard相似度（空集合场景）"""
        set1 = set()
        set2 = set()
        self.assertEqual(jaccard_similar(set1, set2), 0.0)

    def test_jaccard_similar_identical(self):
        """测试Jaccard相似度（相同集合场景）"""
        set1 = {"人工智能", "机器学习"}
        set2 = {"人工智能", "机器学习"}
        self.assertEqual(jaccard_similar(set1, set2), 1.0)

    def test_lcs_similar_identical(self):
        """测试LCS相似度（相同序列场景）"""
        seq1 = ["人工智能", "计算机科学"]
        seq2 = ["人工智能", "计算机科学"]
        self.assertEqual(lcs_similar(seq1, seq2), 1.0)

    def test_read_file_exist(self):
        """测试文件存在时的读取"""
        test_file = "temp_test.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("测试内容")
        self.assertEqual(read_file(test_file), "测试内容")
        os.remove(test_file)

    def test_read_file_not_exist(self):
        """测试文件不存在时的读取"""
        with self.assertRaises(FileNotFoundError):
            read_file("non_exist.txt")

    def test_jaccard_pass_cosine_zero(self):
        """测试Jaccard初筛通过但余弦相似度低的场景"""
        orig_text = "我喜欢苹果"
        copy_text = "我喜欢香蕉"
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertTrue(similarity < 1.00)

    def test_jaccard_pass_lcs_zero(self):
        """测试Jaccard初筛通过但LCS相似度低的场景"""
        orig_text = "红色的花朵"
        copy_text = "蓝色的天空"
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertEqual(type(similarity), float)
        self.assertTrue(similarity < 1.00)

    def test_cosine_special_case(self):
        """测试纯标点文本的混合相似度计算（特殊输入场景）"""
        orig_text = "，，，，，，，，，，"
        copy_text = "。。。。。。。。。。"
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertEqual(type(similarity), float)
        self.assertTrue(similarity < 1.00)
        self.assertTrue(similarity <= 0.1)

    def test_jaccard_special_case(self):
        """测试纯特殊符号文本的混合相似度计算（特殊输入场景）"""
        orig_text = "####"
        copy_text = "$$$$"
        similarity = hybrid_similarity(orig_text, copy_text)
        self.assertEqual(round(similarity, 2), 0.00)


if __name__ == '__main__':
    unittest.main(verbosity=2)