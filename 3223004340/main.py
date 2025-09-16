"""
论文查重主程序
"""
import sys
import os
from difflib import SequenceMatcher
from collections import Counter
import jieba  # type: ignore[import-untyped]
import numpy as np

# -------------------------- 全局配置 --------------------------
jieba.initialize()

# 完整停用词表
GLOBAL_STOPWORDS = {
    '，', '。', '！', '？', '、', '；', '：', '（', '）', '【', '】', '《', '》',
    '.', ',', ' ', '\n', '\t',
    '的', '了', '着', '得', '啊', '呀', '呢', '吧',
    '今天', '我', '要', '去', '看', '是', '我要', '和', '而且', '或者'
}

# 相似度计算配置
JACCARD_THRESHOLD = 0.05
COSINE_WEIGHT = 0.6
LCS_WEIGHT = 0.4


def read_file(file_path):
    """读取文件内容"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径.")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except IOError as e:
        print(f"文件读取失败: {e}")
        sys.exit(1)


def preprocess(text, use_stopwords=True):
    """预处理文本：分词、过滤停用词和数字"""
    words = jieba.lcut(text, cut_all=False)
    stopwords = GLOBAL_STOPWORDS if use_stopwords else set()

    processed = []
    for word in words:
        stripped_word = word.strip()
        if (stripped_word not in stopwords
                and stripped_word
                and not stripped_word.isdigit()):
            processed.append(stripped_word)
    return processed


def jaccard_similar(set1, set2):
    """计算Jaccard相似度"""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0.0


def cosine_similar(vec1, vec2):
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 * norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2 + 1e-8)


def lcs_similar(seq1, seq2):
    """计算最长公共子序列(LCS)相似度"""
    matcher = SequenceMatcher(None, seq1, seq2)
    return matcher.ratio()


def _calculate_vectors(orig_tokens, copy_tokens):
    """计算词频向量"""
    orig_counter = Counter(orig_tokens)
    copy_counter = Counter(copy_tokens)
    vocabulary = list(orig_counter.keys() | copy_counter.keys())

    vec_orig = np.array([orig_counter.get(word, 0) for word in vocabulary])
    vec_copy = np.array([copy_counter.get(word, 0) for word in vocabulary])

    return vec_orig, vec_copy


def hybrid_similarity(orig_text, copy_text):
    """计算综合相似度"""
    # 预处理
    orig_preprocessed = preprocess(orig_text, use_stopwords=True)
    copy_preprocessed = preprocess(copy_text, use_stopwords=True)

    if not orig_preprocessed and not copy_preprocessed:
        return 0.0

    # Jaccard初筛
    orig_set = set(orig_preprocessed)
    copy_set = set(copy_preprocessed)
    jaccard = jaccard_similar(orig_set, copy_set)
    if jaccard < JACCARD_THRESHOLD:
        return 0.0

    # 词频向量计算
    orig_tokens = preprocess(orig_text, use_stopwords=False)
    copy_tokens = preprocess(copy_text, use_stopwords=False)
    vec_orig, vec_copy = _calculate_vectors(orig_tokens, copy_tokens)

    # 计算各相似度并加权融合
    cosine = cosine_similar(vec_orig, vec_copy)
    lcs = lcs_similar(orig_tokens, copy_tokens)

    return (cosine * COSINE_WEIGHT) + (lcs * LCS_WEIGHT)


def main():
    """主函数：解析命令行参数，执行论文查重流程并输出结果"""
    if len(sys.argv) != 4:
        print("参数错误！正确格式：")
        print("python main.py [原文文件路径] [抄袭版文件路径] [结果文件路径]")
        return

    orig_path, copy_path, ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    orig_text = read_file(orig_path)
    copy_text = read_file(copy_path)

    similarity = hybrid_similarity(orig_text, copy_text)
    result = round(similarity, 2)

    with open(ans_path, 'w', encoding='utf-8') as f:
        f.write(f"{result:.2f}")
    print(f"结果已写入 {ans_path}，相似度：{result:.2f}")


if __name__ == "__main__":
    main()
