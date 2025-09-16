import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba._compat")

import sys
import jieba
import numpy as np
from difflib import SequenceMatcher

# 全局配置
JACCARD_THRESHOLD = 0.2  # Jaccard初筛阈值
COSINE_WEIGHT = 0.5  # 余弦相似度权重
LCS_WEIGHT = 0.5  # LCS权重


# 读取文件内容
def read_file(file_path):
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径.")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"文件读取失败: {e}")
        sys.exit(1)


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
    if union == 0:
        print("Jaccard计算中，集合的并集为0，可能存在问题.")
        return 0.0
    return intersection / union


def cosine_similar(vec1, vec2):
    """余弦相似度计算"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 * norm2 == 0:
        print("余弦相似度计算中，向量的模长乘积为0，可能存在问题.")
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


def main():
    if len(sys.argv) != 4:
        print("请输入正确的命令行参数，格式如下：")
        print("python main.py [原始文件][抄袭版论文的文件][答案文件]")
        return

    orig_path, copy_path, ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # 读取原始文本和抄袭文本
    orig_text = read_file(orig_path)
    copy_text = read_file(copy_path)

    # 计算混合相似度
    similarity = hybrid_similarity(orig_text, copy_text)
    result = round(similarity, 2)

    # 写入结果文件
    with open(ans_path, 'w', encoding='utf-8') as f:
        f.write(f"{result:.2f}")
    print(f"检测完成，相似度结果已写入 {ans_path}，结果为：{result:.2f}")


if __name__ == "__main__":
    main()
