"""
论文查重主程序
功能：计算原文与抄袭版论文的重复率，并将结果写入指定文件
输入：命令行参数（原文路径、抄袭版路径、结果路径）
输出：重复率（精确到小数点后两位）
"""
import sys
import os
from difflib import SequenceMatcher
import jieba  # type: ignore[import-untyped]
import numpy as np

# 全局配置
JACCARD_THRESHOLD = 0.2  # Jaccard初筛阈值
COSINE_WEIGHT = 0.5  # 余弦相似度权重
LCS_WEIGHT = 0.5  # LCS权重


def read_file(file_path):
    """读取文件内容"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件 {file_path} 不存在，请检查路径.")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except IOError as e:  # 改为捕获具体异常
        print(f"文件读取失败: {e}")
        sys.exit(1)


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
    if union == 0:
        print("Jaccard计算警告：并集为0")
        return 0.0
    return intersection / union


def cosine_similar(vec1, vec2):
    """余弦相似度计算"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 * norm2 == 0:
        print("余弦计算警告：向量模长为0")
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

    # Jaccard初筛
    orig_set = set(orig_preprocessed)
    copy_set = set(copy_preprocessed)
    jaccard = jaccard_similar(orig_set, copy_set)
    if jaccard < JACCARD_THRESHOLD:
        return 0.0

    # 构建词频向量
    orig_tokens = preprocess(orig_text, use_stopwords=False)
    copy_tokens = preprocess(copy_text, use_stopwords=False)

    vocabulary = list(set(orig_tokens + copy_tokens))
    vec_orig = np.array([orig_tokens.count(word) for word in vocabulary])
    vec_copy = np.array([copy_tokens.count(word) for word in vocabulary])

    # 计算各项相似度并加权
    cosine = cosine_similar(vec_orig, vec_copy)
    lcs = lcs_similar(orig_tokens, copy_tokens)

    return (cosine * COSINE_WEIGHT) + (lcs * LCS_WEIGHT)


def main():
    """主函数：解析参数、计算相似度、输出结果"""
    if len(sys.argv) != 4:
        print("参数错误！正确格式：")
        print("python main.py [原文文件路径] [抄袭版文件路径] [结果文件路径]")
        return

    orig_path, copy_path, ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    # 读取文本
    orig_text = read_file(orig_path)
    copy_text = read_file(copy_path)

    # 计算相似度
    similarity = hybrid_similarity(orig_text, copy_text)
    result = round(similarity, 2)

    # 写入结果
    with open(ans_path, 'w', encoding='utf-8') as f:
        f.write(f"{result:.2f}")
    print(f"结果已写入 {ans_path}，相似度：{result:.2f}")


if __name__ == "__main__":
    main()
