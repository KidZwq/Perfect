import pickle
import numpy as np
from gensim.models import KeyedVectors


# 将词向量文件保存为二进制文件
def trans_bin(path1, path2):
    wv_from_text = KeyedVectors.load_word2vec_format(path1, binary=False)
    # 如果每次都用上面的方法加载，速度非常慢，可以将词向量文件保存成bin文件，以后就加载bin文件，速度会变快
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(path2)


# 构建新的词典和词向量矩阵
def get_new_dict(type_vec_path, type_word_path, final_vec_path, final_word_path):
    """
    构建新的词典和词向量矩阵，并保存为二进制文件。

    Args:
        type_vec_path (str): 原始词向量文件路径，用于加载词向量模型。
        type_word_path (str): 原始词汇表文件路径，包含需要构建词典的词汇列表。
        final_vec_path (str): 输出的词向量矩阵保存路径，保存为二进制文件。
        final_word_path (str): 输出的词典保存路径，保存为二进制文件。

    Returns:
        None

    Raises:
        FileNotFoundError: 如果指定的文件路径不存在时抛出异常。
    """

    # 加载词向量模型
    model = KeyedVectors.load(type_vec_path, mmap='r')

    # 加载词汇表文件
    with open(type_word_path, 'r') as f:
        total_word = eval(f.read())  # 将文本内容转换为Python对象，这里应该是一个词汇列表

    # 初始词典包含预定义的特殊词标记
    word_dict = ['PAD', 'SOS', 'EOS', 'UNK']  # 分别对应0 PAD_ID, 1 SOS_ID, 2 EOS_ID, 3 UNK_ID

    fail_word = []  # 记录加载失败的词汇
    rng = np.random.RandomState(None)

    # 初始化特殊词的词向量
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()  # PAD的词向量初始化为全0
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()  # UNK的词向量随机初始化
    sos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()  # SOS的词向量随机初始化
    eos_embedding = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()  # EOS的词向量随机初始化
    word_vectors = [pad_embedding, sos_embedding, eos_embedding, unk_embedding]

    # 遍历原始词汇表，加载词向量并构建新的词典
    for word in total_word:
        try:
            word_vectors.append(model.wv[word])  # 加载词向量
            word_dict.append(word)  # 将词添加到词典中
        except:
            fail_word.append(word)  # 记录加载失败的词汇

    word_vectors = np.array(word_vectors)  # 转换为numpy数组形式
    word_dict = dict(map(reversed, enumerate(word_dict)))  # 创建反向映射词典，将词映射到索引

    # 将新的词向量矩阵和词典保存为二进制文件
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)

    print("完成")  # 输出提示信息，表示处理完成


# 得到词在词典中的位置
def get_index(type, text, word_dict):
    location = []  # 存储词在词典中的索引位置

    if type == 'code':  # 如果是代码文本
        location.append(1)  # 添加代码起始标记（自定义为1）
        len_c = len(text)  # 获取文本长度
        if len_c + 1 < 350:  # 如果长度小于350
            if len_c == 1 and text[0] == '-1000':  # 特殊情况处理
                location.append(2)  # 添加代码终止标记（自定义为2）
            else:
                for i in range(0, len_c):
                    index = word_dict.get(text[i], word_dict['UNK'])  # 获取词在词典中的索引，未知词使用UNK的索引
                    location.append(index)  # 添加到位置列表中
                location.append(2)  # 添加代码终止标记
        else:
            for i in range(0, 348):  # 限制最大长度为348
                index = word_dict.get(text[i], word_dict['UNK'])  # 获取词在词典中的索引，未知词使用UNK的索引
                location.append(index)  # 添加到位置列表中
            location.append(2)  # 添加代码终止标记

    else:  # 非代码文本（例如普通文本）
        if len(text) == 0:  # 文本为空
            location.append(0)  # 添加空文本标记（自定义为0）
        elif text[0] == '-10000':  # 特定条件的占位符
            location.append(0)  # 添加空文本标记
        else:
            for i in range(0, len(text)):  # 遍历文本中的每个词
                index = word_dict.get(text[i], word_dict['UNK'])  # 获取词在词典中的索引，未知词使用UNK的索引
                location.append(index)  # 添加到位置列表中

    return location  # 返回文本中每个词在词典中的索引位置列表



# 将训练、测试、验证语料序列化
# 查询：25 上下文：100 代码：350
def serialization(word_dict_path, type_path, final_type_path):
    """
    序列化处理函数，将语料转换为指定格式并保存为二进制文件。

    Args:
        word_dict_path (str): 词典路径，包含词汇到索引的映射。
        type_path (str): 待处理语料路径，包含需要处理的语料数据。
        final_type_path (str): 输出序列化后的语料路径，保存为二进制文件。

    Returns:
        None

    Raises:
        FileNotFoundError: 如果指定的文件路径不存在时抛出异常。
    """

    # 加载词典文件
    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)

    # 加载待处理语料文件
    with open(type_path, 'r') as f:
        corpus = eval(f.read())

    total_data = []

    # 遍历每条语料数据
    for i in range(len(corpus)):
        qid = corpus[i][0]  # 获取语料的ID

        # 获取各部分文本数据的词索引列表
        Si_word_list = get_index('text', corpus[i][1][0], word_dict)
        Si1_word_list = get_index('text', corpus[i][1][1], word_dict)
        tokenized_code = get_index('code', corpus[i][2][0], word_dict)
        query_word_list = get_index('text', corpus[i][3], word_dict)

        # 设定固定的块长度和标签
        block_length = 4
        label = 0

        # 调整文本长度，填充至固定长度
        Si_word_list = Si_word_list[:100] if len(Si_word_list) > 100 else Si_word_list + [0] * (100 - len(Si_word_list))
        Si1_word_list = Si1_word_list[:100] if len(Si1_word_list) > 100 else Si1_word_list + [0] * (100 - len(Si1_word_list))

        # 调整代码文本长度，填充至固定长度
        tokenized_code = tokenized_code[:350] + [0] * (350 - len(tokenized_code))

        # 调整查询文本长度，填充至固定长度
        query_word_list = query_word_list[:25] if len(query_word_list) > 25 else query_word_list + [0] * (25 - len(query_word_list))

        # 组装每条数据
        one_data = [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]
        total_data.append(one_data)

    # 将处理后的数据保存为二进制文件
    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)



if __name__ == '__main__':
    # 词向量文件路径
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    # ==========================最初基于Staqc的词典和词向量==========================

    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'

    # get_new_dict(ps_path_bin, python_word_path, python_word_vec_path, python_word_dict_path)
    # get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # =======================================最后打标签的语料========================================

    # sql 待处理语料地址
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'

    # sql最后的词典和对应的词向量
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sqlfinal_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'

    # get_new_dict(sql_path_bin, final_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)
    # get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path,sql_final_word_dict_path)

    staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
    # Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)
    # Serialization(sql_final_word_dict_path, new_sql_large, large_sql_f)

    # python
    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'

    # python最后的词典和对应的词向量
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'

    # get_new_dict(ps_path_bin, final_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)
    # get_new_dict_append(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python, python_final_word_vec_path,python_final_word_dict_path)

    # 处理成打标签的形式
    staqc_python_f = '../hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'
    large_python_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_python_large_multiple_unlable.pkl'
    # Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
    serialization(python_final_word_dict_path, new_python_large, large_python_f)

    print('序列化完毕')
    # test2(test_python1,test_python2,python_final_word_dict_path,python_final_word_vec_path)
