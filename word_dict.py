import pickle


def get_vocab(corpus1, corpus2):
    """
    获取两个语料库中所有的词汇集合。

    参数：
    corpus1 (list): 第一个语料库，每个元素是一个包含问题ID及相关内容的列表。
    corpus2 (list): 第二个语料库，每个元素是一个包含问题ID及相关内容的列表。

    返回：
    word_vocab (set): 包含两个语料库中所有出现过的词汇的集合。

    注意：
    - 语料库的结构应为列表，每个元素是包含问题ID及相关内容的列表，例如 [qid, [Si_word_list, Si1_word_list], [tokenized_code], query_word_list, block_length, label]。
    - 函数会遍历两个语料库，提取出所有的词汇并存放在一个集合中。
    """
    word_vocab = set()  # 用于存放所有的词汇的集合
    for corpus in [corpus1, corpus2]:  # 遍历两个语料库
        for i in range(len(corpus)):
            # 更新词汇集合，包括Si_word_list、Si1_word_list、tokenized_code和query_word_list中的所有词汇
            word_vocab.update(corpus[i][1][0])  # Si_word_list
            word_vocab.update(corpus[i][1][1])  # Si1_word_list
            word_vocab.update(corpus[i][2][0])  # tokenized_code
            word_vocab.update(corpus[i][3])     # query_word_list

    print(len(word_vocab))  # 打印词汇集合的大小
    return word_vocab  # 返回包含两个语料库中所有出现过的词汇的集合



def load_pickle(filename):
    """
    加载 pickle 格式的文件并返回其内容。

    参数：
    filename (str): 要加载的 pickle 文件的路径。

    返回：
    data (object): pickle 文件中存储的数据对象。

    注意：
    - 函数打开指定路径的文件，使用 'rb' 模式（二进制读取），加载 pickle 数据。
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)  # 使用 pickle.load 加载文件内容

    return data  # 返回加载的数据对象



def vocab_processing(filepath1, filepath2, save_path):
    """
    处理词汇表，将总词汇表2中的词汇与总词汇表1进行比较，生成新的词汇集合，并保存到文件中。

    参数：
    filepath1 (str): 第一个词汇表文件的路径，用于排除的词汇集合。
    filepath2 (str): 第二个词汇表文件的路径，用于生成新词汇集合。
    save_path (str): 要保存新词汇集合的文件路径。

    注意：
    - 函数假设输入的词汇表文件内容可以通过 eval() 函数转换为 Python 对象。
    - 词汇表文件中存储的应为列表或集合数据结构。
    - 函数输出生成的新词汇集合，并在指定路径保存为文本文件。
    """
    # 加载并转换词汇表文件内容为集合
    with open(filepath1, 'r') as f:
        total_data1 = set(eval(f.read()))
    with open(filepath2, 'r') as f:
        total_data2 = eval(f.read())

    # 获取第二个词汇表中的词汇集合（不含重复）
    word_set = get_vocab(total_data2, total_data2)

    # 从第一词汇表中排除第二个词汇表中存在的词汇
    excluded_words = total_data1.intersection(word_set)
    word_set = word_set - excluded_words

    # 输出处理后的词汇表信息
    print(f'总词汇表1词汇数量：{len(total_data1)}')
    print(f'新生成的词汇集合数量：{len(word_set)}')

    # 将新的词汇集合保存到文件中
    with open(save_path, 'w') as f:
        f.write(str(word_set))



if __name__ == "__main__":
    python_hnn = './data/python_hnn_data_teacher.txt'
    python_staqc = './data/staqc/python_staqc_data.txt'
    python_word_dict = './data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = './data/sql_hnn_data_teacher.txt'
    sql_staqc = './data/staqc/sql_staqc_data.txt'
    sql_word_dict = './data/word_dict/sql_word_vocab_dict.txt'

    new_sql_staqc = './ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = './ulabel_data/sql_word_dict.txt'

    final_vocab_processing(sql_word_dict, new_sql_large, large_word_dict_sql)
