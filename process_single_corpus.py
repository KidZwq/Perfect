import pickle
from collections import Counter

def load_pickle(filename):
    """
    加载 pickle 文件并返回其中的数据。

    参数：
    filename (str): 要加载的 pickle 文件路径。

    返回：
    data: 从 pickle 文件中加载的数据对象。

    异常：
    IOError: 如果文件无法打开或读取。
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='iso-8859-1')  # 使用ISO-8859-1编码加载pickle文件
    return data


def split_data(total_data, qids):
    """
    将总数据根据问题ID（qids）的出现次数分割成单例数据和多例数据。

    参数：
    total_data (list): 包含所有数据项的列表，每个数据项包含问题ID和相关数据。
    qids (list): 与总数据对应的问题ID列表。

    返回：
    total_data_single (list): 单例数据列表，只包含出现一次的问题ID的数据项。
    total_data_multiple (list): 多例数据列表，包含出现多次的问题ID的数据项。

    注意：
    - total_data 中的每个数据项应包含问题ID作为其第一个元素。
    - qids 应与 total_data 中的问题ID对应。
    """
    result = Counter(qids)  # 统计每个问题ID出现的次数
    total_data_single = []  # 存放单例数据
    total_data_multiple = []  # 存放多例数据

    # 遍历总数据，根据问题ID的出现次数分割数据
    for data in total_data:
        if result[data[0][0]] == 1:  # 如果问题ID只出现一次，将其添加到单例数据中
            total_data_single.append(data)
        else:  # 如果问题ID出现多次，将其添加到多例数据中
            total_data_multiple.append(data)

    return total_data_single, total_data_multiple



def data_staqc_processing(filepath, save_single_path, save_multiple_path):
    """
    对STaQC数据进行处理，将数据根据问题ID的出现次数分为单例数据和多例数据，并分别保存到文件中。

    参数：
    filepath (str): 包含STaQC数据的文件路径，数据格式应为列表形式。
    save_single_path (str): 保存单例数据的文件路径。
    save_multiple_path (str): 保存多例数据的文件路径。

    注意：
    - 文件中的数据格式应为列表形式，每个数据项包含问题ID作为其第一个元素。
    - 保存的单例数据文件（save_single_path）和多例数据文件（save_multiple_path）将会以字符串形式写入。
    """
    with open(filepath, 'r') as f:
        total_data = eval(f.read())  # 读取并解析文件中的数据为列表形式
    qids = [data[0][0] for data in total_data]  # 提取所有数据项的问题ID
    total_data_single, total_data_multiple = split_data(total_data, qids)  # 调用分割数据函数

    # 将单例数据写入文件
    with open(save_single_path, "w") as f:
        f.write(str(total_data_single))

    # 将多例数据写入文件
    with open(save_multiple_path, "w") as f:
        f.write(str(total_data_multiple))



def data_large_processing(filepath, save_single_path, save_multiple_path):
    """
    对大型数据文件进行处理，将数据根据问题ID的出现次数分为单例数据和多例数据，并分别以pickle格式保存到文件中。

    参数：
    filepath (str): 包含大型数据的文件路径，数据应为pickle序列化格式。
    save_single_path (str): 保存单例数据的文件路径。
    save_multiple_path (str): 保存多例数据的文件路径。

    注意：
    - 文件中的数据应为pickle序列化格式，可以通过load_pickle函数加载。
    - 单例数据和多例数据将以pickle格式写入到对应的文件中。
    """
    total_data = load_pickle(filepath)  # 加载大型数据文件
    qids = [data[0][0] for data in total_data]  # 提取所有数据项的问题ID
    total_data_single, total_data_multiple = split_data(total_data, qids)  # 调用分割数据函数

    # 将单例数据以pickle格式写入文件
    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)

    # 将多例数据以pickle格式写入文件
    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)


def single_unlabeled_to_labeled(input_path, output_path):
    """
    将单例无标签数据转换为带标签的数据格式，并按问题ID排序后保存到文件中。

    参数：
    input_path (str): 输入的单例无标签数据文件路径，应为pickle序列化格式。
    output_path (str): 输出带标签数据的文件路径，保存为文本格式。

    注意：
    - 输入的单例无标签数据应为pickle序列化格式，可以通过load_pickle函数加载。
    - 输出的带标签数据将按问题ID进行排序，并以文本格式写入到指定的文件中。
    """
    total_data = load_pickle(input_path)  # 加载单例无标签数据
    labels = [[data[0], 1] for data in total_data]  # 为每条数据添加标签，标签为1表示有标签
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))  # 按问题ID进行排序

    # 将排序后的带标签数据以文本格式写入文件
    with open(output_path, "w") as f:
        f.write(str(total_data_sort))



if __name__ == "__main__":
    staqc_python_path = './ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_single_save = './ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = './ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    data_staqc_processing(staqc_python_path, staqc_python_single_save, staqc_python_multiple_save)

    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_single_save = './ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = './ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    data_staqc_processing(staqc_sql_path, staqc_sql_single_save, staqc_sql_multiple_save)

    large_python_path = './ulabel_data/python_codedb_qid2index_blocks_unlabeled.pickle'
    large_python_single_save = './ulabel_data/large_corpus/single/python_large_single.pickle'
    large_python_multiple_save = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    data_large_processing(large_python_path, large_python_single_save, large_python_multiple_save)

    large_sql_path = './ulabel_data/sql_codedb_qid2index_blocks_unlabeled.pickle'
    large_sql_single_save = './ulabel_data/large_corpus/single/sql_large_single.pickle'
    large_sql_multiple_save = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    data_large_processing(large_sql_path, large_sql_single_save, large_sql_multiple_save)

    large_sql_single_label_save = './ulabel_data/large_corpus/single/sql_large_single_label.txt'
    large_python_single_label_save = './ulabel_data/large_corpus/single/python_large_single_label.txt'
    single_unlabeled_to_labeled(large_sql_single_save, large_sql_single_label_save)
    single_unlabeled_to_labeled(large_python_single_save, large_python_single_label_save)
