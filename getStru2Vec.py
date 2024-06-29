import pickle
import multiprocessing
from python_structured import *  # 导入Python语言解析函数模块
from sqlang_structured import *  # 导入SQL语言解析函数模块


# 多进程处理Python查询语句
def multipro_python_query(data_list):
    return [python_query_parse(line) for line in data_list]


# 多进程处理Python代码片段
def multipro_python_code(data_list):
    return [python_code_parse(line) for line in data_list]


# 多进程处理Python上下文信息
def multipro_python_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])  # 如果是特殊标记'-10000'，直接添加到结果中
        else:
            result.append(python_context_parse(line))  # 解析Python上下文信息并添加到结果中
    return result


# 多进程处理SQL查询语句
def multipro_sqlang_query(data_list):
    return [sqlang_query_parse(line) for line in data_list]


# 多进程处理SQL代码片段
def multipro_sqlang_code(data_list):
    return [sqlang_code_parse(line) for line in data_list]


# 多进程处理SQL上下文信息
def multipro_sqlang_context(data_list):
    result = []
    for line in data_list:
        if line == '-10000':
            result.append(['-10000'])  # 如果是特殊标记'-10000'，直接添加到结果中
        else:
            result.append(sqlang_context_parse(line))  # 解析SQL上下文信息并添加到结果中
    return result


# 解析数据函数，包括上下文、查询语句和代码片段
def parse(data_list, split_num, context_func, query_func, code_func):
    pool = multiprocessing.Pool()  # 创建多进程池
    split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]  # 分割数据列表

    # 使用多进程处理上下文信息
    results = pool.map(context_func, split_list)
    context_data = [item for sublist in results for item in sublist]  # 合并多进程处理结果
    print(f'context条数：{len(context_data)}')

    # 使用多进程处理查询语句
    results = pool.map(query_func, split_list)
    query_data = [item for sublist in results for item in sublist]  # 合并多进程处理结果
    print(f'query条数：{len(query_data)}')

    # 使用多进程处理代码片段
    results = pool.map(code_func, split_list)
    code_data = [item for sublist in results for item in sublist]  # 合并多进程处理结果
    print(f'code条数：{len(code_data)}')

    pool.close()  # 关闭进程池
    pool.join()  # 等待所有子进程执行完毕

    return context_data, query_data, code_data


# 主函数，负责加载数据、解析数据并保存处理结果
def main(lang_type, split_num, source_path, save_path, context_func, query_func, code_func):
    with open(source_path, 'rb') as f:
        corpus_lis = pickle.load(f)  # 加载数据列表

    # 解析数据，包括上下文、查询语句和代码片段
    context_data, query_data, code_data = parse(corpus_lis, split_num, context_func, query_func, code_func)
    qids = [item[0] for item in corpus_lis]  # 获取每条数据的qid

    # 构建完整的处理结果数据结构
    total_data = [[qids[i], context_data[i], code_data[i], query_data[i]] for i in range(len(qids))]

    # 将处理结果保存为二进制文件
    with open(save_path, 'wb') as f:
        pickle.dump(total_data, f)


if __name__ == '__main__':
    # 定义Python语言数据源和保存路径
    staqc_python_path = '.ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.pkl'

    # 定义SQL语言数据源和保存路径
    staqc_sql_path = './ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = './ulabel_data/staqc/sql_staqc_unlabled_data.pkl'

    # 处理Python语言数据
    main(python_type, split_num, staqc_python_path, staqc_python_save, multipro_python_context, multipro_python_query,
         multipro_python_code)

    # 处理SQL语言数据
    main(sqlang_type, split_num, staqc_sql_path, staqc_sql_save, multipro_sqlang_context, multipro_sqlang_query,
         multipro_sqlang_code)

    # 处理大规模Python语言数据
    large_python_path = './ulabel_data/large_corpus/multiple/python_large_multiple.pickle'
    large_python_save = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.pkl'

    main(python_type, split_num, large_python_path, large_python_save, multipro_python_context, multipro_python_query,
         multipro_python_code)

    # 处理大规模SQL语言数据
    large_sql_path = './ulabel_data/large_corpus/multiple/sql_large_multiple.pickle'
    large_sql_save = './ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.pkl'

    main(sqlang_type, split_num, large_sql_path, large_sql_save, multipro_sqlang_context, multipro_sqlang_query,
         multipro_sqlang_code)
