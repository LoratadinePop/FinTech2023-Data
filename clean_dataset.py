import pickle

import pandas as pd


def load_dataset():
    customer_base_info_df = pd.read_csv('/data/train_base.csv', encoding='gbk')
    app_histrory_df = pd.read_csv('/data/train_view.csv', encoding='gbk')
    transaction_df = pd.read_csv('/data/train_trx.csv', encoding='gbk')

    # 删除city_code
    customer_base_info_df = customer_base_info_df.drop('cty_cd', axis=1)
    # 训练阶段删除nan值
    customer_base_info_df = customer_base_info_df.dropna()
    customer_base_info_df.sort_values("cust_wid", inplace=True)

    # 删除 nan，将日期格式转换为年月日
    app_histrory_df = app_histrory_df.dropna()
    app_histrory_df['acs_day'] = app_histrory_df['acs_tm'].str.split().str[0]

    # 初始化一次，之后直接load
    counts = app_histrory_df['page_id'].value_counts()
    # 过滤出总数小于 10000 的行，并删除它们
    app_histrory_df = app_histrory_df[app_histrory_df['page_id'].isin(counts[counts >= 10000].index)]
    print(app_histrory_df['page_id'].value_counts())

    transaction_df = transaction_df.dropna()
    transaction_df['trx_day'] = transaction_df['trx_tm'].str.split().str[0]
    # 初始化一次，之后直接load
    counts = transaction_df['trx_cd'].value_counts()
    # 过滤出总数小于 10000 的行，并删除它们
    transaction_df = transaction_df[transaction_df['trx_cd'].isin(counts[counts >= 10000].index)]
    print(transaction_df['trx_cd'].value_counts())

    # 计算最大值和最小值
    max_val = transaction_df["trx_amt"].max()
    min_val = transaction_df["trx_amt"].min()

    # 定义归一化函数
    def normalize(x):
        return (x - min_val) / (max_val - min_val)

    # 将数据归一化
    transaction_df["trx_amt"] = transaction_df["trx_amt"].apply(normalize)

    # 排序
    customer_base_info_df.sort_values("cust_wid", inplace=True)
    app_histrory_df.sort_values(['cust_wid', 'acs_tm', 'page_id'], ascending=[True, True, True], inplace=True)
    transaction_df.sort_values(
        ['cust_wid', 'trx_tm', 'trx_cd', 'trx_amt'], ascending=[True, True, True, True], inplace=True
    )

    return customer_base_info_df, app_histrory_df, transaction_df


def init_apphistory_table(app_histrory_df):
    page_ids = sorted(app_histrory_df['page_id'].unique())
    print('page_ids', len(page_ids))
    acs_days = sorted(app_histrory_df['acs_tm'].str.split().str[0].unique())
    print('acs_days', len(acs_days))
    pageid_map = {}
    acsday_map = {}
    # idx = 0 为 padding_idx
    for idx, pageid in enumerate(page_ids):
        pageid_map[pageid] = idx + 1
    for idx, acsday in enumerate(acs_days):
        acsday_map[acsday] = idx + 1

    return pageid_map, acsday_map


def init_transaction_table(trx_df):
    trx_types = sorted(trx_df['trx_cd'].unique())  # 41
    print('trx_types', len(trx_types))
    trx_days = sorted(trx_df['trx_tm'].str.split().str[0].unique())  # 31
    print('trx_days', len(trx_days))
    trxtype_map = {}
    trxday_map = {}
    for idx, trxtype in enumerate(trx_types):
        trxtype_map[trxtype] = idx + 1
    for idx, trxday in enumerate(trx_days):
        trxday_map[trxday] = idx + 1

    return trxtype_map, trxday_map


def clean_train_set():
    print("cleaning training set...")
    customer_base_info_df, app_histrory_df, transaction_df = load_dataset()
    page_id_map, acs_day_map = init_apphistory_table(app_histrory_df)
    trx_type_map, trx_day_map = init_transaction_table(transaction_df)

    # customer_base_info_df.to_csv('train_userinfo.csv', index=False)
    # app_histrory_df.to_csv('trian_app_his.csv', index=False)
    # transaction_df.to_csv('train_trx_his.csv', index=False)

    customer_base_info_df.to_parquet('/work/train_userinfo.parquet', index=False)
    app_histrory_df.to_parquet('/work/trian_app_his.parquet', index=False)
    transaction_df.to_parquet('/work/train_trx_his.parquet', index=False)

    with open('/work/train_page_id_map.pickle', 'wb') as f:
        pickle.dump(page_id_map, f)
    with open('/work/train_acs_day_map.pickle', 'wb') as f:
        pickle.dump(acs_day_map, f)
    with open('/work/train_trx_type_map.pickle', 'wb') as f:
        pickle.dump(trx_type_map, f)
    with open('/work/train_trx_day_map.pickle', 'wb') as f:
        pickle.dump(trx_day_map, f)


def clean_test_set():
    customer_base_info_df = pd.read_csv('/data/testb_base.csv', encoding='gbk')
    app_histrory_df = pd.read_csv('/data/testb_view.csv', encoding='gbk')
    transaction_df = pd.read_csv('/data/testb_trx.csv', encoding='gbk')

    # 删除city_code
    customer_base_info_df = customer_base_info_df.drop('cty_cd', axis=1)
    # 测试集补全 userinfo
    customer_base_info_df.fillna({'age': -1, 'gdr_cd': 'M'}, inplace=True)
    customer_base_info_df.sort_values("cust_wid", inplace=True)

    # 删除 nan，将日期格式转换为年月日
    app_histrory_df = app_histrory_df.dropna()
    app_histrory_df['acs_day'] = app_histrory_df['acs_tm'].str.split().str[0]

    app_histrory_df = app_histrory_df.drop(app_histrory_df[app_histrory_df['acs_day'] > '1492-08-31'].index)

    # 根据训练集的label set清空outliers
    with open('/work/train_page_id_map.pickle', 'rb') as f:
        page_id_map = pickle.load(f)
    with open('/work/train_trx_type_map.pickle', 'rb') as f:
        trx_type_map = pickle.load(f)

    app_histrory_df = app_histrory_df[app_histrory_df['page_id'].isin(list(page_id_map.keys()))]

    transaction_df = transaction_df.dropna()
    transaction_df['trx_day'] = transaction_df['trx_tm'].str.split().str[0]
    transaction_df = transaction_df[transaction_df['trx_cd'].isin(list(trx_type_map.keys()))]

    # 计算最大值和最小值
    max_val = transaction_df["trx_amt"].max()
    min_val = transaction_df["trx_amt"].min()

    # 定义归一化函数
    def normalize(x):
        return (x - min_val) / (max_val - min_val)

    # 将数据归一化
    transaction_df["trx_amt"] = transaction_df["trx_amt"].apply(normalize)

    # 排序
    customer_base_info_df.sort_values("cust_wid", inplace=True)
    app_histrory_df.sort_values(['cust_wid', 'acs_tm', 'page_id'], ascending=[True, True, True], inplace=True)
    transaction_df.sort_values(
        ['cust_wid', 'trx_tm', 'trx_cd', 'trx_amt'], ascending=[True, True, True, True], inplace=True
    )

    # customer_base_info_df.to_csv('test_userinfo.csv', index=False)
    # app_histrory_df.to_csv('test_app_his.csv', index=False)
    # transaction_df.to_csv('test_trx_his.csv', index=False)
    customer_base_info_df.to_parquet('/work/test_userinfo.parquet', index=False)
    app_histrory_df.to_parquet('/work/test_app_his.parquet', index=False)
    transaction_df.to_parquet('/work/test_trx_his.parquet', index=False)


# clean_train_set()
# clean_test_set()
