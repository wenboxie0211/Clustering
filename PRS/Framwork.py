from PRS import Core
import numpy as np

source = None

def iterating(data,k):

    # 数据集分割
    sub_data, data_index = Core.data_partition(data, num_sub=k)

    cluster, R = [], []
    for si in range(len(sub_data)):
        c, r = Core.aggregation(sub_data[si])

        # 映射
        cluster.append(c)
        R.append(r)






