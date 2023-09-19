# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import print_function

import argparse
from collections import OrderedDict
import csv
import math
import os

import sys
sys.path.append("..")
import graph
import utils

'''
all_As 就是动态规划的结果，示例如下：

all_As = {list: 2}  
 0 = {list: 100} 
  000 = {list: 99} 
   00 = {list: 5} [(0.0070220000000000005, None, 1), (0.1689894, None, 2), (0.14943257777777777, None, 3), (0.1258643, None, 4), (0.107310576, None, 5)]
   01 = {list: 5} [(0.012285, None, 1), (0.0070220000000000005, (0, 0), 1), (0.0865995, (0, 0), 2), (0.07639255555555556, (0, 0), 3), (0.06429175000000001, (0, 0), 4)]
   02 = {list: 5} [(0.012558, None, 1), (0.0070220000000000005, (0, 0), 1), (0.0070220000000000005, (1, 1), 1), (0.0070220000000000005, (1, 1), 2), (0.0070220000000000005, (1, 1), 3)]
   03 = {list: 5} [(0.021096, None, 1), (0.012285, (1, 0), 1), (0.008538, (2, 1), 1), (0.008538, (2, 2), 1), (0.008538, (2, 3), 1)]
   ......
  __len__ = {int} 100
  
1 = {list: 100} 
 000 = {list: 99} 
  00 = {list: 5} [(0.107310576, None, 1), (0.080131832, None, 2), (0.05930489777777778, None, 3), (0.046685052000000005, None, 4), (0.03840710336000001, None, 5)]
  01 = {list: 5} [(0.06429175000000001, None, 1), (0.072057299, None, 2), (0.05690740466666667, None, 3), (0.0460065055, None, 4), (0.03840166136, None, 5)]
  02 = {list: 5} [(0.0070220000000000005, None, 1), (0.043422424, None, 2), (0.037817488, None, 3), (0.031689068, None, 4), (0.026947711359999998, None, 5)]
  03 = {list: 5} [(0.008538, None, 1), (0.0419991328, (2, 0), 1), (0.043422424, (2, 1), 1), (0.0396227304, None, 4), (0.033697556608, None, 5)]
 ......
  __len__ = {int} 100
 __len__ = {int} 2
 
 
4.2.3 区别
我们接下来要分析代码作者两个相似名字变量之间的区别。

activation_sizes ：某个节点所有前置节点的activation_size 之和。

    for predecessor in all_predecessors:
        states[i].compute_time += ((predecessor.forward_compute_time +
                                    predecessor.backward_compute_time) / 1000.0)
        states[i].activation_size += predecessor.activation_size
        states[i].parameter_size += predecessor.parameter_size
        
用来计算stashed数据大小，用来看看是否超过了节点配置的内存额度。

    stashed_data_size = (activation_sizes[k+1][j]) + last_stage_parameter_size
    stashed_data_size *= math.ceil((num_machines - (m+1)) / m_prime)
    if use_memory_constraint and stashed_data_size > memory_size:
            continue
            
output_activation_sizes : 某个节点所有增强反链的activation_size之和。

    for i in range(len(states)):
        for antichain_node in states[i].antichain:
            states[i].output_activation_size += gr.nodes[antichain_node].activation_size
            
用来计算输出传播时间和输入传播时间。

    input_transfer_time = (2.0 * output_activation_sizes[k]) / \
        (bandwidth * m_prime)
    output_transfer_time = None
    if j < len(output_activation_sizes) -1:
        output_transfer_time = (2.0 *
            output_activation_sizes[j]) / (bandwidth * m_prime)
        
        
'''
def compute_partitioning(compute_times, activation_sizes, parameter_sizes,
                         output_activation_sizes, all_predecessor_ids,
                         num_machines, num_machines_within_machine,
                         bandwidth, final_level=True):
    # 初始化
    A = []
    for i in range(len(compute_times)):  # 遍历所有节点
        row_A = []
        for j in range(len(compute_times[0])):  # 所有后续节点（即第一个节点的所有后续节点）
            row_row_A = []
            for m in range(num_machines):  # 机器数目
                row_row_A.append((None, None, None))
            row_A.append(row_row_A)
        A.append(row_A)

    # 得到计算时间
    for i in range(len(compute_times)):  # 遍历所有节点
        for j in range(i, len(compute_times[0])):  # 所有后续节点
            cum_compute_time = compute_times[i][j]  # i --> j 的计算时间
            cum_activation_size = activation_sizes[i][j]  # i --> j 的激活大小
            cum_parameter_size = parameter_sizes[i][j]  # i --> j 的参数大小
            max_m = 1 if straight_pipeline else num_machines  # 线性还是并行流水线
            for m in range(max_m):  # 遍历流水线下一阶段的机器
                # 存储的数据大小
                stashed_data_size = math.ceil((num_machines - (m + 1)) / (m + 1)) * \
                                    (cum_activation_size + cum_parameter_size)
                # memory_size 是用户传进来的参数，就是每个机器有效的内存
                # use_memory_constraint 也是用户传进来的参数，就是使用的内存限制
                if use_memory_constraint and stashed_data_size > memory_size:
                    continue
                # 数据并行通讯时间依据参数尺寸，带宽，下一阶段机器数量计算
                data_parallel_communication_time = (4 * m * cum_parameter_size) / (bandwidth * (m + 1))
                # 除以本阶段机器数量，如果本阶段机器多，当然就是分开计算了
                data_parallel_communication_time /= num_machines_within_machine

                if cum_compute_time is None:
                    # 需要计算下一阶段中，每个机器的计算时间，所以还要除以(m+1)
                    A[i][j][m] = (None, None, None)  # 直接赋值
                else:
                    # 三元组，分别是[(计算时间 + 通信时间), None，(m+1)]，对应的意义是 min_pipeline_time, optimal_split, optimal_num_machines，就对应了前面的公式 2
                    A[i][j][m] = (sum([cum_compute_time,
                                       data_parallel_communication_time]) / (m + 1), None, (m + 1))

    # 需要得到最小计算时间
    min_machines = 1
    max_i = len(compute_times) if not final_level else 1
    for i in range(max_i):  # 遍历节点
        for m in range(min_machines, num_machines):  # 遍历下一阶段机器的可能选择
            for j in range(i + 1, len(compute_times[0])):  # 遍历 i 的后续节点
                (min_pipeline_time, optimal_split, optimal_num_machines) = A[i][j][m]
                if use_fewer_machines and m > 0 and (  # 如果设置了用尽量少的机器，则如果小于min_pipeline_time，就设置新的 min_pipeline_time
                        min_pipeline_time is None or A[i][j][m - 1][0] < min_pipeline_time):
                    (min_pipeline_time, optimal_split, optimal_num_machines) = A[i][j][m - 1]
                # 遍历 j 节点的前置机器 k，注意，j 是 i 的后续节点之一
                # 就是在 i --> k --> j 之间找到一个计算时间最小的，其中A[i][k][m-m_prime][0]已经是一个最优子问题了
                for k in all_predecessor_ids[j]:
                    # 如果k已经在之前计算过了，就跳过
                    if i > 0 and k in all_predecessor_ids[i - 1]:
                        continue
                    # 设置质数
                    max_m_prime = 2 if straight_pipeline else (m + 1)
                    for m_prime in range(1, max_m_prime):  # prime就是看看如何分割
                        # 输入传输时间 input_transfer_time 使用 k 的输出激活尺寸计算
                        input_transfer_time = (2.0 * output_activation_sizes[k]) / \
                                              (bandwidth * m_prime)
                        # 输出传输时间 output_transfer_time 使用 j 的输出激活尺寸计算
                        output_transfer_time = None
                        if j < len(output_activation_sizes) - 1:
                            output_transfer_time = (2.0 *
                                                    output_activation_sizes[j]) / (bandwidth * m_prime)
                        # last_stage_time 设置为 k 到 j 的计算时间, compute_times[k+1] 就对应了k的输出
                        last_stage_time = compute_times[k + 1][j]
                        if last_stage_time is None:
                            continue
                        # 设置为 k 到 j 的下一阶段参数尺寸
                        last_stage_parameter_size = parameter_sizes[k + 1][j]
                        # 设置为 k 到 j 的存储数据尺寸
                        stashed_data_size = (activation_sizes[k + 1][j]) + last_stage_parameter_size
                        # 依据机器数据计算
                        stashed_data_size *= math.ceil((num_machines - (m + 1)) / m_prime)
                        # 超过机器内存就跳过
                        if use_memory_constraint and stashed_data_size > memory_size:
                            continue
                        # 加上传输时间，所以 last_stage_time 是 (k 到 j 的计算时间) + 传输时间
                        last_stage_time = sum([last_stage_time,
                                               ((4 * (m_prime - 1) *
                                                 last_stage_parameter_size) / (bandwidth * m_prime))])
                        last_stage_time /= m_prime

                        # 如果从i到k没有边，则跳过
                        if A[i][k][m - m_prime][0] is None:
                            continue
                        # 如果i到k已经有计算时间，则选一个较大的
                        pipeline_time = max(A[i][k][m - m_prime][0], last_stage_time)
                        if activation_compression_ratio is not None:  # 如果压缩
                            # 在(A[i][k][m-m_prime][0], last_stage_time, output_transfer_time, input_transfer_time 之中选一个最大的)
                            input_transfer_time /= activation_compression_ratio
                            # output_transfer_time 也压缩
                            if output_transfer_time is not None:
                                output_transfer_time /= activation_compression_ratio
                            # 选一个大的
                            pipeline_time = max(pipeline_time, input_transfer_time)
                            if output_transfer_time is not None:
                                pipeline_time = max(pipeline_time, output_transfer_time)

                        # 如果比min_pipeline_time小，则设定 min_pipeline_time，为了下一次循环
                        if min_pipeline_time is None or min_pipeline_time > pipeline_time:
                            optimal_split = (k, m - m_prime)  # 选一个优化分割点
                            optimal_num_machines = m_prime
                            min_pipeline_time = pipeline_time
                # 设置
                A[i][j][m] = (min_pipeline_time, optimal_split, optimal_num_machines)

    return A

# 5.2 分析阶段
# 分析阶段具体可以参见下面注释。
def analyze_partitioning(A, states, start, end, network_bandwidth, num_machines,
                         activation_compression_ratio, print_configuration, verbose):
    # start，end 是本组节点的起始点，终止点
    metadata = A[start][end - 1][num_machines - 1]  # 这是个三元组  (min_pipeline_time, optimal_split, optimal_num_machines)
    next_split = metadata[1]  # metadata[1] 是 optimal_split，即 (k, m-m_prime)
    remaining_machines_left = num_machines
    splits = []
    replication_factors = []
    prev_split = end - 1  # 前一个分割点

    while next_split is not None:  # 是否继续分割
        num_machines_used = metadata[2]  # optimal_num_machines
        if verbose:
            print("-------------------------------------")
            print("Number of machines used: %d..." % num_machines_used)
            print("Split between layers %d and %d..." % (next_split[0], next_split[0] + 1))
            print("Split before antichain %s..." % (states[next_split[0] + 1].antichain))
        splits.append(next_split[0] + 1)  # 得到了 k + 1，这是关键点，因为最后返回的是splits
        compute_time = states[prev_split - 1].compute_time - \
                       states[next_split[0]].compute_time
        parameter_size = states[prev_split - 1].parameter_size - \
                         states[next_split[0]].parameter_size

        dp_communication_time = (4 * (num_machines_used - 1) * parameter_size) \
                                / (network_bandwidth * num_machines_used)
        pp_communication_time_input = (  # 下个阶段的数据输入时间
                                              2.0 * states[next_split[0]].output_activation_size *
                                              (1.0 / float(num_machines_used))) / network_bandwidth
        pp_communication_time_output = (  # 上个阶段的数据输出时间
                                               2.0 * states[prev_split - 1].output_activation_size *
                                               (1.0 / float(num_machines_used))) / network_bandwidth
        # 如果需要压缩，就进行压缩
        if activation_compression_ratio is not None:
            pp_communication_time_input /= activation_compression_ratio
            pp_communication_time_output /= activation_compression_ratio
        if activation_compression_ratio is None:
            pp_communication_time_input = 0.0
            pp_communication_time_output = 0.0

        compute_time /= num_machines_used  # 本阶段计算时间
        dp_communication_time /= num_machines_used  # 数据并行时间

        if verbose:
            print(("Compute time = %f, Data-parallel communication time = %f, "
                   "Pipeline-parallel communication time = %f...") % (
                      compute_time, dp_communication_time,
                      max(pp_communication_time_input, pp_communication_time_output)))
        prev_split = splits[-1]  # 设定新的前一分割点
        # next_split 格式是 (k, m-m_prime)，就是 optimal_split 的格式
        # A[i][j][m] 格式是 (min_pipeline_time, optimal_split, optimal_num_machines)
        metadata = A[start][next_split[0]][next_split[1]]
        next_split = metadata[1]  # 设定新的下一次分割点，就是 optimal_split
        replication_factors.append(num_machines_used)  # 每个阶段的 replication factor
        remaining_machines_left -= num_machines_used  # 剩余机器
    if verbose:
        print("-------------------------------------")
        print("Number of machines used: %d..." % metadata[2])

    #
    num_machines_used = metadata[2]
    remaining_machines_left -= num_machines_used  # 剩余的机器
    compute_time = states[prev_split - 1].compute_time
    parameter_size = states[prev_split - 1].parameter_size
    dp_communication_time = ((4 * (num_machines_used - 1) * parameter_size) /
                             (network_bandwidth * num_machines_used))
    compute_time /= num_machines_used  # 计算时间
    dp_communication_time /= num_machines_used  # 数据并行通信时间

    if verbose:
        print("Compute time = %f, Data-parallel communication time = %f..." %
              (compute_time, dp_communication_time))
        print("-------------------------------------")
    if print_configuration:
        print("Number of machines in budget not used: %d..." %
              remaining_machines_left)
        print()
        print("(Split start, split end) / compute time taken per stage "
              "/ replication factor per stage:")
    # 下面就是打印 (Split start, split end) / compute time taken per stage / replication factor per stage
    prev_split = start
    splits.reverse()  #
    splits.append(end)
    replication_factors.append(num_machines_used)
    replication_factors.reverse()
    for i in range(len(splits)):
        time = 0.0
        if prev_split > 0:
            time = states[splits[i] - 1].compute_time - states[prev_split - 1].compute_time
        else:
            time = states[splits[i] - 1].compute_time
        if print_configuration:
            print((prev_split, splits[i]), time, replication_factors[i])
        prev_split = splits[i]
    if print_configuration:
        print()
    return splits[:-1]  # 最后一个不返回
'''
3.1 main函数入口
我们首先从 main 函数看起。main函数第一部分是构建反链和拓扑排序，具体如下：

    从图中移除source节点。目的是排除干扰，因为input必然在第一层，
    没必要让优化器再来选择把输入放在哪里，所以先去除，后续转换模型时候会再加上。
    
    对图的输出进行处理，移除没有用到的输出。
    
    得到反链DAG。
    
    对反链DAG进行拓扑排序，得到一个排序好的节点列表。
    
具体代码如下：

....

0x04 计算分区
至此，图已经依据后续反链被分割成若干状态（states），每个状态很重要的一个属性是其增强反链。
states 就是对增强反链进行拓扑排序之后的结果，按照这个顺序进行训练是符合逻辑的。

自动分区算法具体分为两部分。

    compute_partitioning 是使用动态规划算法对于这些状态得出一个最优化结果，但是没有做具体分区。
    
    analyze_partitioning 是利用最优化结果来做具体分区，排序后得到了一个偏序结果。
    
下面我们逐一分析。

4.1 main函数的逻辑
main函数接下来与计算分区相关的逻辑如下：
    
    为每个状态设置index。
    
    给每个状态计算出输出激活值大小，具体是通过遍历其反链（增强反链），可以认为就是其必要前序节点给自己的输出。
    
    给每个状态计算其信息，比如计算时间，激活大小，参数大小等等，都是通过前置节点完成的 。
    
    得到总体输出大小 output_activation_sizes & 所有前置节点id，后面计算分区时候需要。
    
    依据profile估计出系统内部的计算时间，compute_times_row 是 i 节点到 后续节点（i+1, i+2, ...）的计算时间，下面类似。
    
    依据profile估计出系统内部的激活值大小。
    
    依据profile估计出系统内部的参数大小。
    
    遍历机器集&网络带宽组合。流水线可以是straight（数目为1）或者并行（数目为num_machines），
    依据目前的信息，以及机器数量，网络带宽等，使用动态规划算法计算分区。
    假如机器集&网络带宽组合有两个，则会用每个组合进行一次动态规划算法，
    最后 all_As.append(A) 这里就是两个动态规划的结果，就是考虑到各种必要因素之后的最优结果。

具体代码如下：
...

0x05 分析分区
5.1 main函数逻辑
前面计算分区只是得到了一个动态规划优化结果，需要在analyze_partitioning之中进行分析划分之后，赋予到各个层（stage）。

main函数接下来与计算分区相关的逻辑如下：

    states是反链DAG的结果，all_As 就是动态规划得到的优化结果，可能是多个。
    
    splits 初始化时候就只有一个二元组元素：最初的划分 (0, len(states))。
    
    遍历all_As的动态优化结果，对于每个动态优化结果，遍历其各个逻辑关系，调用 analyze_partitioning 对分区进行分析，
    在splits分割中遍历，splits会逐步更新（分割点逐步逐阶段细化），analyze_partitioning 返回一个 partial_splits。
    
    遍历 partial_splits，对于每一个分割点，获取其增强反链（states）的所有前置节点，
    给这些节点打上stage_id。这里是从前往后遍历，所以stage_id数值是逐步增加。
    
    把图写到文件之中。后续 convert_graph_to_model.py 会把这个文件转换成模型。
    
    做分析对比。
    
具体代码如下：

'''


def main(all_num_machines, profile_filename, network_bandwidths, memory_size,
         straight_pipeline, use_memory_constraint, use_fewer_machines,
         activation_compression_ratio, output_directory,
         print_configuration=True, verbose=False):
    gr = graph.Graph.from_str(open(profile_filename, 'r').read())

    # Zero out all metadata associated with inputs in graph, since the optimizer
    # shouldn't really get a choice with where to place the input (should always
    # be in the first stage).
    # 排除干扰，因为input必然在第一层，没必要让优化器再来选择把输入放在哪里，所以先去除，后续会再加上。
    sources = gr.sources()  # 对图的输入进行处理
    nodes_to_remove = OrderedDict()
    for source in sources:
        if source.node_desc.startswith("Input"):  # 只处理input
            source.forward_compute_time = 0.0
            source.backward_compute_time = 0.0
            source.activation_size = 0.0
            source.parameter_size = 0.0
            nodes_to_remove[source] = []
            for out_node in gr.edges[source.node_id]:
                nodes_to_remove[source].append(out_node)  # 记录这些删除source对应了哪些out节点，因为后续还要处理
            gr.remove_node(source)  # 在图中移除这些input source

    # Remove all unneeded sinks that are not used, makes code generation and
    # optimization easier.
    sinks = gr.sinks()  # 对图的输出进行处理，移除没有用到的输出
    for sink in sinks:
        if sink.node_desc.startswith("__getitem__"):
            gr.remove_node(sink)

    antichain_gr = gr.antichain_dag()  # 得到反链DAG
    states = antichain_gr.topological_sort()  # 拓扑排序，得到一个排序好的节点列表
    if verbose:
        print("Total number of states: %d" % len(states))

    ###########################################################################
    # 计算阶段
    ###########################################################################

    states_indices = {}  # 为每个状态设置index
    for i in range(len(states)):
        states_indices[states[i]] = i

    ##################################### 运行时如下
    # states_indices = {dict: 99}
    # antichain_0 -- ['node4'] = {int} 0
    # antichain_1 -- ['node5'] = {int} 1
    # antichain_2 -- ['node6'] = {int} 2
    # antichain_3 -- ['node7'] = {int} 3
    # antichain_4 -- ['node8'] = {int} 4
    # ......

    # 给每个状态计算出输出激活值大小，具体是通过遍历其反链（增强反链），可以认为就是其必要前序节点给自己的输出
    for i in range(len(states)):
        for antichain_node in states[i].antichain:
            states[i].output_activation_size += gr.nodes[antichain_node].activation_size

    # 给每个状态计算其信息，比如计算时间，激活大小，参数大小等等，都是通过前置节点完成的
    for i in range(len(states)):
        antichain = states[i].antichain
        all_predecessors = gr.all_predecessors(antichain)
        states[i].compute_time = 0.0
        states[i].activation_size = 0.0
        states[i].parameter_size = 0.0
        for predecessor in all_predecessors:  # 计算所有前置节点的信息
            states[i].compute_time += ((predecessor.forward_compute_time +
                                        predecessor.backward_compute_time) / 1000.0)
            states[i].activation_size += predecessor.activation_size
            states[i].parameter_size += predecessor.parameter_size
    gr.reset()

    # 得到总体输出大小 & 所有前置节点id，后面计算分区时候需要
    output_activation_sizes = [state.output_activation_size for state in states]
    all_predecessor_ids = [[states_indices[predecessor] for predecessor in
                            antichain_gr.predecessors(states[i].node_id)]
                           for i in range(len(states))]

    ##################################### 运行时如下
    # output_activation_sizes = {list: 99}
    # 00 = {float} 6291456.0
    # 01 = {float} 12582912.0
    # 02 = {float} 12582912.0
    # 03 = {float} 6553600.0
    # .....
    # all_predecessor_ids = {list: 99}
    #  00 = {list: 0} []
    #  01 = {list: 1} [0]
    #  02 = {list: 2} [0, 1]
    #  03 = {list: 3} [0, 1, 2]
    #  04 = {list: 4} [0, 1, 2, 3]
    #  05 = {list: 5} [2, 3, 4, 0, 1]
    #  06 = {list: 6} [2, 3, 4, 0, 1, 5]
    #  07 = {list: 7} [6, 2, 3, 4, 0, 1, 5]
    # ......

    compute_times = []  # 初始化计算时间
    activation_sizes = []  # 初始化激活值大小
    parameter_sizes = []  # 初始化参数值大小
    for i in range(len(states) + 1):  # 具体计算每一个节点的信息，去除他之前节点的影响
        compute_times_row = []
        activation_sizes_row = []
        parameter_sizes_row = []
        for j in range(len(states)):  # 去除之前的节点
            if i == 0:  # 列表中第一个节点
                compute_times_row.append(states[j].compute_time)  # i 到 j 的计算时间
                activation_sizes_row.append(states[j].activation_size)
                parameter_sizes_row.append(states[j].parameter_size)
            else:  # 列表中后续节点
                if j > (i - 1):
                    compute_times_row.append(states[j].compute_time -
                                             states[i - 1].compute_time)  # i 到 j 的计算时间
                    activation_sizes_row.append(states[j].activation_size -
                                                states[i - 1].activation_size)
                    parameter_sizes_row.append(states[j].parameter_size -
                                               states[i - 1].parameter_size)
                else:
                    compute_times_row.append(None)
                    activation_sizes_row.append(None)
                    parameter_sizes_row.append(None)

        # 依据profile估计出系统内部的计算时间，compute_times_row 是 i 节点到 后续节点（i+1, i+2, ...）的计算时间，下面类似
        compute_times.append(compute_times_row)
        activation_sizes.append(activation_sizes_row)  # 依据profile估计出系统内部的激活值大小
        parameter_sizes.append(parameter_sizes_row)  # 依据profile估计出系统内部的参数大小

    ##################################### 运行时如下
    # compute_times = {list: 100}
    # 000 = {list: 99} [0.0070220000000000005, 0.012285, 0.012558, 0.021096000000,...
    # 001 = {list: 99} [None, 0.005263, 0.005535999999999999, 0.014074000000000003, ...
    # 002 = {list: 99} [None, None, 0.00027299999999999894, 0.008811000000000003, ...
    # 003 = {list: 99} [None, None, None, 0.008538000000000004, 0.008538, ...
    # 004 = {list: 99} [None, None, None, None, -3.469446951953614e-18, 0.000191999999...

    counter = 1
    all_As = []
    num_machines_in_machine = 1  # 第一个节点就是1
    # all_num_machines, network_bandwidths 是用户在输入中指定
    # 遍历机器集&网络带宽组合。流水线可以是straight（数目为1）或者并行（数目为num_machines）
    for num_machines, network_bandwidth in zip(all_num_machines, network_bandwidths):
        print("Solving optimization problem with %d machines with inter-machine bandwidth of %.2f GB/s" % (
        num_machines, network_bandwidth / 10 ** 9))
        import numpy as np
        print(np.array(compute_times))
        # 依据目前的信息，以及机器数量，网络带宽等计算分区
        A = compute_partitioning(compute_times, activation_sizes, parameter_sizes,
                                 output_activation_sizes, all_predecessor_ids,
                                 num_machines, num_machines_in_machine,
                                 network_bandwidth,
                                 final_level=(counter == len(network_bandwidths)))
        num_machines_in_machine = num_machines  # 因为计算完了，所以设置为本阶段的机器数目
        for i in range(len(compute_times)):  # 遍历机器
            for j in range(len(compute_times[0])):  # 后续机器
                compute_times[i][j] = A[i][j][-1][0]  # 记录计算时间（本阶段最后一个机器的计算时间）
        counter += 1
        all_As.append(A)  # 添加逻辑关系，就是里面包括了不同阶段的优化逻辑
    print(np.array(compute_times))
    '''
    其中compute_times 是一个计算时间的二维数组，也可以认为是矩阵，具体举例如下。

    [w12,w13,w14,w15], // 第一个节点到后续节点的计算时间

    [None, w23,w24,w25], // 第二个节点到后续节点的计算时间

    [None, None, w34, w35], // 第三个节点到后续节点的计算时间

    [None, None, None, w45], // 第四个节点到后续节点的计算时间

    activation_sizes 和 parameter_sizes 与之类似。
    '''

    # 分析阶段
    # 在 analyze_partitioning 内部做了具体分析
    # 这里最重要的是对 gr.all_predecessors 做设置，就是设置 gr 之中每个node的stage_id，这样就是利用stage_id把初始流水线重新划分
    splits = [(0, len(states))]  # 如何分割，states是反链DAG的结果，所以 splits 初始化时候就只有一个二元组元素：最初的划分 (0, len(states))
    i = len(all_As) - 1  # all_As 就是动态规划得到的优化结果
    while i >= 0:  # 遍历优化的出来的各个逻辑关系

        print("======================================")
        print("Level %d" % (i + 1))
        print("======================================")
        new_splits = []
        stage_id = 0  # 在后续的convert_graph_to_model.py 之中会使用到
        for (start, end) in splits:  # 在分割中遍历，splits会逐步更新
            # 依据新的splits中的二元组重新计算
            partial_splits = \
                analyze_partitioning(all_As[i], states, start, end,
                                     network_bandwidths[i], all_num_machines[i],
                                     activation_compression_ratio,
                                     print_configuration, verbose)
            start_point = start  # 起始点
            for split in partial_splits:  # 遍历分析得出的节点
                new_splits.append((start_point, split))  # 添加一个新的二元祖
                if i == 0:
                    predecessors = gr.all_predecessors(states[split - 1].antichain)
                    for predecessor in predecessors:
                        if predecessor.stage_id is None:
                            predecessor.set_stage_id(stage_id)  # 设置所在阶段  # 打上stage id
                start_point = split  # 下一个阶段
                stage_id += 1  # 增加所在阶段
            new_splits.append((start_point, end))  # 添加一个新的二元祖  # 遍历这个偏序列表
            if i == 0:  # 最终的while
                # 针对每个节点，找到每个节点的所有反链
                predecessors = gr.all_predecessors(states[end - 1].antichain)
                for predecessor in predecessors:
                    if predecessor.stage_id is None:
                        predecessor.set_stage_id(stage_id)  # 设置所在阶段  # 打上stage id
            stage_id += 1  # 增加所在阶段

        print("Total number of stages: %d" % stage_id)
        splits = new_splits  # 加入新的分割
        i -= 1

    # 以下是为了把图写到文件之中。后续convert_graph_to_model.py会把这个文件转换成模型
    for source in nodes_to_remove:  # 之前移除了input节点，现在需要加回到图中
        for out_node in nodes_to_remove[source]:  # input对应的哪些输出
            source.stage_id = 0
            gr.add_edge(source, out_node)

    if output_directory is not None:
        total_num_machines = 1
        for num_machines in all_num_machines:
            total_num_machines *= num_machines
        gr.to_dot(os.path.join(output_directory, "gpus=%d" % total_num_machines))
        gr_str = str(gr)
        with open(os.path.join(output_directory, "gpus=%d.txt" % total_num_machines), 'w') as f:
            f.write(gr_str)

    # 以下是为了做分析对比
    # 计算数据并行需要的时间，以便接下来做比较，这个时间要比动态规划时间长。
    total_time = states[-1].compute_time  # 最后一个阶段的计算时间，是没有经过优化的最初计算时间
    total_parameter_size = states[-1].parameter_size
    data_parallel_total_time = total_time  # 先赋值为最后一阶段的计算时间
    num_machines_in_machine = 1  # 本阶段的机器数目
    # 遍历流水线上各个阶段，因为没有优化，所以就是严格按照用户原始配置的流水线阶段来逐一计算
    for (num_machines, network_bandwidth) in zip(all_num_machines, network_bandwidths):
        # 计算传输时间。num_machines是下一阶段流水线机器数目，所以带宽需要乘以这个数字
        data_parallel_communication_time = (
                                                   (4 * (num_machines - 1) * total_parameter_size) /
                                                   (network_bandwidth * num_machines)) / num_machines_in_machine
        # 总时间需要加上传输时间
        data_parallel_total_time = sum(
            [data_parallel_total_time, data_parallel_communication_time]) / num_machines
        # 下个迭代中，本阶段的机器数目需要设置为num_machines
        num_machines_in_machine = num_machines

    # 这个是用动态规划算法得出来的优化时间
    pipeline_parallel_total_time = A[0][len(states) - 1][num_machines - 1][0]

    # 可以看到用户需要注意哪些数据
    if verbose:
        print()
        print("Time taken by single-stage pipeline:", total_time)
        print("Time per stage in pipeline:", pipeline_parallel_total_time)
        print("Throughput increase (compared to single machine):",
              total_time / pipeline_parallel_total_time)
        dp_str = ",".join([str(elem) for elem in all_num_machines])
        print(("[Note that single-machine and (%s)-machine DP might not fit "
               "given memory constraints]") % dp_str)
        print("Throughput increase of (%s)-machine DP compared to single "
              "machine:" % dp_str, total_time / data_parallel_total_time)
        print("Throughput increase (compared to (%s)-machine DP):" % dp_str,
              data_parallel_total_time / pipeline_parallel_total_time)
    return pipeline_parallel_total_time, data_parallel_total_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Run PipeDream's optimizer for replicated settings")
    )
    parser.add_argument('-n', "--all_num_machines", nargs='+', type=int,
                        help="Number of machines available")
    parser.add_argument('-f', "--profile_filename", required=True,
                        help="Profile filename")
    parser.add_argument('-b', "--network_bandwidths", type=float, nargs='+', default=[1000000000],
                        help="Available network bandwidth in bytes/sec")
    parser.add_argument('-s', "--memory_size", type=float, default=16000000000,
                        help="Amount of memory available on each machine")
    parser.add_argument("--straight_pipeline", action='store_true',
                        help="No replication across stages")
    parser.add_argument('-o', "--output_directory", default=None, type=str,
                        help="Output directory to dump processed graph")
    parser.add_argument("--use_memory_constraint", action='store_true',
                        help="Enforce memory constraint per machine")
    parser.add_argument("--use_fewer_machines", action='store_true',
                        help="Use fewer machines, if possible")
    parser.add_argument("--activation_compression_ratio", default=None, type=float,
                        help="Compression ratio for activations")

    args = parser.parse_args()
    args = vars(args)

    all_num_machines = args["all_num_machines"]
    profile_filename = args["profile_filename"]
    network_bandwidths = args["network_bandwidths"]
    assert(len(all_num_machines) == len(network_bandwidths))
    memory_size = args["memory_size"]
    straight_pipeline = args["straight_pipeline"]
    output_directory = args["output_directory"]
    use_memory_constraint = args["use_memory_constraint"]
    use_fewer_machines = args["use_fewer_machines"]
    activation_compression_ratio = args["activation_compression_ratio"]

    main(all_num_machines, profile_filename, network_bandwidths, memory_size,
         straight_pipeline, use_memory_constraint, use_fewer_machines,
         activation_compression_ratio, output_directory,
         verbose=True)
