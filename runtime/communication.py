# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import threading
import torch
import torch.distributed as dist
import sys

import threadsafe_counter
import threadsafe_queue


NCCL='nccl'
GLOO='gloo'

'''
0x02 类定义
CommunicationHandler 负责在阶段（Stage）之间的通信。

    如果阶段位于不同机器上，就使用 PyTorch p2p 的 send/recv。
    
    如果阶段位于同一个机器上，则使用 PyTorch p2p 的 broadcast。
    
下面代码中，主要就是初始化各种成员变量，我们目前最熟悉的是和DDP相关的，比如init_process_group。
'''
class CommunicationHandler(object):
    """ Handles communication between stages.

    For stages on different machines, use send/recv.
    For stages on same machine, use broadcast.
    """
    def __init__(self, master_addr, master_port, rank,
                 local_rank, num_ranks_in_server,
                 world_size, fp16, backend):
        """ Set up process groups.

        Note: To turn off broadcasting, set num_ranks_in_server = 1.
        """
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self.num_ranks_in_server = num_ranks_in_server
        self.world_size = world_size
        self.fp16 = fp16
        assert num_ranks_in_server > 0

        # Initialize the distributed environment.
        # 以下是为了 DDP
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        assert dist.get_world_size() == self.world_size
        print("Finished initializing process group; backend: %s, rank: %d, "
              "world_size: %d" % (backend, rank, world_size))

        # Stores list of ranks of GPUs on the same server.
        self.ranks_in_server = []

        if num_ranks_in_server == 1:
            return

        # Stores information about tensors sent directly GPU-to-GPU.
        self.connection_list = []

        # Stores process groups (for broadcast() connections).
        self.process_groups = {}

        # Populate ranks_in_server.
        rank_of_first_gpu_in_server = rank - rank % num_ranks_in_server
        for connected_rank in range(
            rank_of_first_gpu_in_server,
            rank_of_first_gpu_in_server + num_ranks_in_server):
            if connected_rank == rank:
                continue
            self.ranks_in_server.append(connected_rank)
        assert len(self.ranks_in_server) == num_ranks_in_server - 1, \
            self.ranks_in_server

    def is_gpu_to_gpu_comm(self, connected_rank):
        if connected_rank in self.ranks_in_server:
            return True
        return False

    '''
    注意，每个张量有唯一一个tag，针对这个张量和这个唯一的tag，注册 [tag, rank] 到 connection_list。
    '''
    def register_tensor(self, connected_rank, tag):
        """
        Builds connections list of tensors that are communicated GPU to GPU.

        For tensors that are sent GPU-to-GPU (intra-server for GLOO backend),
        make a list of destination/source ranks and the corresponding tag.
        This information is then used to crate process groups.
        """
        if not self.is_gpu_to_gpu_comm(connected_rank=connected_rank):
            return
        connection_info = [tag, connected_rank]
        self.connection_list.append(connection_info)

    '''
    0x03 构建
    3.1 初始化
    前面章节中提到，当生成了CommunicationHandler之后，会调用initialize进行初始化。
    
    在初始化代码之中，完成如下操作，主要是：
    
        构建通信需要的queue。
        构建发送消息的次序。
        构建进程组。
    '''
    def initialize(self, receive_ranks, send_ranks,
                   tensor_tags, target_tensor_names,
                   training_tensor_dtypes,
                   rank_in_stage,
                   num_ranks_in_stage,
                   ranks_in_previous_stage,
                   ranks_in_next_stage):
        """
        Initialize state needed for CommunicationHandler.
        """
        self.receive_ranks = receive_ranks
        self.send_ranks = send_ranks
        self.tensor_tags = tensor_tags
        self.target_tensor_names = target_tensor_names
        self.training_tensor_dtypes = training_tensor_dtypes
        self.rank_in_stage = rank_in_stage
        self.num_ranks_in_stage = num_ranks_in_stage
        self.ranks_in_previous_stage = ranks_in_previous_stage
        self.num_ranks_in_previous_stage = len(ranks_in_previous_stage)
        self.ranks_in_next_stage = ranks_in_next_stage
        self.num_ranks_in_next_stage = len(ranks_in_next_stage)

        self.setup_queues() # 构建通信需要的queue
        self.setup_messaging_schedule() # 构建发送消息的次序
        self.create_process_groups() # 构建进程组

'''
3.2 创建queue
Queue 的作用是作为 send，receive 的基础，系统通过index找到哪一个queue，然后进行相应操作。

initialize 函数传入了两个ranks列表。
    
    receive_ranks 就是本节点的输入rank。
    send_ranks 就是本节点的输出rank。
    
ranks 列表举例如下：

    receive_ranks = {dict: 3}  # 这里就是每个tensor对应的接收目标rank
     'out8' = {list: 1} [2] # out8 是tensor name, {list: 1} [2] 是 out8 对应的 ranks
     'out9' = {list: 1} [2] # 就是这几个张量都要从 rank 2 接收
     'out10' = {list: 1} [2]
     __len__ = {int} 3
     
setup_queues 相应一共建立了4个queue列表：

    forward_receive_queues ：
    前向传播过程中，接受张量的queue。对应了 receive_ranks。
    
    backward_send_queues : 
    后向传播过程中，发送张量的queue。对应了 receive_ranks。因为前向传播中接受的对象，就是后向传播中发送的目标。
    
    forward_send_queues : 
    前向传播过程中，发送张量的queue。对应了 send_ranks。
    
    backward_receive_queues ：
    后向传播过程中，接受张量的queue。对应了 send_ranks。因为前向传播中发送的目标就是后向传播中接受的对象。

大致逻辑如下：

    forward_receive_queues <-----> receive_ranks <------->  backward_send_queues
    forward_send_queues  <------>  send_ranks    <------->  backward_receive_queues
    
以 forward_receive_queues 为例。

    forward_receive_queues 这个列表包括多个queue。
    
    receive_ranks 列表中包括多个 rank，每个rank在通信过程之中，对应了一个张量，
    可以认为 receive_ranks 包括多个张量，由一个张量名字来对应。
    张量名字类似于：target_tensor_names = {"target", "target_length"}。
    
    forward_receive_queues 列表之中，每一个queue对应了receive_ranks 之中的一个 张量。
    
    每个张量，对应一个唯一的tag，PipeDream的目的是让每一个tag都有自己的process group，因为任何一个stage都有可能并行。
    
    针对这个张量和这个唯一的tag，注册 [tag, rank] 到 connection_list。
具体如下：
'''
def setup_queues(self):
    """
    Setup queues for communication between main compute thread
    and helper communication threads. One queue per tensor
    in forward / backward direction.
    """
    self.forward_receive_queues = {}
    self.backward_receive_queues = {}
    self.forward_send_queues = {}
    self.backward_send_queues = {}
    self.num_forward_threads = 0
    self.num_backward_threads = 0

    self.target_receive_rank_counts = {}
    self.target_send_rank_counts = {}
    # Setup queues for each tensor to be received and sent.
    for input_name in self.receive_ranks:  # 遍历张量
        # 与 input_name 张量对应的queue，input_name 是张量名字
        self.forward_receive_queues[input_name] = []
        self.backward_send_queues[input_name] = []
        # 遍历该张量对应的每个 ranks
        for i in range(len(self.receive_ranks[input_name])):
            self.forward_receive_queues[input_name].append(
                threadsafe_queue.Queue())
            self.backward_send_queues[input_name].append(
                threadsafe_queue.Queue())
            # 得到 rank
            target_receive_rank = self.receive_ranks[input_name][i]
            # 针对 rank，注册张量
            self.register_tensor(
                connected_rank=target_receive_rank,
                tag=self.tensor_tags[input_name])
            if target_receive_rank not in self.target_receive_rank_counts:
                self.target_receive_rank_counts[target_receive_rank] = 0
            self.target_receive_rank_counts[target_receive_rank] += 1
            self.num_forward_threads += 1
            self.num_backward_threads += 1

    for output_name in self.send_ranks:  # 遍历张量
        # 与 output_name 张量对应的queue
        self.backward_receive_queues[output_name] = []
        self.forward_send_queues[output_name] = []
        # 遍历该张量对应的每个 ranks
        for i in range(len(self.send_ranks[output_name])):
            self.backward_receive_queues[output_name].append(
                threadsafe_queue.Queue())
            self.forward_send_queues[output_name].append(
                threadsafe_queue.Queue())
            # 得到 rank
            target_send_rank = self.send_ranks[output_name][i]
            # 针对 rank，注册张量
            self.register_tensor(
                connected_rank=target_send_rank,
                tag=self.tensor_tags[output_name])
            if target_send_rank not in self.target_send_rank_counts:
                self.target_send_rank_counts[target_send_rank] = 0
            self.target_send_rank_counts[target_send_rank] += 1
            self.num_forward_threads += 1
            self.num_backward_threads += 1

    # 单独处理目标tensor
    for target_tensor_name in self.target_tensor_names:
        # Queues for target in forward pass.
        self.forward_receive_queues[target_tensor_name] = []
        self.forward_send_queues[target_tensor_name] = []

        if self.num_ranks_in_previous_stage > 0:
            self.receive_ranks[target_tensor_name] = self.ranks_in_previous_stage
            for i in range(len(self.receive_ranks[target_tensor_name])):
                # 针对 rank，注册张量
                self.register_tensor(
                    connected_rank=self.receive_ranks[target_tensor_name][i],
                    tag=self.tensor_tags[target_tensor_name])
                self.forward_receive_queues[target_tensor_name].append(
                    threadsafe_queue.Queue())
                self.num_forward_threads += 1

        if self.num_ranks_in_next_stage > 0:
            self.send_ranks[target_tensor_name] = self.ranks_in_next_stage
            for i in range(len(self.send_ranks[target_tensor_name])):
                self.register_tensor(
                    connected_rank=self.send_ranks[target_tensor_name][i],
                    tag=self.tensor_tags[target_tensor_name])
                self.forward_send_queues[target_tensor_name].append(
                    threadsafe_queue.Queue())
                self.num_forward_threads += 1

    print("Send ranks: ", self.send_ranks)
    print("Receive ranks: ", self.receive_ranks)

    # Queues for ack for forward pass-only runs as a clocking mechanism.
    # 单独处理 ack 情况
    self.num_ack_threads = 0
    if "ack" in self.tensor_tags:
        self.backward_receive_queues["ack"] = []
        self.backward_send_queues["ack"] = []
        for i in range(self.num_ranks_in_previous_stage):
            # 针对 rank，注册张量
            self.register_tensor(
                connected_rank=self.ranks_in_previous_stage[i],
                tag=self.tensor_tags["ack"])
            self.backward_send_queues["ack"].append(
                threadsafe_queue.Queue())
            self.num_ack_threads += 1
        for i in range(self.num_ranks_in_next_stage):
            # 针对 rank，注册张量
            self.register_tensor(
                connected_rank=self.ranks_in_next_stage[i],
                tag=self.tensor_tags["ack"])
            self.backward_receive_queues["ack"].append(
                threadsafe_queue.Queue())
            self.num_ack_threads += 1
        '''
        于是，此时逻辑如下，我们仅仅以部分 ranks，queue等举例，
        forward_receive_queues 之中的这几个queue 就是用来作为对应张量的buffer。

+------------------------+         'out8' = {list: 1} [2]
|                        |
|     receive_ranks +----------->  'out9' = {list: 1} [2]
|                        |
+------------------------+         'out10' = {list: 1} [2]



+--------------------------+
|                          |         'out8' ： Queue
| forward_receive_queues+-------->
|                          |         'out9' ： Queue
+--------------------------+
                                     'out10' : Queue




+--------------------------+       'out8' : rank 2
|                          |
|    connection_list  +--------->  'out9' : rank 2
|                          |
+--------------------------+       'out10' : rank 2

        '''

    def set_tensor_shapes(self, tensor_shapes):
        self.tensor_shapes = tensor_shapes

    def set_counter(self, counter):
        self.counter = threadsafe_counter.Counter(counter)

    def wait(self):
        self.counter.wait()

    def num_iterations_for_helper_threads(self, num_iterations):
        """ Scales the number of iterations a helper thread is run for.

        Since we start a helper thread for each worker in previous/next stage,
        the number of iterations for each thread should be scaled by
        the number of workers in previous/next stage.

        TODO: don't current support uneven configurations.
        """
        forward_num_iterations = num_iterations
        backward_num_iterations = num_iterations

        if self.num_ranks_in_next_stage > 0:
            assert forward_num_iterations % self.num_ranks_in_next_stage == 0
            forward_num_iterations = forward_num_iterations // \
                self.num_ranks_in_next_stage
        else:
            forward_num_iterations = 0

        if self.num_ranks_in_previous_stage > 0:
            assert backward_num_iterations % self.num_ranks_in_previous_stage == 0
            backward_num_iterations = backward_num_iterations // \
                self.num_ranks_in_previous_stage
        else:
            backward_num_iterations = 0

        return forward_num_iterations, backward_num_iterations

    '''
    3.5 启动助手线程
    使用 start_helper_threads 来进行启动助手线程。这些助手线程是为了 P2P 使用。
    
    首先，ranks举例，可以看出来，key 是张量名字，value 是ranks列表。
    
        receive_ranks = {dict: 3}  # 这里就是每个tensor对应的接收目标rank
         'out8' = {list: 1} [2]
         'out9' = {list: 1} [2]
         'out10' = {list: 1} [2]
         __len__ = {int} 3
         
    3.5.1 建立线程
    回忆一下之前建立的 4 个queues：
    
        forward_receive_queues ：
        前向传播过程中，接受张量的queue。对应了 receive_ranks。
        
        backward_send_queues : 
        后向传播过程中，发送张量的queue。对应了 receive_ranks。因为前向传播中接受的对象，就是后向传播中发送的目标。
        
        forward_send_queues : 
        前向传播过程中，发送张量的queue。对应了 send_ranks。
        
        backward_receive_queues ：
        后向传播过程中，接受张量的queue。对应了 send_ranks。因为前向传播中发送的目标就是后向传播中接受的对象。
        
    这 4 个queue 其实就对应了 4 个不同的助手线程。
    
    思路是：
    
        针对接受ranks进行处理，即遍历 receive_ranks 中的张量
            遍历张量对应的ranks，对于每一个rank
                需要后向处理，所以建立后向发送线程
                建立接受助手线程
                
        针对发送ranks进行处理，即遍历 send_ranks 中的张量
            遍历张量对应的ranks，对于每一个rank
                需要后向处理，所以建立后向接受线程
                建立发送助手线程
                
        针对target进行处理
        
        如果只有前向，则需要补齐ack
    
    具体代码是：
    
    
    具体线程建立函数为：

    def start_helper_thread(self, a  ....
    '''

    def start_helper_threads(self, num_iterations, forward_only):
        """
        Start helper communication threads, one for each queue.
        """
        if forward_only:
            self.set_counter(self.num_forward_threads +
                             self.num_ack_threads)
            # For validation, receive acks in backward pass from next stage, send
            # acks in backward pass to next stage.
            self.receive_ranks["ack"] = self.ranks_in_previous_stage
            self.send_ranks["ack"] = self.ranks_in_next_stage
        else:
            self.set_counter(self.num_forward_threads +
                             self.num_backward_threads)
            if "ack" in self.receive_ranks:
                del self.receive_ranks["ack"]
            if "ack" in self.send_ranks:
                del self.send_ranks["ack"]

        (num_iterations_for_forward_threads,
         num_iterations_for_backward_threads) = \
            self.num_iterations_for_helper_threads(
                num_iterations=num_iterations)
        dtype = torch.float16 if self.fp16 else torch.float32

        # Setup queues for each tensor to be received and sent.
        # 针对接受rank进行处理
        for input_name in self.receive_ranks:
            if input_name in self.target_tensor_names or input_name == "ack":
                continue

            # 遍历张量对应的ranks
            for i in range(len(self.receive_ranks[input_name])):
                if not forward_only:
                    # 需要后向处理，所以建立后向发送线程
                    self.start_helper_thread(
                        self.send_helper_thread_args,
                        send_helper_thread,
                        [input_name, i, True],
                        num_iterations_for_backward_threads)
                # 建立接受助手线程
                self.start_helper_thread(
                    self.recv_helper_thread_args,
                    recv_helper_thread,
                    [input_name,
                     i,
                     self.training_tensor_dtypes[input_name],
                     False],
                    num_iterations_for_backward_threads)

        # 针对发送ranks进行处理
        for output_name in self.send_ranks:
            if output_name in self.target_tensor_names or output_name == "ack":
                continue

            # 遍历张量对应的ranks
            for i in range(len(self.send_ranks[output_name])):
                if not forward_only:
                    # 需要后向处理，所以建立后向接受线程
                    self.start_helper_thread(
                        self.recv_helper_thread_args,
                        recv_helper_thread,
                        [output_name, i,
                         self.training_tensor_dtypes[output_name],
                         True],
                        num_iterations_for_forward_threads)
                # 发送助手线程
                self.start_helper_thread(
                    self.send_helper_thread_args,
                    send_helper_thread,
                    [output_name, i, False],
                    num_iterations_for_forward_threads)

        # 针对target进行处理
        for target_tensor_name in self.target_tensor_names:
            if self.num_ranks_in_previous_stage > 0:
                for i in range(len(self.receive_ranks[target_tensor_name])):
                    self.start_helper_thread(
                        self.recv_helper_thread_args,
                        recv_helper_thread,
                        [target_tensor_name, i, torch.int64,
                         False],
                        num_iterations_for_backward_threads)

            if self.num_ranks_in_next_stage > 0:
                for i in range(len(self.send_ranks[target_tensor_name])):
                    self.start_helper_thread(
                        self.send_helper_thread_args,
                        send_helper_thread,
                        [target_tensor_name, i, False],
                        num_iterations_for_forward_threads)

        # Start helper threads for ack for forward pass-only run as a clocking
        # mechanism.
        # 如果只有前向，则需要补齐ack
        if forward_only:
            # 有前向就补齐 ack
            if "ack" in self.receive_ranks:
                for i in range(len(self.receive_ranks["ack"])):
                    self.start_helper_thread(self.send_helper_thread_args,
                                             send_helper_thread,
                                             ["ack", i, True],
                                             num_iterations_for_backward_threads)
            if "ack" in self.send_ranks:
                for i in range(len(self.send_ranks["ack"])):
                    self.start_helper_thread(self.recv_helper_thread_args,
                                             recv_helper_thread,
                                             ["ack", i, torch.int64, True],
                                             num_iterations_for_forward_threads)

    def start_helper_thread(self, args_func, func, args_func_args, num_iterations):
        """
        Start passed-in func on a helper thread.
        """
        args_func_args += [num_iterations]
        args = args_func(*args_func_args)  # 需要注意的是使用函数来获取对应的参数
        helper_thread = threading.Thread(target=func, # 用线程主函数来执行线程
                                         args=args)
        helper_thread.start()

    '''
    3.4 建立进程组
    目的是：针对每个张量，设置两个进程组，一个用于前向，一个用于后向。
    每一个张量有一个自己的tag。每一个tag都有自己的两个process group，因为任何一个stage都有可能并行。
    
    3.4.1 设计
    首先，我们看看注释，学习一下为何这么设计。
    
    create_process_groups 方法在所有rank之中以同样顺序建立进程组。
    为了以同样顺序建立进程组，每个worker都会收集其他所有workers的connection_list（GPU to GPU）。
    为了做到这一点，每个worker收集所有其他workers的连接列表connection_list（L）的最大大小。
    然后每个worker创建一个大小为Lx2的张量，其中每行表示一个连接，并根据“它本身连接列表大小”来填充此张量。
    拥有最大连接列表的worker将填充整个张量。
    
    构建此列表后，将执行all_gather操作，然后每个worker都拥有一个相同的 NxLx2 输出，其中N是worker 数量（world_size），
    输出的每个index代表一个worker的连接列表。对于 i=self.rank，输出将与本worker的本地连接列表相同。
    
    每个worker以相同的顺序在连接列表上进行迭代，检查是否已创建每个连接（每个连接都将在输出中出现两次），
    如果连接不存在，则对于前向和后向都创建一个新的进程组。
    既然在进程组中rank永远是一致的，所以小rank排在前面，大的rank排在后面。
    
    3.4.2 代码
    回到代码上，我们仔细分析下。
    
    +--------------------------+       'out8' : rank 2
    |                          |
    |    connection_list  +--------->  'out9' : rank 2
    |                          |
    +--------------------------+       'out10' : rank 2
    
    这里就用到了 connection_list。具体逻辑是：
    
        找到 workers 之中最大的 connection_list
        
        获取到 connection_list 的大小，即 connection_list_size
        
        用集合通信来对 connection_list_size 进行聚合，
        最后得到的gathered_connection_list_sizes就是所有节点上的 connection_list_size 集合
        
        得到connection_list的最大数值
        
        利用最大数值来构建张量列表 connection_list_tensor
        
        把张量移动到GPU之上
        
        用集合通信来对 connection_list_tensor进行聚合，得到aggregated_connection_list
        
        在每个worker之上，利用 dist.new_group 建立同样的进程组
        
        遍历aggregated_connection_list中的每一个connection
        
            得到张量对应的tag
            
            针对每个张量，设置两个进程组，一个前向，一个后向
    
    因此，目的就是在每个 worker 之中建立同样的进程组，针对每个张量，设置两个进程组，一个前向，一个后向。
    
    具体代码如下：
    
    具体 如何使用进程组？在 recv_helper_thread_args 等函数会使用，比如：....
    '''

    def create_process_groups(self):
        """ Create process groups in the same order across all ranks.

        To create process groups in the same order, each worker collects
        the connection_list of all other workers. To do this, every worker
        gathers the largest size of all other worker's connection_lists (L).
        Then every worker creates a tensor of size Lx2, where each row
        represents a connection, and fills up this tensor depending on how
        large its own connection list is. The worker(s) w/ the largest
        connection list will fill up the entire tensor.

        After constructing this list, an all_gather is performed, after which
        each worker has an identical NxLx2 output, where N is the number of
        workers (world_size), and each index of output represents a worker's
        connection list. For i=self.rank, the output will be identical to the
        workers local connection list.

        Each worker then iterates in the same order over the connections list,
        checking if each connection has been created yet (every connection will
        appear twice in the output), and creating a new process group if one
        doesn't exist for that connection, for both the forward and backward
        direction. Since ranks within process groups must always be identical,
        the smaller rank always goes first, followed by the larger rank.
        """
        if self.num_ranks_in_server == 1:
            return

        print("Setting up process groups for broadcasts...")

        # Figure out the size of the largest connection list that any worker
        # has (L).
        # 找到最大的 connection_list
        # 获取到 connection_list 的大小，即 connection_list_size
        connection_list_size = torch.tensor(
            len(self.connection_list), dtype=torch.int)
        if self.backend == NCCL:
            connection_list_size = connection_list_size.cuda()
        gathered_connection_list_sizes = [
            torch.ones_like(connection_list_size)
            for _ in range(self.world_size)]

        # 用集合通信来对 connection_list_size 进行聚合，最后得到的gathered_connection_list_sizes就是所有节点上的 connection_list_size 集合
        dist.all_gather(gathered_connection_list_sizes,
                        connection_list_size)
        # 得到最大数值
        max_connection_list_size = max(
            gathered_connection_list_sizes)

        if max_connection_list_size == 0:
            return

            # 利用最大数值来构建张量列表 connection_list_tensor
        # Build tensor to send local connection list to all other workers.
        connection_list_tensor = torch.ones([max_connection_list_size, 2],
                                            dtype=torch.int) * -1
        # 把张量移动到GPU之上
        if self.backend == NCCL:
            connection_list_tensor = connection_list_tensor.cuda()
        if len(self.connection_list) > 0:
            connection_list_tensor[0:len(self.connection_list)] = \
                torch.IntTensor(self.connection_list)

        # 用集合通信来对 connection_list_tensor进行聚合
        # Gather connection lists of all workers.
        aggregated_connection_list = [
            torch.ones_like(connection_list_tensor)
            for _ in range(self.world_size)]
        dist.all_gather(aggregated_connection_list,
                        connection_list_tensor)

        # 在每个worker之上，利用 dist.new_group 建立同样的进程组
        # Construct identical process groups on each worker.
        local_rank_connections = 0
        for src_rank in range(len(aggregated_connection_list)):
            for connection in aggregated_connection_list[src_rank]:
                # 得到张量对应的tag
                tag = int(connection[0])
                dst_rank = int(connection[1])

                if tag == -1:
                    assert dst_rank == -1
                    continue

                min_rank = min(src_rank, dst_rank)
                max_rank = max(src_rank, dst_rank)
                assert min_rank != max_rank

                if min_rank not in self.process_groups:
                    self.process_groups[min_rank] = {}

                if max_rank not in self.process_groups[min_rank]:
                    self.process_groups[min_rank][max_rank] = {}

                '''
                3.5.3 构建参数
                回忆一下，在 create_process_groups 方法中，
                有如下代码，这里就给每一个 tag 设定了 进程组，在助手线程之中，就要利用这些进程组来完成逻辑：
                '''
                if tag not in self.process_groups[min_rank][max_rank]:
                    # 用到了pytorch p2p 的api
                    sub_process_group_fwd = dist.new_group(
                        ranks=[min_rank, max_rank])
                    sub_process_group_bwd = dist.new_group(
                        ranks=[min_rank, max_rank])

                    # 针对每个张量，设置进程组
                    self.process_groups[min_rank][max_rank][tag] = {
                        'forward': sub_process_group_fwd,
                        'backward': sub_process_group_bwd
                    }

                    if min_rank == self.rank or max_rank == self.rank:
                        local_rank_connections += 1
        assert local_rank_connections == len(self.connection_list)

    '''
    3.3 前向后向顺序
    接下来建立消息传递的前后向顺序，其目的是为了让每个 worker 记录如何处理由前向层/后向层传来的rank。
    
    3.3.1 建立顺序
    setup_messaging_schedule 方法就是用来建立：
    
        前向传播时接受的顺序。
        
        后向传播时发送的顺序。
        
    这里的重点是：如果前一层数目比本层数目多，
    就把 i对应的前一层rank 和 i + (本层rank数目) * n 对应的前一层rank 都加入到本层 i 的计划（self.message_schedule）。
    n 等于 num_ranks_in_stage。
    
    最终把顺序放入 self.messaging_schedule 成员变量。
    假如本stage是拥有 3 个rank，则 self.messaging_schedule 就是这三个rank 分别的 message_schedule，
    每个 message_schedule 里面都是对应的上一层 某些 ranks。
    
    再细化一下：
    
        self.messaging_schedule 是一个列表。
        
        self.messaging_schedule 其中每一个item又是一个列表。
        self.messaging_schedule[ i ] 就表示比如 本层 第 i 个 rank 对应的 schedule（message_schedule）。
        
        schedule（message_schedule）是上一层 或者 下一层 的某些ranks。
        
        message_schedule包括的ranks是本stage所包括ranks的一个index。
        因为是内部使用，所以不需要是真正的 rank 数值，只要能和内部的queue等其他内部数据结构映射上即可。
        
    代码如下：
    
    具体逻辑如下：

+-------------------+                 +--------------------------------------------------+
| Stage 0           |                 | Stage 1                                          |
|                   |                 |                                                  |
|                   |                 |                                                  |
|                   |                 |     +----------------------------------------+   |
|                   |   send_ranks    |     | messaging_schedule                     |   |
|  ranks:           |                 |     |                                        |   |
|                   +---------------> |     |                                        |   |
|  [0,1,2,3,4,5,    |                 |     |   message_schedule +---> [0,1,2,9]     |   |
|  6,7,8,9,10,11,12]|                 |     |                                        |   |
|                   |                 |     |   message_schedule +---> [3,4,5,6,10]  |   |
|                   |                 |     |                                        |   |
|                   |                 |     |   message_schedule +---> [6,7,8,11]    |   |
|                   |                 |     |                                        |   |
|                   |                 |     +----------------------------------------+   |
|                   |                 |                                                  |
+-------------------+                 +--------------------------------------------------+

    '''
    def setup_messaging_schedule(self):
        """ Order in which to receive forward and send backwards.

        Separate indexes of ranks in previous stage based on their
        corresponding offset in this stage. Then each worker will go
        in increasing order within a subset, and process subsets in
        a decreasing order.

        This is done so that messages are processed in the order
        that they are sent. Backwards send is done so that that it
        matches up with forward receive.
        """
        self.messaging_schedule = []
        for i in range(self.num_ranks_in_stage): # 本stage的并行数目
            idx = i
            message_schedule = []
            while idx < self.num_ranks_in_previous_stage: # 上一个stage的并行数目
                message_schedule.append(idx)
                # 如果前一层比本层多，就把 i, i + (本层rank) * n 对应的前一层rank都加入到本层 i 的计划
                idx += self.num_ranks_in_stage
            if len(message_schedule) > 0:
                self.messaging_schedule.append(message_schedule)

        self.fwd_messaging_scheduling_row = self.rank_in_stage # 自己的rank index
        self.fwd_messaging_scheduling_col = 0 # receive forward
        self.bwd_messaging_scheduling_row = self.rank_in_stage # 自己的rank index
        self.bwd_messaging_scheduling_col = 0 # send backwards

        # For cases where previous stage has less workers than current stage.
        while self.fwd_messaging_scheduling_row >= \
            len(self.messaging_schedule):
            self.fwd_messaging_scheduling_row -= 1
            self.bwd_messaging_scheduling_row -= 1

#3.3.2 获取消息序列
# get_messaging_index 方法是用来获取本次传递的对象，就是应该和哪个rank进行交互。
#     哪里用到了get_messaging_index？原来是send, recv函数，就是和前一层打交道时候会用到。
#     比如： def recv(self, tensor_name, forward_minibatch_id
    def get_messaging_index(self, sending):
        if sending:
            connection_rank = self.messaging_schedule[
                self.bwd_messaging_scheduling_row][
                    self.bwd_messaging_scheduling_col]
        else:
            connection_rank = self.messaging_schedule[
                self.fwd_messaging_scheduling_row][
                    self.fwd_messaging_scheduling_col]

        return connection_rank

    '''
    3.3.3 增加消息序列
    increment_messaging_index 方法用来增加消息序列，就是得到下一次应该使用哪个消息。
    
    其中，两个参数需要说明：
    
        bwd_messaging_scheduling_col 表示上游具体哪一个 rank index。
        
        bwd_messaging_scheduling_row 表示自己的 rank index。
    
    方法如下：
    
哪里会用到？在以下几个函数中会用到：

    def receive_tensors_forward(self):

    def send_tensors_backward(self):
 
    def run_ack(self):
        
    '''
    def increment_messaging_index(self, sending):
        if sending:
            self.bwd_messaging_scheduling_col += 1 # send backwards 对应的下一个 rank
            if self.bwd_messaging_scheduling_col == len(
                    self.messaging_schedule[
                        self.bwd_messaging_scheduling_row]):
                self.bwd_messaging_scheduling_col = 0
                self.bwd_messaging_scheduling_row -= 1 # 自己的rank index
                if self.bwd_messaging_scheduling_row == -1:
                    self.bwd_messaging_scheduling_row = \ # 重置回self.messaging_schedule，继续新的一轮本地 rank通讯
                        len(self.messaging_schedule) - 1
        else:
            self.fwd_messaging_scheduling_col += 1 # receive forward 对应的下一个 rank
            if self.fwd_messaging_scheduling_col == len(
                    self.messaging_schedule[
                        self.fwd_messaging_scheduling_row]):
                self.fwd_messaging_scheduling_col = 0
                self.fwd_messaging_scheduling_row -= 1 # 自己的rank index
                if self.fwd_messaging_scheduling_row == -1:
                    self.fwd_messaging_scheduling_row = \ # 重置回self.messaging_schedule，继续新的一轮本地 rank通讯
                        len(self.messaging_schedule) - 1
    '''
    使用如下函数来完成对线程主函数参数的获取。基本逻辑就是：
    
        利用张量名字，获取到对应的rank
        利用张量名字，获取到对应的tag
        使用tag来获取到对应的进程组
        利用张量名字和index得到对应的queue
        返回参数
    '''
    def recv_helper_thread_args(self, tensor_name, index, dtype,
                                backward, num_iterations):
        # 利用张量名字，获取到对应的rank
        if backward:
            src_rank = self.send_ranks[tensor_name][index]
        else:
            src_rank = self.receive_ranks[tensor_name][index]

        # 利用张量名字，获取到对应的tag
        sub_process_group = None
        tag = self.tensor_tags[tensor_name]

        # 使用tag来获取到对应的进程组
        if self.is_gpu_to_gpu_comm(connected_rank=src_rank) and tensor_name != "ack":
            min_rank = min(self.rank, src_rank)
            max_rank = max(self.rank, src_rank)
            if src_rank > self.rank:
                sub_process_group = \
                    self.process_groups[min_rank][max_rank][tag]['backward']
            else:
                sub_process_group = \
                    self.process_groups[min_rank][max_rank][tag]['forward']
            assert sub_process_group

        # 得到对应的queue
        if backward:
            queue = self.backward_receive_queues[tensor_name][index]
        else:
            queue = self.forward_receive_queues[tensor_name][index]
        tensor_shape = self.tensor_shapes[tensor_name]

        # 返回参数
        return (queue, self.counter, self.local_rank, tensor_name,
                src_rank, tag, tensor_shape, dtype, sub_process_group,
                num_iterations)


    def send_helper_thread_args(self, tensor_name, index,
                                backward, num_iterations):
        # 利用张量名字得到对应的rank
        if backward:
            dst_rank = self.receive_ranks[tensor_name][index]
            num_ranks_in_connected_stage = self.num_ranks_in_previous_stage
        else:
            dst_rank = self.send_ranks[tensor_name][index]
            num_ranks_in_connected_stage = self.num_ranks_in_next_stage

        # 使用tag来获取到对应的进程组
        sub_process_group = None
        tag = self.tensor_tags[tensor_name]
        if self.is_gpu_to_gpu_comm(connected_rank=dst_rank) and tensor_name != "ack":
            min_rank = min(self.rank, dst_rank)
            max_rank = max(self.rank, dst_rank)
            if dst_rank > self.rank:
                sub_process_group = \
                    self.process_groups[min_rank][max_rank][tag]['forward']
            else:
                sub_process_group = \
                    self.process_groups[min_rank][max_rank][tag]['backward']
            assert sub_process_group

        # 得到对应的queue
        if backward:
            queue = self.backward_send_queues[tensor_name][index]
        else:
            queue = self.forward_send_queues[tensor_name][index]

        # 返回参数
        return (queue, self.counter, self.local_rank, tensor_name, self.rank,
                dst_rank, tag, sub_process_group, num_iterations)

    # #4.3 recv
    # 这里其实就是从对应的queue之中，依据张量名字来获取对应的张量。
    # 在运行时 receive_tensors_forward，receive_tensors_backward 函数中，
    # 会调用到 recv 函数，从对应的queue 拿到已经存的张量。比如：
    def recv(self, tensor_name, forward_minibatch_id,
             backward_minibatch_id, backward=False):
        if backward:
            index = (backward_minibatch_id + self.rank_in_stage) % \
                len(self.backward_receive_queues[tensor_name])
            tensor = self.backward_receive_queues[tensor_name][
                index].remove()
            return tensor
        else:
            # 这里会使用到，获取与哪一个rank进行交互
            # 前向时候，需要知道从前一层的哪一个index获取
            index = self.get_messaging_index(sending=False)
            # 然后得到使用哪个张量，从queue之中提取对应的最新张量
            tensor = self.forward_receive_queues[tensor_name][
                index].remove()
            if tensor.dtype == torch.float32:
                tensor = tensor.requires_grad_()
            return tensor

    #4.4 send
    # 这里是把张量放置在对应的queue之中。
    # send_tensors_backward，send_tensors_forward
    # 之中会使用，比如： def send_tensors_backward(self):
    def send_tensors_backward(self):
        def send(self, tensor_name, tensor, forward_minibatch_id,
                 backward_minibatch_id, backward=False):
            if backward:
                # 后向时候，需要知道发送给前一层的哪一个index
                index = self.get_messaging_index(sending=True)
                dst_rank = self.receive_ranks[tensor_name][index]
                self.backward_send_queues[tensor_name][index].add(tensor)
            else:
                index = (forward_minibatch_id + self.rank_in_stage) % \
                    len(self.send_ranks[tensor_name])
                self.forward_send_queues[tensor_name][index].add(tensor)
'''
3.5.2 线程主函数
recv_helper_thread 和 send_helper_thread 分别是 接受助手线程 和 发送助手线程。分别调用 _recv 和 _send 来完成具体业务工作。

需要注意的是使用函数来获取对应的参数。就是使用 recv_helper_thread_args 和 send_helper_thread_args 来获取参数。
'''
def recv_helper_thread(queue, counter, local_rank, tensor_name,
                       src_rank, tag, tensor_shape, dtype,
                       sub_process_group, num_iterations):
    torch.cuda.set_device(local_rank)
    # This method is to be executed from a helper daemon thread.
    for i in range(num_iterations):
        tensor = _recv(
            tensor_name, src_rank, tensor_shape=tensor_shape,
            dtype=dtype, tag=tag,
            sub_process_group=sub_process_group)
        queue.add(tensor)  # 获取到张量之后，放入queue
    counter.decrement()

def send_helper_thread(queue, counter, local_rank, tensor_name,
                       src_rank, dst_rank, tag,
                       sub_process_group, num_iterations):
    torch.cuda.set_device(local_rank)
    # This method is to be executed from a helper daemon thread.
    for i in range(num_iterations):
        tensor = queue.remove()
        # 从queue提取张量，发送出去。
        _send(tensor, tensor_name, src_rank, dst_rank,
              tag=tag,
              sub_process_group=sub_process_group)
    counter.decrement()

# 4.5 _recv
# _recv 参数中，sub_process_group 就是上面代码中构建的。
# 如果在同一个节点上，就使用dist.broadcast，否则使用dist.recv。
#在 recv_helper_thread 之中会调用 _recv。
def _recv(tensor_name, src_rank, tensor_shape=None, dtype=torch.float32,
          tensor=None, tag=None, sub_process_group=None):
    """
    Receives tensor by calling PyTorch's recv() call.

    Tensor will be copied to GPU prior to return.
    """
    assert tag is not None
    if tensor is None:
        assert tensor_shape is not None
        assert dtype is not None
        assert dtype != torch.float16

    if sub_process_group is not None:
        # Receive tensor shape.
        received_tensor_shape = torch.zeros(len(tensor_shape),
                                            dtype=torch.int)
        dist.broadcast(tensor=received_tensor_shape,
                       src=src_rank,
                       group=sub_process_group)
        received_tensor_shape = list(map(lambda x: int(x),
                                         received_tensor_shape))

        # Receive tensor.
        tensor = torch.zeros(received_tensor_shape, dtype=dtype).cuda()
        dist.broadcast(tensor=tensor,
                       src=src_rank,
                       group=sub_process_group)
    else:
        # Receive tensor shape.
        received_tensor_shape = torch.zeros(len(tensor_shape),
                                            dtype=torch.int)
        dist.recv(tensor=received_tensor_shape,
                  src=src_rank,
                  tag=tag)
        received_tensor_shape = list(map(lambda x: int(x),
                                         received_tensor_shape))

        # Receive tensor.
        tensor = torch.zeros(received_tensor_shape, dtype=dtype)
        dist.recv(tensor=tensor,
                  src=src_rank,
                  tag=tag)
        tensor = tensor.cuda()

    assert tensor.is_cuda
    return tensor

# 4.6 _send
# 如果在同一个节点上，就使用dist.broadcast，否则使用dist.send。
# recv_helper_thread 使用 _send获取张量。
def _send(tensor, tensor_name, src_rank, dst_rank, tag, sub_process_group=None):
    """
    Sends tensor by calling PyTorch's send() call.

    If tensor is being sent not via broadcast(), it will
    be first copied to the CPU.
    """
    if sub_process_group is not None:
        assert tensor.is_cuda

        # Send tensor shape.
        tensor_shape = torch.tensor(tensor.shape, dtype=torch.int)
        dist.broadcast(tensor=tensor_shape, src=src_rank,
                      group=sub_process_group)

        # Send tensor.
        contiguous_tensor = tensor.detach().clone()
        dist.broadcast(tensor=contiguous_tensor.contiguous(),
                       src=src_rank,
                       group=sub_process_group)
    else:
        assert tensor.is_cuda
        tensor = tensor.cpu()

        # Send tensor shape.
        tensor_shape = torch.tensor(tensor.shape, dtype=torch.int)
        dist.send(tensor=tensor_shape, dst=dst_rank, tag=tag)

        # Send tensor.
        dist.send(tensor=tensor, dst=dst_rank, tag=tag)
