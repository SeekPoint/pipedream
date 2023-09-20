# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import collections
import itertools
import time
import torch
import torch.distributed as dist

import communication
import runtime_utilities

IMAGE_CLASSIFICATION = "image_classification"
TRANSLATION = "translation"
SPEECH_TO_TEXT = "speech_to_text"

'''
4.2.6 设置module
接下来会处理module相关操作，这里具体会：

    首先使用 ModulesWithDependencies 对模型进行继续处理，把输入，输出配置出来。
    
    然后调用 cuda 把模型和参数移动到 GPU。
    
    如果需要进行处理，针对 fp16 进行转换。
    
关于 ModulesWithDependencies 部分，我们重点说明。
之前我们代码中有如下，就是得倒本stage对应的modules index。

    modules = stage_to_module_map[self.stage] # 这里得到 [3,4]，后续会用到。

stage_to_module_map 就是设置 stage 到 modules 的关系，目的是为了得到本stage所对应的modules。

回忆一下配置文件，本stage（数值为 3）对应的是 index 为 3，4 的两个 module，就是下面的 3 ,3

    module_to_stage_map = {list: 5} [0, 1, 2, 3, 3]
    
接下来要通过如下代码拿到本stage具体的modules，包括每个module的输入，输出。

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()
            if self.fp16:
                import apex.fp16_utils as fp16_utils
                modules[i] = fp16_utils.BN_convert_float(modules[i].half())
                
运行之后如下

    modules = {list: 2} 
     0 = {Stage3} Stage3(\n  (layer5): LSTM(2048, 1024)\n  (layer8): Classifier(\n    (classifier): Linear(in_features=1024, out_features=32320, bias=True)\n  )\n)
     1 = {LabelSmoothing} LabelSmoothing()
     __len__ = {int} 2
     
具体 ModulesWithDependencies 如下：
'''
class ModulesWithDependencies:
    def __init__(self, modules_with_dependencies):
        self._modules = []
        self._all_input_names = []
        self._all_output_names = []
        for (module, input_names, output_names) in modules_with_dependencies:
            self._modules.append(module)
            self._all_input_names.append(input_names)
            self._all_output_names.append(output_names)

    def modules(self):
        return self._modules

    def all_input_names(self):
        return self._all_input_names

    def all_output_names(self):
        return self._all_output_names

    def is_input_tensor(self, tensor_name):
        for module_input_names in self._all_input_names:
            if tensor_name in module_input_names:
                return True
        return False

# 4.1 StageRuntime
# StageRuntime定义如下，可以看到其主要成员变量为在此stage内部进行前向后向操作所需要的元数据，比如：
# 张量，梯度，分布式后端，loss scale，训练数据的张量类型，输出值张量形状等等。
class StageRuntime:
    def __init__(self, model, distributed_backend, fp16, loss_scale,
                 training_tensor_shapes, eval_tensor_shapes,
                 training_tensor_dtypes, inputs_module_destinations,
                 target_tensor_names, configuration_maps, master_addr,
                 rank, local_rank, num_ranks_in_server, verbose_freq,
                 model_type, enable_recompute=False):
        # Metadata needed for forward and backward pass within this stage.
        self.tensors = []
        self.gradients = {}
        self.distributed_backend = distributed_backend
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shapes = eval_tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.model_type = model_type
        self.target_tensor_names = target_tensor_names

        self.initialize(model, inputs_module_destinations, configuration_maps,
                        master_addr, rank, local_rank, num_ranks_in_server)

        self.verbose_freq = verbose_freq
        self.forward_only = False

        self.forward_stats = runtime_utilities.RuntimeStats(forward=True)
        self.backward_stats = runtime_utilities.RuntimeStats(forward=False)

        # Enable recomputation to prevent the need to save activations
        # computed from the forward pass for the backward pass.
        self.enable_recompute = enable_recompute

        # Disable recomputation for the last stage.
        if rank == num_ranks_in_server - 1:
            self.enable_recompute = False

    def initialize(self, model, inputs_module_destinations,
                   configuration_maps, master_addr, rank,
                   local_rank, num_ranks_in_server):
        self.send_ranks = {}
        self.receive_ranks = {}
        self.rank = rank
        self.local_rank = local_rank
        self.stage = None
        self.tensor_tags = {}
        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0
        self.criterion_input_name = str(model[-1][1][0])

        tensor_tag = 1
        # 遍历模型中每一层，每一层的格式是 (_, input_tensors, output_tensors)
        for (_, input_tensors, output_tensors) in model:
            # 遍历输入
            for input_tensor in input_tensors:
                if input_tensor not in self.tensor_tags:
                    self.tensor_tags[input_tensor] = tensor_tag
                    tensor_tag += 1 # 设置 tag
            # 遍历输出
            for output_tensor in output_tensors:
                if output_tensor not in self.tensor_tags:
                    self.tensor_tags[output_tensor] = tensor_tag
                    tensor_tag += 1 # 设置 tag
        for target_tensor_name in sorted(self.target_tensor_names):
            self.tensor_tags[target_tensor_name] = tensor_tag
            tensor_tag += 1 # 设置 tag
        self.tensor_tags["ack"] = tensor_tag
        tensor_tag += 1 # 设置 tag

        '''
        输入是：

            target_tensor_names = {set: 2} {'target_length', 'target'}
             {str} 'target_length'
             {str} 'target'
             __len__ = {int} 2
            
            model = {list: 5} 
              0 = {Stage0} Stage0(\n  (layer4): Embedding(32320, 1024, padding_idx=0)\n  (layer5): EmuBidirLSTM(\n    (bidir): LSTM(1024, 1024, bidirectional=True)\n    (layer1): LSTM(1024, 1024)\n    (layer2): LSTM(1024, 1024)\n  )\n  (layer6): Dropout(p=0.2, inplace=False)\n  (layer7): LS
              1 = {list: 2} ['input0', 'input1']
              2 = {list: 2} ['out2', 'out1']
              __len__ = {int} 3
                                  
             1 = {tuple: 3} 
              0 = {Stage1} Stage1(\n  (layer6): LSTM(1024, 1024)\n  (layer9): Embedding(32320, 1024, padding_idx=0)\n  (layer11): Dropout(p=0.2, inplace=False)\n  (layer12): LSTM(1024, 1024)\n  (layer15): RecurrentAttention(\n    (rnn): LSTM(1024, 1024)\n    (attn): BahdanauAttention(\n      (linear_q): Linear(in_features=1024, out_features=1024, bias=False)\n      (linear_k): Linear(in_features=1024, out_features=1024, bias=False)\n      (dropout): Dropout(p=0, inplace=False)\n    )\n    (dropout): Dropout(p=0, inplace=False)\n  )\n)
              1 = {list: 4} ['out2', 'input1', 'input2', 'out1']
              2 = {list: 2} ['out3', 'out7']
              __len__ = {int} 3
                                  
             2 = {tuple: 3} 
              0 = {Stage2} Stage2(\n  (layer7): Dropout(p=0.2, inplace=False)\n  (layer9): LSTM(2048, 1024)\n  (layer11): Dropout(p=0.2, inplace=False)\n  (layer13): LSTM(2048, 1024)\n  (layer16): Dropout(p=0.2, inplace=False)\n)
              1 = {list: 2} ['out3', 'out7']
              2 = {list: 3} ['out8', 'out9', 'out10']
              __len__ = {int} 3
                                  
             3 = {tuple: 3} 
              0 = {Stage3} Stage3(\n  (layer5): LSTM(2048, 1024)\n  (layer8): Classifier(\n    (classifier): Linear(in_features=1024, out_features=32320, bias=True)\n  )\n)
              1 = {list: 3} ['out8', 'out9', 'out10']
              2 = {list: 1} ['out12']
              __len__ = {int} 3
                                  
             4 = {tuple: 3} 
              0 = {LabelSmoothing} LabelSmoothing()
              1 = {list: 1} ['out12']
              2 = {list: 1} ['loss']
              __len__ = {int} 3
             __len__ = {int} 5
         
        得到：
        
            tensor_tags = {dict: 15} 
             'input0' = {int} 1
             'input1' = {int} 2
             'out2' = {int} 3
             'out1' = {int} 4
             'input2' = {int} 5
             'out3' = {int} 6
             'out7' = {int} 7
             'out8' = {int} 8
             'out9' = {int} 9
             'out10' = {int} 10
             'out12' = {int} 11
             'loss' = {int} 12
             'target' = {int} 13
             'target_length' = {int} 14
             'ack' = {int} 15
             __len__ = {int} 15
        '''

        module_to_stage_map = configuration_maps['module_to_stage_map']
        stage_to_rank_map = configuration_maps['stage_to_rank_map']
        stage_to_depth_map = configuration_maps['stage_to_depth_map']

        if module_to_stage_map is None:
            # If IP addresses not specified, resort to all layers on
            # single machine.
            assert self.rank is None
            self.modules_with_dependencies = ModulesWithDependencies(model)
            self.is_criterion = True
            self.rank_in_stage = 0
            self.num_ranks = 1
            self.num_ranks_in_first_stage = 1
            self.num_ranks_in_previous_stage = 0
            self.num_ranks_in_next_stage = 0
            self.num_stages = 1
            self.num_ranks_in_stage = 1
            self.num_warmup_minibatches = 0
            self.comm_handler = None
        else:
            assert len(module_to_stage_map) == len(model)
            assert self.rank is not None

            '''
            4.2.3 找到自己的配置
            因为在命令行设置了本地的 local_rank 和 rank，所以接下来runtime从配置文件中依据rank找到自己的东西，
            对自己进一步做配置。
            '''
            stage_to_module_map = collections.defaultdict(list)
            for module in range(len(module_to_stage_map)):
                # 这里配置了哪个stage拥有哪些module
                stage_to_module_map[module_to_stage_map[module]].append(module)

            rank_to_stage_map = {}
            for stage in stage_to_rank_map:
                for rank in stage_to_rank_map[stage]:
                    # 配置了哪个 rank 拥有哪些 stage
                    rank_to_stage_map[rank] = stage

            # Now, use this mapping to determine the modules contained in
            # each stage.
            assert 0 <= self.rank < len(rank_to_stage_map)
            self.num_ranks = len(rank_to_stage_map)  # 就是得到了world_size，因为有多少个rank，就是有多少个训练进程，就是world size
            self.num_stages = len(stage_to_module_map)  # 多少个阶段
            self.stage = rank_to_stage_map[self.rank]  # 通过自己的rank得到自己的stage
            self.rank_in_stage = stage_to_rank_map[self.stage].index(self.rank) # 本rank在stage之中排第几个
            self.num_ranks_in_stage = len(stage_to_rank_map[self.stage]) #得到自己stage的rank数目，就是数据并行数目，可以得到本层的数据并行次数
            self.num_ranks_in_first_stage = len(stage_to_rank_map[0])
            self.num_ranks_in_previous_stage = 0
            self.ranks_in_previous_stage = []
            if self.stage > 0:
                self.num_ranks_in_previous_stage = len(
                    stage_to_rank_map[self.stage - 1])
                self.ranks_in_previous_stage = stage_to_rank_map[self.stage - 1]
            self.num_ranks_in_next_stage = 0
            self.ranks_in_next_stage = []
            if self.stage < self.num_stages - 1:
                self.num_ranks_in_next_stage = len(
                    stage_to_rank_map[self.stage + 1])
                self.ranks_in_next_stage = stage_to_rank_map[self.stage + 1]

            modules = stage_to_module_map[self.stage] # 这里得到 [3,4]，后续会用到。

            self.modules_with_dependencies = ModulesWithDependencies(
                [model[module] for module in modules])
            self.is_criterion = self.stage == (self.num_stages - 1)
            if stage_to_depth_map is not None:
                self.num_warmup_minibatches = stage_to_depth_map[
                    str(self.stage)]
            else:
                self.num_warmup_minibatches = self.num_ranks - 1
                for i in range(self.stage):
                    self.num_warmup_minibatches -= len(
                        stage_to_rank_map[i])
                self.num_warmup_minibatches = self.num_warmup_minibatches // \
                    self.num_ranks_in_stage
            '''
变量为：

self = {StageRuntime} 
 backward_minibatch_id = {int} 0
 criterion_input_name = {str} 'out12'
 distributed_backend = {NoneType} None
 eval_tensor_shapes = {dict: 13} {'input0': (50, 128), 'input1': (128,), 'input2': (50, 128), 'target': (6400,), 'target_length': (128,), 'out2': (50, 128, 1024), 'out1': (50, 128, 1024), 'out3': (50, 128, 1024), 'out7': (50, 128, 1024), 'out8': (50, 128, 1024), 'out9': (50, 128, 1024), '
 forward_minibatch_id = {int} 0
 fp16 = {bool} False
 gradients = {dict: 0} {}
 is_criterion = {bool} True
 local_rank = {int} 3
 loss_scale = {int} 1
 model_type = {str} 'translation'
 modules_with_dependencies = {ModulesWithDependencies}
  _all_input_names = {list: 2} [['out8', 'out9', 'out10'], ['out12']]
  _all_output_names = {list: 2} [['out12'], ['loss']]
  _modules = {list: 2} 
   0 = {Stage3} Stage3(\n  (layer5): LSTM(2048, 1024)\n  (layer8): Classifier(\n    (classifier): Linear(in_features=1024, out_features=32320, bias=True)\n  )\n)
   1 = {LabelSmoothing} LabelSmoothing()
   __len__ = {int} 2                                  
 num_ranks = {int} 4
 num_ranks_in_first_stage = {int} 1
 num_ranks_in_next_stage = {int} 0
 num_ranks_in_previous_stage = {int} 1
 num_ranks_in_stage = {int} 1
 num_stages = {int} 4
 num_warmup_minibatches = {int} 0
 rank = {int} 3
 rank_in_stage = {int} 0
 ranks_in_next_stage = {list: 0} []
 ranks_in_previous_stage = {list: 1} [2]
 receive_ranks = {dict: 0} {}
 send_ranks = {dict: 0} {}
 stage = {int} 3
 target = {str} 'python-ce/helpers/pydev/_pydevd_bundle/pydevd_resolver.py", line 178, in _getPyDictionary\n    attr = getattr(var, n)\n  File "../runtime.py", line 295, in target\n    r
 target_tensor_names = {set: 2} {'target', 'target_length'}
 tensor_tags = {dict: 15} {'input0': 1, 'input1': 2, 'out2': 3, 'out1': 4, 'input2': 5, 'out3': 6, 'out7': 7, 'out8': 8, 'out9': 9, 'out10': 10, 'out12': 11, 'loss': 12, 'target': 13, 'target_length': 14, 'ack': 15}
 tensors = {list: 0} []
 training_tensor_dtypes = {dict: 13} {'input0': torch.int64, 'input1': torch.int64, 'input2': torch.int64, 'target': torch.int64, 'target_length': torch.int32, 'out2': torch.float32, 'out1': torch.float32, 'out3': torch.float32, 'out7': torch.float32, 'out8': torch.float32, 'out9': torch.floa
 training_tensor_shapes = {dict: 13} {'input0': (50, 128), 'input1': (128,), 'input2': (50, 128), 'target': (6400,), 'target_length': (128,), 'out2': (50, 128, 1024), 'out1': (50, 128, 1024), 'out3': (50, 128, 1024), 'out7': (50, 128, 1024), 'out8': (50, 128, 1024), 'out9': (50, 128, 1024), '
我们看看几个变量如何使用。

4.2.3.1 num_ranks
首先，看看 num_ranks 如何使用。在后续代码中有使用，比如：

    world_size=self.num_ranks # 依据 num_ranks 得到 world_size
    
    self.num_warmup_minibatches = self.num_ranks - 1 # 依据 num_ranks 得到热身batch数目
    
4.2.3.2 rank_in_stage
其次，再看看 rank_in_stage 如何使用？
前面有

    self.rank_in_stage = stage_to_rank_map[self.stage].index(self.rank)  # 本rank在stage之中排第几个

rank_in_stage 会传递给 Comm 模块。

    self.comm_handler.initialize(
        self.receive_ranks,
        self.send_ranks,
        self.tensor_tags,
        self.target_tensor_names,
        self.training_tensor_dtypes,
        self.rank_in_stage, # 在这里作为参数传入，在函数里面代表本节点，后续会详细介绍
        self.num_ranks_in_stage,
        self.ranks_in_previous_stage,
        self.ranks_in_next_stage)            
            '''

            #4.2.4 设置通信模块
            # 接下来对通信模块进行配置。
            '''
            4.2.5 设置生产者和消费者
接下来对发送，接受的rank进行配置，receive_ranks 和 send_ranks 就是在本阶段各个张量对应的发送，接收目标 rank。

前面已经提到，在 PipeDream开发时候，PyTorch 并没有发布稳定的RPC，所以 PipeDream （2019年发布论文）只能自己实现一套通信逻辑关系，
或者说是分布式计算图。生产者和消费者就是分布式计算图的重要组成部分。

逻辑抽象如下：
    
    遍历模型的 model，假定是 model [i]，注意，这里的 model[i] 是具体的 layer。一个stage可以包括多个layer，
    比如 [layer1, layer 2, layer3]，这个stage又可以在多个rank上进行数据并行，
    比如 rank 1 和 rank 2 都会运行 [layer1, layer 2, layer3]。
    
    对于每个model [i]，遍历model [i] 之后的model，假定是 model [j]。
    
    对于model [i] 的输出进行遍历，假定是 tensor_name。
    
        如果 tensor_name 也在 modle[j] 的输入之中，
        即 tensor_name即在 model[i] 的输出，也在 module[j]的输入，就说明他们之间可以建立联系。
        因为如果一个 张量只有输入或者只有输出，就不需要为这个张量建立任何 通信机制。
        
            如果 model[i] 和 modle[j] 在同一个stage 之中，就是同一个节点 或者 若干节点但是用 DDP 控制，这样就用不到 通信机制。
            
            如果 tensor_name 是 modle[j]的输入，且module[j] 位于本节点上，
            说明本节点的 receive_ranks 就包括 module[j] 的输入（当然也可能包括其他model的输入）。
            
                所以tensor_name的输入rank包括model[j] 对应的rank。
            
            tensor_name 是module[i] 的输出，且 module[i] 位于本节点上，
            说明 本节点的 send_ranks 就包括 module[i] 的输出（当然也可能包括其他model的输出）。
            
                所以tensor_name的输出rank包括 model[i] 对应的rank。
具体代码如下：
            '''

            # To determine where tensors should be sent and received, first
            # determine the "producing" and "consuming" module IDs of each
            # tensor. We then use the corresponding machine ranks to send
            # and receive tensors.
            master_port = 12345
            self.comm_handler = communication.CommunicationHandler(
                master_addr=master_addr,
                master_port=master_port,
                rank=self.rank,
                local_rank=self.local_rank,
                num_ranks_in_server=num_ranks_in_server,
                world_size=self.num_ranks,
                fp16=self.fp16,
                backend=self.distributed_backend)

            # 设置生产者和消费者部分，我们下面会详细分析
            # 设置接受ranks
            for i in range(len(model)): # 遍历层
                for j in range(i+1, len(model)):  # 遍历 i 层之后的若干层
                    for tensor_name in model[i][2]:  # 找出前面层 output 的tensor
                        if tensor_name in model[j][1]:  # 看看 output 在不在input之中  # 看看 tensor_name 在不在input之中，即tensor_name 是不是 modle[j]的输入
                            # tensor_name即在 model[i] 的输出，也在 module[j]的输入，就说明他们之间可以建立联系
                            if module_to_stage_map[i] == \
                                module_to_stage_map[j]: # 两个module在一个node上，不用通信机制
                                continue
                            # For now, assume that each stage is served by only
                            # a single machine.
                            # tensor_name 是 modle[j]的输入，且module[j]位于本节点上，说明可以和本节点的 receive_ranks 建立联系
                            if module_to_stage_map[j] == self.stage:
                                # 所以tensor_name的输入rank包括rank i
                                self.receive_ranks[tensor_name] = \
                                    stage_to_rank_map[module_to_stage_map[i]]
                            # tensor_name 是module[i]的输出，且module[i]位于本节点上，说明可以和本节点的 send_ranks 建立联系
                            if module_to_stage_map[i] == self.stage:
                                # 所以tensor_name的输出rank包括rank j
                                self.send_ranks[tensor_name] = \
                                    stage_to_rank_map[module_to_stage_map[j]]

            # 设置发送ranks
            for model_inputs in inputs_module_destinations.keys():
                destination_stage = module_to_stage_map[
                    inputs_module_destinations[model_inputs]]
                if destination_stage > self.stage:
                    self.send_ranks[model_inputs] = \
                        self.ranks_in_next_stage

                if 0 < self.stage <= destination_stage:
                    self.receive_ranks[model_inputs] = \
                        self.ranks_in_previous_stage

                if destination_stage > 0:
                    if model_inputs not in self.tensor_tags:
                        self.tensor_tags[model_inputs] = tensor_tag
                        tensor_tag += 1
            '''
            得到变量如下：

                num_ranks = {int} 4
                num_ranks_in_first_stage = {int} 1
                num_ranks_in_next_stage = {int} 0
                num_ranks_in_previous_stage = {int} 1
                num_ranks_in_stage = {int} 1
                num_stages = {int} 4
                num_warmup_minibatches = {int} 0
                rank = {int} 3
                rank_in_stage = {int} 0
                ranks_in_next_stage = {list: 0} []
                ranks_in_previous_stage = {list: 1} [2]
                receive_ranks = {dict: 3}  # 这里就是每个tensor对应的接收目标rank
                 'out8' = {list: 1} [2]
                 'out9' = {list: 1} [2]
                 'out10' = {list: 1} [2]
                 __len__ = {int} 3
                send_ranks = {dict: 0} {} # 这里就是每个tensor对应的发送目标rank
                 __len__ = {int} 0
                stage = {int} 3
            '''

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()
            if self.fp16:
                import apex.fp16_utils as fp16_utils
                modules[i] = fp16_utils.BN_convert_float(modules[i].half())

        '''
        4.2.7 设置group
        接下来针对每个stage的并行数目，建立group。
        
        ranks就是每个stage的并行 rank，比如 stage 0 对应的就是 [0, 1, 2]。
        
            {
                "module_to_stage_map": [0, 1, 1],
                "stage_to_rank_map": {"0": [0, 1, 2], "1": [3]} # 每个stage的rank，这里目的是得到并行的机器
            }
            
        遍历stage，针对每个stage，调用new_group() 建立进程组。new_group() 函数使用所有进程的任意子集来创建新的进程组，
        该方法返回一个分组句柄，可作为 collectives （用于特定编程模式中的信息交换）相关分布式函数的 group 参数 。
        
        这里就是最开始问题中提到的：为了数据并行，每个stage都需要建立并且管理自己的进程组。
        '''
        # Initialize all groups in the same order on every worker.
        if stage_to_rank_map is not None:
            groups = []
            for stage in range(self.num_stages):  # 遍历stage
                ranks = stage_to_rank_map[stage]  # 与stage的数据并行对应，比如得到 [0, 1, 2]
                if len(ranks) > 1:  # 与后面的 ddp 相对应
                    groups.append(dist.new_group(ranks=ranks))
                else:
                    groups.append(None)
            group = groups[self.stage]
        else:
            group = None

        '''
        4.2.8 设置数据并行
        最后调用 DistributedDataParallel 进行处理。这里参数 process_group=group 就是前面 “设定group” 返回的。
        
        就是针对每一个group建立一套 DistributedDataParallel。
        '''
        # self.modules_with_dependencies contains a list of PyTorch
        # modules, along with a list of user-defined input and output
        # tensor names. We use our module_executor.ModuleExecutor
        # class to wrap these dependencies, and use run_forward and
        # run_backward methods downstream.
        num_parameters = 0
        for i in range(len(modules)):
            if group is not None:
                if ((i < (len(modules)-1) and self.is_criterion)
                    or not self.is_criterion):
                    num_parameters += \
                        sum(x.size()[0] * x.size()[1]
                            if len(x.size()) > 1 else x.size()[0]
                            for x in modules[i].parameters() if x.size())
                    # 建立分布式数据并行
                    modules[i] = torch.nn.parallel.DistributedDataParallel(
                        modules[i],
                        process_group=group,
                        device_ids=[local_rank],
                        output_device=local_rank)
        if self.num_ranks_in_stage > 1:
            module_size = 4. * num_parameters
            print("Replicating stage: ranks=%d, module_size=%.3f" % (
                self.num_ranks_in_stage, module_size))

        if self.fp16:
            self.master_parameters = []
            self.model_parameters = []
            for i in range(len(modules)):
                import apex.fp16_utils as fp16_utils
                module_parameters, module_master_parameters = \
                    fp16_utils.prep_param_lists(modules[i])
                self.master_parameters.extend(module_master_parameters)
                self.model_parameters.extend(module_parameters)
        else:
            self.master_parameters = list(self.parameters())
            self.model_parameters = None

        #4.2.9 初始化通信函数
        # 最后，针对这个通信模块，进行初始化。
        if self.comm_handler is not None:
            self.comm_handler.initialize(
                self.receive_ranks,
                self.send_ranks,
                self.tensor_tags,
                self.target_tensor_names,
                self.training_tensor_dtypes,
                self.rank_in_stage,
                self.num_ranks_in_stage,
                self.ranks_in_previous_stage,
                self.ranks_in_next_stage)
        ''''
我们还是使用论文中的图片为例来看看运行时引擎初始化之后的结果：
17.png

如果针对本文再细化，则是：

 
                                         +----------------------------------------+
                                         | Stage 2                   StageRuntime |
                                         |                                        |
                                         |           CommunicationHandler         |
                                         |                                        |
                                         |      +----------------------------+    |
                                         |      | +------------------------+ |    |
                                         |      | |Rank 2                  | |    |
                                         |      | |                        | |    |
                                         |      | |                        | |    |
+-----------------------------+          |      | |  Layer 3 +---> Layer 4 | |    |
| Stage 1        StageRuntime |          |      | |                        | |    |       +---------------------------+
|                             |          |      | |                        | |    |       | Stage 3      StageRuntime |
|                             |          |      | +------------------------+ |    |       |                           |
|     CommunicationHandler    |          |      | +------------------------+ |    |       |   CommunicationHandler    |
|                             |          |      | |Rank 3                  | |    |       |                           |
|  +-----------------------+  |          | DDP  | |                        | |    |       | +-----------------------+ |
|  |Rank 1                 |  +---------------->+ |                        | +----------> | | Rank 4                | |
|  |                       |  |          |      | |  Layer 3 +---> Layer 4 | |    |       | |                       | |
|  | Layer 1 +---> Layer 2 |  |          |      | |                        | |    |       | | Layer 5 +---> Layer 6 | |
|  |                       |  |          |      | |                        | |    |       | |                       | |
|  |                       |  |          |      | +------------------------+ |    |       | |                       | |
|  +-----------------------+  |          |      | +------------------------+ |    |       | +-----------------------+ |
|                             |          |      | |Rank 4                  | |    |       |                           |
|                             |          |      | |                        | |    |       |                           |
+-----------------------------+          |      | |                        | |    |       +---------------------------+
                                         |      | |  Layer 3 +---> Layer 4 | |    |
                                         |      | |                        | |    |
                                         |      | |                        | |    |
                                         |      | +------------------------+ |    |
                                         |      +----------------------------+    |
                                         +----------------------------------------+



手机如下：  18.png


        '''

    @property
    def target(self):
        return self.tensors[-1]["target"]

    def modules(self):
        return self.modules_with_dependencies.modules()

    def parameters(self):
        parameter_iterators = []
        for module in self.modules_with_dependencies.modules():
            parameter_iterators.append(module.parameters())
        return itertools.chain(*parameter_iterators)

    def state_dict(self):
        state_dict = collections.OrderedDict()
        for i, module in enumerate(self.modules_with_dependencies.modules()):
            state_dict["module%d" % i] = module.state_dict()
        if self.fp16:
            state_dict["master_parameters"] = self.master_parameters
        return state_dict

    def load_state_dict(self, state_dict):
        for i, module in enumerate(self.modules_with_dependencies.modules()):
            module.load_state_dict(state_dict["module%d" % i])
        if self.fp16:
            saved_master_parameters = state_dict["master_parameters"]
            for master_parameter, saved_master_parameter in zip(
                self.master_parameters, saved_master_parameters):
                master_parameter.data.copy_(saved_master_parameter.data)

    def cuda(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()

    def zero_grad(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].zero_grad()

    def train(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.training_tensor_shapes
        self.forward_only = False

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            self.comm_handler.start_helper_threads(
                num_iterations, forward_only=False)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].train()

    def eval(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.eval_tensor_shapes
        self.tensor_shapes["ack"] = (1,)
        self.forward_only = True

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            self.comm_handler.start_helper_threads(
                num_iterations, forward_only=True)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].eval()

    def set_loader(self, loader):
        if loader is not None:
            self.loader_iter = iter(loader)
        else:
            self.loader_iter = None

    '''
    4.3 功能函数
    我们这里只是介绍基础功能函数。另外有几个业务功能函数，比如 run_forward 会在1F1B文章中一并介绍。
    
    以下这几个功能函数都是调用通讯模块完成功能。
    
    4.3.1 receive_tensors_forward
    receive_tensors_forward 就是在前向传播中，从前面层获取张量。
    
    前向传播中，张量记录在本实例的 self.tensors 之中。
    '''
    def receive_tensors_forward(self):
        if self.forward_only and len(self.tensors) > 0:
            self.tensors.pop(0) # 弹出以前
        self.tensors.append({})
        if self.loader_iter is not None:  # 前向传播第一层，需要加载数据
            input = next(self.loader_iter)  # 加载新的
            if self.model_type == TRANSLATION:
                (input, target) = input
                src, src_length = input
                tgt, tgt_length = target

                self.tensors[-1]["input0"] = src.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = torch.LongTensor(src_length).cuda(
                    non_blocking=True)
                self.tensors[-1]["input2"] = tgt[:-1].cuda(non_blocking=True)
                self.tensors[-1]["target"] = tgt[1:].cuda().contiguous().view(-1)
                self.tensors[-1]["target_length"] = \
                    torch.tensor([int(sum(torch.LongTensor(tgt_length) - 1))],
                                 dtype=torch.int).cuda()
            elif self.model_type == IMAGE_CLASSIFICATION:
                (input, target) = input
                if self.fp16:
                    input = input.half()
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
            elif self.model_type == SPEECH_TO_TEXT:
                input, target, input_percentages, target_sizes = input
                input_sizes = input_percentages.mul_(int(input.size(3))).int()
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = input_sizes.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
                self.tensors[-1]["target_length"] = target_sizes.cuda(
                    non_blocking=True)
        else:
            # Receive all required tensors from upstream machines.
            for input_name in self.receive_ranks: # 遍历本stage对应的接受rank,从前面层获取
                if input_name == "ack":
                    continue

                self.tensors[-1][input_name] = \
                    self.comm_handler.recv(
                        input_name,
                        forward_minibatch_id=self.forward_minibatch_id,
                        backward_minibatch_id=self.backward_minibatch_id,
                        backward=False)

                self.forward_stats.stats['receive_tensors_size'] += \
                    (self.tensors[-1][input_name].element_size() *
                     self.tensors[-1][input_name].nelement())

            # Used to track where to receive forward from.
            self.comm_handler.increment_messaging_index(
                sending=False)

    '''
    4.3.2 send_tensors_forward
    send_tensors_forward就是在前向传播中，向后面层发送张量。
    '''
    def send_tensors_forward(self):
        # Send all required tensors downstream.
        for output_name in self.send_ranks:  # 遍历本stage对应的发送rank，进行发送
            if output_name == "ack":
                continue

            self.comm_handler.send(
                output_name,
                self.tensors[-1][output_name],
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=False)

            self.forward_stats.stats['send_tensors_size'] += \
                (self.tensors[-1][output_name].element_size() *
                 self.tensors[-1][output_name].nelement())

    '''
    4.3.3 receive_tensors_backward
    后向传播中，梯度保存在 self.gradients。
    
    receive_tensors_backward 就是在后向传播中，从前面层获取张量。
    
    注意，这里对应的是self.send_ranks，就是前向过程中的发送rank，它们在反向过程中就是接受rank
    
    '''
    def receive_tensors_backward(self):
        # Receive all required gradients from downstream
        # machines.
        for output_name in self.send_ranks: # 遍历本stage对应的发送rank（前向），进行接受
             if output_name in self.target_tensor_names:
                continue

             # 获取梯度
             self.gradients[output_name] = \
                self.comm_handler.recv(  # 这里使用了  def recv(self
                    output_name,
                    forward_minibatch_id=self.forward_minibatch_id,
                    backward_minibatch_id=self.backward_minibatch_id,
                    backward=True)

             self.backward_stats.stats['receive_tensors_size'] += \
                 (self.gradients[output_name].element_size() *
                  self.gradients[output_name].nelement())

    '''
    4.3.4 send_tensors_backward
    后向传播中，梯度保存在 self.gradients。
    
    send_tensors_forward就是在后向传播中，向后面层发送梯度张量。
    
    注意，这里对应的是self.receive_ranks，就是前向过程中的接受rank，它们在反向过程中就是发送rank
    
    '''
    def send_tensors_backward(self):
        # Send all required gradients upstream.
        for input_name in self.receive_ranks: # 遍历本stage对应的接受rank，进行发送
            if input_name in self.target_tensor_names:
                continue

            self.comm_handler.send(
                input_name,
                self.gradients[input_name],
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)

            self.backward_stats.stats['send_tensors_size'] += \
                (self.gradients[input_name].element_size() *
                 self.gradients[input_name].nelement())

        if self.num_ranks_in_previous_stage > 0:
            # Used to track where to send tensors in the
            # backward pass.
            self.comm_handler.increment_messaging_index(
                sending=True)

    def run_forward(self, recompute_step=False):
        """Run forward pass.
        """
        # Receive tensors from previous worker.
        self.receive_tensors_forward()
        tensors = self.tensors[-1]

        # Run forward pass.
        self._run_forward(tensors)

        # Send tensors forward.
        self.send_tensors_forward()
        if self.verbose_freq > 0 and self.forward_minibatch_id % self.verbose_freq == 0:
            self.forward_stats.print_stats()
        self.forward_stats.reset_stats()
        self.forward_minibatch_id += 1

    def _run_forward(self, tensors):
        # Perform forward pass through model (self.modules_with_dependencies already
        # has modules in topological order).
        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()
        for i, (module, input_names, output_names) in \
                enumerate(zip(modules, all_input_names, all_output_names)):
            if i == (len(modules) - 1) and self.is_criterion:
                # If layer is criterion (loss).
                if self.model_type == SPEECH_TO_TEXT:
                    output = tensors["output"].transpose(0, 1).float()
                    output_sizes = tensors["output_sizes"].cpu()
                    target = tensors["target"].cpu()
                    target_sizes = tensors["target_length"].cpu()
                    input0_size = tensors["input0_size"].cpu()
                    module_outputs = [module(output, target, output_sizes, target_sizes) / input0_size[0]]
                else:
                    module_outputs = [module(tensors[input_name],
                                             tensors["target"])
                                      for input_name in input_names]
                    module_outputs = [sum(module_outputs)]
            else:
                # If layer is non-criterion.
                module_outputs = module(*[tensors[input_name]
                                          for input_name in input_names])
                if not isinstance(module_outputs, tuple):
                    module_outputs = (module_outputs,)
                module_outputs = list(module_outputs)

            for (output_name, module_output) in zip(output_names, module_outputs):
                tensors[output_name] = module_output

        self.output = tensors[input_names[0]]
        if self.is_criterion and self.model_type == TRANSLATION:
            loss_per_batch = tensors[output_names[0]] * tensors[self.criterion_input_name].size(1)
            loss_per_token = loss_per_batch / tensors["target_length"][0].item()
            self.loss = loss_per_token
        elif self.is_criterion:
            self.loss = tensors[output_names[0]]
        else:
            self.loss = 1

    def run_backward(self):
        # Receive input gradients needed for backward pass.
        self.receive_tensors_backward()
        # Backward pass through modules in reverse order.
        inputs = {}
        outputs = {}
        input_gradients = {}
        output_gradients = {}

        # Get input and output names spanning all modules in this stage.
        all_input_names_set = set()
        all_output_names_set = set()

        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()

        for (input_names, output_names) in zip(all_input_names, all_output_names):
            for input_name in input_names:
                all_input_names_set.add(input_name)
            for output_name in output_names:
                all_output_names_set.add(output_name)

        tensors = self.tensors.pop(0)
        # Set inputs, outputs, and output_gradients.
        # Only set outputs/output_gradients for tensors that are not inputs of
        # other modules in this stage.
        # Similarly, only set inputs for tensors that are not outputs of other
        # modules in this stage.
        for (module, input_names, output_names) in \
            zip(reversed(modules), reversed(all_input_names), reversed(all_output_names)):
            for output_name in output_names:
                if output_name not in all_input_names_set:
                    if output_name not in self.gradients:
                        output_gradients[output_name] = None
                    else:
                        output_gradients[output_name] = self.gradients[output_name]
                    if tensors[output_name].requires_grad:
                        outputs[output_name] = tensors[output_name]
            for input_name in input_names:
                if input_name not in all_output_names_set:
                    inputs[input_name] = tensors[input_name]

        # Hook to record input gradients.
        def hook_wrapper(input_name):
            def hook(input_gradient):
                input_gradients[input_name] = input_gradient
            return hook

        for input_name in inputs:
            if input_name != "input0" and input_name != "input1" and input_name != "input2" \
                    and inputs[input_name].requires_grad:
                inputs[input_name].register_hook(hook_wrapper(input_name))

        if "loss" in outputs:
            outputs["loss"] *= self.loss_scale

        # Perform backward pass.
        torch.autograd.backward(tuple([outputs[output_name] for output_name in outputs]),
                                grad_tensors=tuple([output_gradients[output_name]
                                                    for output_name in outputs]))

        # Input tensors don't need gradients.
        for input_name in inputs:
            if not inputs[input_name].requires_grad:
                self.gradients[input_name] = inputs[input_name]
                continue

            if input_name != "input0" and input_name != "input1" and input_name != "input2" and input_name != "input":
                self.gradients[input_name] = input_gradients[input_name]

        # Send output gradients.
        self.send_tensors_backward()
        if self.verbose_freq > 0 and self.backward_minibatch_id % self.verbose_freq == 0:
            self.backward_stats.print_stats()
        self.backward_stats.reset_stats()
        self.backward_minibatch_id += 1

    def num_tokens(self):
        return self.tensors[-1]["target_length"][0].item()

    '''
    4.3.5 run_ack
    run_ack就是在传播中，给前面层，后面层回应一个确认。
    
    至此，运行时引擎我们介绍完毕其静态信息和初始化，下一篇我们介绍通信模块。
    '''
    def run_ack(self):
        # No need for ack if running on a single worker.
        if self.rank is None:
            return

        # Receive ack from next stage. Send ack to previous stage.
        if self.stage < (self.num_stages-1):
            self.comm_handler.recv(
                "ack",
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)
        if self.stage > 0:
            self.comm_handler.send(
                "ack",
                torch.zeros(self.tensor_shapes["ack"],
                            dtype=torch.int64).cuda(),
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)

            # Used to track where to receive forward from.
            self.comm_handler.increment_messaging_index(sending=True)

        self.backward_minibatch_id += 1

    def wait(self):
        if self.comm_handler is not None:
            self.comm_handler.wait()

    def num_iterations(self, loader_size):
        """ Determines number of iterations for this stage

        TODO: don't currently support uneven configurations.
        """
        if self.stage == 0 or self.stage is None:
            return loader_size

        num_iterations = loader_size * self.num_ranks_in_first_stage
        assert num_iterations % self.num_ranks_in_stage == 0
        num_iterations = num_iterations // self.num_ranks_in_stage

        return num_iterations

    def get_adjusted_learning_rate(self, base_lr):
        if self.stage == 0:
            return base_lr

        adjusted_lr = float(base_lr) * float(self.num_ranks_in_stage) \
                      / float(self.num_ranks_in_first_stage)

        return adjusted_lr
