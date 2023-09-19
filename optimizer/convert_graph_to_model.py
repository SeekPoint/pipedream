# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import re
import subprocess
import sys

sys.path.append("..")
import graph

'''
2.2 支撑逻辑
首先我们看看两个数组。
    declaration_whitelist 是一个白名单，如果某节点在这个白名单之中，则不需要在 init 函数之中进行处理。
    declaration_specialcase 数组包括一些特殊定义，如果某节点是需要特殊定义的，就进行特殊转换，比如import，层定义等等。
'''
declaration_whitelist = [
    "hidden",
    "__getitem__",
    "Add",
    "Mul",
    "Concat",
    "Input",
    "Size",
    "View",
    "Transpose",
    "self.get_seq_lens"
]

declaration_specialcase = [
    "EmuBidirLSTM",
    "RecurrentAttention",
    "Classifier",
    "MaskConv",
    "ResizeInput",
    "InferenceBatchSoftmax",
    "BatchRNN",
    "SequenceWise"
]

def get_output_tuple_str(outputs):
    if len(outputs) == 1:
        return outputs[0]
    return "(%s)" % ", ".join(outputs)

def get_tensor_names_list(names):
    return [names[node_id] for node_id in sorted(names.keys())]

# 其次我们看看get_input_names方法。
# get_input_names 函数遍历graph的节点，找到这个子图的输入。
def get_input_names(graph, full_graph, check_stages=True):
    # Figure out the inputs to this sub-graph, which are the predecessors of
    # nodes in the sub-graph not in the sub-graph.
    # input_names is a dict mapping each predecessor's node_id to assigned
    # variable name.
    nodes = graph.nodes
    input_names = {}
    counter = 0
    for node_id in nodes:
        if (node_id in full_graph.in_edges and
            len(full_graph.in_edges[node_id]) > 0):
            for in_node in full_graph.in_edges[node_id]:
                if in_node.stage_id != nodes[node_id].stage_id and check_stages:
                    # Skip hidden inputs.
                    if full_graph.nodes[in_node.node_id].node_desc.startswith("hidden"):
                        continue
                    input_names[in_node.node_id] = "input%d" % counter
                    counter += 1
        else:
            if graph.nodes[node_id].node_desc.startswith("Input"):
                input_names[node_id] = "input%d" % counter
                counter += 1
    return input_names

def get_output_names(graph, full_graph, counter):
    # Figure out the outputs of this sub-graph, which are the nodes in the
    # sub-graph with edges out of the sub-graph.
    nodes = graph.nodes
    output_names = {}
    for node_id in nodes:
        if (node_id in full_graph.edges and
            len(full_graph.edges[node_id]) > 0):
            for out_node in full_graph.edges[node_id]:
                if out_node.stage_id != nodes[node_id].stage_id:
                    if full_graph.nodes[node_id].node_desc.startswith("hidden"):
                        continue
                    output_names[node_id] = "out%d" % counter
                    counter += 1
        else:
            output_names[node_id] = "out%d" % counter
            counter += 1
    return output_names, counter
'''
3.2.1 转换Module
转换Module逻辑如下：

    get_input_names 函数遍历graph，找到这个子图的输入。
    
    如果节点在输入中，则构建forward函数定义部分，为后续生成代码做准备。
    得到 function_definition 类似为 ['out0 = input0.clone()', 'out1 = input1.clone()']。
    
    遍历图中的节点，做如下操作，基本就是依据节点性质，生成各种python语句：
    
        得到每一层的相关信息，比如名字，输出，是不是inplace操作。
        
        如果某节点是需要特殊定义的，就进行特殊转换，比如import，层定义等等
        
        归并import语句
        
        如果节点描述不在声明白名单之中，则记录，后续会在init方法生成时候对这些节点生成构建语句。
        
        得到节点入边
        
        如果节点在内置运算符之中，直接构造python语句。
        
        如果不是内置运算，就直接设置，比如 'out2 = self.layer2(out0, out1)'。
        
    确保模块输出是按照原始模型的顺序输出。
    
    如果需要初始化权重，则做处理。
    
    应用模版文件生成模型，就是把前面生成的各种python语句填充到模版文件之中。
    
    写入模型python文件。
    
下面代码注解中，有部分运行时变量的打印。
'''


def convert_subgraph_to_module(graph, full_graph, num_subgraphs, module_name, initialize_weights,
                               model_template_filename, output_filename):
    model_template = open(model_template_filename, 'r').read()
    nodes = graph.topological_sort()
    import_statements = []
    module_methods = []

    counter = 0
    layer_names = {}
    layer_names_and_declarations = []
    function_definition = []
    # get_input_names 函数遍历graph，找到这个子图的输入
    input_names = get_input_names(graph, full_graph)
    num_inputs = len(input_names)
    output_names = input_names.copy()
    sources = graph.sources()

    # Now, generate expressions for each node.
    # Iterate through nodes in topological order, and add output_name mappings for
    # each expression. Use this output_name mapping when generating expressions
    # in the model's implementation file.
    # TODO: Make sure that nodes with multiple inputs have the inputs in the
    # right order (even though this probably does not matter in practice).

    # 构建forward函数定义部分，为后续生成代码做准备
    for node_id in input_names:
        output_name = "out%d" % counter
        function_definition.append("%s = %s.clone()" % (output_name,
                                                        input_names[node_id]))
        output_names[node_id] = output_name
        counter += 1
    # 得到 function_definition 为 ['out0 = input0.clone()', 'out1 = input1.clone()']

    # 遍历图中的节点
    for node in nodes:
        # 层相关信息
        layer_call = None
        layer_name = "self.layer%d" % counter
        output_name = "out%d" % counter
        layer_declaration = "torch.nn.%s" % (
            node.node_desc.replace("inplace", "inplace=True"))
        layer_names[node.node_id] = layer_name
        if node.node_id not in output_names:
            output_names[node.node_id] = output_name

        # Skip layers that don't need a declaration (example: '+=').
        for declaration in declaration_specialcase:
            # 如果某节点是需要特殊定义的，就进行特殊转换，比如import，层定义等等
            if node.node_desc.startswith(declaration):
                found = True
                if declaration == "EmuBidirLSTM":
                    m = re.search(r'.*LSTM\((\d+), (\d+)\).*', node.node_desc)
                    input_size = int(m.group(1))
                    hidden_size = int(m.group(2))
                    layer_declaration = "EmuBidirLSTM(%d, %d)" % (input_size, hidden_size)
                    import_statements.append("from seq2seq.models.encoder import EmuBidirLSTM")
                    # 这里得到 import_statements 为 ['from seq2seq.models.encoder import EmuBidirLSTM']，layer_declaration 为 'EmuBidirLSTM(1024, 1024)'

                elif declaration == "RecurrentAttention":
                    m = re.search(r'.*LSTM\((\d+), (\d+)\).*', node.node_desc)
                    input_size = int(m.group(1))
                    hidden_size = int(m.group(2))
                    m = re.search(r'.*in_features=(\d+), out_features=(\d+).*', node.node_desc)
                    context_size = int(m.group(1))
                    layer_declaration = "RecurrentAttention(%d, %d, %d)" % (input_size, hidden_size, context_size)
                    import_statements.append("from seq2seq.models.decoder import RecurrentAttention")
                elif declaration == "Classifier":
                    m = re.search(r'.*in_features=(\d+), out_features=(\d+).*', node.node_desc)
                    in_features = int(m.group(1))
                    out_features = int(m.group(2))
                    layer_declaration = "Classifier(%d, %d)" % (in_features, out_features)
                    import_statements.append("from seq2seq.models.decoder import Classifier")
                elif declaration == "MaskConv":
                    node_desc = node.node_desc
                    modules = node_desc.split("    ")[1:-1]
                    module_declarations = []
                    for module in modules:
                        module_declaration = "torch.nn." + module.split(": ")[1].replace("inplace", "inplace=True")
                        module_declarations.append(module_declaration)
                    layer_declaration = "MaskConv(torch.nn.Sequential(%s))" % ",\n            ".join(
                        module_declarations)
                    import_statements.append("from model import MaskConv")
                    module_methods.append("""def get_seq_lens(self, input_length):
        seq_len = input_length
        for m in %s.modules():
            if type(m) == torch.nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()""" % layer_name)
                elif declaration == "BatchRNN":
                    if "batch_norm" in node.node_desc:
                        batch_norm = True
                    else:
                        batch_norm = False
                    if "LSTM" in node.node_desc:
                        rnn_type = "torch.nn.LSTM"
                        m = re.search(r'LSTM\((\d+), (\d+), bidirectional=([a-zA-Z]+)\)', node.node_desc)
                        input_size = int(m.group(1))
                        hidden_size = int(m.group(2))
                        bidirectional = m.group(3)
                    elif "GRU" in node.node_desc:
                        rnn_type = "torch.nn.GRU"
                        m = re.search(r'GRU\((\d+), (\d+), bidirectional=([a-zA-Z]+)\)', node.node_desc)
                        input_size = int(m.group(1))
                        hidden_size = int(m.group(2))
                        bidirectional = m.group(3)
                    else:
                        # TODO: Do something else?
                        pass
                    # TODO: Pass remaining arguments.
                    # TODO: Get hidden and input size.
                    layer_declaration = "BatchRNN(%d, %d, rnn_type=%s, batch_norm=%s, bidirectional=%s)" % (
                        input_size, hidden_size, rnn_type, batch_norm, bidirectional)
                    import_statements.append("from model import BatchRNN")
                elif declaration == "ResizeInput":
                    layer_declaration = "ResizeInput()"
                    import_statements.append("from model import ResizeInput")
                elif declaration == "SequenceWise":
                    node_desc = node.node_desc
                    modules = node_desc[:-2].split("  ")[1:]
                    module_declarations = []
                    for module in modules:
                        module_declaration = "torch.nn." + module.split(": ")[1].replace("inplace", "inplace=True")
                        module_declarations.append(module_declaration)
                    layer_declaration = "SequenceWise(torch.nn.Sequential(%s))" % ",\n            ".join(
                        module_declarations)
                    import_statements.append("from model import SequenceWise")
                elif declaration == "InferenceBatchSoftmax":
                    layer_declaration = "InferenceBatchSoftmax()"
                    import_statements.append("from model import InferenceBatchSoftmax")
                break

        # 归并import语句
        import_statements = list(set(import_statements))
        # 如果节点描述不在声明白名单之中，则处理
        found = False
        for declaration in declaration_whitelist:
            if node.node_desc.startswith(declaration):
                found = True
        if not found:
            layer_names_and_declarations.append((layer_name, layer_declaration))

        # 得到节点入边
        if node.node_id in full_graph.in_edges:
            in_edges = full_graph.in_edges[node.node_id]
        else:
            in_edges = []
        if len(in_edges) == 0 and node.node_desc.startswith("Input"):
            pass  # Don't need to do anything for this case.
        else:
            # 看看节点是否在内置运算符之中
            # node_desc 为 'EmuBidirLSTM(  (bidir): LSTM(1024, 1024, bidirectional=True)  (layer1): LSTM(1024, 1024)  (layer2): LSTM(1024, 1024))'

            if node.node_desc.startswith("Size"):
                assert (len(in_edges) == 1)
                m = re.search(r'Size\((-?\d+)\)', node.node_desc)
                idx = int(m.group(1))
                layer_call = "%s = %s.size(%d)" % (output_name,
                                                   output_names[in_edges[0].node_id],
                                                   idx)
            elif node.node_desc.startswith("View"):
                size_node_ids = []
                input_node_id = None
                for i in range(len(in_edges)):
                    if in_edges[i].node_desc.startswith("Size"):
                        size_node_id = in_edges[i].node_id
                        size_node_ids.append(size_node_id)
                    else:
                        input_node_id = in_edges[i].node_id
                m = re.search(r'View\((-?\d+)\)', node.node_desc)
                if m is None:
                    size_output_names = [output_names[size_node_id] for size_node_id in size_node_ids]
                    layer_call = "%s = %s.view(%s)" % (output_name,
                                                       output_names[input_node_id],
                                                       ", ".join(size_output_names))
                else:
                    size = int(m.group(1))
                    layer_call = "%s = %s.view(%s, %d)" % (output_name,
                                                           output_names[input_node_id],
                                                           output_names[size_node_id],
                                                           size)
            elif node.node_desc.startswith("__getitem__"):
                assert (len(in_edges) == 1)
                m = re.search(r'__getitem__\((\d+)\)', node.node_desc)
                idx = int(m.group(1))
                if "hidden" in in_edges[0].node_desc:
                    layer_call = "%s = None" % output_name
                else:
                    layer_call = "%s = %s[%d]" % (output_name,
                                                  output_names[in_edges[0].node_id],
                                                  idx)
            elif node.node_desc.startswith("Add"):
                assert (len(in_edges) == 2)
                node1 = in_edges[0]
                node2 = in_edges[1]
                if len(full_graph.edges[node1.node_id]) > 1:
                    tmp = node1
                    node1 = node2
                    node2 = tmp
                layer_call = "%s = %s + %s" % (output_names[node1.node_id],
                                               output_names[node1.node_id],
                                               output_names[node2.node_id])
                output_names[node.node_id] = output_names[node1.node_id]
            elif node.node_desc.startswith("Mul"):
                assert (len(in_edges) == 2)
                node1 = in_edges[0]
                node2 = in_edges[1]
                if len(full_graph.edges[node1.node_id]) > 1:
                    tmp = node1
                    node1 = node2
                    node2 = tmp
                layer_call = "%s = %s * %s" % (output_names[node1.node_id],
                                               output_names[node1.node_id],
                                               output_names[node2.node_id])
                output_names[node.node_id] = output_names[node1.node_id]
            elif node.node_desc.startswith("Concat"):
                m = re.search(r'Concat\((-?\d+)\)', node.node_desc)
                dim = int(m.group(1))
                layer_call = "%s = torch.cat([%s], %d)" % (
                    output_name,
                    ", ".join([output_names[in_node.node_id]
                               for in_node in in_edges]), dim)
            elif node.node_desc.startswith("Transpose"):
                m = re.search(r'Transpose\((.+)\)', node.node_desc)
                args = m.group(1)
                assert (len(in_edges) == 1)
                node1 = in_edges[0]
                layer_call = "%s = %s.transpose(%s)" % (output_name, output_names[node1.node_id],
                                                        args)
            elif node.node_desc.startswith("hidden"):
                pass
            elif node.node_desc == "self.get_seq_lens":
                assert (len(in_edges) == 1)
                in_node = in_edges[0]
                layer_call = "%s = %s(%s)" % (output_name, node.node_desc, output_names[in_node.node_id])
            else:
                # 如果不是内置运算，就直接设置，这里为 'out2 = self.layer2(out0, out1)'
                layer_call = "%s = %s(%s)" % (output_name, layer_name,
                                              ", ".join([output_names[in_node.node_id]
                                                         for in_node in in_edges]))
        if layer_call is not None:
            function_definition.append(layer_call)
        counter += 1

    # Ensure that outputs of a module are returned in the same order as
    # the original model implementation.
    # TODO: This might not work as intended for sub-graphs.
    # 确保模块输出是按照原始模型的顺序输出
    full_graph.populate_depths()
    graph_output_names, _ = get_output_names(graph, full_graph, 0)
    for key in graph_output_names:
        graph_output_names[key] = output_names[key]
    output_names_list = get_tensor_names_list(graph_output_names)
    num_outputs = len(output_names_list)
    function_definition.append("return %s" %
                               get_output_tuple_str(output_names_list))
    # function_definition 是 ['out0 = input0.clone()', 'out1 = input1.clone()', 'out2 = self.layer2(out0, out1)', 'out3 = self.layer3(out2)', 'return out3']

    # Layer declarations are added to the constructor of the module.
    # Function definitions are added to the `forward()' method of the
    # module.
    layer_declarations_str = "\n        ".join([
        "%s = %s" % (x[0], x[1]) for x in layer_names_and_declarations])

    # 如果需要初始化权重，则做处理
    if initialize_weights:
        layer_declarations_str += "\n        self._initialize_weights()"
        module_methods.append("""def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)""")
    function_definition_str = "\n        ".join(function_definition)
    # function_definition_str 为 "\n        ".join(function_definition) ['out0 = input0.clone()', 'out1 = input1.clone()', 'out2 = self.layer2(out0, out1)', 'out3 = self.layer3(out2)', 'return out3']
    input_names_list = get_tensor_names_list(input_names)
    input_names = ", ".join(input_names_list)
    # input_names 为 'input1, input0'
    # 应用模版文件生成模型
    model = model_template % {"layer_declarations": layer_declarations_str,
                              "function_definition": function_definition_str,
                              "module_name": module_name,
                              "inputs": input_names,
                              "import_statements": "\n".join(import_statements),
                              "module_methods": "\n\n".join(module_methods)}
    '''
    3.2.3 生成文件
    前面代码之中，如下语句会生成若干模型文件，每一个subgraph会生成一个python文件。
    
    得到的生成模型文件我们举出两个文件如下：

    import torch
    
    class Stage0(torch.nn.Module):
        def __init__(self):
            super(Stage0, self).__init__()
            self.layer6 = torch.nn.Embedding(32320, 1024, padding_idx=0)
    
        def forward(self, input0, input1, input2):
            out0 = input0.clone()
            out1 = input1.clone()
            out2 = input2.clone()
            out6 = self.layer6(out0)
            return (out1, out2, out6)
            
    再比如：

    import torch
    from seq2seq.models.encoder import EmuBidirLSTM
    
    class Stage1(torch.nn.Module):
        def __init__(self):
            super(Stage1, self).__init__()
            self.layer2 = EmuBidirLSTM(1024, 1024)
            self.layer3 = torch.nn.Dropout(p=0.2)
      
        def forward(self, input1, input0):
            out0 = input0.clone()
            out1 = input1.clone()
            out2 = self.layer2(out0, out1)
            out3 = self.layer3(out2)
            return out3
    '''
    # 写入模型python文件
    with open(output_filename, 'w') as f:
        f.write(model)
    return num_inputs, num_outputs

'''
3.3.2 融合模型
fuse_subgraphs_to_module 生成一个gnmt.py文件，具体逻辑如下：
    加载模版。
    归并模块名称。
    处理函数定义和层定义。
    遍历子图，构建输出和输入。
    添加输出信息。
    添加import信息。
    应用模版文件。
    输出文件。
源码如下：
'''
def fuse_subgraphs_to_module(graph, subgraphs, model_name, initialize_weights,
                             model_template_filename, output_filename):
    # 加载模版
    model_template = open(model_template_filename, 'r').read()

    # PyTorch modules are the names given to the generated stages (which are
    # of type torch.nn.Module).
    # Python modules are the names given to the filenames containing these
    # generated torch.nn.Modules.
    # 归并模块名称
    pytorch_modules = []
    python_modules = []
    for i in range(len(subgraphs)):
        pytorch_modules.append("Stage%d" % i)
        python_modules.append("stage%d" % i)
    '''
python_modules = {list: 10} ['stage0', 'stage1', 'stage2', 'stage3',
 'stage4', 'stage5', 'stage6', 'stage7', 'stage8', 'stage9']
 
pytorch_modules = {list: 10} ['Stage0', 'Stage1', 'Stage2', 'Stage3', 
'Stage4', 'Stage5', 'Stage6', 'Stage7', 'Stage8', 'Stage9']    
    '''

    # 处理函数定义和层定义
    layer_declarations = []
    function_definition = []
    for i, pytorch_module in enumerate(pytorch_modules):
        layer_declarations.append("self.stage%d = %s()" % (
            i, pytorch_module))
    if initialize_weights:
        layer_declarations.append("self._initialize_weights()")
    '''
		# function_definition = {list: 0} []
		# layer_declarations = {list: 10} ['self.stage0 = Stage0()', 'self.stage1 = Stage1()', 
		'self.stage2 = Stage2()', 'self.stage3 = Stage3()', 'self.stage4 = Stage4()', 
		'self.stage5 = Stage5()', 'self.stage6 = Stage6()', 'self.stage7 = Stage7()', 
		'self.stage8 = Stage8()', 'self.stage9 = Stage9    
    '''

    output_counter = 0
    output_names = {}
    graph_input_names = get_input_names(graph, graph, check_stages=False)
    for key in graph_input_names:
        output_names[key] = graph_input_names[key]
    subgraph_inputs = []
    subgraph_outputs = []

    # 遍历子图，构建输出和输入
    for i, subgraph in enumerate(subgraphs):
        subgraph_input_names = get_input_names(subgraph, graph)
        subgraph_output_names, output_counter = get_output_names(
            subgraph, graph, output_counter)
        for key in subgraph_input_names:
            subgraph_input_names[key] = output_names[key]
        for key in subgraph_output_names:
            output_names[key] = subgraph_output_names[key]

        function_definition.append("%s = self.stage%d(%s)" % (
            get_output_tuple_str(get_tensor_names_list(subgraph_output_names)),
            i, ", ".join(get_tensor_names_list(subgraph_input_names))))
        subgraph_inputs.append(get_tensor_names_list(subgraph_input_names))
        subgraph_outputs.append(get_tensor_names_list(subgraph_output_names))

    # 添加输出信息
    function_definition.append("return %s" %
        get_output_tuple_str(get_tensor_names_list(subgraph_output_names)))
    function_definition_str = "\n        ".join(function_definition)

    # 添加import信息
    import_statements = ["from .%s import %s" % (python_module, pytorch_module)
                         for (python_module, pytorch_module) in zip(python_modules, pytorch_modules)]
    input_names = get_input_names(graph, graph, check_stages=False)
    input_names = ", ".join(get_tensor_names_list(input_names))

    # 应用模版文件
    model = model_template % {"layer_declarations": "\n        ".join(layer_declarations),
                              "function_definition": function_definition_str,
                              "module_name": model_name,
                              "inputs": input_names,
                              "import_statements": "\n".join(import_statements),
                              "module_methods": ""}  # TODO: Figure out if we need to pass in other module_methods here?

    print("Done with sub-graph fusion...")

    # 输出文件
    with open(output_filename, 'w') as f:
        f.write(model)
    '''
3.3.3 输出
最终融合结果如下：

import torch
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .stage4 import Stage4
from .stage5 import Stage5
from .stage6 import Stage6
from .stage7 import Stage7
from .stage8 import Stage8
from .stage9 import Stage9

class pd(torch.nn.Module):
    def __init__(self):
        super(pd, self).__init__()
        self.stage0 = Stage0()
        self.stage1 = Stage1()
        self.stage2 = Stage2()
        self.stage3 = Stage3()
        self.stage4 = Stage4()
        self.stage5 = Stage5()
        self.stage6 = Stage6()
        self.stage7 = Stage7()
        self.stage8 = Stage8()
        self.stage9 = Stage9()
   
    def forward(self, input0, input1, input2):
        (out2, out3, out0) = self.stage0(input0, input1, input2)
        out4 = self.stage1(out2, out0)
        out5 = self.stage2(out4)
        (out7, out6) = self.stage3(out5)
        (out8, out9) = self.stage4(out7, out6)
        (out10, out12, out11) = self.stage5(out8, out9, out2, out3)
        (out14, out15, out16) = self.stage6(out10, out12)
        (out17, out18, out19) = self.stage7(out14, out15, out16, out11)
        out20 = self.stage8(out14, out17, out18, out19)
        out21 = self.stage9(out20)
        return out21
    '''
    return python_modules, pytorch_modules, subgraph_inputs, subgraph_outputs

'''
0x02 合成模型
具体合成模型代码在 optimizer/convert_graph_to_model.py。
2.1 主体逻辑
主体逻辑如下：
    获取配置
    从graph文件中加载，得到一个图
    分割图为一系列子图
    把子图转换成模块
    合并子图，生成一个总体模型文件
    生成__init__.py
    生成配置文件
我们先看看源码，后续会仔细分析。
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert profile graphs to generated model description")
    parser.add_argument('-f', "--profile_filename", required=True,
                        help="Input profile filename")
    parser.add_argument("--model_template_filename", default="templates/model.py.template",
                        help="Model template filename")
    parser.add_argument("--init_template_filename", default="templates/__init__.py.template",
                        help="__init__.py template filename")
    parser.add_argument("--conf_template_filename", default="templates/conf.json.template",
                        help="Conf template filename")
    parser.add_argument("--stage_to_num_ranks_map", type=str, default=None,
                        help="Stage split")
    parser.add_argument('-n', "--model_name", required=True,
                        help="Name of model class")
    parser.add_argument('-a', "--arch", required=True,
                        help="Human-readable architecture name")
    parser.add_argument('-o', "--output_directory", required=True,
                        help="Full path of output model directory")
    args = parser.parse_args()

    # mkdir output_directory.
    subprocess.check_output("mkdir -p %s" % args.output_directory, shell=True)

    '''
    0x03 模型转换
    我们接下来看看具体模型转换。
    
    3.1 分离子图
    首先，main函数需要按照stage来分离子图。
    
    因为在前文中，已经把模型的层分配到了各个Stage之上，所以本阶段就是使用 partition_graph 把每个Stage所包含的层分离出来。
    '''
    # 从graph文件中加载，得到一个图
    input_node = graph.Node("input_node", node_desc="Input")
    full_graph = graph.Graph.from_str(open(args.profile_filename, 'r').read())
    initialize_weights = (args.arch == "vgg16" or args.arch == "resnet50")
    input_node.stage_id = 0
    sinks = full_graph.sinks()
    # Remove all unneeded sinks that are not used, makes code generation easier.
    # 去除没有使用的sink
    for sink in sinks:
        if sink.node_desc.startswith("__getitem__"):
            full_graph.remove_node(sink)

    # 分割图为一系列子图
    subgraphs = full_graph.partition_graph()

    '''
    3.2 转换模型
    在main函数之中，对于每个子图，将其转换为一个Pytorch Module，就对应着一个python文件。
    就是说，每一层都是这个 Module 的一个子模块。
    
    转换模型逻辑如下：假如输入为一个graph，里面包含了若干nodes，convert_subgraph_to_module 会把这个graph 转换成为一个module。

        graph = {Graph} 
         edges = {dict: 1} 
         in_edges = {dict: 1} 
         marked_nodes = {set: 2} 
         nodes = {dict: 2} 
          'node5' = {Node} node5 -- EmuBidirLSTM(  (bidir): LSTM(1024, 1024, bidirectional=True)  (layer1): LSTM(1024, 1024)  (layer2): LSTM(1024, 1024)) -- forward_compute_time=5.247, backward_compute_time=0.016, activation_size=12582912.0, parameter_size=67174400.000 -- stage_id=1
          'node6' = {Node} node6 -- Dropout(p=0.2) -- forward_compute_time=0.077, backward_compute_time=0.196, activation_size=12582912.0, parameter_size=0.000 -- stage_id=1
          __len__ = {int} 2
    我们逐一分析。
    '''
    for i, subgraph in enumerate(subgraphs): # 遍历每一个子图
        module_name = "Stage%d" % i
        module_filename = "stage%d.py" % i
        if len(subgraphs) == 1:
            module_name = args.model_name
            module_filename = "%s.py" % args.arch

        # 把这个子图转换成一个module
        num_inputs, num_outputs = convert_subgraph_to_module(subgraph, full_graph, len(subgraphs),
                                                             module_name, initialize_weights,
                                                             args.model_template_filename,
                                                             os.path.join(args.output_directory,
                                                                          module_filename))
        print("Done generating %s..." % module_filename)

    '''
    3.3 融合模型
    前一部分中，生成了若干module的python文件，都对应了一个subgraph，本节的作用就是把这些子图合并成一个大图，
    对应到python代码，就是生成一个新python文件，里面把各个subgraph的python 引入，生成一个总module文件。
    
    3.3.1 main函数逻辑
    
    main函数逻辑为：
    
        导入参数配置，得到类似 ['from .gnmt import pd']。
        把子图融合成一个总体module。
        依据融合好的结果拓展 model。
        依据融合好的结果拓展import 语句。
    '''
    # 合并子图，生成一个总体模型文件。
    model = []
    # 导入参数配置，得到类似 ['from .gnmt import pd']
    import_statements = ["from .%s import %s" % (args.arch, args.model_name)]
    pytorch_modules = None
    if len(subgraphs) > 1:
        # 把子图融合成一个总体module
        python_modules, pytorch_modules, subgraph_inputs, subgraph_outputs = \
            fuse_subgraphs_to_module(full_graph, subgraphs, args.model_name,
                                     initialize_weights,
                                     args.model_template_filename,
                                     os.path.join(args.output_directory,
                                                  "%s.py" % args.arch))

        # 依据融合好的结果拓展 model
        model = ["(%s(), [%s], [%s])" % (x[0],
                                         ", ".join(["\"%s\"" % y for y in x[1]]),
                                         ", ".join(["\"%s\"" % y for y in x[2]]))
                 for x in zip(pytorch_modules, subgraph_inputs,
                              subgraph_outputs)]
        model.append("(criterion, [\"%s\"], [\"loss\"])" % subgraph_outputs[-1][0])

        # 依据融合好的结果拓展import 语句
        import_statements.extend(
            ["from .%s import %s" % (python_module, pytorch_module)
             for (python_module, pytorch_module) in zip(python_modules, pytorch_modules)])
    else:
        inputs = ["\"input%d\"" % i for i in range(num_inputs)]
        assert(num_outputs == 1)
        model.append("(%s.%s(), [%s], [\"output\"])" % (args.arch, args.model_name, ", ".join(inputs)))
        model.append("(criterion, [\"output\"], [\"loss\"])")

    '''
    3.4 init 文件
    为了便于使用，系统又生成了 __init__文件。
    就是依据之前的 import_statements，model 等变量进行生成：
    变量如下：
    
        model = {list: 11} 
         00 = {str} '(Stage0(), ["input0", "input1", "input2"], ["out2", "out3", "out0"])'
         01 = {str} '(Stage1(), ["out2", "out0"], ["out4"])'
         02 = {str} '(Stage2(), ["out4"], ["out5"])'
         03 = {str} '(Stage3(), ["out5"], ["out7", "out6"])'
         04 = {str} '(Stage4(), ["out7", "out6"], ["out8", "out9"])'
         05 = {str} '(Stage5(), ["out8", "out9", "out2", "out3"], ["out10", "out12", "out11"])'
         06 = {str} '(Stage6(), ["out10", "out12"], ["out14", "out15", "out16"])'
         07 = {str} '(Stage7(), ["out14", "out15", "out16", "out11"], ["out17", "out18", "out19"])'
         08 = {str} '(Stage8(), ["out14", "out17", "out18", "out19"], ["out20"])'
         09 = {str} '(Stage9(), ["out20"], ["out21"])'
         10 = {str} '(criterion, ["out21"], ["loss"])'
         __len__ = {int} 11
         
         
         import_statements = {list: 1} ['from .gnmt import pd']
         0 = {str} 'from .gnmt import pd'
         
    代码如下：
    
    '''
    # 生成__init__.py
    with open(os.path.join(args.output_directory, "__init__.py"), 'w') as f1, \
         open(args.init_template_filename, 'r') as f2:
        template = f2.read()
        init = template % {
            "arch": args.arch,
            "import_statements": "\n".join(import_statements),
            "model": ",\n        ".join(model),
            "full_model": "%s()" % args.model_name
        }
        f1.write(init)
    '''
得到的__init__文件如下：

from .gnmt import pd
from .stage0 import Stage0
from .stage1 import Stage1
from .stage2 import Stage2
from .stage3 import Stage3
from .stage4 import Stage4
from .stage5 import Stage5
from .stage6 import Stage6
from .stage7 import Stage7
from .stage8 import Stage8
from .stage9 import Stage9

def arch():
    return "gnmt"

def model(criterion):
    return [
        (Stage0(), ["input0", "input1", "input2"], ["out2", "out3", "out0"]),
        (Stage1(), ["out2", "out0"], ["out4"]),
        (Stage2(), ["out4"], ["out5"]),
        (Stage3(), ["out5"], ["out7", "out6"]),
        (Stage4(), ["out7", "out6"], ["out8", "out9"]),
        (Stage5(), ["out8", "out9", "out2", "out3"], ["out10", "out12", "out11"]),
        (Stage6(), ["out10", "out12"], ["out14", "out15", "out16"]),
        (Stage7(), ["out14", "out15", "out16", "out11"], ["out17", "out18", "out19"]),
        (Stage8(), ["out14", "out17", "out18", "out19"], ["out20"]),
        (Stage9(), ["out20"], ["out21"]),
        (criterion, ["out21"], ["loss"])
    ]

def full_model():
    return pd()
    
    '''

    '''
    3.5 配置文件
    接下来会生成配置文件，这个是为后续程序运行准备。具体可能会生成 "dp_conf.json", "mp_conf.json", "hybrid_conf.json" 这几个文件。
    文件具体内容大致就是：哪个module配置到哪个stage之上，哪个stage配置到rank之上。
    3.5.1 代码逻辑
    主要逻辑是：
        如果程序输入中已经设置了如何把stage配置到rank之上，就进行设置。
        依据pytorch_modules进行设置stage数目和module数目。
        对具体rank, stage, module的分配作出设置。
        写入配置文件。
    其中，pytorch_modules 是fuse_subgraphs_to_module返回的结果。
    
        pytorch_modules = {list: 10} ['Stage0', 'Stage1', 'Stage2', 'Stage3', 'Stage4', 'Stage5', 'Stage6', 'Stage7', 'Stage8', 'Stage9']
    
    '''
    # 生成配置文件
    if args.stage_to_num_ranks_map is not None:
        # 如果程序输入中已经设置了如何把stage配置到rank之上，就进行设置
        stage_to_num_ranks_map = args.stage_to_num_ranks_map.split(",")
        stage_to_num_ranks_map = [(int(x.split(":")[0]), int(x.split(":")[1]))
                      for x in stage_to_num_ranks_map]
        num_stages = 0
        for (stage_id, replication_factor) in stage_to_num_ranks_map:
            num_stages += replication_factor
        assert(len(stage_to_num_ranks_map) == len(pytorch_modules))
        num_modules = len(pytorch_modules) + 1  # Add 1 for criterion.
    elif pytorch_modules is None:
        # 依据pytorch_modules进行设置stage数目和module数目
        num_stages = 1
        num_modules = 2  # Add 1 for criterion.
    else:
        num_stages = len(pytorch_modules)
        num_modules = len(pytorch_modules) + 1  # Add 1 for criterion.

    # 对具体rank, stage, module的分配作出设置
    all_template_args = []

    # 对数据并行进行设置
    all_template_args.append({
        "module_to_stage_map": [0] * num_modules,
        "stage_to_rank_map": str({"0": list(range(num_stages))}).replace("'", "\"")
    })

    # 对模型配置进行设置
    all_template_args.append({
        "module_to_stage_map": list(range(num_modules-1)) + [num_modules-2],
        "stage_to_rank_map": str({str(i): [i] for i in range(num_modules-1)}).replace("'", "\"")
    })
    '''
运行时变量如下：
all_template_args = 
 0 = {dict: 2} 
  'module_to_stage_map' = {list: 11} [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  'stage_to_rank_map' = {str} '{"0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}'
 1 = {dict: 2} 
  'module_to_stage_map' = {list: 11} [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
  'stage_to_rank_map' = {str} '{"0": [0], "1": [1], "2": [2], "3": [3], 
        "4": [4], "5": [5], "6": [6], "7": [7], "8": [8], "9": [9]}'   
    '''

    # 如果程序参数做设置，进行处理
    if args.stage_to_num_ranks_map is not None:
        stage_to_rank_map = {}
        ranks_so_far = 0
        for i in range(num_modules-1):
            stage_to_rank_map[str(i)] = list(range(ranks_so_far,
                                                   ranks_so_far + stage_to_num_ranks_map[i][1]))
            ranks_so_far += stage_to_num_ranks_map[i][1]
        stage_to_rank_map = str(stage_to_rank_map).replace("'", "\"")
        all_template_args.append({
            "module_to_stage_map": list(range(num_modules-1)) + [num_modules-2],
            "stage_to_rank_map": stage_to_rank_map
        })

    # 写入配置文件
    for conf_filename, template_args in zip(
        ["dp_conf.json", "mp_conf.json", "hybrid_conf.json"], all_template_args):
        with open(os.path.join(args.output_directory, conf_filename), 'w') as f1, \
             open(args.conf_template_filename, 'r') as f2:
            template = f2.read()
            conf = template % template_args
            f1.write(conf)

'''
3.5.2 数据并行
dp_config.json是专门为数据并行生成的配置文件，举例如下。

    {
        "module_to_stage_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "stage_to_rank_map": {"0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
    }
3.5.3 模型并行
mp_config.json 是专门为模型并行生成的配置文件，举例如下。

    {
        "module_to_stage_map": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
        "stage_to_rank_map": {"0": [0], "1": [1], "2": [2], "3": [3], "4": 
                [4], "5": [5], "6": [6], "7": [7], "8": [8], "9": [9]}
    }
'''