# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import graphviz
import os

# 2.1 Graph
# Graph就是图的数据结构，其主要成员包括：
#
#     nodes ：图内节点；
#     edges ：图内每个节点的输出边；
#     in_edges ：图的每个节点的输入边；
#     _predecessors ：每个节点的前序节点；
#     _successors ：每个节点的后序节点；
#     _antichain_dag ：反链DAG；
class Graph(object):
    def __init__(self, node=None):
        self.nodes = {}  # 节点
        if node is not None:
            self.nodes[node.node_id] = node
        self.edges = {} # 出边
        self.in_edges = {} # 入边

        self._predecessors = {} #每个节点的前序节点
        self._successors = {}  # 每个节点的后序节点
        self._augmented_antichains = {}
        self._deaugmented_augmented_antichains = {}
        self._next_antichains = {}
        self._antichain_dag = None  # 反链DAG

        self._colors = ['lightblue', 'green', 'grey', 'firebrick1',
                        'gold', 'chocolate1', 'beige']

        if node is not None:
            self.in_edges[node.node_id] = list()

    def copy(self):
        gr = Graph()
        for node_id in self.in_edges:
            for node2 in self.in_edges[node_id]:
                gr.add_edge(node2, self.nodes[node_id])
        return gr

    def sources(self):
        sources = []
        for node_id in self.nodes:
            if node_id not in self.in_edges or len(self.in_edges[node_id]) == 0:
                sources.append(self.nodes[node_id])
        return sources

    def add_node(self, node):
        self.nodes[node.node_id] = node

    def remove_node(self, node):
        del self.nodes[node.node_id]
        if node.node_id in self.edges:
            out_nodes = self.edges[node.node_id]
            del self.edges[node.node_id]
            for out_node in out_nodes:
                self.in_edges[out_node.node_id].remove(node)
        if node.node_id in self.in_edges:
            in_nodes = self.in_edges[node.node_id]
            del self.in_edges[node.node_id]
            for in_node in in_nodes:
                self.edges[in_node.node_id].remove(node)

    def sinks(self):
        sinks = []
        for node_id in self.nodes:
            if node_id not in self.edges or len(self.edges[node_id]) == 0:
                sinks.append(self.nodes[node_id])
        return sinks

    def reset(self):
        self._predecessors = {}
        self._successors = {}

    def add_edge(self, node1, node2):
        if node1.node_id not in self.nodes:
            self.nodes[node1.node_id] = node1
        if node2.node_id not in self.nodes:
            self.nodes[node2.node_id] = node2

        if node2.node_id not in self.in_edges:
            self.in_edges[node2.node_id] = list()
        self.in_edges[node2.node_id].append(node1)
        if node1.node_id not in self.edges:
            self.edges[node1.node_id] = list()
        self.edges[node1.node_id].append(node2)

    def remove_edge(self, node1, node2):
        self.edges[node1.node_id].remove(node2)
        self.in_edges[node2.node_id].remove(node1)

    def populate_depths(self):
        # Helper method that annotates each node in the graph with its depth from the sink.
        sources = self.sources()
        sources[0].depth = 1
        queue = [sources[0]]
        while len(queue) > 0:
            node = queue.pop(-1)
            if node.node_id not in self.edges: continue
            for out_node in self.edges[node.node_id]:
                if out_node.depth is None or out_node.depth < (node.depth + 1):
                    out_node.depth = node.depth + 1
                queue.append(out_node)

    def populate_heights(self):
        # Helper method that annotates each node in the graph with its height from the further
        # away sink.
        sinks = self.sinks()
        for sink in sinks: sink.height = 1
        queue = sinks
        visited = set()
        while len(queue) > 0:
            node = queue.pop(-1)
            visited.add(node.node_id)
            if node.node_id not in self.in_edges: continue
            for in_node in self.in_edges[node.node_id]:
                if in_node.height is None or in_node.height < (node.height + 1):
                    in_node.height = node.height + 1
                if in_node.node_id not in visited:
                    queue.append(in_node)

    '''
    partition_graph 对应的代码具体逻辑为：
    
        遍历节点，找到所有的stage。
        得到所有stage id之后，按照stage id来构建子图，具体就是针对给定的stage，在所有节点中查找对应stage的节点，构建一个子图。
    '''
    def partition_graph(self):
        stage_ids = set()

        # 遍历节点，找到所有的stage
        for node_id in self.nodes:
            stage_ids.add(self.nodes[node_id].stage_id)

        # stage_ids 为 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if len(stage_ids) == 1:
            return [self.copy()]
        subgraphs = []

        # 按照stage构建子图
        for stage_id in stage_ids:
            subgraphs.append(self.partition_graph_helper(stage_id))
        return subgraphs
    '''
    得到子图为：

        subgraphs = {list: 10} 
        
         00 = {Graph} 
           'node4' = {Node} node4 -- Embedding(32320, 1024, padding_idx=0) -- forward_compute_time=0.073, backward_compute_time=6.949, activation_size=6291456.0, parameter_size=132382720.000 -- stage_id=0
           'node1' = {Node} node1 -- Input0 -- forward_compute_time=0.000, backward_compute_time=0.000, activation_size=0.0, parameter_size=0.000 -- stage_id=0
           'node2' = {Node} node2 -- Input1 -- forward_compute_time=0.000, backward_compute_time=0.000, activation_size=0.0, parameter_size=0.000 -- stage_id=0
           'node3' = {Node} node3 -- Input2 -- forward_compute_time=0.000, backward_compute_time=0.000, activation_size=0.0, parameter_size=0.000 -- stage_id=0
           __len__ = {int} 4
            
         01 = {Graph} node5 
          edges = {dict: 1} {'node5': [<graph.graph.Node object at 0x7f9c5be91438>]}
          in_edges = {dict: 1} {'node6': [<graph.graph.Node object at 0x7f9c5be91470>]}
          nodes = {dict: 2} {'node5': <graph.graph.Node object at 0x7f9c5be91470>, 'node6': <graph.graph.Node object at 0x7f9c5be91438>}
           'node5' = {Node} node5 -- EmuBidirLSTM(  (bidir): LSTM(1024, 1024, bidirectional=True)  (layer1): LSTM(1024, 1024)  (layer2): LSTM(1024, 1024)) -- forward_compute_time=5.247, backward_compute_time=0.016, activation_size=12582912.0, parameter_size=67174400.000 -- stage_id=1
           'node6' = {Node} node6 -- Dropout(p=0.2) -- forward_compute_time=0.077, backward_compute_time=0.196, activation_size=12582912.0, parameter_size=0.000 -- stage_id=1
           __len__ = {int} 2
            
        ......
    '''

    # 针对给定的stage，在所有节点中查找对应stage的节点，构建一个子图
    def partition_graph_helper(self, stage_id):
        subgraph = Graph()
        for node1_id in self.nodes:
            if self.nodes[node1_id].stage_id == stage_id:
                subgraph.add_node(self.nodes[node1_id])
                if node1_id not in self.edges: continue
                for node2 in self.edges[node1_id]:
                    if node2.stage_id == stage_id:
                        subgraph.add_edge(self.nodes[node1_id], node2)
        return subgraph

    def compress_branch_helper(self, node, new_node_id):
        if len(self.in_edges[node.node_id]) > 1:
            return None, node
        new_node = Node("compressed_node%d" % new_node_id,
                        node_desc=("Branch %d" % new_node_id))
        chain_length = 0
        # Assumption here is that node has edges coming into it, since this is how
        # compress_branch_helper was called on it.
        while (len(self.in_edges[node.node_id]) == 1 and node.node_id in self.edges
               and len(self.edges[node.node_id]) == 1):
            chain_length += 1
            next_node = self.edges[node.node_id][0] # Since node has a single out-neighbor.
            # Compute time and parameter size are added; latest node's activation_size is used.
            new_node.forward_compute_time += node.forward_compute_time
            new_node.backward_compute_time += node.backward_compute_time
            new_node.activation_size = node.activation_size
            new_node.parameter_size += node.parameter_size
            # If next_node has more than one predecessor, then can't continue merging
            # next_node into new_node.
            if len(self.in_edges[next_node.node_id]) > 1:
                break
            node = next_node
            if node.node_id not in self.edges:
               return new_node, node
        if chain_length == 0:
            return node, node
        if chain_length == 1:
            new_node.node_desc = node.node_desc

        # If node can't be compressed into `new_node` because it has multiple
        # out-neighbors, make sure to compress `node` into `new_node` as well.
        if node.node_id in self.edges and len(self.edges[node.node_id]) > 1:
            new_node.forward_compute_time += node.forward_compute_time
            new_node.backward_compute_time += node.backward_compute_time
            new_node.activation_size = node.activation_size
            new_node.parameter_size += node.parameter_size

        # Return the new_node along with the last merged-in node which is now
        # effectively replaced in the input graph.
        return new_node, node

    def compress_branches(self):
        nodes = self.sources() # Start exploration with the input graph's source node.
        new_gr = Graph() # Create new graph, that will be returned.
        i = 0
        seen_node_ids = set()
        new_node_mapping = dict() # Map old nodes to the new compressed nodes.
        while len(nodes) > 0:
            node = nodes.pop(0)
            if node.node_id in seen_node_ids:
                continue
            if node.node_id in self.edges and len(self.edges[node.node_id]) > 1:
                for out_node in self.edges[node.node_id]:
                    # Each out_node is now a branch that needs to be compressed.
                    compressed_node, old_node = self.compress_branch_helper(
                        out_node, i)
                    i += 1
                    if compressed_node is None:
                        # Now, add an edge between `node` (or the node that replaces `node`)
                        # and `out_node`, since node compression didn't take place.
                        if node.node_id in new_node_mapping:
                            new_gr.add_edge(new_node_mapping[node.node_id], out_node)
                        else:
                            new_gr.add_edge(node, out_node)
                    else:
                        new_node_mapping[old_node.node_id] = compressed_node
                        # Add an edge between `node` (or the node that replaces `node`)
                        # and `compressed_node`.
                        if node.node_id in new_node_mapping:
                            new_gr.add_edge(new_node_mapping[node.node_id], compressed_node)
                        else:
                            new_gr.add_edge(node, compressed_node)
                    if old_node.node_id not in seen_node_ids:
                        nodes.append(old_node)
            else:
                # No branching -- copy graph to output graph.
                if node.node_id in self.edges:
                    for out_node in self.edges[node.node_id]:
                        in_node = node
                        if node.node_id in new_node_mapping:
                             in_node = new_node_mapping[node.node_id]
                        if out_node.node_id in new_node_mapping:
                            new_gr.add_edge(in_node, new_node_mapping[out_node.node_id])
                        else:
                            new_gr.add_edge(in_node, out_node)
                        if out_node.node_id not in seen_node_ids:
                            nodes.append(out_node)
            seen_node_ids.add(node.node_id)
        return new_gr

    def is_series_parallel(self, arch):
        gr_copy = self.copy()
        chain_nodes = gr_copy.chain_nodes()
        while len(chain_nodes) > 0:
            node = chain_nodes[0]
            predecessor = next(iter(gr_copy.in_edges[node.node_id]))
            successor = next(iter(gr_copy.edges[node.node_id]))
            if successor not in gr_copy.edges[predecessor.node_id]:
                gr_copy.add_edge(predecessor, successor)
            del gr_copy.nodes[node.node_id]
            gr_copy.remove_edge(node, successor)
            gr_copy.remove_edge(predecessor, node)
            chain_nodes = gr_copy.chain_nodes()
        gr_copy.to_dot("%s/%s" % (arch, arch))
        return len(gr_copy.nodes) == 2

    def chain_nodes(self):
        chain_nodes = list()
        for node in self.nodes.values():
            if node.node_id in self.edges and len(self.edges[node.node_id]) == 1 \
                and node.node_id in self.in_edges and len(self.in_edges[node.node_id]) == 1:
                chain_nodes.append(node)
        return chain_nodes

    def aggregate(self, sum_activations=False):
        forward_compute_time = 0.0
        backward_compute_time = 0.0
        parameter_size = 0.0
        activation_size = 0.0
        for node in self.nodes.values():
           forward_compute_time += node.forward_compute_time
           backward_compute_time += node.backward_compute_time
           parameter_size += node.parameter_size
           if sum_activations:
               activation_size += node.activation_size
           else:
               if node.node_id not in self.in_edges or len(self.in_edges[node.node_id]) == 0:
                   activation_size += node.activation_size
        return [forward_compute_time, backward_compute_time, parameter_size, activation_size]

    def check_fidelity(self, other):
        self_aggregate = self.aggregate()
        other_aggregate = other.aggregate()
        for i in xrange(len(self_aggregate)):
            assert(0.9999 <= (self_aggregate[i] / other_aggregate[i]) <= 1.0001)

    def check_isomorphism(self, other):
        # Hack to check for isomorphism (break ties when exploring out-neighbors with "height"
        # [longest path from one of the sinks]).
        self.populate_heights()
        other.populate_heights()
        self_topological_sort = self.topological_sort()
        other_topological_sort = other.topological_sort()
        assert(len(self_topological_sort) == len(other_topological_sort))

        for (self_node, other_node) in zip(self_topological_sort, other_topological_sort):
            assert(self_node.node_desc == other_node.node_desc)
            if self_node.node_id in self.edges:
                assert(len(self.edges[self_node.node_id]) == len(other.edges[other_node.node_id]))
            if self_node.node_id in self.in_edges:
                assert(len(self.in_edges[self_node.node_id]) == len(other.in_edges[other_node.node_id]))

    '''
    3.5 拓扑排序
    得到了增强反链之后，需要进行拓扑排序之后才能使用。
    
        antichain_gr = gr.antichain_dag()
        states = antichain_gr.topological_sort()
    
    得出拓扑排序的目的是：如果按照拓扑序列的顶点次序，在到达某节点之前，可以保证它的所有前序活动都已经完成，
    从而整个工程顺序执行，不会冲突。
    
    在图论中，拓扑排序（Topological Sorting）是一个有向无环图（DAG, Directed Acyclic Graph）的所有顶点的线性序列。
    且该序列必须满足下面两个条件：
        
        1 每个顶点出现且只出现一次。
        
        2 若存在一条从顶点 A 到顶点 B 的路径，那么在序列中顶点 A 出现在顶点 B 的前面。
        
    有向无环图（DAG）才有拓扑排序，非DAG图没有拓扑排序一说。一个有向无环图可以有一个或多个拓扑排序序列。
    
    例如，下面这个图：
    
    +--------+                  +--------+
    |        +----------------> |        |
    |   1    |                  |   4    +------------+
    |        |    +-----------> |        |            |
    +-----+--+    |             +---+----+            |
          |       |                 |                 v
          |       |                 |              +--+--+
          |       |                 |        +---> |  5  |
          |       |                 |        |     +-----+
          v       |                 |        |
                  |                 v        |
    +--------+    |             +---+-----+  |
    |        +----+             |         |  |
    |    2   +----------------->+    3    +--+
    |        |                  |         |
    +--------+                  +---------+
    
    得到拓扑排序后的结果是 { 1, 2, 4, 3, 5 }。
    
    这里的拓扑排序算法使用的是深度优先排序。
    
    最终结果举例如下，可以和上面的反链DAG antichain_dag 比对，看看异同：
    
    states = {list: 99} 
     00 = {AntichainNode} antichain_0 -- ['node4']
     01 = {AntichainNode} antichain_1 -- ['node5']
     02 = {AntichainNode} antichain_2 -- ['node6']
     03 = {AntichainNode} antichain_3 -- ['node7']
     04 = {AntichainNode} antichain_4 -- ['node8']
     05 = {AntichainNode} antichain_5 -- ['node8', 'node10']
     06 = {AntichainNode} antichain_7 -- ['node8', 'node11']
     07 = {AntichainNode} antichain_10 -- ['node8', 'node12']
     08 = {AntichainNode} antichain_6 -- ['node14']
     09 = {AntichainNode} antichain_8 -- ['node14', 'node15']
     10 = {AntichainNode} antichain_11 -- ['node14', 'node16']
     11 = {AntichainNode} antichain_13 -- ['node14', 'node17']
     12 = {AntichainNode} antichain_9 -- ['node19']
     13 = {AntichainNode} antichain_12 -- ['node20', 'node23']
     14 = {AntichainNode} antichain_18 -- ['node23', 'node20', 'node26']
     15 = {AntichainNode} antichain_17 -- ['node23', 'node20', 'node24']
     16 = {AntichainNode} antichain_32 -- ['node23', 'node20', 'node28']
     17 = {AntichainNode} antichain_31 -- ['node23', 'node20', 'node26', 'node24']
     18 = {AntichainNode} antichain_63 -- ['node23', 'node20', 'node26', 'node28']
     19 = {AntichainNode} antichain_33 -- ['node20', 'node26', 'node29']
     20 = {AntichainNode} antichain_16 -- ['node20', 'node43', 'node23']
     21 = {AntichainNode} antichain_30 -- ['node23', 'node20', 'node43', 'node26']
     22 = {AntichainNode} antichain_29 -- ['node23', 'node20', 'node43', 'node24']
     23 = {AntichainNode} antichain_59 -- ['node23', 'node20', 'node43', 'node28']
     
    我们 也可以和如下增强反链比对，
    看到 states 就是对增强反链DAG进行拓扑排序之后的结果，
    按照这个顺序进行训练是符合逻辑的。
    
    _augmented_antichains = {dict: 99} 
     ('node4',) = {list: 1} ['node4']
     ('node5',) = {list: 1} ['node5']
     ('node6',) = {list: 1} ['node6']
     ('node7',) = {list: 1} ['node7']
     ('node8',) = {list: 1} ['node8']
     ('node10',) = {list: 2} ['node8', 'node10']
     ('node14',) = {list: 1} ['node14']
     ('node11',) = {list: 2} ['node8', 'node11']
     ('node15',) = {list: 2} ['node14', 'node15']
     ('node19',) = {list: 1} ['node19']
     ('node12',) = {list: 2} ['node8', 'node12']
     ('node16',) = {list: 2} ['node14', 'node16']
     ('node23',) = {list: 2} ['node20', 'node23']
     ('node17',) = {list: 2} ['node14', 'node17']
     ('node23', 'node30') = {list: 3} ['node20', 'node30', 'node23']
     ('node23', 'node36') = {list: 3} ['node20', 'node36', 'node23']
     ('node23', 'node43') = {list: 3} ['node20', 'node43', 'node23']
     ('node24',) = {list: 3} ['node23', 'node20', 'node24']
     ('node26',) = {list: 3} ['node23', 'node20', 'node26']
     ('node23', 'node30', 'node36') = {list: 4} ['node20', 'node36', 'node30', 'node23']
     ('node23', 'node30', 'node43') = {list: 4} ['node20', 'node43', 'node30', 'node23']
     ('node31',) = {list: 3} ['node20', 'node26', 'node31']
     ('node24', 'node30') = {list: 4} ['node23', 'node20', 'node30', 'node24']
     ('node26', 'node30') = {list: 4} ['node23', 'node20', 'node30', 'node26']
     ('node23', 'node36', 'node43') = {list: 4} ['node20', 'node43', 'node36', 'node23']
     ('node37',) = {list: 4} ['node32', 'node20', 'node26', 'node37']
     ('node24', 'node36') = {list: 4} ['node23', 'node20', 'node36', 'node24']
     ('node26', 'node36') = {list: 4} ['node23', 'node20', 'node36', 'node26']
     ('node44',) = {list: 2} ['node40', 'node44']
     ('node24', 'node43') = {list: 4} ['node23', 'node20', 'node43', 'node24']
     ('node26', 'node43') = {list: 4} ['node23', 'node20', 'node43', 'node26']
     ('node24', 'node26') = {list: 4} ['node23', 'node20', 'node26', 'node24']    

    3.6 总结
    因为目前的算法比较复杂，所以我们暂时总结一下目前为止的工作：
    
        计算出了每个节点的增强反链，最终得到增强反链组合 _augmented_antichains 。
        
        计算出了每个节点的后续反链。寻找某节点后续反链的目的就是找到下一个图分割点 A，然后为了确定 A 的运行时间（或者其他信息），
        需要找到 A 的增强反链（一些增强反链就是一些状态）。
        _next_antichains 是后续反链组合。
        
        antichain_dag 函数依据 _next_antichains 和 _augmented_antichains 进行处理，
        构建一个反链 DAG，就是变量 antichain_dag。
        
        得到了增强反链DAG之后，需要进行拓扑排序之后才能使用。
        得出拓扑排序的目的是：如果按照拓扑序列的顶点次序，在到达某节点之前，可以保证它的所有前序活动都已经完成，
        从而整个工程顺序执行，不会冲突。
        
        states 就是对增强反链DAG进行拓扑排序之后的结果，按照这个顺序进行训练是符合逻辑的。
        所以后续工作就是在 states 基础上运行。
             
    '''
    def topological_sort(self):
        # Algorithm from https://en.wikipedia.org/wiki/Topological_sorting
        self.sorted_nodes = []
        self.marked_nodes = set()
        self.temporarily_marked_nodes = set()
        nodes = list(self.nodes.values())
        nodes.sort(key=lambda x: x.node_desc)
        for node in nodes:
            if node.node_id in self.marked_nodes:
                continue
            self.topological_sort_helper(node.node_id)
        return [self.nodes[node_id] for node_id in self.sorted_nodes]

    def topological_sort_helper(self, node_id):
        if node_id in self.marked_nodes:
            return
        if node_id in self.temporarily_marked_nodes:
            raise Exception("Graph has a cycle")
        self.temporarily_marked_nodes.add(node_id)
        if node_id in self.edges:
            out_nodes = list(self.edges[node_id])
            out_nodes.sort(key=lambda x: (x.node_desc, x.height))
            for out_node in out_nodes:
                self.topological_sort_helper(out_node.node_id)
        self.marked_nodes.add(node_id)
        self.temporarily_marked_nodes.remove(node_id)
        self.sorted_nodes.insert(0, node_id)

    def predecessors(self, node):
        if node in self._predecessors:
            return self._predecessors[node]
        predecessors = set()
        if node not in self.in_edges:  # Source node
            return predecessors
        for in_node in self.in_edges[node]:
            predecessors.add(in_node)
            predecessors.update(self.predecessors(in_node.node_id))
        self._predecessors[node] = predecessors
        return self._predecessors[node]

    def all_predecessors(self, antichain):
        all_predecessors = set()
        for antichain_node in antichain:
            all_predecessors.update(self.predecessors(antichain_node))
            all_predecessors.add(self.nodes[antichain_node])
        return all_predecessors

    def successors(self, node):
        if node in self._successors:
            return self._successors[node]
        successors = set()
        if not node in self.edges:  # Sink node
            return successors
        for out_node in self.edges[node]:
            successors.add(out_node)
            successors.update(self.successors(out_node.node_id))
        self._successors[node] = successors
        return self._successors[node]
    '''
    3.2 增强反链
    首先要介绍先增强反链概念。每个节点的增强反链包括：本身节点 + 部分前序节点。
    这个前序节点的选取算法是：
    
        获取本节点的全部前序节点列表；
        
        如果一个前序节点的"出边目的节点"不在全部前序节点列表，且"出边目的节点"不为本身，则选取此前序节点为增强反链的一部分。
        
    从下面图例中可以看出来，如果某一个节点 A，其前置节点中有一个分叉节点 Z，
    且这个分叉之中，有一个分叉绕过了节点 A，则对于节点 A，他的增强反链就是 [A, Z]。
    
    对于增强反链概念，可以理解为：对于节点 A，他只有把节点 Z 一起考虑，才能唯一确定自己节点的运行时间。
    因为如果思考节点 A 的运行时间，我理解的大致思路是：
    
        因为各个阶段可以流水线并行，所以 A 的运行时间应该是以下三个时间的最大值：A的计算时间，A的输入时间，A的输出时间。
        
        A 的输入时间是以下两个时间的最大值： X --> A 节点输出时间，Z --> A 节点的输出时间。
        
        但是因为不清楚 Z 的内部运行机制，所以不能确定 Z 的两个输出之间是否有依赖关系，
        比如 "必须先完成 Z--> D，才能输出 Z--> A"， 所以，也需要考虑 Z --> D 的传输时间。
        
    所以，需要把 [ A，Z ] 放在一起作为一个状态考虑，事实上 PipeDream 就是这么处理的，用 [ A，Z ] 这个状态来统一计算。
    
    因为作为一个状态考虑，所以给节点 A 计算输出激活值大小，
    具体是通过遍历其反链（增强反链）来计算，就是把其增强反链的前序节点给自己的输出都叠加起来。
    
        +-----+            +-----+
        |  X  |            |  Z  |
        +--+--+            +--+-++
           |                  | |
           |                  | |
           +------+   +-------+ |
                  |   |         |
                  v   v         |
                 ++---++        |
                 |  A  |        |
                 ++-+--+        |
                  | |           |
        +---------+ |           |
        |           |           |
        v           v           v
    +---+-+      +--+--+      +-+---+
    |  B  |      |  C  |      |  D  |
    +-----+      +-----+      +-----+
    在代码之中，_augmented_antichains 是增强反链，也是一个字典类，key是节点名字，value是 key 节点的增强反链，比如：
    '''
    def augment_antichain(self, antichain):
        # 参数 antichain 是一个节点列表
        antichain_key = tuple(sorted(antichain))

        # 如果key已经在扩大反链之中，就直接返回对应key的增强反链
        if antichain_key in self._augmented_antichains:
            return self._augmented_antichains[antichain_key]
        extra_nodes = set()
        all_predecessors = set()

        # 遍历参数list之中的反链节点，获取每个节点的前置节点，归并在all_predecessors之中。
        for antichain_node in antichain:
            predecessors = self.predecessors(antichain_node)
            all_predecessors = all_predecessors.union(predecessors)

        # 遍历参数list之中的反链节点
        for antichain_node in antichain:
            # 获取每个反链节点的前置节点列表
            predecessors = self.predecessors(antichain_node)

            # 遍历每个前置节点
            for predecessor in predecessors:
                # 看每个前置节点的出边，如果出边不在前置节点列表之中，且 出边节点不等于本反链节点
                for out_node in self.edges[predecessor.node_id]:
                    if out_node not in predecessors and out_node.node_id != antichain_node:
                        # 把这个前置节点插入到附加节点列表中
                        extra_nodes.add(predecessor.node_id)

        # 最终把个附加节点列表插入到增强节点之中
        self._augmented_antichains[antichain_key] = list(extra_nodes) + antichain
        return self._augmented_antichains[antichain_key]
    '''
比如对应下图中的逻辑，初始化之后，_augmented_antichains 就是
    
    _augmented_antichains = {dict: 1}
     ('node4',) = {list: 1} ['node4']
     
后续迭代node 5之后，_augmented_antichains 就是

    _augmented_antichains = {dict: 2}
     ('node4',) = {list: 1} ['node4']
     ('node5',) = {list: 1} ['node5']
     __len__ = {int} 2
 
继续迭代，增强反链为：

    _augmented_antichains = {dict: 7}
    ('node4',) = {list: 1} ['node4'] # node4的增强反链只有自己
    ('node5',) = {list: 1} ['node5'] # node5的增强反链只有自己
    ('node6',) = {list: 1} ['node6']
    ('node7',) = {list: 1} ['node7']
    ('node8',) = {list: 1} ['node8']
    ('node10',) = {list: 2} ['node8', 'node10'] # node10的增强反链是'node8', 'node10'
    ('node14',) = {list: 1} ['node14']
    ('node11',) = {list: 2} ['node8', 'node11'] # node11的增强反链是'node8', 'node11'
    ('node15',) = {list: 2} ['node14', 'node15']
    ('node19',) = {list: 1} ['node19']
    ('node12',) = {list: 2} ['node8', 'node12']
    ('node16',) = {list: 2} ['node14', 'node16']
    ('node23',) = {list: 2} ['node20', 'node23']
    ('node17',) = {list: 2} ['node14', 'node17']
    
图例中可以看出来，因为有 node 8的出边 [node 8，node 14] 存在，
对于 node 10, node 11, node 12 来说，他们必须把 node 8 加入自己的增强反链之中。

对于 node 10，我们可以认为，必须结合 node 8之后，node 10 才能确定 node 10 的运行时间。
下面图上标记出来了 node 10 的 augmented 反链（本身节点 + 部分前序节点）。

+-------+       +-------+
| node1 |       | node2 |
+---+---+       +---+---+
    |               |
    |               |
    |               |
    v               v
+---+---+       +---+---+        +-------+        +-------+
| node4 +-----> | node5 +------> | node6 +------->+ node7 |
+-------+       +-------+        +-------+        +-+-+---+
                                                    | |
                                                    | |
                                      +-------------+ |
                                      |               |
                                      v               v  augmented
                                 +----+--+        +---+---+
                                 | node9 |        | node8 +-----+
                                 +-------+        +---+---+     |
                                                      |         |
                    +---------------------------------+         |
                    |                                           |
                    v                                           |
               +----+---+       +--------+        +--------+    |
     antichain | node10 +-----> | node11 +------> | node12 |    |
               +--------+       +---+----+        +----+---+    |
             augmented              |                  |        |
                                    |                  |        |
                                    v                  v        |
                                +---+----+        +----+---+    |
                                | node13 |        | node14 +<---+
                                +--------+        +-+----+-+
                                                    |    |
                                             +------+    +---+
                                             |               |
                                             v               v
                                        +----+---+        +--+-----+
                                        | node15 |        | node19 |
                                        +--------+        +--------+    
    '''

    def deaugment_augmented_antichain(self, augmented_antichain):
        augmented_antichain_key = tuple(sorted(augmented_antichain))
        if augmented_antichain_key in self._deaugmented_augmented_antichains:
            return self._deaugmented_augmented_antichains[augmented_antichain_key]
        nodes_to_remove = set()
        all_successors = set()
        for augmented_antichain_node in augmented_antichain:
            successors = self.successors(augmented_antichain_node)
            for augmented_antichain_node_prime in augmented_antichain:
                if self.nodes[augmented_antichain_node_prime] in successors:
                    nodes_to_remove.add(augmented_antichain_node)
        antichain = list()
        for augmented_antichain_node in augmented_antichain:
            if (augmented_antichain_node not in nodes_to_remove and \
                augmented_antichain_node not in antichain):
                antichain.append(augmented_antichain_node)
        self._deaugmented_augmented_antichains[augmented_antichain_key] = antichain
        return self._deaugmented_augmented_antichains[augmented_antichain_key]

    #is_next_antichain 方法用来判断某新节点是否为后续反链。
    def is_next_antichain(self, augmented_antichain, new_node):
        successors = self.successors(new_node)
        augmented_antichain_set = set(augmented_antichain)

        # 遍历新节点的后续节点
        for successor in successors:
            # 如果后续节点有一个在增强节点之中，就返回false，说明不是后续反链
            if successor.node_id in augmented_antichain_set:
                return False
        # 否则就是后续反链
        return True

    def construct_antichain(self, augmented_antichain, old_node, new_node):
        new_antichain = [x if x != old_node else new_node for x in augmented_antichain]
        return self.deaugment_augmented_antichain(new_antichain)

    '''
    3.3 后续反链
    在代码之中，_next_antichains 是一个字典类，key是节点名字，value是 key 节点的后续反链。
    
    比如，对于 node A 来说，下一个反链是 [ node B, node C ]，
    其中 node B 和 node C 彼此之间无法排序。寻找反链的目的就是找到下一个图分割点。
    
        +-----+            +-----+
        |  X  |            |  Z  |
        +--+--+            +--+-++
           |                  | |
           |                  | |
           +------+   +-------+ |
                  |   |         |
                  v   v         |
                 ++---++        |
                 |  A  |        |
                 ++-+--+        |
                  | |           |
        +---------+ |           |
        |           |           |
        v           v           v
    +---+-+      +--+--+      +-+---+
    |  B  |      |  C  |      |  D  |
    +-----+      +-----+      +-----+
    对于每个节点 antichain ，next_antichains 函数获取其后续反链。
    
 ...
 
 _next_antichains举例如下，大家可以结合之前的增强反链对比看看。

    以 node 10 为例，其增强节点为：[ node 8，node 10 ]，
    
    遍历这些增强节点，看每一个增强节点的出边。8 的出边 [ node 10，node 14 ]，10 的出边是 [ node 11]。
    
    所以有三个点 node 10，node 11，node 14 可以继续看。其中node 10 已经在[ node 8，node 10 ]之中，所以不考虑。
    
    用 14 调用 is_next_antichain。
    
        is_next_antichain 之中，augmented_antichain 为 [ node 8, node 10]，new_node 是 node 14。
        
        得到 successors 集合为 [ node31，node16，node23，node44，node48 ....] 等22个节点，
        这些节点都不在 [ node 8, node 10] 之中，所以 is_next_antichain 为 true，14 是后续反链节点之一。
        
    用 11 调用 is_next_antichain。
        is_next_antichain 之中，augmented_antichain 为 [ node 8, node 10]，new_node 是 node 11。
        
        得到 successors 集合为 [ node16，node40，node23，....] 等节点，
        这些节点都不在 [ node 8, node 10] 之中，所以 is_next_antichain 为 true，11 是后续反链节点之一。

所以 node 10 的后续反链是 [ ['node14'] ，[ 'node11'] ]。

对比 看看，node 10 的增强反链是 ['node8', 'node10']，

    _next_antichains = {dict: 99} 
     ('node4',) = {list: 1} [['node5']]
     ('node5',) = {list: 1} [['node6']]
     ('node6',) = {list: 1} [['node7']]
     ('node7',) = {list: 1} [['node8']]
     ('node8',) = {list: 2} [['node10'], ['node14']]
     ('node10',) = {list: 2} [['node14'], ['node11']] # 这里
     ('node14',) = {list: 2} [['node15'], ['node19']]
     ('node11',) = {list: 2} [['node14'], ['node12']]
     ('node15',) = {list: 2} [['node19'], ['node16']]
     ('node19',) = {list: 1} [['node23']]
     ('node12',) = {list: 2} [['node14'], ['node14']]
     ('node16',) = {list: 2} [['node19'], ['node17']]
     
具体如下图，可以看出来，node 11和 node 14确实是 node 10的后续反链，就是在这两个节点上可以对于图进行分割。

可以这么理解：对于 node 10 来说，下一个反链是 [ node 11, node 14]，其中 node 11 和 node 14 彼此之间无法排序。
寻找后续反链的目的就是找到下一个图分割点。

+-------+       +-------+
| node1 |       | node2 |
+---+---+       +---+---+
    |               |
    |               |
    |               |
    v               v
+---+---+       +---+---+        +-------+        +-------+
| node4 +-----> | node5 +------> | node6 +------->+ node7 |
+-------+       +-------+        +-------+        +-+-+---+
                                                    | |
                                                    | |
                                      +-------------+ |
                                      |               |
                                      v               v  augmented
                                 +----+--+        +---+---+
                                 | node9 |        | node8 +-----+
                                 +-------+        +---+---+     |
                                                      |         |
                    +---------------------------------+         |
                    |                                           |
                    v              next                         |
               +----+---+       +--------+        +--------+    |
     antichain | node10 +-----> | node11 +------> | node12 |    |
               +--------+       +---+----+        +----+---+    |
             augmented              |                  |        |
                                    |                  |        |
                                    v             next v        |
                                +---+----+        +----+---+    |
                                | node13 |        | node14 +<---+
                                +--------+        +-+----+-+
                                                    |    |
                                             +------+    +---+
                                             |               |
                                             v               v
                                        +----+---+        +--+-----+
                                        | node15 |        | node19 |
                                        +--------+        +--------+
   
    
    '''
    def next_antichains(self, antichain):
        # 构建antichain的反链key，其实就是 antichain 自己作为key
        antichain_key = tuple(sorted(antichain))

        # 如果key已经在后续反链之中，则返回这个后续反链
        if antichain_key in self._next_antichains:
            return self._next_antichains[antichain_key]

        next_antichains = []
        antichain_set = set(antichain)

        # 获取 antichain 的增强反链
        augmented_antichain = self.augment_antichain(antichain)

        # 遍历增强反链
        for augmented_antichain_node in augmented_antichain:
            # 遍历增强反链某节点的出边
            next_nodes = self.edges[augmented_antichain_node] if augmented_antichain_node in self.edges else []

            # 遍历增强反链某节点的出边
            for next_node in next_nodes:
                # 如果出边节点已经在反链集合之中，跳过，进入下一循环
                if next_node.node_id in antichain_set:
                    continue

                # 如果出边节点是后续反链，则假如到反链列表
                if self.is_next_antichain(augmented_antichain, next_node.node_id):
                    next_antichain = self.construct_antichain(augmented_antichain,
                                                              augmented_antichain_node,
                                                              next_node.node_id)
                    next_antichains.append(next_antichain)

        # 最终把反链列表设置为key对应的反链]
        self._next_antichains[antichain_key] = next_antichains
        return self._next_antichains[antichain_key]

    '''
    3.4 总体构建
    antichain_dag 的目的是依据 增强反链列表 和 后续反链列表来构建一个反链 DAG。
    我们以上面的图例进行讲解，以 node 8 为例。

    这里其实目的是设置 antichain_mapping。
    
    流程是：
    
        从 antichain_queue 弹出第一个节点，赋值为 antichain，这里为 node 8。
        
        获取 antichain 的后续反链，对于8，这里是[[10],[14]]。
        
        遍历后续反链 [10,14]。
        
        以 10 为例，设置下一个反链节点的key 为 10。
        
        下一反链节点 10 被设置为其增强节点 [ 8, 10 ]，即 ('node10',) = {AntichainNode} antichain_5 -- ['node8', 'node10']。
    
    可以看到，寻找某节点后续反链的目的就是找到下一个图分割点 A，然后为了确定 A 的运行时间（或者其他信息），
    需要找到 A 的增强反链（一些增强反链就是一些状态），A 的 antichain_mapping 就是其增强反链。
    
    antichain_mapping 示例如下：
    
        antichain_mapping = {dict: 99} 
         ('node4',) = {AntichainNode} antichain_0 -- ['node4']
         ('node5',) = {AntichainNode} antichain_1 -- ['node5']
         ('node6',) = {AntichainNode} antichain_2 -- ['node6']
         ('node7',) = {AntichainNode} antichain_3 -- ['node7']
         ('node8',) = {AntichainNode} antichain_4 -- ['node8']
         ('node10',) = {AntichainNode} antichain_5 -- ['node8', 'node10'] # 最新设置
         ('node14',) = {AntichainNode} antichain_6 -- ['node14']
         ('node11',) = {AntichainNode} antichain_7 -- ['node8', 'node11']
         ('node15',) = {AntichainNode} antichain_8 -- ['node14', 'node15']
         ('node19',) = {AntichainNode} antichain_9 -- ['node19']
         ('node12',) = {AntichainNode} antichain_10 -- ['node8', 'node12']
         ('node16',) = {AntichainNode} antichain_11 -- ['node14', 'node16']
         ('node23',) = {AntichainNode} antichain_12 -- ['node20', 'node23']
         ('node17',) = {AntichainNode} antichain_13 -- ['node14', 'node17']
     
    antichain_dag 示例如下，可以认为就是增强反链DAG：
    
        antichain_dag = {Graph}
            nodes = {dict: 99} 
           'antichain_0' = {AntichainNode} antichain_0 -- ['node4']
           'antichain_1' = {AntichainNode} antichain_1 -- ['node5']
           'antichain_2' = {AntichainNode} antichain_2 -- ['node6']
           'antichain_3' = {AntichainNode} antichain_3 -- ['node7']
           'antichain_4' = {AntichainNode} antichain_4 -- ['node8']
           'antichain_5' = {AntichainNode} antichain_5 -- ['node8', 'node10']
           'antichain_6' = {AntichainNode} antichain_6 -- ['node14']
           'antichain_7' = {AntichainNode} antichain_7 -- ['node8', 'node11']
           'antichain_8' = {AntichainNode} antichain_8 -- ['node14', 'node15']
           'antichain_9' = {AntichainNode} antichain_9 -- ['node19']
           'antichain_10' = {AntichainNode} antichain_10 -- ['node8', 'node12']
           'antichain_11' = {AntichainNode} antichain_11 -- ['node14', 'node16']
           'antichain_12' = {AntichainNode} antichain_12 -- ['node20', 'node23']
           'antichain_13' = {AntichainNode} antichain_13 -- ['node14', 'node17']
           'antichain_14' = {AntichainNode} antichain_14 -- ['node20', 'node30', 'node23']
           'antichain_15' = {AntichainNode} antichain_15 -- ['node20', 'node36', 'node23']
           'antichain_16' = {AntichainNode} antichain_16 -- ['node20', 'node43', 'node23']
           'antichain_17' = {AntichainNode} antichain_17 -- ['node20', 'node23', 'node24']
           
    '''
    def antichain_dag(self):
        if self._antichain_dag is not None:
            return self._antichain_dag

        antichain_dag = Graph()
        antichain_id = 0
        antichain = [self.sources()[0].node_id]  # 获取source第一个节点。

        # 构建首节点，同时利用 augment_antichain 来往_augmented_antichains 之中添加首节点。
        source_node = AntichainNode("antichain_%d" % antichain_id, self.augment_antichain(antichain))
        antichain_dag.source = source_node

        antichain_queue = [antichain] # 把第一个节点插入queue
        antichain_mapping = {tuple(sorted(antichain)): source_node}

        # 如果queue之中还有节点
        while len(antichain_queue) > 0:
            antichain = antichain_queue.pop(0) # 弹出第一个节点，赋值为 antichain，这里为 node 8

            # key就是由 antichain 节点名字构建，比如 antichain_key = {tuple: 1} node8
            antichain_key = tuple(sorted(antichain))

            # 如果 antichain_key 已经位于self._next_antichains之中，
            # 即 antichain_key 的后续反链已经被记录，就跳过去
            if antichain_key in self._next_antichains:
                continue

            # 获取 antichain 的后续反链，对于8，这里是[[10],[14]]
            next_antichains = self.next_antichains(antichain)

            # 遍历后续反链[10,14]
            for next_antichain in next_antichains:
                # 下一个反链节点的key 10
                next_antichain_key = tuple(sorted(next_antichain))
                if next_antichain_key not in antichain_mapping:  # 如果存在，就跳过
                    antichain_id += 1

                    # 下一反链节点 10 被设置为其增强节点 [ 8, 10 ]
                    next_antichain_node = \
                        AntichainNode("antichain_%d" % antichain_id, self.augment_antichain(next_antichain))

                    # 设置 antichain_mapping
                    antichain_mapping[next_antichain_key] = next_antichain_node

                # 向 反链DAG 插入边：
                antichain_dag.add_edge(antichain_mapping[antichain_key],
                                       antichain_mapping[next_antichain_key])

                # 把最新反链节点插入queue，下次迭代使用
                antichain_queue.append(next_antichain)

        self._antichain_dag = antichain_dag
        return antichain_dag

    def __str__(self):
        strs = []
        for node in self.nodes.values():
            strs.append(str(node))
        for node in self.nodes.values():
            if node.node_id not in self.in_edges:
                continue
            for in_node in self.in_edges[node.node_id]:
                strs.append("\t%s -- %s" % (in_node.node_id, node.node_id))
        return "\n".join(strs)

'''
2.2 构建图
图是由profile文件的字符串构建出来。找出来profile文件内容我们就可以知道，具体是针对每行进行不同处理。

    node1 -- Input0 -- forward_compute_time=0.000, backward_compute_time=0.000, activation_size=0.0, parameter_size=0.000
    node4 -- Embedding(32320, 1024, padding_idx=0) -- forward_compute_time=0.073, backward_compute_time=6.949, activation_size=6291456.0, parameter_size=132382720.000
    node5 -- EmuBidirLSTM(  (bidir): LSTM(1024, 1024, bidirectional=True)  (layer1): LSTM(1024, 1024)  (layer2): LSTM(1024, 1024)) -- forward_compute_time=5.247, backward_compute_time=0.016, activation_size=12582912.0, parameter_size=67174400.000
        node1 -- node4
        node4 -- node5
        node2 -- node5
构建图具体代码如下：
'''
    @staticmethod
    def from_str(graph_str):
        gr = Graph()
        graph_str_lines = graph_str.strip().split('\n')
        for graph_str_line in graph_str_lines:  # 逐行处理
            if not graph_str_line.startswith('\t'):
                node = Node.from_str(graph_str_line.strip())  # 构建节点
                gr.nodes[node.node_id] = node
            else:
                # 构建边
                [in_node_id, node_id] = graph_str_line.strip().split(" -- ")
                if node_id not in gr.in_edges:  # 每个节点的输入边
                    gr.in_edges[node_id] = [gr.nodes[in_node_id]]
                else:
                    gr.in_edges[node_id].append(gr.nodes[in_node_id])
                if in_node_id not in gr.edges: # 每个节点的输出边
                    gr.edges[in_node_id] = [gr.nodes[node_id]]
                else:
                    gr.edges[in_node_id].append(gr.nodes[node_id])
        return gr

    # 具体调用了 graph.py 的函数完成，这里摘录 to_dot函数如下：
    def to_dot(self, arch):
        dot = graphviz.Digraph()
        for node in self.nodes.values():
            node_desc = "%s\n[forward_compute_time=%.3f,backward_compute_time=%.3f,activation_size=%s,parameter_size=%.1f]" % (
                node.node_desc, node.forward_compute_time, node.backward_compute_time,
                node.activation_size, node.parameter_size)
            if node.stage_id is not None:
                color = self._colors[node.stage_id % len(self._colors)]
                dot.node(node.node_id, node_desc,
                   color=color, style='filled')
            else:
                dot.node(node.node_id, node_desc)
        for node in self.nodes.values():
            if node.node_id not in self.edges:
                continue
            for out_node in self.edges[node.node_id]:
                dot.edge(node.node_id, out_node.node_id)
        dot.render(arch)

    def plot_cdfs(self, cdfs, output_directory):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import seaborn as sns
        matplotlib.rc('text', usetex=True)
        sns.set_style('ticks')
        sns.set_style({'font.family':'sans-serif'})
        flatui = ['#002A5E', '#FD151B', '#8EBA42', '#348ABD', '#988ED5', '#777777', '#8EBA42', '#FFB5B8']
        sns.set_palette(flatui)
        paper_rc = {'lines.linewidth': 2, 'lines.markersize': 10}
        sns.set_context("paper", font_scale=3,  rc=paper_rc)
        current_palette = sns.color_palette()

        plt.figure(figsize=(10, 4))
        ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

        labels = ["Compute", "Activations", "Parameters"]
        for i in range(3):
            cdf = [cdfs[j][i] for j in range(len(cdfs))]
            ax.plot(range(len(cdfs)), cdf,  label=labels[i],
                    linewidth=2)
        ax.set_xlim([0, None])
        ax.set_ylim([0, 100])

        ax.set_xlabel("Layer ID")
        ax.set_ylabel("CDF (\%)")
        plt.legend()

        with PdfPages(os.path.join(output_directory, "cdf.pdf")) as pdf:
            pdf.savefig(bbox_inches='tight')

    def plot_bar_graph(self, all_values, ylabel, legend, output_template, output_directory):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import seaborn as sns
        matplotlib.rc('text', usetex=True)
        sns.set_style('ticks')
        sns.set_style({'font.family':'sans-serif'})
        flatui = ['#002A5E', '#FD151B', '#8EBA42', '#348ABD', '#988ED5', '#777777', '#8EBA42', '#FFB5B8']
        sns.set_palette(flatui)
        paper_rc = {'lines.linewidth': 2, 'lines.markersize': 10}
        sns.set_context("paper", font_scale=3,  rc=paper_rc)
        current_palette = sns.color_palette()

        labels = ["Compute_times", "Activations", "Parameters"]
        ylabels = ["Compute time\n(milliseconds)", "Activation size\n(bytes)", "Parameter size\n(bytes)"]
        for i in range(3):
            plt.figure(figsize=(10, 4))
            ax = plt.subplot2grid((1, 1), (0, 0), colspan=1)

            values_sum = sum([all_values[j][i] for j in range(len(all_values))])
            # Truncate the number of values plotted, since bars become very thin otherwise.
            values = [all_values[j][i] for j in range(len(all_values))][:400]
            if legend:
                ax.bar(range(len(values)), values, label="Sum: %.1f" % values_sum)
            else:
                ax.bar(range(len(values)), values)
            ax.set_xlim([0, None])
            ax.set_ylim([0, None])

            ax.set_xlabel("Layer ID")
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel(ylabels[i])
            if legend:
                plt.legend()

            with PdfPages(os.path.join(output_directory,
                          (output_template % labels[i].lower()))) as pdf:
                pdf.savefig(bbox_inches='tight')

    def render_bar_graphs_and_cdfs(self, output_directory):
        topological_ordering = self.topological_sort()[1:]  # Skip input node.
        cdfs = []
        raw_values = []
        pdfs = []
        for node in topological_ordering:
            activation_size = node.activation_size
            if isinstance(activation_size, list):
                activation_size = sum(activation_size)
            if len(cdfs) == 0:
                cdfs.append([node.forward_compute_time + node.backward_compute_time,
                             activation_size, node.parameter_size])
            else:
                cdfs.append([cdfs[-1][0] + node.forward_compute_time + node.backward_compute_time,
                             cdfs[-1][1] + activation_size,
                             cdfs[-1][2] + node.parameter_size])

        for node in topological_ordering:
            activation_size = node.activation_size
            if isinstance(activation_size, list):
                activation_size = sum(activation_size)
            raw_values.append((node.forward_compute_time + node.backward_compute_time,
                               activation_size, node.parameter_size))
        self.plot_bar_graph(raw_values, None, True, "%s.pdf", output_directory)

        for node in topological_ordering:
            activation_size = node.activation_size
            if isinstance(activation_size, list):
                activation_size = sum(activation_size)
            pdfs.append(((node.forward_compute_time + node.backward_compute_time) / (cdfs[-1][0] / 100.0),
                         activation_size / (cdfs[-1][1] / 100.0),
                         node.parameter_size / (cdfs[-1][2] / 100.0)))
        self.plot_bar_graph(pdfs, "PDF (\%)", False, "%s_pdf.pdf", output_directory)

        for i in range(len(cdfs)):
            cdfs[i][0] /= (cdfs[-1][0] / 100.0)
            cdfs[i][1] /= (cdfs[-1][1] / 100.0)
            cdfs[i][2] /= (cdfs[-1][2] / 100.0)
        self.plot_cdfs(cdfs, output_directory)

'''
节点定义如下，里面就是从profile获取到的结构，比如：
    forward_compute_time : 前向传播时间；
    backward_compute_time ：反向传播时间；
    activation_size : 激活值大小；
    parameter_size : 参数大小；
'''
class Node(object):
    def __init__(self, node_id, node_desc="", forward_compute_time=0.0,
                 backward_compute_time=0.0, activation_size=0.0, parameter_size=0.0,
                 stage_id=None):
        self.node_id = node_id
        self.node_desc = node_desc
        self.forward_compute_time = forward_compute_time
        self.backward_compute_time = backward_compute_time
        self.activation_size = activation_size
        self.parameter_size = parameter_size
        self.stage_id = stage_id
        self.depth = None
        self.height = None

    def set_stage_id(self, stage_id):
        self.stage_id = stage_id

    def __str__(self):
        stage_id_str = " -- stage_id=%d" % self.stage_id if self.stage_id is not None else ""
        node_desc = self.node_desc.replace('\n', "")
        activation_size = ("%s" % self.activation_size).replace(", ", "; ")
        return "%s -- %s -- forward_compute_time=%.3f, backward_compute_time=%.3f, activation_size=%s, parameter_size=%.3f%s" % (
            self.node_id, node_desc, self.forward_compute_time, self.backward_compute_time,
            activation_size, self.parameter_size, stage_id_str)

    # 构建节点具体代码如下：
    @staticmethod
    def from_str(node_str):
        node_str_tokens = node_str.strip().split(" -- ")
        node_id = node_str_tokens[0]   # 节点名字
        node_desc = node_str_tokens[1] # 节点描述
        node_metadata = node_str_tokens[2] # 元数据
        stage_id = None
        if len(node_str_tokens) > 3:
            stage_id = int(node_str_tokens[3].split("=")[1]) # 阶段信息
        [forward_compute_time, backward_compute_time, activation_size, parameter_size] = node_metadata.split(", ")
        forward_compute_time = float(forward_compute_time.split("=")[1])  # 前向传播计算时间
        backward_compute_time = float(backward_compute_time.split("=")[1])# 后向传播计算时间
        if "[" in activation_size:
            activation_size = activation_size.split("=")[1]   # 激活值大小
            activation_size = sum([float(x) for x in activation_size.lstrip("[").rstrip("]").split("; ")])
        else:
            activation_size = float(activation_size.split("=")[1])
        parameter_size = float(parameter_size.split("=")[1])  # 参数大小
        # 构建节点
        return Node(node_id, node_desc, forward_compute_time=forward_compute_time,
                    backward_compute_time=backward_compute_time, activation_size=activation_size,
                    parameter_size=parameter_size, stage_id=stage_id)
# 2.3 反链
# 在有向无环图中，有如下的一些概念：
#
#     链 ：一条链是一些点的集合，在此链上的任意两个点x, y，满足以下条件：或者 x 能到达 y ，或者 y 能到达 x 。
#         也可以认为是某一个偏序集S的全序子集（所谓全序是指其中任意两个元素可以比较）
#
#     反链 ：一条反链也是一些点的集合，在此链上任意两个点x, y，满足如下条件： x 不能到达 y，且 y 也不能到达 x。
#         也可以认为是某一个偏序集S的子集，其中任意两个元素不可比较。
#
# 在PipeDream的图数据结构之中，也有反链的概念。反链节点定义如下：

#因为此处过于复杂，所以我们会在下面用一节专门分析。3.1 main函数入口  这里再取出反链节点定义如下，可以看出来和代码对应关系。
class AntichainNode(Node):
    def __init__(self, node_id, antichain, node_desc=""):
        self.antichain = antichain
        self.output_activation_size = 0.0
        super(AntichainNode, self).__init__(node_id, node_desc)

    def __str__(self):
        return "%s -- %s" % (self.node_id, self.antichain)
