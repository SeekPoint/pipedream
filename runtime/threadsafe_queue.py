# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import threading

"""
Implementation of a thread-safe queue with one producer and one consumer.
"""
'''
0x04 功能函数
以下功能函数就是最终被使用完成 流水线 RPC 逻辑的函数。

这里有一个通过queue完成的解耦合：

    recv 和 send 就会对于 queue 进行操作，往queue里面添加或者提取张量。
    助手线程会调用 _recv 和 _send 对 queue 进行操作。
    
所以我们要先看看这个Queue的实现，可以看到，无论是 add 还是 remove，都使用了 threading.Condition，
就说明几个线程可以在 Queue 上通过 add / remove 实现等待，阻塞，即生产者和消费者。
'''
class Queue:
    def __init__(self):
        self.queue = []
        self.cv = threading.Condition()

    def add(self, tensor):
        self.cv.acquire()
        self.queue.append(tensor)
        self.cv.notify()
        self.cv.release()

    def remove(self):
        self.cv.acquire()
        while len(self.queue) == 0:
            self.cv.wait()
        tensor = self.queue.pop(0)
        self.cv.release()
        return tensor
