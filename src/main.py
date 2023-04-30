import argparse
#import torch
import numpy as np
from data_loader import load_data
from train import train
import os
from train import show_and_explain
from explanation import make_graph
import argparse
#import torch
import json
import numpy as np
from data_loader import load_data
from train import train
from explanation import make_graph

import socket
import sys

import threading
import json
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='music', help='which dataset to use (music, book, movie, restaurant)')
parser.add_argument('--n_epoch', type=int, default=20, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--n_layer', type=int, default=3, help='depth of layer')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')

parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
parser.add_argument('--user_triple_set_size', type=int, default=16, help='the number of triples in triple set of user')
parser.add_argument('--item_triple_set_size', type=int, default=16, help='the number of triples in triple set of item')
parser.add_argument('--agg', type=str, default='concat', help='the type of aggregator (sum, pool, concat)')

parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
parser.add_argument('--random_flag', type=bool, default=False,  help='whether using random seed or not')
parser.add_argument('--train', type=bool, default=True, help='whether have trained or not')
parser.add_argument('--userid', type=int, default=101,  help='whether using random seed or not')
args = parser.parse_args()


# def set_random_seed(np_seed, torch_seed):
#     np.random.seed(np_seed)
#     torch.manual_seed(torch_seed)
#     torch.cuda.manual_seed(torch_seed)
#     torch.cuda.manual_seed_all(torch_seed)
#
# if not args.random_flag:
#     set_random_seed(136, 2022)

def ggson(user_id):

    data_info = load_data(args)
    userid = user_id
    print(user_id)
    # train(args, data_info)

    getuser = userid[0:userid.find('-')]
    gettop = userid[userid.find('-')+1:len(userid)]
    print('fenge',getuser,gettop)
    path = '../big_graph/' + args.dataset + '/' + str(getuser)

    if args.train:
        if os.path.exists(path):
            # show_and_explain(args, data_info, user_id)
            make_graph(args.dataset, getuser,gettop)

        else:
            show_and_explain(args, data_info, getuser)
            make_graph(args.dataset, getuser,gettop)
    else:
        train(args, data_info)
        show_and_explain(args, data_info, getuser)


def main():
        # 创建服务器套接字
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 获取本地主机名称
        host = socket.gethostname()
        # 设置一个端口
        port = 12345
        # 将套接字与本地主机和端口绑定
        serversocket.bind((host, port))
        # 设置监听最大连接数
        serversocket.listen(5)
        # 获取本地服务器的连接信息
        myaddr = serversocket.getsockname()
        print("服务器地址:%s" % str(myaddr))
        # 循环等待接受客户端信息
        while True:
            # 获取一个客户端连接
            clientsocket, addr = serversocket.accept()
            print("连接地址:%s" % str(addr))
            try:
                t = ServerThreading(clientsocket)  # 为每一个请求开启一个处理线程
                t.start()
                pass
            except Exception as identifier:
                print(identifier)
                pass
            pass
        serversocket.close()
        pass


class ServerThreading(threading.Thread):
    # words = text2vec.load_lexicon()
    def __init__(self, clientsocket, recvsize=1024 * 1024, encoding="utf-8"):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        pass

    def run(self):
        print("开启线程.....")
        try:
            # 接受数据
            msg = ''
            while True:
                # 读取recvsize个字节
                rec = self._socket.recv(self._recvsize)
                # print(rec)
                # 解码
                msg += rec.decode(self._encoding)
                # print(msg)
                # 文本接受是否完毕，因为python socket不能自己判断接收数据是否完毕，
                # 所以需要自定义协议标志数据接受完毕
                if msg.strip().endswith('over'):
                    msg = msg[:-4]
                    break
            # 解析json格式的数据
            print(msg)
            re = str(msg)
            print(re)
            # print(re)
            #print("接收到信息：" + str(re) + "准备执行gson(" + str(re) + ")")
            print('this1')
            ggson(str(re))
            print('this2')
            pass
        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            print(identifier)
            pass
        finally:
            self._socket.close()
        print("任务结束.....")

        pass

    def __del__(self):

        pass


if __name__ == "__main__":
    main()