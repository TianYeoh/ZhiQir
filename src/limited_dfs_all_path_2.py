
from graphviz import Digraph
class total2():
    def __init__(self):
        dataset = 'music'
        self.dot = Digraph(comment='Gragh2Print')  # 新建图
        self.dot.edge_attr.update(arrowhead='none')  # 去除箭头
        self.dot.graph_attr['rankdir'] = 'LR'  # 左右顺序排列

        self.edgeLinks = dict()
        self.size = 0
        self.outlist = []

        self.stack = []
        self.total = 0
        self.nodeNumber = 0
        self.length = 0

    def printRoute(self,stackList):  # 输出路径

        self.nodeNumber += 1
        self.dot.node(str(self.nodeNumber), stackList[0])
        for node in stackList[1:]:
            self.nodeNumber += 1
            self.dot.node(str(self.nodeNumber), node)
            self.dot.edge(str(self.nodeNumber - 1), str(self.nodeNumber))

    def addEdge(self,a, b):

        if a not in self.edgeLinks: self.edgeLinks[a] = set()
        if b not in self.edgeLinks: self.edgeLinks[b] = set()
        self.edgeLinks[a].add(b)
        self.edgeLinks[b].add(a)

    def loadGraph(self,fileName):
        try:
            f = open(fileName, 'r')
        except:
            print("打开文件失败, 请检查文件名是否正确或程序是否有权限访问")
        self.size, edgeCount = map(int, f.readline().split())
        print("节点:", self.size, "边数:", edgeCount)
        for i in range(1, self.size + 1):
            self.dot.node(str(i), str(i))
        for i in range(edgeCount):
            a, b = f.readline().split()
            self.addEdge(a, b)
            self.dot.edge(a, b)
        re = f.readline()
        f.close()
        return re

    def subrelation(self,a):
        list = []
        for i in range(len(a)):
            if i < len(a) - 1:
                left = a[i]
                right = a[i + 1]
                list.append((left, right))
        return list

    def findAllRoutes(self,start, end):


        self.stack.append(start)
        self.length = len(self.stack)
        if start == end:
            print("找到路径:", self.stack, '长度为:', len(self.stack))
            # print(subrelation(stack))
            self.outlist.append(self.subrelation(self.stack))


            self.total += 1
            self.length = 0
            self.printRoute(self.stack)
            self.stack.pop()
        elif self.length < 10:
            for nextPoint in self.edgeLinks[start]:  # 相邻点
                if nextPoint not in self.stack:
                    self.findAllRoutes(nextPoint, end)
            self.stack.pop()
        elif self.length >= 10:
            self.stack.pop()

    def rmRoute2Itself(self,start):
        for point in self.edgeLinks:
            if point != start and start in self.edgeLinks[point]:
                self.edgeLinks[point].remove(start)

    def contrlstring(self,string):
        stringout = string[0:len(string) - 1]
        return stringout

    def body_2(self,file):

        a, b = self.loadGraph(file).split()
        self.rmRoute2Itself(a)
        self.nodeNumber = self.size + 1
        self.findAllRoutes(a, b)
        return self.outlist, self.total

    def dis(self):
        self.edgeLinks = dict()
        self.size = 0
        self.outlist = []

        self.stack = []
        self.total = 0
        self.nodeNumber = 0
        self.length = 0
    def __del__(self):
        print("meile")




