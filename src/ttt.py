#
# length = 0
# def searchGraph(graph,start,end):
#     results = []
#     generatePath(graph,[start],end,results)
#     results.sort(key=lambda x:len(x))
#     return results
#
# def generatePath(graph,path,end,results):
#     global length
#     state = path[-1]
#     length = len(path)
#     if state == end:
#         print(length,path)
#         results.append(path)
#         length = 0
#     elif length<10:
#         if state in graph.keys():
#             for arc in graph[state]:
#                 if arc not in path:
#                     generatePath(graph, path + [arc], end, results)
# # def searchGraph(graph,start,end):
# #     results = []
# #     generatePath(graph,[start],end,results)
# #     results.sort(key=lambda x:len(x))
# #     return results
# #
# # def generatePath(graph,path,end,results):
# #     state = path[-1]
# #     if state == end:
# #         results.append(path)
# #     else:
# #         for arc in graph[state]:
# #             if arc not in path:
# #                 generatePath(graph,path + [arc],end,results)
#
# if __name__ == '__main__':
#     # Graph = {'A':['B','C','D'],
#     #          'B':['E'],
#     #          'C':['D','F'],
#     #          'D':['B','E','G'],
#     #          'E':[],
#     #          'F':['D','G'],
#     #          'G':['E']}
#     # graph= dict()
#     # graph['1'] = ['3','4']
#     # graph['2'] = ['5']
#     # graph['3'] = ['6']
#     # graph['4'] = ['7']
#     # graph['5'] = ['8']
#     # graph['6'] = ['9']
#     # graph['7'] = ['10']
#     # graph['8'] = ['11']
#     # graph['9'] = ['11']
#     # graph['10'] = ['11']
#     # graph['11'] = []
#     graph_list = []
#     graph_data = '../big_graph/' + 'music' + '/' + '102' + '/' + '102.txt'
#     graph = dict()
#
#     for line in open(graph_data, encoding='utf-8').readlines():
#         left = line.strip().split('\t')[0]
#         right = line.strip().split('\t')[1]
#         left_str = str(left)
#         right_str = str(right)
#         graph_list.append((left_str,right_str))
#     for i in range(1,len(graph_list)):
#         if i!=len(graph_list)-1:
#             if graph_list[i][0] not in graph.keys():
#                 graph[graph_list[i][0]] = []
#                 graph[graph_list[i][0]].append(graph_list[i][1])
#             elif graph_list[i][0] in graph.keys():
#                 graph[graph_list[i][0]].append(graph_list[i][1])
#         else:
#             start = graph_list[i][0]
#             end = graph_list[i][1]
#     print(graph)
#     print(start,end)
#     r = searchGraph(graph,start,end)
#
#     # print('*************************')
#     # print(' path A to E')
#     # print('*************************')
#     #
#     if not r:
#         print('很遗憾，无此路径...')
#     else:
#         for i in r:
#             print(i)
a = '123'
b = a[1:len(a)-1]
print(b)

