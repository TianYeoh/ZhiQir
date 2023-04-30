from py2neo import *


#########################################

# from train import show_and_explain
# import argparse
# from data_loader import load_data

# parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--dataset', type=str, default='music', help='which dataset to use (music, book, movie, restaurant)')
# parser.add_argument('--n_epoch', type=int, default=20, help='the number of epochs')
# parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
# parser.add_argument('--n_layer', type=int, default=3, help='depth of layer')
# parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
# parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
#
# parser.add_argument('--dim', type=int, default=64, help='dimension of entity and relation embeddings')
# parser.add_argument('--user_triple_set_size', type=int, default=16, help='the number of triples in triple set of user')
# parser.add_argument('--item_triple_set_size', type=int, default=16, help='the number of triples in triple set of item')
# parser.add_argument('--agg', type=str, default='concat', help='the type of aggregator (sum, pool, concat)')
#
# parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
# parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
# parser.add_argument('--random_flag', type=bool, default=False,  help='whether using random seed or not')
# parser.add_argument('--train', type=bool, default=True, help='whether have trained or not')
# parser.add_argument('--userid', type=int, default=101,  help='whether using random seed or not')
# args = parser.parse_args()
##########################################
def make_graph(dataset,userid,top):

    graph_data = '../big_graph/' + dataset + '/' + userid + '/' + 'explanation_graph_'+ top +'.txt'  # kg中的   head  relation  tail
    # id2name = '../big_graph/' + dataset + '/' + str(userid) + '/' + 'id2name.json'
    # writer_id2name = open(id2name, 'w', encoding='utf-8')
    # writer_id2name.write("{ ")
    artist = '../data/' + dataset + '/' + 'artists.txt'
    name_index = dict()
    total_list = []

    for line in open(artist, encoding='utf-8').readlines():
        left = line.strip().split('\t')[0]
        right = line.strip().split('\t')[1]
        name_index[str(left)] = str(right)


    graph = Graph('http://localhost:7474', auth=('neo4j', '13868593638sgr'))
    graph.delete_all()
    relation_index = '../big_graph/' + dataset + '/' + 'relation_index.txt'
    relation = dict()
    for line in open(relation_index, encoding='utf-8').readlines():
        relation[line.strip().split('\t')[0]] = line.strip().split('\t')[1]
    tx = graph.begin()
    matcher = NodeMatcher(tx)
    sum_len = 0
    if userid=='999':
        for line in open(graph_data, encoding='utf-8').readlines():
            left = line.strip().split('\t')[0]
            right = line.strip().split('\t')[1]
            if left not in total_list:
                total_list.append(str(left))
            if right not in total_list:
                total_list.append(str(right))
            if left[0] == 'u' and right[0] == 'i':
                name_a = str(left)
                name_b = str(right)
                name_ab = str('交互')

                list_a = list(matcher.match('user', name=name_a[1:len(name_a)]))
                list_b = list(matcher.match('item', name=name_b[1:len(name_b)]))

                if len(list_a) > 0:
                    a = list_a[0]
                else:
                    a = Node('user', name=name_a[1:len(name_a)])
                    tx.create(a)
                if len(list_b) > 0:
                    b = list_b[0]
                else:
                    b = Node('item', name=name_b[1:len(name_b)])
                    tx.create(b)
                ab = Relationship(a, name_ab, b)
                tx.create(ab)
            elif left[0] == 'e' and right[0] == 'e':
                name_a = str(left)
                name_b = str(right)
                name_total = name_a + name_b
                if name_total in relation.keys():
                    name_ab = relation[name_total]
                else:
                    name_total = name_b + name_a
                    name_ab = relation[name_total]
                list_a = list(matcher.match('entity', name=name_a[1:len(name_a)]))
                list_b = list(matcher.match('entity', name=name_b[1:len(name_b)]))
                if len(list_a) > 0:
                    a = list_a[0]
                else:
                    a = Node('entity', name=name_a[1:len(name_a)])
                    tx.create(a)
                if len(list_b) > 0:
                    b = list_b[0]
                else:
                    b = Node('entity', name=name_b[1:len(name_b)])
                    tx.create(b)
                ab = Relationship(a, name_ab, b)
                tx.create(ab)
            elif left[0] == 'i' and right[0] == 'e':
                name_a = str(left)
                name_b = str(right)
                name_total = name_a + name_b
                if name_total in relation.keys():
                    name_ab = relation[name_total]
                else:
                    name_total = name_b + name_a
                    name_ab = relation[name_total]
                list_a = list(matcher.match('item', name=name_a[1:len(name_a)]))
                list_b = list(matcher.match('entity', name=name_b[1:len(name_b)]))
                if len(list_a) > 0:
                    a = list_a[0]
                else:
                    a = Node('item', name=name_a[1:len(name_a)])
                    tx.create(a)
                if len(list_b) > 0:
                    b = list_b[0]
                else:
                    b = Node('entity', name=name_b[1:len(name_b)])
                    tx.create(b)
                ab = Relationship(a, name_ab, b)
                tx.create(ab)
            elif left[0] == 'e' and right[0] == 'i':
                name_a = str(left)
                name_b = str(right)
                name_total = name_a + name_b
                if name_total in relation.keys():
                    name_ab = relation[name_total]
                else:
                    name_total = name_b + name_a
                    name_ab = relation[name_total]
                list_a = list(matcher.match('entity', name=name_a[1:len(name_a)]))
                list_b = list(matcher.match('item', name=name_b[1:len(name_b)]))
                if len(list_a) > 0:
                    a = list_a[0]
                else:
                    a = Node('entity', name=name_a[1:len(name_a)])
                    tx.create(a)
                if len(list_b) > 0:
                    b = list_b[0]
                else:
                    b = Node('item', name=name_b[1:len(name_b)])
                    tx.create(b)
                ab = Relationship(a, name_ab, b)
                tx.create(ab)
        graph.commit(tx)
    else:
        for line in open(graph_data, encoding='utf-8').readlines():
            left = line.strip().split('\t')[0]
            right = line.strip().split('\t')[1]
            if left not in total_list:
                total_list.append(str(left))
            if right not in total_list:
                total_list.append(str(right))
            if left[0] == 'u' and right[0] == 'i':
                name_a = str(left)
                name_b = str(right)
                name_ab = str('user.interact')

                list_a = list(matcher.match('user', name=name_a))
                list_b = list(matcher.match('item', name=name_b))

                if len(list_a) > 0:
                    a = list_a[0]
                else:
                    a = Node('user', name=name_a)
                    tx.create(a)
                if len(list_b) > 0:
                    b = list_b[0]
                else:
                    b = Node('item', name=name_b)
                    tx.create(b)
                ab = Relationship(a, name_ab, b)
                tx.create(ab)
            elif left[0] == 'e' and right[0] == 'e':
                name_a = str(left)
                name_b = str(right)
                name_total = name_a + name_b
                if name_total in relation.keys():
                    name_ab = relation[name_total]
                else:
                    name_total = name_b + name_a
                    name_ab = relation[name_total]
                list_a = list(matcher.match('entity', name=name_a))
                list_b = list(matcher.match('entity', name=name_b))
                if len(list_a) > 0:
                    a = list_a[0]
                else:
                    a = Node('entity', name=name_a)
                    tx.create(a)
                if len(list_b) > 0:
                    b = list_b[0]
                else:
                    b = Node('entity', name=name_b)
                    tx.create(b)
                ab = Relationship(a, name_ab, b)
                tx.create(ab)
            elif left[0] == 'i' and right[0] == 'e':
                name_a = str(left)
                name_b = str(right)
                name_ab = str('item.entity')
                list_a = list(matcher.match('item', name=name_a))
                list_b = list(matcher.match('entity', name=name_b))
                if len(list_a) > 0:
                    a = list_a[0]
                else:
                    a = Node('item', name=name_a)
                    tx.create(a)
                if len(list_b) > 0:
                    b = list_b[0]
                else:
                    b = Node('entity', name=name_b)
                    tx.create(b)
                ab = Relationship(a, name_ab, b)
                tx.create(ab)
            elif left[0] == 'e' and right[0] == 'i':
                name_a = str(left)
                name_b = str(right)
                name_ab = str('entity.item')
                list_a = list(matcher.match('entity', name=name_a))
                list_b = list(matcher.match('item', name=name_b))
                if len(list_a) > 0:
                    a = list_a[0]
                else:
                    a = Node('entity', name=name_a)
                    tx.create(a)
                if len(list_b) > 0:
                    b = list_b[0]
                else:
                    b = Node('item', name=name_b)
                    tx.create(b)
                ab = Relationship(a, name_ab, b)
                tx.create(ab)
        graph.commit(tx)

if __name__ == "__main__":
    make_graph('music','999','0')