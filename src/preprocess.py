import argparse
import numpy as np
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

RATING_FILE_NAME = dict({'music': 'user_artists.dat', 'book': 'BX-Book-Ratings.csv', 'movie': 'ratings.csv'})
SEP = dict({'music': '\t', 'book': ';', 'movie': ','})
THRESHOLD = dict({'music': 0, 'book': 0, 'movie': 4})


def read_item_index_to_entity_id_file(dataset):
    file = '../data/' + dataset + '/item_index2entity_id.txt'
    logging.info("reading item index to entity id file: %s", file)
    item_index_old2new = dict()
    entity_id2index = dict()

    new2_item_index_old = dict()
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]  #item_index
        satori_id = line.strip().split('\t')[1]   #entity_id
        item_index_old2new[item_index] = i  #item_index=》new_item_index的索引
        entity_id2index[satori_id] = i  #entity_id=》entity_index的索引
        i += 1
    list_new2item_index = []
    file_new2item_index = '../big_graph/' + dataset + '/new2item_index.txt'
    writer_file_new2item_index = open(file_new2item_index, 'w', encoding='utf-8')
    for key in item_index_old2new.keys():
        # list_new2item_index.append((item_index_old2new[key],key))
        # new2_item_index_old[item_index_old2new[key]] = key
        writer_file_new2item_index.write('%d\t%d\n' % (item_index_old2new[key], int(key)))
    writer_file_new2item_index.close()
    return item_index_old2new, entity_id2index  #[1,item1,entity1][2,item2,entity2]....


def convert_rating(dataset, item_index_old2new, entity_id2index):
    file = '../data/' + dataset + '/' + RATING_FILE_NAME[dataset]
    logging.info("reading rating file: %s", file)
    
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()
    
    for line in open(file, encoding='ISO-8859-1').readlines()[1:]:
        array = line.strip().split(SEP[dataset])  #一行分别为 userID	artistID	weight
        # remove prefix and suffix quotation marks for BX dataset
        if dataset == 'book':
            array = list(map(lambda x: x[1:-1], array))
        item_index_old = array[1]  #item的旧下标
        
        # if the item is not in the final item set
        if item_index_old not in item_index_old2new.keys():  
            continue
        item_index = item_index_old2new[item_index_old]  #单纯的下标变换
        
        user_index_old = array[0]#user的旧下标
        rating = float(array[2])  #评分
        if rating >= THRESHOLD[dataset]:  #正负训练集
            if user_index_old not in user_pos_ratings:#不在就创建
                user_pos_ratings[user_index_old] = set() #创建一个字典项
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    write_file = '../data/' + dataset + '/ratings_final.txt'
    logging.info("converting rating file to: %s", write_file)
    writer = open(write_file, 'w', encoding='utf-8')
    writer_idx = 0
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():#user下标以及对应的正评分item列表
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]#这里和上面的item一样得到新的user下标
        for item in pos_item_set:
            writer_idx += 1
            writer.write('%d\t%d\t1\n' % (user_index, item))    #rating_final里面就是（user，item，rating（0 or 1））
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer_idx += 1
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    
    logging.info("number of users: %d", user_cnt)
    logging.info("number of items: %d", len(item_set))
    logging.info("number of interactions: %d", writer_idx)
    return user_cnt+len(item_set)

def convert_kg(dataset, entity_id2index):
    file = '../data/' + dataset + '/' + 'kg.txt'
    logging.info("reading kg file: %s", file)
    file_relation_index = '../big_graph/' + dataset + '/' + 'relation_index.txt'
    write_file = '../data/' + dataset + '/' + 'kg_final.txt'
    logging.info("converting kg file to: %s", write_file)
    
    entity_cnt = len(entity_id2index)
    relation_id2index = dict()

    index2entity_id = dict()
    relation_cnt = 0
    
    writer = open(write_file, 'w', encoding='utf-8')
    writer_relation = open(file_relation_index, 'w', encoding='utf-8')
    writer_idx = 0
    for line in open(file, encoding='utf-8').readlines():
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]
        insting = 'e'+str(head_old)+'e'+str(tail_old)
        writer.write('%d\t%d\t%d\n' % (head, relation, tail))#head和tail都是下标
        writer_relation.write('%s\t%s\n' % (insting,relation_old))
        writer_idx += 1
    writer.close()
    
    logging.info("number of entities (containing items): %d", entity_cnt)
    logging.info("number of relations: %d", relation_cnt)
    logging.info("number of triples: %d", writer_idx)
    #这里写index到id
    file_index2entity_id = '../big_graph/' + dataset + '/index2entity_id.txt'
    writer_file_index2entity_id = open(file_index2entity_id, 'w', encoding='utf-8')
    for key in entity_id2index.keys():
        # index2entity_id[entity_id2index[key]] = key
        writer_file_index2entity_id.write('%d\t%d\n' % (entity_id2index[key], int(key)))
    writer_file_index2entity_id.close()
    return entity_id2index, relation_id2index,entity_cnt


if __name__ == '__main__':
    # we use the same random seed as RippleNet, KGCN, KGNN-LS for better comparison
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='music', help='which dataset to preprocess')
    args = parser.parse_args()

    item_index_old2new, entity_id2index = read_item_index_to_entity_id_file(args.dataset)
    n_user_and_item = convert_rating(args.dataset, item_index_old2new, entity_id2index)
    entity_id2index, relation_id2index ,entity_cnt= convert_kg(args.dataset, entity_id2index)

    logging.info("data %s preprocess: done.",args.dataset)
