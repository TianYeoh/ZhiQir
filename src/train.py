import os
import numpy as np
import torch
import torch.nn as nn 
from sklearn.metrics import roc_auc_score, f1_score
from model import CKAN
import logging
from delete_recommend_list import load_data_delete
from limited_dfs__all_path import total
from limited_dfs_all_path_2 import total2
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def train(args, data_info):
    logging.info("================== training CKAN ====================")
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    user_triple_set = data_info[5]
    # print('user_triple_set',user_triple_set)
    item_triple_set = data_info[6]
    model, optimizer, loss_func = _init_model(args, data_info)
    # print('entity',n_entity)
    for step in range(args.n_epoch):
        np.random.shuffle(train_data)
        start = 0
        while start < train_data.shape[0]:
            labels = _get_feed_label(args, train_data[start:start + args.batch_size, 2])
            scores = model(*_get_feed_data(args, train_data, user_triple_set, item_triple_set, start, start + args.batch_size))
            loss = loss_func(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            start += args.batch_size
        eval_auc, eval_f1 = ctr_eval(args, model, eval_data, user_triple_set, item_triple_set)
        test_auc, test_f1 = ctr_eval(args, model, test_data, user_triple_set, item_triple_set)
        ctr_info = 'epoch %.2d    eval auc: %.4f f1: %.4f    test auc: %.4f f1: %.4f'
        logging.info(ctr_info, step, eval_auc, eval_f1, test_auc, test_f1)
        # if args.show_topk:
        #     topk_eval(args, model, train_data, test_data, user_triple_set, item_triple_set)
    save_path = '../model_save/' + args.dataset + '.pt'
    torch.save(model, save_path)
def topk_eval(args, model, train_data, test_data, user_triple_set, item_triple_set,userid):

    # logging.info('calculating recall ...')
    k_list = [5, 10, 20, 50, 100]
    recall_list = {k: [] for k in k_list}

    item_set = set(train_data[:,1].tolist() + test_data[:,1].tolist())#1维全是item
    train_record = _get_user_record(args, train_data, True)#
    test_record = _get_user_record(args, test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    user_num = 100
    # if len(user_list) > user_num:#这里是随机选择100个
    #     np.random.seed()
    #     user_list = np.random.choice(user_list, size=user_num, replace=False)
    user_list = user_list[0:user_num]

    model.eval()

    every_user_topk_list = dict()
    user = int(userid)
    # print(userid)
    # print("asdasd")
    # print(train_record[int(userid)])
    # print(train_record[int(userid)].dtype)
    # print(item_set - set(train_record[user]))
    # print("safjoifgjfgh")
    test_item_list = list((item_set - set(train_record[int(user)])))


    item_score_map = dict()
    top_n = 0
    out_list = []
    start = 0
    while start + args.batch_size <= len(test_item_list):
        items = test_item_list[start:start + args.batch_size]
        input_data = _get_topk_feed_data(user, items)


        #todo ls
        scores = model(*_get_feed_data(args, input_data, user_triple_set, item_triple_set, 0, args.batch_size))

        for item, score in zip(items, scores):
            item_score_map[item] = score
        start += args.batch_size
    # padding the last incomplete mini-batch if exists

    if start < len(test_item_list):  # 对剩余小于一个batch的做处理
        res_items = test_item_list[start:] + [test_item_list[-1]] * (args.batch_size - len(test_item_list) + start)
        input_data = _get_topk_feed_data(user, res_items)
        scores = model(*_get_feed_data(args, input_data, user_triple_set, item_triple_set, 0, args.batch_size))
        for item, score in zip(res_items, scores):
            item_score_map[item] = score

    item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
    for i in item_score_pair_sorted:
        if top_n == 10:
            break
        else:
            out_list.append((i[0], i[1].item()))
            top_n += 1
    every_user_topk_list[user] = out_list
    item_sorted = [i[0] for i in item_score_pair_sorted]

    for k in k_list:
        hit_num = len(set(item_sorted[:k]) & set(test_record[user]))
        recall_list[k].append(hit_num / len(set(test_record[user])))
    model.train()
    recall = [np.mean(recall_list[k]) for k in k_list]
    _show_recall_info(zip(k_list, recall))
    return every_user_topk_list

def show_and_explain(args,data_info,userid):
    userid = int(userid)
    load_path = '../model_save/' + args.dataset + '.pt'
    kg_file = '../data/' + args.dataset + '/' + 'kg.txt'
    index2entity_id = dict()
    new2item_index = dict()
    item2entity = dict()
    file_index2entity_id = '../big_graph/' + args.dataset + '/index2entity_id.txt'
    file_new2item_index = '../big_graph/' + args.dataset + '/new2item_index.txt'
    file_item2entity = '../data/' + args.dataset + '/item_index2entity_id.txt'
    for line in open(file_item2entity, encoding='utf-8').readlines():
        item = line.strip().split('\t')[0]
        eneity = line.strip().split('\t')[1]
        item2entity[item] = eneity
    for line_index2entity_id in open(file_index2entity_id, encoding='utf-8').readlines():
        index = line_index2entity_id.strip().split('\t')[0]
        entity_id = line_index2entity_id.strip().split('\t')[1]
        index2entity_id[index] = entity_id
    for line_new2item_index in open(file_new2item_index, encoding='utf-8').readlines():
        new = line_new2item_index.strip().split('\t')[0]
        item_index = line_new2item_index.strip().split('\t')[1]
        new2item_index[new] = item_index

    model = torch.load(load_path)
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    user_triple_set = data_info[5]
    item_triple_set = data_info[6]
    user_history_item_dict = data_info[7]#eval_indices,test_indices,
    eval_indices =data_info[8]
    test_indices = data_info[9]
    train_indices = data_info[10]
    list_user = user_history_item_dict.keys()
    list_item = item_triple_set.keys()
    n_node = len(list_item)+len(list_user)+n_entity

    recommend_list = topk_eval(args, model, train_data, test_data, user_triple_set, item_triple_set, userid)

    start_user = userid
    str_start_user = 'u' + str(userid)
    topk=10

    for j in range(topk):
        print('create',j)
        traget_item = recommend_list[userid][j][0]
        print('推荐对象',traget_item)
        ################################################################
        str_traget_item = 'i' + str(new2item_index[str(traget_item)])
        big_graph_l_dfs = []
        big_graph = []
        big_graph_temp = []
        big_graph_explanation = []
        for item in user_history_item_dict[userid]:
            edge_head_ui = 'u' + str(userid)
            edge_tail_ui = 'i' + str(new2item_index[str(item)])
            big_graph.append((edge_head_ui, edge_tail_ui))
        for line in open(kg_file, encoding='utf-8').readlines():
            array = line.strip().split('\t')
            head_old = index2entity_id[array[0]]
            relation_old = array[1]
            tail_old = index2entity_id[array[2]]
            edge_head_ee = 'e' + str(head_old)
            relation = relation_old

            edge_tail_ee = 'e' + str(tail_old)
            big_graph.append((edge_head_ee, edge_tail_ee))
            big_graph_temp.append((edge_head_ee, edge_tail_ee))
        # 这里放item->entity
        for key_item in item2entity.keys():
            catch_entity = item2entity[key_item]
            edge_head_ie = 'i' + str(key_item)
            edge_tail_ie = 'e' + str(catch_entity)
            big_graph.append((edge_head_ie, edge_tail_ie))
        big_graph_l_dfs.append((n_node, len(big_graph)))
        for edge in big_graph:
            big_graph_l_dfs.append(edge)

        big_graph_l_dfs.append((str_start_user, str_traget_item))
# 到这里得到用于深搜的形式
#######################################################################
# 这里是userid.txt（未经过dfs）
        path = '../big_graph/' + args.dataset + '/' + str(start_user)
        if os.path.isdir(path):
            big_graph_file = path + '/' + str(start_user) + '_' + str(j) + '.txt'
        else:
            os.mkdir(path)
            big_graph_file = path + '/' + str(start_user)+ '_' + str(j) + '.txt'
        writer = open(big_graph_file, 'w', encoding='utf-8')
        for each_one in big_graph_l_dfs:
            left = each_one[0]
            right = each_one[1]
            writer.write('%s\t%s\n' % (left, right))
        writer.close()
        print(big_graph_file)
##########################################################
# 得到了未经过删除的图,只是把user-item对从整个图中分离出来还没进行删减子图操作
        mid = total()
        each_pair,num_total = mid.body('../big_graph/' + args.dataset + '/' + str(userid) + '/' +str(userid)+ '_' + str(j) + '.txt')
    # mid.dis()
    #这里进行limited_dfs
        print('num_not_explanation:',num_total)
        file_total_graph = '../big_graph/' + args.dataset + '/' + str(userid) + '/'+ 'total_graph'+ '_' + str(j)+'.txt'
        writer_total_graph = open(file_total_graph, 'w', encoding='utf-8')
        for path in each_pair:
            for pair in path:
                writer_total_graph.write('%s\t%s\n' % (pair[0], pair[1]))
        writer_total_graph.close()
#########################################################################

    # print('recommand item',recommend_list[userid][0][0])#这个是推荐目标（第一高的评分item）
        recommend_all_list = dict()
        recommend_all_list['none'] = recommend_list
        for k in user_history_item_dict[userid]:#每次删除k
            data_info_delete = load_data_delete(args,k,eval_indices,test_indices,train_indices)
            user_triple_set_delete = data_info_delete[5]
            item_triple_set_delete = data_info_delete[6]
            recommend_list_each_k = topk_eval(args, model, train_data, test_data, user_triple_set_delete, item_triple_set_delete, userid)
            recommend_all_list[k] = recommend_list_each_k
    #这里就得到所有删除后的推荐列表结果
    # for i in recommend_all_list.keys():
    #     print(recommend_all_list[i][userid])
        test = []

        for i in user_history_item_dict[userid]:
            test.append(new2item_index[str(i)])
    # print(2,test)
        list_out = delete_graph(recommend_all_list,userid,num_total)#这里就是要删除的user交互节点
###########################################################
    # print(3,new2item_index[str(list_out[0])])
        for item in user_history_item_dict[userid]:
            if item not in list_out:
                edge_head_ui = 'u' + str(userid)
                edge_tail_ui = 'i' + str(new2item_index[str(item)])
                # print(edge_head_ui, edge_tail_ui)
                big_graph_temp.append((edge_head_ui, edge_tail_ui))
    #entity2entity在上面已经加入
        for key_item in item2entity.keys():
            catch_entity = item2entity[key_item]
            edge_head_ie = 'i' + str(key_item)
            edge_tail_ie = 'e' + str(catch_entity)
            big_graph_temp.append((edge_head_ie,edge_tail_ie))
        big_graph_explanation.append((n_node-len(list_out),len(big_graph_temp)))
        for edge in big_graph_temp:
            big_graph_explanation.append(edge)
        big_graph_explanation.append((str_start_user,str_traget_item))
##############################################################
        path_explanation = '../big_graph/' + args.dataset + '/' + str(start_user)
        if os.path.isdir(path_explanation):
            big_graph_explanation_file = path_explanation + '/' + str(start_user) + '_explanation'+ '_' + str(j) +'.txt'
        else:
            os.mkdir(path_explanation)
            big_graph_explanation_file = path_explanation + '/' + str(start_user) + '_explanation'+ '_' + str(j) +'.txt'
        writer = open(big_graph_explanation_file, 'w', encoding='utf-8')
        for each_one in big_graph_explanation:
            left = each_one[0]
            right = each_one[1]
            writer.write('%s\t%s\n' % (left, right))
        writer.close()
    #########################################################
    #这里要根据列表进行重新画图 body一下
    #########################################################
        mid2 = total()
        each_pair_explanation , num_explanation= mid2.body(big_graph_explanation_file)
    # mid2.dis()

        file_explanation_graph = '../big_graph/' + args.dataset + '/' + str(userid) +'/' + 'explanation_graph'+ '_' + str(j)+'.txt'
        writer_explanation_graph = open(file_explanation_graph, 'w', encoding='utf-8')
        for path in each_pair_explanation:
            for pair in path:
                writer_explanation_graph.write('%s\t%s\n' % (pair[0], pair[1]))
        writer_explanation_graph.close()

        file_name = '../data/music/artists.txt'
        id_name = dict()
        for line in open(file_name, encoding='utf-8').readlines():
            id = line.strip().split('\t')[0]
            name = line.strip().split('\t')[1]
            url = line.strip().split('\t')[2]
            if id not in id_name.keys():
                id_name[id] = (name, url)
#以上需要循环
#############################################################

    list_to_show = '../big_graph/' + args.dataset + '/' + str(start_user) + '/' + 'recommend_list.json'
    writer_listout = open(list_to_show, 'w', encoding='utf-8')



    writer_listout.write("{ ")

    sum_len = 0

    print("json loading!")
    for i_pair in recommend_list[userid]:


        id = new2item_index[str(i_pair[0])]
        # writer_listout.write('%s\t%s\t%s\n' % (id_name[id][0], str(i_pair[1]), id_name[id][1]))
        writer_listout.write(" \""+"user" + str(sum_len)+ "\":\"" +id_name[id][0]+"\""+
                             " , \"" + "score" + str(sum_len)+"\":" + "\"" + str(i_pair[1]) + "\"" +
                             " , \"" + "url" + str(sum_len) + "\":" + "\"" + id_name[id][1]+ "\""
                            )
        sum_len += 1
        print(len(recommend_list[userid]))
        print(sum_len)
        if sum_len != len(recommend_list[userid]):
            writer_listout.write(",")

    writer_listout.write("}")
    writer_listout.close()
    print("json success!")
##############################################################################

def delete_graph(recommend_all_list,userid,num):  #用来删除边并且得到子图
    recommend_item = recommend_all_list['none'][userid][0][0]
    item_score = recommend_all_list['none'][userid][0][1]
    delete_list = []
    if num>2:
        for key in recommend_all_list.keys():  # keys是删除的节点
            point_traget = 0
            judge_exist = 0
            if key != 'none':
                for i in range(len(recommend_all_list[key][userid])):
                    if recommend_all_list[key][userid][i][0] == recommend_item:
                        point_traget = i
                        judge_exist = 1
                        break

                if point_traget == 0 and judge_exist == 1 and item_score == \
                        recommend_all_list[key][userid][point_traget][1]:  # 清晰掉完全没有影响的（排名和分数）
                    delete_list.append(key)
                elif point_traget == 0 and judge_exist == 1 and item_score != \
                        recommend_all_list[key][userid][point_traget][1]:  # 清洗掉对推荐结果有没有影响的
                    delete_list.append(key)
                elif point_traget != 0 and judge_exist == 1 and item_score >= \
                        recommend_all_list[key][userid][point_traget][1]:  # 清晰对推荐结果有负向影响的
                    delete_list.append(key)
    return delete_list



def ctr_eval(args, model, data, user_triple_set, item_triple_set):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    while start < data.shape[0]:
        labels = data[start:start + args.batch_size, 2]
        scores = model(*_get_feed_data(args, data, user_triple_set, item_triple_set, start, start + args.batch_size))
        scores = scores.detach().cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        auc_list.append(auc)
        f1_list.append(f1)
        start += args.batch_size
    model.train()  
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1



    
def _init_model(args, data_info):
    n_entity = data_info[3]
    n_relation = data_info[4]
    model = CKAN(args, n_entity, n_relation)
    if args.use_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
        weight_decay = args.l2_weight,
    )
    loss_func = nn.BCELoss()
    return model, optimizer, loss_func
    
    
def _get_feed_data(args, data, user_triple_set, item_triple_set, start, end):
    # origin item
    items = torch.LongTensor(data[start:end, 1])
    if args.use_cuda:
        items = items.cuda()
    # kg propagation embeddings
    users_triple = _get_triple_tensor(args, data[start:end,0], user_triple_set)
    items_triple = _get_triple_tensor(args, data[start:end,1], item_triple_set)
    return items, users_triple, items_triple


def _get_feed_label(args,labels):
    labels = torch.FloatTensor(labels)
    if args.use_cuda:
        labels = labels.cuda()
    return labels


def _get_triple_tensor(args, objs, triple_set):#(args, data[start:end,0或1], user_triple_set)
    # [h,r,t]  h: [layers, batch_size, triple_set_size]
    h,r,t = [], [], []
    for i in range(args.n_layer):
        h.append(torch.LongTensor([triple_set[obj][i][0] for obj in objs]))#2048个head
        r.append(torch.LongTensor([triple_set[obj][i][1] for obj in objs]))
        t.append(torch.LongTensor([triple_set[obj][i][2] for obj in objs]))
        if args.use_cuda:
            h = list(map(lambda x: x.cuda(), h))
            r = list(map(lambda x: x.cuda(), r))
            t = list(map(lambda x: x.cuda(), t))
    return [h,r,t]


def _get_user_record(args, data, is_train):
    user_history_dict = dict()
    for rating in data:
        user = rating[0]
        item = rating[1]
        label = rating[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict


def _get_topk_feed_data(user, items):
    res = list()
    for item in items:
        res.append([user,item])
    return np.array(res)


def _show_recall_info(recall_zip):
    res = ""
    for i,j in recall_zip:
        res += "K@%d:%.4f  "%(i,j)
    logging.info(res)



