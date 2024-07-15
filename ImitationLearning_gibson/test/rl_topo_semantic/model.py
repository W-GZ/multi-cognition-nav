import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import no_grad
import logging
import torchvision
import lpips
import numpy as np
import dgl
import networkx as nx
from yolov3.yolov3_model_create import create
from utils.graph_pooling_utils import normalize, GCN_3_layers_, GraphPooling_
from utils.gcn_utils import GCN_3_layers, GraphPooling

logging.getLogger().setLevel(logging.INFO)


class E2EModel(nn.Module):
    def __init__(self, action_space, device='cpu', category_num=26, category_bgr=None, test_scene=None, all_scene=None):
        super(E2EModel, self).__init__()
        self.test_scene = test_scene
        self.all_scene = all_scene
        self.device = device
        '''Resnet50'''
        self.resnet = torchvision.models.resnet50(pretrained=True).to(device)  # b*3*144*192, bgr, 0-1--->b*1000
        self.resnet_fc = ResnetFC().to(device)  # b*1000--b*256
        # print("resnet loaded")

        '''yolov3'''
        self.yolov3 = create(name='../../category_bgr/best.pt', pretrained=True,  # b*144*192*3, bgr, 0-255
                             channels=3, classes=category_num,
                             autoshape=True, verbose=True).to(device)
        self.yolov3_fc = YoloV3FC().to(device)  # b*5--b*256
        # print("yolov3 loaded")

        '''RetrievalNetwork'''
        # model_path = 'RetrievalNet/best.pth'
        # model_state_dict = torch.load(model_path)
        # self.alexnet = lpips.LPIPS(pretrained=False, net='alex', version='0.1', lpips=True, spatial=False,
        #                            pnet_rand=False, pnet_tune=True,
        #                            use_dropout=True, model_path=None, eval_mode=True).to(device)
        # self.alexnet.load_state_dict(model_state_dict, strict=True)

        '''policy module'''
        self.resizeNet = ResizeNet(768)  # feature_vector ---> 512
        self.policyNet = PolicyNet(512, action_space).to(device)  # 512 ---> action_prob
        self.valueNet = ValueNet(512).to(device)  # 512 ---> value

        self.default_value = 1e-6
        '''Semantic Graph'''
        self.SemanticGraph = dgl.DGLGraph().to(device)
        # nodes and nodes features
        self.SemanticGraph.add_nodes(category_num, {'x': self.default_value * torch.ones(category_num, 768).to(device)})
        # self.SemanticGraph.add_nodes(category_num,
        # {'x': torch.tensor(np.random.random((category_num, 517)), dtype=torch.float).to(device)})
        self.categories_visual_feature = None
        for i in range(len(category_bgr)):
            temp_input = torch.Tensor(category_bgr[i].transpose((2, 0, 1))).unsqueeze(0).to(device)  # 1*3*144*192m
            with no_grad():
                temp_output = self.resnet(temp_input)  # 1*1000
            if self.categories_visual_feature is None:
                self.categories_visual_feature = temp_output
            else:
                self.categories_visual_feature = torch.cat([self.categories_visual_feature, temp_output], 0)

        # edges
        a_raw = torch.load("Semantic/adjmat_manual_gibson.dat")
        x, y = np.where(a_raw == 1)
        self.SemanticGraph.add_edges(x, y)

        self.gcn_semantic = GCN_3_layers(768, 512, 512, 256).to(device)

        self.gcn_semantic_ = GCN_3_layers(768, 512, 512, 1).to(device)
        self.gcn_semantic_fc = Gcn_Semantic_FC(category_num).to(device)  # 26-->256
        self.current_obj_predict = CurrentObjPredict(category_num).to(device)  # 256-->26
        # self.next_obj_predict = NextObjPredict(category_num).to(device)  # 256-->26

        '''Topological Graph'''
        # type_ = 'manual_gibson/'  # 'auto'  'manual'
        # edges__ = np.fromfile("RetrievalNet/" + type_ + scene_id + "/edges.npy", dtype='int64').reshape(2, -1)
        # resnet_current_input = np.fromfile("RetrievalNet/" + type_ + scene_id + "/resnet_current_input.npy", dtype='float32').reshape(-1, 3, 144, 192)  # N*3*144*192, bgr, 0-1
        # alex_current_input = np.fromfile("RetrievalNet/" + type_ + scene_id + "/alex_current_input.npy", dtype='float32').reshape(-1, 3, 144, 192)  # N*3*144*192  -scaled to [-1,1]
        # position_yaw = np.fromfile("RetrievalNet/" + type_ + scene_id + "/position_yaw.npy", dtype='float32').reshape(-1, 3)
        #
        # self.TopologicalGraph = dgl.DGLGraph().to(device)
        #
        # with no_grad():
        #     self.topological_visual_feature = self.resnet(torch.tensor(resnet_current_input, dtype=torch.float).to(device))  # N*3*144*192--N*1000
        # topological_visual_feature = self.resnet_fc(self.topological_visual_feature)  # N*256
        #
        # node_features = torch.cat([topological_visual_feature, torch.zeros(len(resnet_current_input), 2).to(device)], 1)  # N*258
        #
        # self.TopologicalGraph.add_nodes(len(resnet_current_input), {'x': node_features,
        #                                                             'y': torch.tensor(alex_current_input, dtype=torch.float).to(device),
        #                                                             'position_yaw': torch.tensor(position_yaw, dtype=torch.float).to(device)
        #                                                             })
        # self.TopologicalGraph.add_edges(edges__[0], edges__[1])
        type_ = 'manual_gibson/'  # 'auto'  'manual'
        self.TopologicalGraph = []
        self.topological_visual_feature = []
        self.gcn_topological_fc = []
        self.gcn_fc = []

        for i in range(len(test_scene)):
            temp_topological_graph = dgl.DGLGraph().to(device)

            edges__ = np.fromfile("../../" + type_ + test_scene[i] + "/edges.npy", dtype='int64').reshape(2, -1)
            resnet_current_input = np.fromfile("../../" + type_ + test_scene[i] + "/resnet_current_input.npy",
                                               dtype='float32').reshape(-1, 3, 144, 192)  # N*3*144*192, bgr, 0-1
            alex_current_input = np.fromfile("../../" + type_ + test_scene[i] + "/alex_current_input.npy",
                                             dtype='float32').reshape(-1, 3, 144, 192)  # N*3*144*192  -scaled to [-1,1]
            position_yaw = np.fromfile("../../" + type_ + test_scene[i] + "/position_yaw.npy",
                                       dtype='float32').reshape(-1, 3)

            with no_grad():
                temp_topological_visual_feature = self.resnet(
                    torch.tensor(resnet_current_input, dtype=torch.float).to(device))  # N*3*144*192--N*1000
                self.topological_visual_feature.append(temp_topological_visual_feature)
            topological_visual_feature = self.resnet_fc(temp_topological_visual_feature)  # N*256

            node_features = torch.cat(
                [topological_visual_feature, torch.zeros(len(resnet_current_input), 2).to(device)], 1)  # N*258

            temp_topological_graph.add_nodes(len(resnet_current_input), {'x': node_features,
                                                                         'y': torch.tensor(alex_current_input,
                                                                                           dtype=torch.float).to(
                                                                             device),
                                                                         'position_yaw': torch.tensor(position_yaw,
                                                                                                      dtype=torch.float).to(
                                                                             device)
                                                                         })
            temp_topological_graph.add_edges(edges__[0], edges__[1])

            self.TopologicalGraph.append(temp_topological_graph)
            self.gcn_topological_fc.append(Gcn_Topological_FC(len(resnet_current_input)).to(device))
            self.gcn_fc.append(GcnFC(category_num + temp_topological_graph.number_of_nodes()).to(device))  # 142-->256

        self.gcn_topological = GCN_3_layers(258, 256, 256, 256).to(device)

        self.gcn_topological_ = GCN_3_layers(258, 256, 256, 1).to(device)
        self.dir_predict = DirPredict().to(device)  # 256-->8

        '''graph pooling'''
        # self.gcn_1 = GCN_3_layers_(517, 128, 32, 4).to(device)
        # self.graph_pooling = GraphPooling_(517, 128, 32, 4).to(device)
        self.gcn_1 = GCN_3_layers(256, 128, 32, 4).to(device)
        self.graph_pooling = GraphPooling(256, 128, 32, 4).to(device)

        '''GCN'''
        self.gcn_2 = GCN_3_layers_(256, 128, 32, 1).to(device)

        self.time_semantic = []
        self.time_topo = []
        self.time_fusing = []
        self.time_policy = []
        self.time_total = []

        self.step = 0
        self.all_detected = 0
        self.detected = 0
        self.acc = []

    def graph_init(self):
        """Semantic Graph"""
        categories_bgr_visual_feature = self.resnet_fc(self.categories_visual_feature)  # 42*256
        node_features = torch.cat([categories_bgr_visual_feature, self.default_value * torch.ones(26, 256+256).to(self.device)], 1)  # 42*768
        self.SemanticGraph.ndata['x'] = node_features

        '''Topological Graph'''
        topological_visual_feature = self.resnet_fc(self.topological_visual_feature)  # N*256
        node_features = torch.cat([topological_visual_feature, torch.zeros(self.TopologicalGraph.number_of_nodes(), 2).to(self.device)], 1)  # N*258
        self.TopologicalGraph.ndata['x'] = node_features

    def forward(self, current_bgr=None, target_bgr=None, current_position_yaw=None, target_position_yaw=None,
                scene_id=None,
                obj_detect_results=None, current_visual_feature_=None, target_visual_feature_=None,
                name='gathering'):
        if name == 'gathering':
            import time
            start_time = time.time()  # 获取当前时间作为开始时间

            with no_grad():
                '''resnet'''
                resnet_current_input = torch.Tensor(current_bgr.transpose((0, 3, 1, 2)).copy()).to(self.device)  # b*3*144*192, bgr, 0-1
                resnet_target_input = torch.Tensor(target_bgr.transpose((0, 3, 1, 2)).copy()).to(self.device)  # b*3*144*192, bgr, 0-1
                current_visual_feature_ = self.resnet(resnet_current_input)  # batch_size*3*144*192 --> b*1000
                target_visual_feature_ = self.resnet(resnet_target_input)  # batch_size*3*144*192 --> b*1000
                '''yolov3'''
                yolov3_input = np.round(current_bgr * 255).astype(np.uint8)  # 1*144*192*3, 0-255
                obj_detect_results = self.yolov3(yolov3_input[0]).detect_result()
                '''RetrievalNetwork'''
                # temp1 = np.array(current_bgr) * 255  # b*144*192*3  -scaled to [0,255]
                # alex_current_input = torch.Tensor((temp1 / 127.5 - 1)[:, :, :, ].transpose((0, 3, 1, 2))).to(self.device)  # 1*3*144*192  -scaled to [-1,1]
                # temp2 = np.array(target_bgr) * 255  # b*144*192*3  -scaled to [0,255]
                # alex_target_input = torch.Tensor((temp2 / 127.5 - 1)[:, :, :, ].transpose((0, 3, 1, 2))).to(self.device)  # 1*3*144*192  -scaled to [-1,1]

            

            test_scene_index = self.test_scene.index(scene_id)

            current_visual_feature = self.resnet_fc(current_visual_feature_)  # b*1000 --> b*256
            target_visual_feature = self.resnet_fc(target_visual_feature_)  # b*1000 --> b*256
            categories_bgr_visual_feature = self.resnet_fc(self.categories_visual_feature)  # 42*256
            topological_visual_feature = self.resnet_fc(self.topological_visual_feature[test_scene_index])  # N*256

            time1 = time.time()

            '''根据categories_bgr_visual_feature、检测结果obj_detect_results、target_visual_feature更新语义图'''
            # detect_features = self.default_value * np.ones((42, 5 + 256))  # detect_features = np.random.random((42, 5 + 256))
            # target_visual_f = target_visual_feature[0].detach().cpu().numpy()  # 256
            # if obj_detect_results[0] != 0:
            #     for j in range(len(obj_detect_results[0])):
            #         index = int(obj_detect_results[0][j]['cls'])
            #         detect_features[index] = np.concatenate([np.array(obj_detect_results[0][j]['box']),
            #                                                  np.array(target_visual_f)], axis=0)
            # node_features = torch.cat([categories_bgr_visual_feature, torch.Tensor(detect_features).to(self.device)], 1)

            detect_features = np.zeros((26, 5))
            target_current_obj_detect = np.zeros(26)
            count = 0
            if obj_detect_results[0] != 0:
                for j in range(len(obj_detect_results[0])):
                    index = int(obj_detect_results[0][j]['cls'])
                    detect_features[index] = np.array(obj_detect_results[0][j]['box'])
                    count += 1
                    target_current_obj_detect[index] = 1.0
            
            detect_feature = self.yolov3_fc(torch.tensor(detect_features, dtype=torch.float).to(self.device))  # 42*256

            target_visual_f = target_visual_feature
            for k in range(self.SemanticGraph.number_of_nodes()-1):
                target_visual_f = torch.cat([target_visual_f, target_visual_feature], 0)

            node_features = torch.cat([categories_bgr_visual_feature, detect_feature, target_visual_f.to(self.device)], 1)

            self.SemanticGraph.ndata['x'] = node_features  # 26*768

            semantic_graph_temp = dgl.DGLGraph().to(self.device)
            semantic_graph_temp.add_nodes(self.SemanticGraph.number_of_nodes(), {'x': self.default_value * torch.ones(self.SemanticGraph.number_of_nodes(), 256).to(self.device)})
            semantic_graph_temp.add_edges(self.SemanticGraph.all_edges()[0].cpu().detach().numpy(),
                                          self.SemanticGraph.all_edges()[1].cpu().detach().numpy())
            semantic_graph_temp.ndata['x'] = self.gcn_semantic(self.SemanticGraph, self.SemanticGraph.ndata['x'])  # 26*768-->26*256

            semantic_feature = self.gcn_semantic_(self.SemanticGraph, self.SemanticGraph.ndata['x']).transpose(1, 0)  # 1*26
            semantic_feature = self.gcn_semantic_fc(semantic_feature)  # 1*256
            current_obj_predict = self.current_obj_predict(semantic_feature)
            # next_obj_predict = self.next_obj_predict(semantic_feature)

            time2 = time.time()

            '''拓扑图'''
            '''position'''
            # # current
            current_locate = torch.zeros(self.TopologicalGraph[test_scene_index].number_of_nodes(), 1)
            for j in range(self.TopologicalGraph[test_scene_index].number_of_nodes()):
                temp_position_yaw = self.TopologicalGraph[test_scene_index].ndata['position_yaw'][j].cpu().detach().numpy()
                temp_dis = np.sqrt((current_position_yaw[0] - temp_position_yaw[0]) ** 2 + (
                            current_position_yaw[1] - temp_position_yaw[1]) ** 2)
                temp_delta_yaw = abs(current_position_yaw[2] - temp_position_yaw[2])

                if temp_dis <= 1.5 and temp_delta_yaw <= np.pi/4:
                    current_locate[j] = 1.0
                    break
            # print('current:', np.argmin(np.array(diatance)))

            # # target
            target_locate = torch.zeros(self.TopologicalGraph[test_scene_index].number_of_nodes(), 1)
            for j in range(self.TopologicalGraph[test_scene_index].number_of_nodes()):
                temp_position_yaw = self.TopologicalGraph[test_scene_index].ndata['position_yaw'][j].cpu().detach().numpy()
                temp_dis = np.sqrt((target_position_yaw[0] - temp_position_yaw[0]) ** 2 + (
                            target_position_yaw[1] - temp_position_yaw[1]) ** 2)
                temp_delta_yaw = abs(target_position_yaw[2] - temp_position_yaw[2])

                if temp_dis <= 1.5 and temp_delta_yaw <= np.pi/4:
                    target_locate[j] = 1.0
                    break
            # print('target:', np.argmin(np.array(diatance)))

            # current_locate = torch.zeros(self.TopologicalGraph.number_of_nodes(), 1)
            # diatance = []
            # for j in range(self.TopologicalGraph.number_of_nodes()):
            #     temp_position_yaw = self.TopologicalGraph.ndata['position_yaw'][j].cpu().detach().numpy()
            #     temp_dis = np.sqrt((current_position_yaw[0] - temp_position_yaw[0]) ** 2 + (current_position_yaw[1] - temp_position_yaw[1]) ** 2)
            #     diatance.append(temp_dis)
            # current_locate[np.argmin(np.array(diatance))] = 1.0
            # # print('current:', np.argmin(np.array(diatance)))
            #
            # # # target
            # target_locate = torch.zeros(self.TopologicalGraph.number_of_nodes(), 1)
            # diatance = []
            # for j in range(self.TopologicalGraph.number_of_nodes()):
            #     temp_position_yaw = self.TopologicalGraph.ndata['position_yaw'][j].cpu().detach().numpy()
            #     temp_dis = np.sqrt((target_position_yaw[0] - temp_position_yaw[0]) ** 2 + (target_position_yaw[1] - temp_position_yaw[1]) ** 2)
            #     diatance.append(temp_dis)
            # target_locate[np.argmin(np.array(diatance))] = 1.0
            # # print('target:', np.argmin(np.array(diatance)))

            '''alexnet'''
            # # # current_node
            # current_locate = torch.zeros(self.TopologicalGraph.number_of_nodes(), 1)
            # likely = []
            # for j in range(self.TopologicalGraph.number_of_nodes()):
            #     with no_grad():
            #         likely.append(torch.clamp(self.alexnet.forward(alex_current_input[0].unsqueeze(0),
            #                                                        self.TopologicalGraph.ndata['y'][j],
            #                                                        retPerLayer=False), 0, 1).cpu().detach().numpy())
            # current_locate[np.argmin(np.array(likely))] = 1.0
            # # print('current_node:', np.argmin(np.array(likely)))
            #
            # # # target_node
            # target_locate = torch.zeros(self.TopologicalGraph.number_of_nodes(), 1)
            # likely = []
            # for j in range(self.TopologicalGraph.number_of_nodes()):
            #     with no_grad():
            #         likely.append(torch.clamp(self.alexnet.forward(alex_target_input[0].unsqueeze(0),
            #                                                        self.TopologicalGraph.ndata['y'][j],
            #                                                        retPerLayer=False), 0, 1).cpu().detach().numpy())
            # target_locate[np.argmin(np.array(likely))] = 1.0
            # # print('target_node:', np.argmin(np.array(likely)))

            '''locate_vector'''
            locate_vector = torch.cat([current_locate, target_locate], 1)  # N*2

            node_features = torch.cat([topological_visual_feature, locate_vector.to(self.device)], 1)  # N*258
            self.TopologicalGraph[test_scene_index].ndata['x'] = node_features

            topological_graph_temp = dgl.DGLGraph().to(self.device)
            topological_graph_temp.add_nodes(self.TopologicalGraph[test_scene_index].number_of_nodes(), {'x': self.default_value * torch.ones(self.TopologicalGraph[test_scene_index].number_of_nodes(), 256).to(self.device)})
            topological_graph_temp.add_edges(self.TopologicalGraph[test_scene_index].all_edges()[0].cpu().detach().numpy(),
                                             self.TopologicalGraph[test_scene_index].all_edges()[1].cpu().detach().numpy())

            topological_graph_temp.ndata['x'] = self.gcn_topological(self.TopologicalGraph[test_scene_index], self.TopologicalGraph[test_scene_index].ndata['x'])  # N*258-->N*256

            topological_feature = self.gcn_topological_(self.TopologicalGraph[test_scene_index], self.TopologicalGraph[test_scene_index].ndata['x']).transpose(1, 0)  # N*258-->1*N
            topological_feature = self.gcn_topological_fc[test_scene_index](topological_feature)  # 1*256
            dir_predict = self.dir_predict(topological_feature)

            time3 = time.time()

            '''graph pooling'''
            nx_semantic = semantic_graph_temp.to_networkx()
            A = nx.adjacency_matrix(nx_semantic).todense()
            # A_normed = normalize(torch.FloatTensor(A), True).to(self.device)  # 26*26
            # S = self.gcn_1(A_normed, self.SemanticGraph.ndata['x'])  # 26*4
            S = self.gcn_1(semantic_graph_temp, semantic_graph_temp.ndata['x'])  # 26*4

            nx_topological = topological_graph_temp.to_networkx()
            B = nx.adjacency_matrix(nx_topological).todense()
            # B_normed = normalize(torch.FloatTensor(B), True).to(self.device)
            # C = self.graph_pooling(B_normed, self.TopologicalGraph.ndata['x'], S)  # 100*26
            C = self.graph_pooling(topological_graph_temp, topological_graph_temp.ndata['x'], S)  # 100*26

            '''GCN'''
            A_1 = torch.cat([torch.FloatTensor(A).to(self.device), C.transpose(1, 0)], dim=1)  # 42*(42+100)
            A_2 = torch.cat([C, torch.FloatTensor(B).to(self.device)], dim=1)  # 100*(42+100)
            # A_1 = torch.cat([A_normed, C.transpose(1, 0)], dim=1)  # 42*(42+100)
            # A_2 = torch.cat([C, B_normed], dim=1)  # 100*(42+100)
            A_ = torch.cat([A_1, A_2], dim=0)  # (42+100)*(42+100)
            Anormed = normalize(A_.cpu(), True).to(self.device)

            node_features_ = torch.cat([semantic_graph_temp.ndata['x'],
                                        topological_graph_temp.ndata['x']], dim=0)  # (42+100)*256

            topological_semantic_feature = self.gcn_2(Anormed, node_features_).transpose(1, 0)  # 1*(42+100)
            # st, end = self.TopologicalGraph.all_edges()
            # # edge_index = torch.cat([st.unsqueeze(1), end.unsqueeze(1)], dim=1)
            # edge_index = torch.cat([st.unsqueeze(0), end.unsqueeze(0)], dim=0).to(self.device)  # 2*num-of-node
            # topological_semantic_feature = self.gcn_2(node_features_, edge_index).transpose(1, 0)

            topological_semantic_feature = self.gcn_fc[test_scene_index](topological_semantic_feature)  # 1*(42+100)-->1*256

            time4 = time.time()

            '''policy module'''
            feature_vector = torch.cat([current_visual_feature, target_visual_feature, topological_semantic_feature], dim=1)  # 1*768
            feature_vector = self.resizeNet(feature_vector)  # 1*768-->1*512

            value = self.valueNet(feature_vector)  # 1*512 --> 1*1
            pi = self.policyNet(feature_vector)  # 1*512 --> 1*3

            # return pi, detect_features, current_visual_feature_, target_visual_feature_
            
            end_time = time.time()  # 获取当前时间作为结束时间

            self.time_semantic.append((time2 - time1) * 1000)
            self.time_topo.append((time3 - time2) * 1000)
            self.time_fusing.append((time4 - time3) * 1000)
            self.time_policy.append((end_time - time4) * 1000)
            self.time_total.append((end_time - start_time) * 1000)

            # print(f"\ntime_semantic{np.mean(self.time_semantic):.2f} 毫秒")
            # print(f"time_topo{np.mean(self.time_topo):.2f} 毫秒")
            # print(f"time_fusing{np.mean(self.time_fusing):.2f} 毫秒")
            # print(f"time_policy{np.mean(self.time_policy):.2f} 毫秒")
            # print(f"time_total{np.mean(self.time_total):.2f} 毫秒")
            

            '''aux_task_acc '''
            # current_obj_predict = current_obj_predict[0].cpu().numpy()
            # indices = np.argpartition(current_obj_predict, -count)[-count:]
            
            # current_obj_predict__ = np.zeros(26)
            # for i in range(count):
            #     current_obj_predict__[indices[i]] = 1.0
            # # print(target_current_obj_detect)
            # # print(current_obj_predict__)

            # # self.all_detected += count
            # # for i in range(26):
            # #     if target_current_obj_detect[i] == 1 and current_obj_predict__[i] == 1:
            # #         self.detected += 1


            # self.all_detected += 26
            # for i in range(26):
            #     if target_current_obj_detect[i] == current_obj_predict__[i]:
            #         self.detected += 1
            
            # acc = float(self.detected / self.all_detected)

            # self.step += 1
            # self.acc.append(acc)

            # print(self.step, acc)  # 0.728730633324

            # # if self.step == 1000:
            # #     # 打开一个文件以写入模式
            # #     with open('aux_task_acc.txt', 'w') as file:
            # #         # 使用循环将列表中的每个元素写入文件
            # #         for item in self.acc:
            # #             file.write(str(item) + '\n')
            
            return pi, current_obj_predict, dir_predict

        elif name == 'training':
            current_visual_feature = self.resnet_fc(current_visual_feature_.squeeze(0).to(self.device))  # b*1000 --> b*256
            target_visual_feature = self.resnet_fc(target_visual_feature_.squeeze(0).to(self.device))  # b*1000 --> b*256
            categories_bgr_visual_feature = self.resnet_fc(self.categories_visual_feature)  # 42*256

            '''根据categories_bgr_visual_feature、detect_features更新语义图'''
            # target_visual_features = np.ones((42, 256))
            # target_visual_f = target_visual_feature[0].detach().cpu().numpy()  # 256
            # for j in range(len(target_visual_features)):
            #     target_visual_features[j] = target_visual_f
            # node_features = torch.cat([categories_bgr_visual_feature, obj_detect_results.squeeze(0).to(self.device), torch.Tensor(target_visual_features).to(self.device)], 1)

            detect_feature = self.yolov3_fc(torch.tensor(obj_detect_results.squeeze(0)).to(self.device))  # 42*256
            target_visual_f = target_visual_feature
            for k in range(self.SemanticGraph.number_of_nodes()-1):
                target_visual_f = torch.cat([target_visual_f, target_visual_feature], 0)

            node_features = torch.cat([categories_bgr_visual_feature, detect_feature, target_visual_f.to(self.device)], 1)

            self.SemanticGraph.ndata['x'] = node_features  # 42*768

            semantic_feature = self.gcn_semantic(self.SemanticGraph, self.SemanticGraph.ndata['x']).transpose(1, 0)  # 1*42
            semantic_feature = self.gcn_semantic_fc(semantic_feature)  # 1*256

            '''graph pooling'''
            '''GCN'''
            '''policy module'''
            feature_vector = torch.cat([current_visual_feature, target_visual_feature, semantic_feature], dim=1)  # 1*768
            feature_vector = self.resizeNet(feature_vector)  # 1*768-->1*512

            value = self.valueNet(feature_vector)  # 1*512 --> 1*1
            pi = self.policyNet(feature_vector)  # 1*512 --> 1*3

            return value, pi


class ResizeNet(nn.Module):  # actor
    def __init__(self, input_size):
        super(ResizeNet, self).__init__()

        self.mlp_ = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 512)
        )

    def forward(self, feature_vector):
        output_ = self.mlp_(feature_vector)
        return output_


class PolicyNet(nn.Module):  # actor
    def __init__(self, input_size, action_size):
        super(PolicyNet, self).__init__()

        self.mlp_action_prob = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, action_size)
        )

    def forward(self, feature_vector):
        action_prob = F.softmax(self.mlp_action_prob(feature_vector), dim=1)
        return action_prob


class ValueNet(nn.Module):  # critic
    def __init__(self, input_size):
        super(ValueNet, self).__init__()

        self.mlp_value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(16, 1)
        )

    def forward(self, feature_vector):
        value = self.mlp_value(feature_vector)
        return value


class ResnetFC(nn.Module):
    def __init__(self):
        super(ResnetFC, self).__init__()

        self.resnet_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1000, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 256),
        )

    def forward(self, resnet_output):
        visual_feature = self.resnet_fc(resnet_output)
        return visual_feature


class YoloV3FC(nn.Module):
    def __init__(self):
        super(YoloV3FC, self).__init__()

        self.yolov3_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 256),
        )

    def forward(self, detect_features_):
        detect_features = self.yolov3_fc(detect_features_)
        return detect_features


class GcnFC(nn.Module):
    def __init__(self, input_):
        super(GcnFC, self).__init__()

        self.gcn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 256),
        )

    def forward(self, node_features_):
        topological_semantic_feature = self.gcn_fc(node_features_)
        return topological_semantic_feature


class Gcn_Semantic_FC(nn.Module):
    def __init__(self, input_size):
        super(Gcn_Semantic_FC, self).__init__()

        self.gcn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 256),
        )

    def forward(self, node_features_):
        output_feature = self.gcn_fc(node_features_)
        return output_feature


class Gcn_Topological_FC(nn.Module):
    def __init__(self, input_size):
        super(Gcn_Topological_FC, self).__init__()

        self.gcn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 256),
        )

    def forward(self, node_features_):
        output_feature = self.gcn_fc(node_features_)
        return output_feature


class CurrentObjPredict(nn.Module):
    def __init__(self, output_size):
        super(CurrentObjPredict, self).__init__()

        self.mlp_ = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, output_size)
        )

    def forward(self, input_vector):
        output_vector = self.mlp_(input_vector)
        return output_vector


class NextObjPredict(nn.Module):
    def __init__(self, output_size):
        super(NextObjPredict, self).__init__()

        self.mlp_ = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, output_size)
        )

    def forward(self, input_vector):
        output_vector = self.mlp_(input_vector)
        return output_vector


class DirPredict(nn.Module):
    def __init__(self):
        super(DirPredict, self).__init__()

        self.mlp_ = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 8)
        )

    def forward(self, input_vector):
        output_vector = self.mlp_(input_vector)
        return output_vector

