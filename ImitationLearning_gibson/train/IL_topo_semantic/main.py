import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch import tensor
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from datetime import datetime
from model import E2EModel
import os
import logging
import warnings
from main_util import myFunctions as Func
from category_bgr.category import scene_name_list, category_name_list, target_name_list

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(threshold=np.inf)

'''util'''
func_util = Func()
scene_name_list = scene_name_list()  # 12
category_name_list = category_name_list()  # 26
target_name_list = target_name_list()

'''create path for model and summary'''
date = str(datetime.today())[:10]
model_path = './record/' + date + '/models/'
summary_path = './record/' + date + '/summary/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(summary_path):
    os.makedirs(summary_path)

'''parameters 模型实例化'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
action_space = 3
category_num = 26

'''category_'''
category_bgr = []
for i in range(len(category_name_list)):
    category_name = category_name_list[i]
    file_bgr = None
    for j in range(len(scene_name_list)):
        if os.path.exists('../../category_bgr/' + scene_name_list[j] + '/' + category_name + '.npy'):
            temp = np.fromfile('../../category_bgr/' + scene_name_list[j] + '/' + category_name + '.npy',
                               dtype='float32').reshape(144, 192, 3)
            if file_bgr is None:
                file_bgr = temp
            else:
                file_bgr = np.concatenate([file_bgr, temp], axis=1)  # 144*192m*3

    category_bgr.append(file_bgr)

'''first train'''
train_scene = ['Cooperstown',
               'Denmark',
               'Elmira',
               'Hometown',
               'Maben',
               'Matoaca',
               'Rabbit',
               'Ribera']  # 'Rosser', 'Rs', 'Seward', 'Sumas'
model = E2EModel(action_space=action_space, device=device, category_num=category_num, category_bgr=category_bgr,
                 train_scene=train_scene)
if torch.cuda.is_available():
    model = model.cuda()

# def get_parameter_number(net):
#     total_num = sum(p.numel() for p in net.parameters())
#     trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}
#
#
# print("模型总大小为：{:.3f}".format(get_parameter_number(model)["Total"]))
# print("模型Trainable：{:.3f}".format(get_parameter_number(model)["Trainable"]))

initial_lr = 1e-4  # 1e-3

'''loss & optimizer'''
CrossEntropyLoss = nn.CrossEntropyLoss()
MSELoss1 = nn.MSELoss()
MSELoss2 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
stats = {'action_loss': [],
         'current_obj_predict_loss': [],
         'dir_predict_loss': [],
         'total_loss': [],
         'learning_rate': []}

batch_size_train = 32

lr_decay_step = 100000
save_model_step = 30


def train(epoches):
    print('Training Start')
    model.train()
    writer = SummaryWriter(summary_path)

    obs_list, action_list, scene_index_list, traj_index_list, \
    current_obj_detect_list, next_obj_detect_list, pos_theta_list, target_dir_list = func_util.read()
    bgr = tensor(np.array(obs_list)).to(device)
    action_list = tensor(np.array(action_list, dtype=int)).to(device)
    scene_index_list = tensor(np.array(scene_index_list, dtype=int)).to(device)
    traj_index_list = tensor(np.array(traj_index_list, dtype=int)).to(device)

    current_obj_detect_list = tensor(np.array(current_obj_detect_list, dtype=np.float32)).to(device)

    pos_theta_list = tensor(np.array(pos_theta_list, dtype=np.float32)).to(device)
    target_dir_list = tensor(np.array(target_dir_list, dtype=np.float32)).to(device)

    batch_num = np.float64(int(len(bgr) / batch_size_train) + 1)  # batch_num,  iteration*batch_size=10000
    print('batch_num:', batch_num)

    '''train'''
    for episode in range(1, epoches + 1):
        print('-' * 40)
        print('Epoch {}/{}'.format(episode, epoches))
        action_loss, current_obj_predict_loss, dir_predict_loss, total_loss = 0.0, 0.0, 0.0, 0.0
        learning_rate = func_util.adjust_learning_rate(optimizer, initial_lr, lr_decay_step, episode)
        # model.graph_init()

        for index in BatchSampler(SubsetRandomSampler(range(len(bgr))), batch_size_train, False):
            '''getData, one batch'''
            action_prob = None
            current_obj_predict = None
            dir_predict = None

            for k in range(len(index)):
                current_bgr = bgr[[index[k]]].cpu().numpy()  # 1*144*192*3, 0-1

                scene_index = scene_index_list[[index[k]]].cpu().item()
                traj_index = traj_index_list[[index[k]]].cpu().item()
                target_bgr = np.expand_dims(
                    np.fromfile('../../../Gibson_Dataset_Sample/IL_dataset_gibson_auto/' +
                                scene_name_list[scene_index] + '/%04i' % traj_index + '/target_bgr.npy',
                                dtype=np.float32).reshape(144, 192, 3)
                    , axis=0)  # 1*144*192*3, 0-1

                current_position_yaw = pos_theta_list[[index[k]]].cpu().numpy()

                target_position_yaw = np.fromfile('../../../Gibson_Dataset_Sample/IL_dataset_gibson_auto/' +
                                                  scene_name_list[scene_index] + '/%04i' % traj_index +
                                                  '/target_pos_theta.npy',
                                                  dtype=np.float32).reshape(-1, 3)

                '''train'''
                action_prob_, current_obj_predict_, dir_predict_ = model(current_bgr=current_bgr, target_bgr=target_bgr,
                                                                         current_position_yaw=current_position_yaw[0],
                                                                         target_position_yaw=target_position_yaw[0],
                                                                         scene_index=scene_index,
                                                                         name='gathering')

                if k == 0:
                    action_prob = action_prob_
                    current_obj_predict = current_obj_predict_
                    dir_predict = dir_predict_

                else:
                    action_prob = torch.cat([action_prob, action_prob_], 0)  # batch_size_train*3
                    current_obj_predict = torch.cat([current_obj_predict, current_obj_predict_],
                                                    0)  # batch_size_train*42
                    dir_predict = torch.cat([dir_predict, dir_predict_], 0)  # batch_size_train*8

            target_action = action_list[index]
            target_current_obj_detect = current_obj_detect_list[index]
            target_dir = target_dir_list[index]

            '''loss'''
            action_loss_current = CrossEntropyLoss(action_prob, target_action)
            c_obj_predict_loss_current = MSELoss1(current_obj_predict, target_current_obj_detect)
            dir_predict_loss_current = MSELoss2(dir_predict, target_dir)

            # loss = 0.8 * action_loss_current + 0.1 * c_obj_predict_loss_current + 0.1 * n_obj_predict_loss_current
            loss = action_loss_current + c_obj_predict_loss_current / (
                    c_obj_predict_loss_current / action_loss_current).detach() + dir_predict_loss_current / (
                           dir_predict_loss_current / action_loss_current).detach()

            # print(action_loss_current)

            '''backward'''
            optimizer.zero_grad()  # Delete old gradients
            loss.backward()  # Perform backward step to compute new gradients
            nn.utils.clip_grad_norm_(model.parameters(), 0.4)  # Clip gradients
            optimizer.step()  # Perform training step based on gradients

            action_loss += action_loss_current.cpu().detach().item()
            current_obj_predict_loss += c_obj_predict_loss_current.cpu().detach().item()
            dir_predict_loss += dir_predict_loss_current.cpu().detach().item()
            total_loss += loss.detach().item()

        print('action_loss:{loss1:.6f}  current_obj_predict_loss:{loss2:.6f}  '
              'dir_predict_loss:{loss3:.6f}  '
              'total_loss:{loss4:.6f}'.format(loss1=action_loss / batch_num,
                                              loss2=current_obj_predict_loss / batch_num,
                                              loss3=dir_predict_loss / batch_num,
                                              loss4=total_loss / batch_num))
        stats['action_loss'].append(action_loss / batch_num)
        stats['current_obj_predict_loss'].append(current_obj_predict_loss / batch_num)
        stats['dir_predict_loss'].append(dir_predict_loss / batch_num)
        stats['total_loss'].append(total_loss / batch_num)
        stats['learning_rate'].append(learning_rate)

        # step = episode + 1000
        step = episode
        func_util.writeSummary(writer, stats, step)

        if episode % save_model_step == 0:
            torch.save(model.state_dict(), model_path + str(step) + '.pkl')

    torch.save(model.state_dict(), model_path + str(step) + '.pkl')
    writer.close()
    print('Training Ended!')


if __name__ == '__main__':
    train(lr_decay_step)
