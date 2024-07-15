# import logging
import yaml
import os
import warnings
from trainer import Transition, Trainer
from model import *
from robot import TurtleBotRobot
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from category_bgr.category import scene_name_list, category_name_list, target_name_list
import json
from copy import deepcopy

warnings.filterwarnings("ignore", category=UserWarning)


scene_name_list = scene_name_list()  # 12
category_name_list = category_name_list()  # 26
target_name_list = target_name_list()


def create_txt(txtPath, txtHead):
    pose_file = open(txtPath, mode='w', encoding='utf-8_sig')
    pose_file.write(txtHead)
    # pose_file.write('\n')
    pose_file.close()


def write_txt(txtPath, episode_, epo_length,
              episodeScore,
              target_distance,
              success_rate):
    # msg = str(episode_id) + '\t' + str(step) + '\t' + \
    #       str(episodeScore) + '\t' + str(search_rate) + '\t' + str(re_search_rate) + '\n'
    msg = str(episode_) + '\t\t' + str(epo_length) + '\t\t' + \
          "%.8f" % episodeScore + '\t\t' + "%.8f" % target_distance + '\t\t' + \
          "%.8f" % success_rate + '\n'

    pose_file = open(txtPath, mode='a+', encoding='utf-8_sig')
    pose_file.write(msg)
    pose_file.close()


def progress_bar(current_step, max_step_per_episode, episodecount, episodelimit,
                 episodeScore):
    """Print the progress until the next episode."""
    progress = (((current_step - 1) % max_step_per_episode) + 1) / max_step_per_episode
    fill = int(progress * 40)
    print("\r[{}{}]: {}   "
          "episode:{}/{}  "
          "episodeScore:{:.2f}".format("=" * fill, " " * (40 - fill),
                                       current_step,
                                       episodecount, episodelimit,
                                       episodeScore), end='')


# mode configuration
training = 0  # 0-test / 1-firsttrain / 2-continuetrain

# train mode parameters
max_steps_per_episode = 1000
bufferSize = 512
batchSize = 128
initialLearningRate = 1e-7
lrDecayStep = 1e7
useGreedy = True and training
epsilonGreedyDecayStep = 5e7

# model path for saving
date = str(datetime.today())[:10]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型实例化
# ***********************************************#
action_space = 3
category_num = 26

'''category_'''
category_bgr = []
for i in range(len(category_name_list)):  # 26
    category_name = category_name_list[i]
    file_bgr = None
    for j in range(len(scene_name_list)):
        if os.path.exists('../../category_bgr/' + scene_name_list[j] + '/' + category_name + '.npy'):
            temp = np.fromfile('../../category_bgr/' + scene_name_list[j] + '/' + category_name + '.npy',
                               dtype='float32').reshape(144, 192, 3)  # 144*192*3, bgr, 0-1
            if file_bgr is None:
                file_bgr = temp
            else:
                file_bgr = np.concatenate([file_bgr, temp], axis=1)  # 144*192m*3

    category_bgr.append(file_bgr)

test_scene = ['Cooperstown',
              'Denmark',
              'Elmira',
              'Hometown',
              'Maben',
              'Matoaca',
              'Rabbit',
              'Ribera',
              'Rosser',
              'Seward',
              'Sumas',
              # 'Rs'
              ]
model = E2EModel(action_space=action_space, device=device, category_num=category_num, category_bgr=category_bgr,
                 test_scene=test_scene, all_scene=scene_name_list)

width = 192  # 640*480, 192*144
height = 144
config_data = yaml.load(open("myExample.yaml", "r"), Loader=yaml.FullLoader)
config_data['image_width'] = width
config_data['image_height'] = height
# ***********************************************#


trainer = Trainer(action_space=action_space,
                  model=model,
                  initial_lr=initialLearningRate,
                  batch_size=batchSize,
                  device=device,
                  use_greedy=useGreedy,
                  lr_decay_step=lrDecayStep,
                  epsilon_greedy_decay_step=epsilonGreedyDecayStep)

if not training:
    summary_path = './record_test/' + date + '/'
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)

    # # test mode parameters
    model_path_for_testing = '../../train/IL_topo_semantic/record/2023-05-04/models/930.pkl'
    trainer.load(model_path_for_testing)
    model.eval().to(device)

    total_success_count = 0
    total_episodeCount = 0
    writer = SummaryWriter(summary_path)

    print('Testing start!\r\n')

    for scene_id in test_scene:
        scene_index = scene_name_list.index(scene_id)
        with open('../../../Gibson_Dataset_Sample/test_data/' + scene_id + '.json', 'r') as f:
            content = json.load(f)

        env = TurtleBotRobot(config_file=deepcopy(config_data), shape=(height, width, 3),
                             action_space=action_space,
                             device=device,
                             scene_id=scene_id)  # env.success_count = 0.0

        env.env.reset()

        print('%s load：' % scene_id)

        txt_path = summary_path + scene_id + '.txt'
        txt_head = "epo_id\t\tepo_length\t\tepisodeScore\t\ttarget_distance\t\tsuccess_rate\n"  # 表头信息
        create_txt(txt_path, txt_head)

        episode_id = 0  # 0~ len(content)
        for episode in content:
            start_pos = np.array([episode['startX'], episode['startY'], episode['startZ']])
            start_ori = np.array([0, 0, -np.sin(episode['startAngle'] / 2), np.cos(episode['startAngle'] / 2)])
            target_pos = np.array([episode['goalX'], episode['goalY'], episode['goalZ']])

            env.robot.set_position(target_pos)
            temp, _, _, _ = env.env.step(np.array([0.0, 0.0], dtype='float32'))
            target_bgr = temp['rgb']

            target_position_yaw = np.array([target_pos[0],
                                            target_pos[1],
                                            env.robot.get_rpy()[2]])

            current_bgr = env.reset(start_pos=start_pos, start_ori=start_ori, target_position_yaw=target_position_yaw)
            print('-' * 40)
            print('scene_id：', scene_id)
            print('target position:', env.target_x, env.target_y)
            print('target distance:', env.target_distance)

            # model.graph_init()  # 拓扑图初始化

            step = 0
            for step in range(max_steps_per_episode):

                selectedAction = trainer.work(current_bgr, target_bgr, type_="selectAction",
                                              current_position_yaw=np.array([env.robot.get_position()[0],
                                                                             env.robot.get_position()[1],
                                                                             env.robot.get_rpy()[2]]),
                                              target_position_yaw=target_position_yaw,
                                              scene_id=scene_id
                                              )

                newObservation, reward, done, info = env.step(selectedAction, target_bgr, total_success_count)

                current_bgr = newObservation
                env.episodeScore += reward[0]
                total_success_count = reward[1]

                if done:
                    break
                # if step >= 10:
                #     break

                trainer.global_step += 1
                env.current_step += 1

                progress_bar(env.current_step, max_steps_per_episode, total_episodeCount, len(content),
                             env.episodeScore)

            # # trainer.stats
            trainer.stats['scene_index'].append(scene_index)
            trainer.stats['cumulative_reward'].append(env.episodeScore)
            trainer.stats['episode_length'].append(step + 1)
            trainer.stats['target_dis'].append(env.target_distance)
            trainer.stats['success_rate'].append(total_success_count / float(total_episodeCount + 1))
            trainer.writeSummary(writer, total_episodeCount)

            write_txt(txt_path, episode_id, step + 1, env.episodeScore, env.target_distance,
                      env.success_count / float(episode_id + 1))

            env.episodeScoreList.append(env.episodeScore)
            print("\r\nTesting @ ")
            print("Episode:", episode_id, "score:", env.episodeScore, 'episode_length:', step + 1)
            print("success_rate", env.success_count / float(episode_id + 1), "target_dis:", env.target_distance)

            episode_id += 1
            total_episodeCount += 1

    print('Testing end!\r\n')
