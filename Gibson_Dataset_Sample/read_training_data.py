import os
import torch
import numpy as np
import random
import torch.nn.functional as F
import logging

logging.getLogger().setLevel(logging.INFO)
import torchvision.transforms.functional as f


class myFunctions:
    def __init__(self):
        super().__init__()

    def adjust_learning_rate(self, optimizer, initial_lr, lr_decay_step, global_episode):
        if lr_decay_step > 0:
            learning_rate = 0.9 * initial_lr * (lr_decay_step - global_episode) / lr_decay_step + 0.1 * initial_lr
            if global_episode > lr_decay_step:
                learning_rate = 0.1 * initial_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        else:
            learning_rate = initial_lr
        # stats['learning_rate'].append(learning_rate)
        return learning_rate

    def writeSummary(self, writer, stats, global_step):
        for key in stats:
            if len(stats[key]) > 0:
                stat_mean = float(np.mean(stats[key]))
                writer.add_scalar(tag='Info/{}'.format(key), scalar_value=stat_mean, global_step=global_step)
                stats[key] = []
        writer.flush()

    def read(self):
        # obs_list = None  # length*144*192*3

        action_list = None  # length
        scene_index_list = None  # length
        traj_index_list = None  # length

        # current_obj_detect_list = None  # length*26
        # next_obj_detect_list = None  # length*26
        #
        # pos_theta_list = None  # length*3
        # target_dir_list = None  # length*8

        epi_num = 0

        dataset_path = 'IL_dataset_gibson_auto/'
        train_scene = [
            'Cooperstown',
            'Denmark',
            'Elmira',
            'Hometown',
            'Maben',
            'Matoaca',
            'Rabbit',
            'Ribera',
            'Rosser',
            # 'Rs',
            'Seward',
            'Sumas'
        ]

        for aa in train_scene:
            target_path = dataset_path + aa
            if os.path.exists(target_path):
                traj_file = os.listdir(target_path)
                
                epi_num += len(traj_file)
                print(len(traj_file))

                for traj_index in range(len(traj_file)):
                    traj_index_ = "/%04i" % traj_index
                    traj_path = target_path + traj_index_

                    # obs = np.fromfile(traj_path + '/obs_list.npy', dtype=np.float32).reshape(-1, 144, 192, 3)

                    action = np.fromfile(traj_path + "/action_list.npy", dtype=np.uint8)
                    scene = np.fromfile(traj_path + "/scene_index_list.npy", dtype=np.uint8)
                    traj = np.fromfile(traj_path + "/traj_index_list.npy", dtype=np.uint8)

                    # current_obj_detect = np.fromfile(traj_path + "/current_obj_detect_list.npy", dtype=np.float32).reshape(-1,
                    #                                                                                                   26)
                    # next_obj_detect = np.fromfile(traj_path + "/next_obj_detect_list.npy", dtype=np.float32).reshape(-1, 26)
                    #
                    # pos_theta = np.fromfile(traj_path + "/pos_theta_list.npy", dtype=np.float32).reshape(-1, 3)
                    # target_dir = np.fromfile(traj_path + "/target_dir_list.npy", dtype=np.float32).reshape(-1, 8)

                    if action_list is None:
                        # obs_list = obs
                        action_list = action
                        scene_index_list = scene
                        traj_index_list = traj
                        # current_obj_detect_list = current_obj_detect
                        # next_obj_detect_list = next_obj_detect
                        # pos_theta_list = pos_theta
                        # target_dir_list = target_dir
                    else:
                        # obs_list = np.concatenate((obs_list, obs), axis=0)
                        action_list = np.concatenate((action_list, action), axis=0)
                        scene_index_list = np.concatenate((scene_index_list, scene), axis=0)
                        traj_index_list = np.concatenate((traj_index_list, traj), axis=0)

                        # current_obj_detect_list = np.concatenate((current_obj_detect_list, current_obj_detect), axis=0)
                        # next_obj_detect_list = np.concatenate((next_obj_detect_list, next_obj_detect), axis=0)
                        #
                        # pos_theta_list = np.concatenate((pos_theta_list, pos_theta), axis=0)
                        # target_dir_list = np.concatenate((target_dir_list, target_dir), axis=0)

        # return obs_list, action_list, scene_index_list, traj_index_list, current_obj_detect_list, next_obj_detect_list, pos_theta_list, target_dir_list
        return action_list, scene_index_list, traj_index_list, epi_num

myfunc = myFunctions()
action_list, scene_index_list, traj_index_list, epi_num = myfunc.read()

print("*"*20)
print(epi_num)
print(len(action_list))
print(len(scene_index_list))
print(len(traj_index_list))
# print(len(current_obj_detect_list))
# print(len(next_obj_detect_list))
# print(len(pos_theta_list))
# print(len(target_dir_list))

