import numpy as np
from copy import deepcopy
import time
from igibson.envs.igibson_env import iGibsonEnv
from matplotlib import pyplot as plt
import os
import yaml
import json
from yolov3.yolov3_model_create import create
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
YOLOv3 = create(name='yolov3/best.pt', pretrained=True,  # b*144*192*3, bgr, 0-255
                channels=3, classes=26,
                autoshape=True, verbose=True).to(device)
scene_name_list = ['Cooperstown', 'Denmark', 'Elmira', 'Hometown',
                   'Maben', 'Matoaca', 'Rabbit', 'Ribera',
                   'Rosser', 'Rs', 'Seward', 'Sumas']


def obj_detect(bgr_):
    bgr = np.expand_dims(bgr_, axis=0)  # 1*144*192*3, 0-1
    yolov3_input = np.round(bgr * 255).astype(np.uint8)  # 1*144*192*3, 0-255
    obj_detect_results = YOLOv3(yolov3_input[0]).detect_result()

    detect_features = np.zeros(26)
    if obj_detect_results[0] != 0:
        for j in range(len(obj_detect_results[0])):
            index = int(obj_detect_results[0][j]['cls'])
            detect_features[index] = 1.0

    return detect_features


def get_target_direction(robot_p_t, target_p):
    target_direction = np.zeros(8)

    dx = target_p[0] - robot_p_t[0]
    dy = target_p[1] - robot_p_t[1]
    target_theta = np.arctan2(dy, dx)

    robot_t = robot_p_t[2]

    target_theta = target_theta - robot_t
    if target_theta < -np.pi:
        target_theta = target_theta + 2 * np.pi
    if target_theta > np.pi:
        target_theta = target_theta - 2 * np.pi

    a = np.abs(target_theta - np.pi)
    index = int(np.floor_divide(a, np.pi / 4))
    if index == 8:
        index = 7
    target_direction[index] = 1.0

    return target_direction


dt = [
    np.array([
        [np.cos(-0.08334 * np.pi), -np.sin(-0.08334 * np.pi), 0, 0],
        [np.sin(-0.08334 * np.pi), np.cos(-0.08334 * np.pi), 0, 0],
        [0, 0, 1, 0.007],
        [0, 0, 0, 1]
    ]),
    np.array([
        [np.cos(0.08334 * np.pi), -np.sin(0.08334 * np.pi), 0, 0],
        [np.sin(0.08334 * np.pi), np.cos(0.08334 * np.pi), 0, 0],
        [0, 0, 1, 0.007],
        [0, 0, 0, 1]
    ]),

]


def Quat2Rotation(x, y, z, w):
    l1 = np.stack([1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y], axis=0)
    l2 = np.stack([2 * x * y - 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z + 2 * w * x], axis=0)
    l3 = np.stack([2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2], axis=0)
    T_w = np.stack([l1, l2, l3], axis=0)
    return T_w


def Rotation2Quat(pose):
    m11, m22, m33 = pose[0][0], pose[1][1], pose[2][2]
    m12, m13, m21, m23, m31, m32 = pose[0][1], pose[0][2], pose[1][0], pose[1][2], pose[2][0], pose[2][1]
    x, y, z, w = np.sqrt(m11 - m22 - m33 + 1) / 2, np.sqrt(-m11 + m22 - m33 + 1) / 2, np.sqrt(
        -m11 - m22 + m33 + 1) / 2, np.sqrt(m11 + m22 + m33 + 1) / 2
    Quat_ = np.array([
        [x, (m12 + m21) / (4 * x), (m13 + m31) / (4 * x), (m23 - m32) / (4 * x)],
        [(m12 + m21) / (4 * y), y, (m23 + m32) / (4 * y), (m31 - m13) / (4 * y)],
        [(m13 + m31) / (4 * z), (m23 + m32) / (4 * z), z, (m12 - m21) / (4 * z)],
        [(m23 - m32) / (4 * w), (m31 - m13) / (4 * w), (m12 - m21) / (4 * w), w]
    ], dtype=np.float32)
    index = np.array([x, y, z, w]).argmax()
    Quat = Quat_[index]
    return Quat


def control(theta, scan):
    if 0.458334 <= theta <= 0.541667:  # 0.41667  0.58334
        act = 0
    elif -0.5 <= theta < 0.458334:
        act = 2
    else:
        act = 1

    if not hasattr(control, 'turn'):
        control.turn = -1
        control.turn_count = 0
        control.stright_count = 0

    if scan.min() * 5.6 < 0.3 or control.turn > 0:
        # print('control')
        if control.turn < 0:
            index = scan.argmin()
            index_ = scan.argmax()
            portion = (index / len(scan)) * 90
            portion = portion // 15
            por_ = (index_ / len(scan)) * 90
            por_ = por_ / 15

            control.turn = 2 if por_ < 3 else 1
            control.turn_count = portion + 2
            control.stright_count = 2
            act = control.turn
        else:
            if not control.turn_count <= 0:
                control.turn_count -= 1
                act = control.turn
            elif not control.stright_count <= 0:
                control.stright_count -= 1
                act = 0
            else:
                control.turn = -1

    return act


def step(env, act, robot_T):
    if act == 0:
        state, reward, done, info = env.step(np.array([0.7, 0]))

    elif act == 1 or act == 2:
        T = np.matmul(robot_T, dt[act - 1])
        pos = T[:3, -1]
        ori = Rotation2Quat(T[:3, :3])
        env.robots[0].set_position_orientation(pos, ori)
        state, reward, done, info = env.step(np.array([0.0, 0.0]))

    return state, reward, done, info


def save(count, sc,
         obs_list, action_list, scene_index_list, traj_index_list,
         current_obj_detect_list, next_obj_detect_list, pos_theta_list, target_dir_list,
         dist, pathdist):
    if not os.path.exists('IL_dataset_gibson_auto'):
        os.mkdir('IL_dataset_gibson_auto')
    if not os.path.exists('IL_dataset_gibson_auto/' + sc):
        os.mkdir('IL_dataset_gibson_auto/' + sc)
    if not os.path.exists('IL_dataset_gibson_auto/' + sc + '/' + '%04i' % count):
        os.mkdir('IL_dataset_gibson_auto/' + sc + '/' + '%04i' % count)
    root = 'IL_dataset_gibson_auto/' + sc + '/' + '%04i' % count

    obs_list.tofile(os.path.join(root, 'obs_list.npy'))
    action_list.tofile(os.path.join(root, 'action_list.npy'))
    scene_index_list.tofile(os.path.join(root, 'scene_index_list.npy'))
    traj_index_list.tofile(os.path.join(root, 'traj_index_list.npy'))
    current_obj_detect_list.tofile(os.path.join(root, 'current_obj_detect_list.npy'))
    next_obj_detect_list.tofile(os.path.join(root, 'next_obj_detect_list.npy'))
    pos_theta_list.tofile(os.path.join(root, 'pos_theta_list.npy'))
    target_dir_list.tofile(os.path.join(root, 'target_dir_list.npy'))
    with open(os.path.join(root, 'info.txt'), 'w') as f:
        f.write('distance:%f pathdist:%f' % (dist, pathdist))


root = 'navigation_scenarios/waypoints/full+'

scene = [
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
    'Sumas']

for sc in scene:
    # print(sc)

    scene_index = scene_name_list.index(sc)
    print('*' * 40)
    print(scene_index)

    with open(os.path.join(root, sc + '.json'), 'r') as f:
        content = json.load(f)

    config_data = yaml.load(open("turtlebot_nav.yaml", "r"), Loader=yaml.FullLoader)
    env = iGibsonEnv(config_file=deepcopy(config_data), scene_id=sc, mode='headless', action_timestep=2 / 5)
    count = 0

    for episode in content:
        # print(episode)

        obs_list = []  # length*8

        action_list = []  # length
        scene_index_list = []  # length
        traj_index_list = []  # length

        current_obj_detect_list = []  # length*26
        next_obj_detect_list = []  # length*26

        pos_theta_list = []  # length*3
        target_dir_list = []  # length*8

        target_pos_theta = None
        target_bgr = None

        flag = False

        state = env.reset()
        start_pos = np.array([episode['startX'], episode['startY'], episode['startZ']])
        start_ori = np.array([0, 0, -np.sin(episode['startAngle'] / 2), np.cos(episode['startAngle'] / 2)])
        end_pos = np.array([episode['goalX'], episode['goalY'], episode['goalZ']])

        env.robots[0].set_position_orientation(start_pos, start_ori)
        state, _, _, _ = env.step(np.array([0, 0]))
        episode['waypoints'].append([episode['goalX'], episode['goalY'], episode['goalZ']])
        waypoints = np.array(episode['waypoints'])
        collison_num = 0
        # map = deepcopy(env.scene.floor_map[episode['level']]) / 255
        step_num = 0

        current_bgr = state['rgb']
        current_obj_detect = obj_detect(current_bgr)
        next_bgr = None
        next_obj_detect = None

        for pos in waypoints:
            target_pos = np.array([pos[0], pos[1], pos[2], 1])
            collison_count = 0
            while True:
                robot_pos = env.robots[0].get_position()

                # xxx, yyy = env.scene.world_to_map(robot_pos[:2])
                # map[xxx, yyy] = 3
                # plt.imshow(map)
                # plt.pause(0.1)
                # plt.clf()

                if np.linalg.norm(robot_pos[:2] - target_pos[:2], ord=2) <= 0.2:
                    break
                x, y, z, w = env.robots[0].get_orientation()
                T = np.concatenate([Quat2Rotation(-x, -y, -z, w), robot_pos.reshape(3, 1)], axis=1)
                T = np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0)
                theta = np.matmul(np.linalg.inv(T), target_pos.reshape(4, 1))
                theta = theta.reshape(4)[:2]
                theta /= np.linalg.norm(theta, ord=2)
                theta = np.arctan2(theta[0], -theta[1]) / np.pi

                tar_theta = np.matmul(np.linalg.inv(T), np.concatenate([end_pos, np.array([1])]).reshape(4, 1))
                tar_theta = tar_theta.reshape(4)[:2]
                tar_theta /= np.linalg.norm(tar_theta, ord=2)
                tar_theta = np.arctan2(tar_theta[0], -tar_theta[1]) / np.pi

                T[:3, :3] = Quat2Rotation(x, y, z, w)
                robot_T = T

                act = control(theta, state['scan'])
                # rgb = state["rgb"]
                # observation.append(rgb)
                # target_theta.append(tar_theta)
                # action.append(act)
                # robot_pos_ori.append(deepcopy(robot_T))

                robot_pos_theta = np.array([env.robots[0].get_position()[0],
                                            env.robots[0].get_position()[1],
                                            env.robots[0].get_rpy()[2]])

                state, reward, done, info = step(env, act, robot_T)
                step_num += 1
                # print(state['scan'].min()*5.6, state['scan'].argmin())
                # print(act)

                next_bgr = state['rgb']
                next_obj_detect = obj_detect(next_bgr)

                obs_list.append(current_bgr)
                action_list.append(act)
                scene_index_list.append(scene_index)
                traj_index_list.append(count)

                current_obj_detect_list.append(current_obj_detect)
                next_obj_detect_list.append(next_obj_detect)

                pos_theta_list.append(robot_pos_theta)
                target_dir_list.append(get_target_direction(robot_pos_theta, end_pos))

                target_pos_theta = np.array([env.robots[0].get_position()[0],
                                             env.robots[0].get_position()[1],
                                             env.robots[0].get_rpy()[2]], dtype=np.float32)
                target_bgr = np.array(state['rgb'], dtype=np.float32)

                current_bgr = next_bgr
                current_obj_detect = next_obj_detect

                if info['collision_step'] > collison_num:
                    collison_num = info['collision_step']
                    collison_count += 1
                if collison_count > 5 or step_num > 500:
                    flag = True
                    break

            if flag == True:
                break

        if not flag:
            obs_list = np.stack(obs_list, 0).astype(np.float32)
            action_list = np.array(action_list).astype(np.uint8)
            scene_index_list = np.array(scene_index_list).astype(np.uint8)
            traj_index_list = np.array(traj_index_list).astype(np.uint8)

            current_obj_detect_list = np.stack(current_obj_detect_list, 0).astype(np.float32)
            next_obj_detect_list = np.stack(next_obj_detect_list, 0).astype(np.float32)

            pos_theta_list = np.stack(pos_theta_list, 0).astype(np.float32)
            target_dir_list = np.stack(target_dir_list, 0).astype(np.float32)

            if not os.path.exists('IL_dataset_gibson_auto'):
                os.mkdir('IL_dataset_gibson_auto')
            if not os.path.exists('IL_dataset_gibson_auto/' + sc):
                os.mkdir('IL_dataset_gibson_auto/' + sc)
            if not os.path.exists('IL_dataset_gibson_auto/' + sc + '/' + '%04i' % count):
                os.mkdir('IL_dataset_gibson_auto/' + sc + '/' + '%04i' % count)

            target_bgr.tofile('IL_dataset_gibson_auto/' + sc + '/' + '%04i' % count + '/target_bgr.npy')
            target_pos_theta.tofile('IL_dataset_gibson_auto/' + sc + '/' + '%04i' % count + '/target_pos_theta.npy')

            save(count, sc, obs_list, action_list, scene_index_list, traj_index_list,
                 current_obj_detect_list, next_obj_detect_list, pos_theta_list, target_dir_list,
                 episode['dist'], episode['pathDist'])
            count += 1
            print(sc, episode['dist'], episode['pathDist'])
