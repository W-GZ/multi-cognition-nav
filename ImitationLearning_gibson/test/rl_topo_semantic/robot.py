import numpy as np
import random
import logging
from igibson.envs.igibson_env import iGibsonEnv
import cv2
from collections import Counter
# from yolov3.ig_categories import ig_categories
import pybullet
import torch
from category_bgr.category import scene_name_list, category_name_list, target_name_list

scene_name_list = scene_name_list()  # 12
category_name_list = category_name_list()  # 26
target_name_list = target_name_list()


def minmax(a, low, up):
    return min(max(a, low), up)


class TurtleBotRobot:
    def __init__(self, config_file, shape, action_space, device, scene_id='Cooperstown'):
        super().__init__()
        self.success_count = 0.0
        '''config'''
        self.target_category = None
        self.config_file = config_file
        self.scene_id = scene_id  # scene[random.randint(0, 0)]  'Rs_int', 'Wainscott_0_int', 'Wainscott_1_int',
        self.env = iGibsonEnv(config_file=config_file, scene_id=self.scene_id,
                              mode="headless", action_timestep=0.4)  # gui_non_interactive, headless
        self.robot = self.env.robots[0]
        # self.obj_dict = self.env.scene.objects_by_name  # 每一个object, .keys() .values()
        # self.obj_name_keys = list(self.env.scene.objects_by_name.keys())

        self.action_space = action_space
        # self.key = None

        '''YoloV3'''
        # self.yolov3 = model.yolov3
        '''AlexNet'''
        # self.alexnet = model.alexnet
        self.device = device

        '''obs'''
        self.shape = shape
        self.bgr = np.zeros(shape)  # 144*192*3; 0-1
        '''reward'''
        self.episodeScore = 0
        self.episodeScoreList = []
        '''collision_step'''
        self.collision_step = 0
        self.collision = False
        '''target'''
        # self.target = self.env.task
        self.target_x = 0
        self.target_y = 0
        self.target_yaw = None
        self.target_distance = None
        self.target_distance_last = None
        '''lidar'''
        self.min_dist = 0
        self.left_dist = 0
        self.middle_dist = 0
        self.right_dist = 0
        '''Avoid obstacles'''
        self.action = 0
        self.count = 0
        self.turn = False
        '''env.step'''
        self.state = None
        self.reward = None
        self.done = False
        self.info = None
        self.current_step = 0
        '''collision'''
        self.delta_dis_list = []
        self.last_position = None
        '''Visualization'''
        self.map = np.array([[i for i in j] for j in self.env.scene.floor_map[self.env.task.floor_num]])  # 100*100
        x, y = np.where(self.map == 0)
        self.map[x, y] = 120

    def apply_action(self, action):
        '''action'''
        rotate_theta = np.pi / 12

        current_position = self.robot.get_position()  # x, y, z
        current_rpy = self.robot.get_rpy()  # roll, pitch, yaw

        if action == 0:  # forward
            self.state, self.reward, _, self.info = self.env.step(np.array([0.7, 0.0], dtype='float32'))
        elif action == 1:  # left
            yaw = current_rpy[2] + rotate_theta
            if yaw < -np.pi:
                yaw = yaw + 2 * np.pi
            if yaw > np.pi:
                yaw = yaw - 2 * np.pi

            next_position = current_position
            next_orientation = np.array(pybullet.getQuaternionFromEuler([current_rpy[0],  # roll
                                                                         current_rpy[1],  # pitch
                                                                         yaw]))  # yaw
            self.robot.set_position_orientation(next_position, next_orientation)
            self.state, self.reward, _, self.info = self.env.step(np.array([0.0, 0.0], dtype='float32'))
        elif action == 2:  # right
            yaw = current_rpy[2] - rotate_theta
            if yaw < -np.pi:
                yaw = yaw + 2 * np.pi
            if yaw > np.pi:
                yaw = yaw - 2 * np.pi

            next_position = current_position
            next_orientation = np.array(pybullet.getQuaternionFromEuler([current_rpy[0],  # roll
                                                                         current_rpy[1],  # pitch
                                                                         yaw]))  # yaw
            self.robot.set_position_orientation(next_position, next_orientation)
            self.state, self.reward, _, self.info = self.env.step(np.array([0.0, 0.0], dtype='float32'))
        else:
            self.state, self.reward, _, self.info = self.env.step(np.array([0.0, 0.0], dtype='float32'))

    def update_map(self):
        robot_x, robot_y = self.robot.get_position()[:2]
        # map
        index_x = int(self.map.shape[0] / 2 + 10 * robot_y)
        index_y = int(self.map.shape[1] / 2 + 10 * robot_x)
        index_x = minmax(index_x, 0, self.map.shape[0] - 1)
        index_y = minmax(index_y, 0, self.map.shape[1] - 1)

        if self.map[index_x, index_y] == 255:
            self.map[index_x, index_y] = 0

    def step(self, action, target_bgr, total_success_count):
        # lidar
        if self.state is not None:
            lidar = 5.6 * self.state['scan'].reshape(-1)
            left_dist = np.mean(lidar[-10:])
            middle_dist = np.mean(lidar[330:350])
            right_dist = np.mean(lidar[0:10])
            # print(left_dist, middle_dist, right_dist)

            if not self.turn:
                if left_dist <= 0.2 and middle_dist <= 0.2 and right_dist <= 0.2:  # 0.3
                    self.turn = True
                    if left_dist >= right_dist:
                        self.action = 1
                        self.count = 10  # 32
                        # print('left_back')
                    if left_dist < right_dist:
                        self.action = 2
                        self.count = 10  # 32
                        # print('right_back')
                    action = self.action

                elif min(left_dist, middle_dist, right_dist) <= 0.25:  # 0.55
                    if middle_dist <= left_dist or middle_dist <= right_dist:
                        self.turn = True
                        if left_dist >= right_dist:
                            self.action = 1
                            self.count = 5  # 14
                            # print('turn left')
                        if left_dist < right_dist:
                            self.action = 2
                            self.count = 5  # 10
                            # print('turn right')
                        action = self.action
                    if middle_dist > left_dist and middle_dist > right_dist:
                        # print('forward')
                        self.turn = False
                        # self.action = 0
                        # action = self.action
            elif self.turn:
                action = self.action
                self.count -= 1
                if self.count <= 0:
                    self.turn = False

        self.apply_action(action)

        robot_position = self.robot.get_position()[:2]
        self.delta_dis_list.append(np.sqrt((robot_position[0] - self.last_position[0]) ** 2 + (robot_position[1] - self.last_position[1]) ** 2))
        self.last_position = robot_position

        self.update_map()

        # cv2.namedWindow("map", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("map", 400, 400)  # 设置窗口大小
        # cv2.imshow("map", self.map)

        # # camera_pose = self.robot.get_position()
        # # camera_pose[-1] = 10
        # # view_direction = np.array([0, 0, -1.0])
        # # self.env.simulator.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 1, 0])
        # # frame = cv2.cvtColor(np.concatenate(self.env.simulator.renderer.render(modes="rgb"), axis=1), cv2.COLOR_RGB2BGR)
        # # cv2.imshow("viewer", frame)

        return (
            self.get_observations(),
            self.get_reward(action, target_bgr, total_success_count),
            self.is_done(),
            self.get_info(),
        )

    def get_observations(self):
        self.bgr = self.state['rgb']  # bgr, 144*192*3, 0-1

        return self.bgr

    def get_reward(self, action, target_bgr, total_success_count):
        # live reward
        reward = 0.0
        '''collision'''
        # # if self.info['collision_step'] > self.collision_step:
        # #     self.collision = True
        # #     reward -= 65
        # #     # self.collision_step = self.info['collision_step']
        # #     # if self.collision_step >= 20:
        # #     #     self.collision = True
        # #     # reward -= 10
        # if self.current_step > 60 and np.sum(np.array(self.delta_dis_list[-1:-51:-1])) < 0.1:
        #     self.collision = True
        #     reward -= 65
        # if abs(self.robot.get_rpy()[0]) >= 0.05 or abs(self.robot.get_rpy()[1]) >= 0.05:  # set position and orientation
        #     self.collision = True
        #     reward -= 65

        # if self.current_step > 60 and np.sum(np.array(self.delta_dis_list[-1:-51:-1])) < 0.8:  # set orientation
        #     self.collision = True
        #     reward -= 65

        '''search reward'''
        robot_x, robot_y = self.robot.get_position()[:2]
        self.target_distance = np.sqrt((robot_x - self.target_x) ** 2 + (robot_y - self.target_y) ** 2)
        if self.target_distance <= self.target_distance_last:
            reward += 0.5
            self.target_distance_last = self.target_distance
        if self.target_distance <= 1.0:
            reward += 100
            self.success_count += 1
            total_success_count += 1
            self.done = True

        '''lidar'''
        lidar = 5.6 * self.state['scan'].reshape(-1)
        left_dist = np.mean(lidar[-67:])
        middle_dist = np.mean(lidar[306:374])
        right_dist = np.mean(lidar[0:68])

        min_dist = None
        if action == 0:
            min_dist = (left_dist + middle_dist + right_dist) / 3
            self.min_dist = (self.left_dist + self.middle_dist + self.right_dist) / 3
        if action == 1:
            min_dist = 0.2 * left_dist + 1 / 3 * middle_dist + 7 / 15 * right_dist
            self.min_dist = 0.2 * self.left_dist + 1 / 3 * self.middle_dist + 7 / 15 * self.right_dist
        if action == 2:
            min_dist = 7 / 15 * left_dist + 1 / 3 * middle_dist + 0.2 * right_dist
            self.min_dist = 7 / 15 * self.left_dist + 1 / 3 * self.middle_dist + 0.2 * self.right_dist

        if min(left_dist, middle_dist, right_dist) < 0.7:
            if min_dist - self.min_dist < 0:
                reward += 2 * (min_dist - self.min_dist)
            else:
                reward += self.min_dist - min_dist
        if min(left_dist, middle_dist, right_dist) > 1.5:
            reward += 0.1

        self.left_dist = left_dist
        self.middle_dist = middle_dist
        self.right_dist = right_dist
        self.min_dist = min_dist

        # '''alexnet'''
        # current_bgr = self.state['rgb']  # bgr, 144*192*3, 0-1
        # current_bgr = np.expand_dims(current_bgr, axis=0)  # 1*144*192*3, 0-1
        # target_bgr = np.expand_dims(target_bgr, axis=0)  # 1*144*192*3, 0-1
        #
        # temp1 = np.array(current_bgr) * 255  # b*144*192*3  -scaled to [0,255]
        # alex_current_input = torch.Tensor((temp1 / 127.5 - 1)[:, :, :, ].transpose((0, 3, 1, 2))).to(self.device)  # 1*3*144*192  -scaled to [-1,1]
        # temp2 = np.array(target_bgr) * 255  # b*144*192*3  -scaled to [0,255]
        # alex_target_input = torch.Tensor((temp2 / 127.5 - 1)[:, :, :, ].transpose((0, 3, 1, 2))).to(self.device)  # 1*3*144*192  -scaled to [-1,1]
        # likely = torch.clamp(self.alexnet.forward(alex_current_input, alex_target_input, retPerLayer=False), 0, 1).cpu().detach().numpy()
        # if likely < 0.2:
        #     reward += 1.0

        return reward, total_success_count

    def is_done(self):
        return self.done or self.collision
        # return self.collision

    def get_info(self):
        return

    def generate_target_position(self, target_bgr):
        """
        根据输入的target_bgr 确定目标类别，然后获取其真实位置
        """
        img = np.round(target_bgr * 255).astype(np.uint8)  # 144*192*3, 0-255, bgr
        result = self.yolov3(img).detect_result()
        # categories = ig_categories()
        self.target_category = result[0][0]['label']

        for i in range(len(self.obj_name_keys)):
            obj_name = self.obj_name_keys[i]
            obj_temp = self.obj_dict[obj_name]
            obj_position = obj_temp.get_position()
            obj_category = obj_temp.category
            if obj_category == self.target_category:
                self.target_x = obj_position[0]
                self.target_y = obj_position[1]
                break

        return

    def reset(self, start_pos, start_ori, target_position_yaw):
        """obs"""
        self.bgr = np.zeros(self.shape)  # 144*192*3; 0-1
        '''reward'''
        self.episodeScore = 0
        self.episodeScoreList = []
        '''collision_step'''
        self.collision_step = 0
        self.collision = False

        '''env reset'''
        self.state = self.env.reset()

        self.robot.set_position_orientation(start_pos, start_ori)
        self.state, self.reward, _, self.info = self.env.step(np.array([0.0, 0.0], dtype='float32'))
        self.bgr = self.state['rgb']

        '''target'''
        # # self.generate_target_position(target_bgr)
        # pos_theta = np.fromfile('../../category_bgr/' + scene_name_list[scene_index] + "/pos_theta.npy", dtype='float32').reshape(-1, 3)
        # target_name = category_name_list[target_index]
        #
        # # print(target_l[scene_index].index(target_name))
        #
        # target_position_yaw = pos_theta[target_name_list[scene_index].index(target_name)]

        self.target_x = target_position_yaw[0]
        self.target_y = target_position_yaw[1]
        self.target_yaw = target_position_yaw[2]

        robot_x, robot_y = self.robot.get_position()[:2]
        self.target_distance = np.sqrt((robot_x - self.target_x) ** 2 + (robot_y - self.target_y) ** 2)
        self.target_distance_last = self.target_distance

        '''lidar'''
        self.min_dist = 0
        self.left_dist = 0
        self.middle_dist = 0
        self.right_dist = 0
        '''Avoid obstacles'''
        self.action = 0
        self.count = 0
        self.turn = False
        '''env.step'''
        self.state = None
        self.reward = None
        self.done = False
        self.info = None
        self.current_step = 0
        '''collision'''
        self.delta_dis_list = []
        self.last_position = self.robot.get_position()[:2]
        '''Visualization'''
        self.map = np.array([[i for i in j] for j in self.env.scene.floor_map[self.env.task.floor_num]])
        x, y = np.where(self.map == 0)
        self.map[x, y] = 120

        index_x = int(self.map.shape[0] / 2 + 10 * self.target_y)
        index_y = int(self.map.shape[1] / 2 + 10 * self.target_x)
        index_x = minmax(index_x, 0, self.map.shape[0] - 1)
        index_y = minmax(index_y, 0, self.map.shape[1] - 1)
        self.map[(index_x - 1):(index_x + 2), (index_y - 1):(index_y + 2)] = 0

        print("Resetting environment")

        return self.bgr
