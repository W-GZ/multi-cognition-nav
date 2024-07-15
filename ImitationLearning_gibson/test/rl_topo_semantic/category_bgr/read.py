import numpy as np
import cv2
import os
import torchvision
import torch

from category import scene_name_list, category_name_list, target_name_list

scene_name_list = scene_name_list()
category_name_list = category_name_list()
target_name_list = target_name_list()

'target_bgr'
# scene_index = 0  # 0~11
# target_index = 2  # 0~25
#
# target_bgr = np.fromfile(scene_name_list[scene_index] + '/' + category_name_list[target_index] + '.npy', dtype='float32').reshape(144, 192, 3)
#
# while True:
#     cv2.imshow('%s_%s' % (scene_name_list[scene_index], category_name_list[target_index]), cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB))
#     q = cv2.waitKey(1)
#     if q != -1:
#         break


'''category_'''
# category_bgr = []
# for i in range(len(category_name_list)):  # 26
#     file_name = category_name_list[i]
#     file_bgr = None
#     for j in range(len(scene_name_list)):
#         if os.path.exists(scene_name_list[j] + '/' + file_name + '.npy'):
#             temp = np.fromfile(scene_name_list[j] + '/' + file_name + '.npy', dtype='float32').reshape(144, 192, 3)  # 144*192*3, bgr, 0-1
#             if file_bgr is None:
#                 file_bgr = temp
#             else:
#                 file_bgr = np.concatenate([file_bgr, temp], axis=1)  # 144*192m*3

#     category_bgr.append(file_bgr)
#
# temp = 0
# for i in range(len(category_name_list)):
#     print(category_bgr[i].shape)
#
#     # temp += category_bgr[i].shape[0]/144
#
#     # if category_bgr[i].shape[0] <= 144*4:
#     while True:
#         cv2.imshow('%s_%s' % (i, category_name_list[i]), cv2.cvtColor(category_bgr[i], cv2.COLOR_BGR2RGB))
#         q = cv2.waitKey(1)
#         if q != -1:
#             break
# print(temp)
#
# resnet = torchvision.models.resnet50(pretrained=True)  # b*3*144*192m, bgr, 0-1--->b*1000
# # category_bgr = torch.Tensor(np.array(category_bgr).transpose((0, 3, 1, 2)))  # 26*3*144*192m
# # categories_visual_feature = resnet(category_bgr)  # 26*3*144*192m--26*1000
#
# categories_visual_feature = None
# for i in range(len(category_bgr)):
#     temp_input = torch.Tensor(category_bgr[i].transpose((2, 0, 1))).unsqueeze(0)
#     # print(temp_input.shape)
#
#     temp_output = resnet(temp_input)
#     if categories_visual_feature is None:
#         categories_visual_feature = temp_output
#     else:
#         categories_visual_feature = torch.cat([categories_visual_feature, temp_output], 0)
#
# print(categories_visual_feature.shape)  # 26*1000

'''target_position'''
# scene_index = 5  # 0-11
# target_index = 10  # 0-25
#
# pos_theta = np.fromfile('../../../category_bgr/' + scene_name_list[scene_index] + "/pos_theta.npy", dtype='float32').reshape(-1, 3)
# target_name = category_name_list[target_index]
# target_bgr = np.fromfile('../../../category_bgr/' + scene_name_list[scene_index] + "/" + target_name + ".npy", dtype='float32').reshape(144, 192, 3)
# # while True:
# #     cv2.imshow('%s_%s' % (scene_list[scene_index], category_list[target_index]), cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB))
# #     p = cv2.waitKey(1)
# #     if p != -1:
# #         break
#
# # print('%s' % target_name, target_list[scene_index].index(target_name))
#
# target_position = pos_theta[target_name_list[scene_index].index(target_name)]
# print('%s' % target_name, target_position)
#
# for i in range(len(target_name_list[scene_index])):
#     name = target_name_list[scene_index][i]
#     print('%s' % name, pos_theta[i])
