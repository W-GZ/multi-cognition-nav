import numpy as np
import torch
import dgl
import cv2

# 'Cooperstown', 'Denmark', 'Elmira', 'Hometown',
# 'Maben', 'Matoaca', 'Rabbit', 'Ribera',
# 'Rosser'11, 'Rs'11, 'Seward'18, 'Sumas'11]
scene_id = 'Sumas'

# edges__ = np.fromfile(scene_id + "/edges.npy", dtype='int64').reshape(2, -1)
resnet_current_input = np.fromfile(scene_id + "/resnet_current_input.npy", dtype='float32').reshape(-1, 3, 144, 192)  # (82944,)
alex_current_input = np.fromfile(scene_id + "/alex_current_input.npy", dtype='float32').reshape(-1, 3, 144, 192)  # (82944,)
position_yaw = np.fromfile(scene_id + "/position_yaw.npy", dtype='float32').reshape(-1, 3)
# print(edges__.shape)
# print(edges__.dtype)
# # # print(resnet_current_input__[0])
print(resnet_current_input.shape)
print(resnet_current_input.dtype)
print(alex_current_input.shape)
print(alex_current_input.dtype)

print(position_yaw.shape)
print(position_yaw.dtype)
# # print(position_yaw)

# for i in range(len(resnet_current_input)):
#     while True:
#         cv2.imshow('rgb', resnet_current_input[i].transpose((1, 2, 0)))
#         q = cv2.waitKey(1)
#         if q == ord('q'):
#             break

#
#
''' 4 '''
# G__ = dgl.DGLGraph()
# G__.add_nodes(len(resnet_current_input), {'x': torch.tensor(resnet_current_input, dtype=torch.float),
#                                           'y': torch.tensor(alex_current_input, dtype=torch.float)
#                                           })
#
# G__.add_edges(edges__[0], edges__[1])
#
# #
# edges = G__.all_edges()
# resnet_current_input = G__.ndata['x']  # N*3*144*192, bgr, 0-1
# alex_current_input = G__.ndata['y']
# edges = np.array([edges[0].numpy(),
#                   edges[1].numpy()])
# resnet_current_input = resnet_current_input.numpy()
# alex_current_input = alex_current_input.numpy()
#
# # print(edges)
# # print(edges.dtype)
# # # print(resnet_current_input[0])
# print(resnet_current_input.shape)
# print(resnet_current_input.dtype)
# print(alex_current_input.shape)
# print(alex_current_input.dtype)

# grid_map = np.fromfile(scene_id + "/grid_map.npy", dtype=np.uint8).reshape(160, 160)
# while True:
#     cv2.imshow('grid_map', grid_map)
#     cv2.imshow('grid_map_', cv2.resize(grid_map, (500, 500)))
#     q = cv2.waitKey(1)
#     if q == ord('q'):
#         break
