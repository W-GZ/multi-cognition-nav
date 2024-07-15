import cv2
import numpy as np
import torch

# a_raw = torch.load("adjmat.dat")
# x, y = np.where(a_raw == 1)
#
# # print(a_raw.dtype)  # int64
# # print(len(x))
# # print(x)
# # print(y)
#
# adjmat_manual = np.zeros((len(a_raw), len(a_raw)), dtype=int)
# # print(adjmat_manual.dtype)
# # print(len(adjmat_manual))
# for i in range(len(adjmat_manual)):
#     adjmat_manual[i, i] = 1


'''generate
08	door

00	armchair		00-15  09-15  15-20
09	floor_lamp
15	sofa            
20	tea_table


05	bookshelf		05-06  05-19  06-19  
06	booktable
19	swivel_chair

02	bed			02-03  02-04  02-23  03-04
03	bed_lamp
04	bedside_table
23	wardrobe


07	breakfast_table		07-18
18	straight_chair


10	fridge			10-11  10-12  10-13  10-17  10-24
11	kitchen_cabinet	11-12  11-13  11-17  11-24
12	microwave		12-13  12-17  12-24
13	oven			13-17  13-24
17	stove			17-24
24	washer			

16	standing_tv     16-22
22	tv_cabinet


01	bathtub         01-21  
14	shower          14-21
21	toilet          21-25  
25	washroom_cabinet
'''

'''
08	door            08-02  08-25

00	armchair
09	floor_lamp
15	sofa          	15-00  15-09  15-20  ||  15-19  15-06  15-16  15-22
20	tea_table


05	bookshelf		05-06  05-19  06-19  
06	booktable                             || 06-18
19	swivel_chair

02	bed			    02-03  02-04  02-23  || 02-16  02-22 || 02-25
03	bed_lamp        03-04
04	bedside_table
23	wardrobe


07	breakfast_table		07-18   ||  18-10  18-11  18-17
18	straight_chair


10	fridge			10-11  10-12  10-13  10-17  10-24
11	kitchen_cabinet	11-12  11-13  11-17  11-24
12	microwave		12-13  12-17  12-24
13	oven			13-17  13-24
17	stove			17-24
24	washer			

16	standing_tv     16-22
22	tv_cabinet


01	bathtub         
14	shower          
21	toilet          21-01  21-14  21-25  
25	washroom_cabinet    25-01  25-14
'''
adjmat_manual = np.zeros((26, 26), dtype=int)

index_2 = [
    [0],  # 0
    [1],  # 1
    [3, 4, 23, 16, 22, 25],  # 2
    [4],  # 3
    [4],  # 4
    [6, 19],  # 5
    [19, 18],  # 6
    [18],  # 7
    [2, 25],  # 8
    [9],  # 9
    [11, 12, 13, 17, 24],  # 10
    [12, 13, 17, 24],  # 11
    [13, 17, 24],  # 12
    [17, 24],  # 13
    [14],  # 14
    [0, 9, 20, 19, 6, 16, 22],  # 15
    [22],  # 16
    [24],  # 17
    [10, 11, 17],  # 18
    [19],  # 19
    [20],  # 20
    [1, 14, 25],  # 21
    [22],  # 22
    [23],  # 23
    [24],  # 24
    [1, 14],  # 25
]

for i in range(len(adjmat_manual)):
    for j in range(len(index_2[i])):
        adjmat_manual[i, index_2[i][j]] = 1

x, y = np.where(adjmat_manual == 1)
adjmat_manual[y, x] = 1

for i in range(len(adjmat_manual)):
    adjmat_manual[i, i] = 1

torch.save(adjmat_manual, "adjmat_manual_gibson.dat")

'''load'''
a_raw = torch.load("adjmat_manual_gibson.dat")
x, y = np.where(a_raw == 1)

# print(a_raw)
# print(a_raw.dtype)
# print(len(x))
# print(x)
# print(y)

for i in range(26):
    print(a_raw[i])

cv2.imshow('aa', cv2.resize(np.array(255*adjmat_manual, dtype=np.uint8), (260, 260)))
cv2.waitKey(10000)
