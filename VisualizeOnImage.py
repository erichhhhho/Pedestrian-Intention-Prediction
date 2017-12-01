'''By He Wei Oct23 2017'''

import pickle
import argparse
import os
from utils import DataLoader
import numpy as np
import cv2
import  random


green = (0,255,0)
red = (0,0,255)

parser = argparse.ArgumentParser()
# Observed length of the trajectory parameter
parser.add_argument('--obs_length', type=int, default=8,
                    help='Observed length of the trajectory')
# Predicted length of the trajectory parameter
parser.add_argument('--pred_length', type=int, default=12,
                    help='Predicted length of the trajectory')
# Test dataset
parser.add_argument('--visual_dataset', type=int, default=2,
                    help='Dataset to be tested on')

# Model to be loaded
parser.add_argument('--epoch', type=int, default=56,
                    help='Epoch of model to be loaded')

# Parse the parameters
sample_args = parser.parse_args()

'''KITTI Training Setting'''

#save_directory = '/home/hesl/PycharmProjects/social-lstm-tf-HW/ResultofTrainingKITTI-13NTestonKITTI-17/save'
save_directory = '/home/hesl/PycharmProjects/srnn-pytorch/save/FixedPixel_150epochs/'+str(sample_args.visual_dataset)+'/'
save_directory += 'save_attention'

#save_directory ='/home/hesl/PycharmProjects/social-lstm-tf-HW/ResultofTrainingETH1TestETH0/save/'

with open(os.path.join(save_directory, 'config.pkl'), 'rb') as f:
    saved_args = pickle.load(f)

f = open('/home/hesl/PycharmProjects/srnn-pytorch/save/FixedPixel_150epochs/'+str(sample_args.visual_dataset)+'/save_attention/results.pkl', 'rb')
results = pickle.load(f)

dataset = [sample_args.visual_dataset]
data_loader = DataLoader(1, sample_args.pred_length + sample_args.obs_length, dataset, True, infer=True)

videopath=['/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/eth/eth/',
                '/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/eth/hotel/',
                '/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/ucy/zara01/',
                '/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/ucy/zara02/',
                '/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/ucy/univ/']

H_path=['/media/hesl/OS/Documents and Settings/N1701420F/Desktop/pedestrians/ewap_dataset/seq_eth/H.txt',
        '/media/hesl/OS/Documents and Settings/N1701420F/Desktop/pedestrians/ewap_dataset/seq_hotel/H.txt',
        '/media/hesl/OS/Documents and Settings/N1701420F/Desktop/pedestrians/ucy_crowd/data_zara01/H.txt',
        '/media/hesl/OS/Documents and Settings/N1701420F/Desktop/pedestrians/ucy_crowd/data_zara02/H.txt',
        '/media/hesl/OS/Documents and Settings/N1701420F/Desktop/pedestrians/ucy_crowd/data_students03/H.txt']

H=np.loadtxt(H_path[sample_args.visual_dataset])


skip_frame=10


#[7,16]
#print(data_loader.data[0][0].shape)
#
'''Visualize Ground Truth (u,v)'''
# for j in range(len(data_loader.frameList[0])):
#
#     #sourceFileName = "/home/hesl/PycharmProjects/social-lstm-tf-HW/data/KITTI-17/img1/" + str(j + 1).zfill(6) + ".jpg"
#     #Visualize ETH/hotel
#     #sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/eth/hotel/frame-" + str(int(data_loader.frameList[0][j])).zfill(3) + ".jpg"
#     #Eth/eth
#     #sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/eth/eth/frame-" + str(int(data_loader.frameList[0][j])).zfill(3)+ ".jpeg"
#     #UCY/univ
#     #sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/ucy/univ/frame-" + str(int(data_loader.frameList[0][j])+1).zfill(3) + ".jpg"
#     # UCY/zara01
#     sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/ucy/zara01/frame-" + str(int(data_loader.frameList[0][j])+1).zfill(3) + ".jpg"
#
#     # UCY/zara02
#     #sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/ucy/zara02/frame-" + str(int(data_loader.frameList[0][j])).zfill(3) + ".jpg"
#
#     print(sourceFileName)
#
#     avatar= cv2.imread(sourceFileName)
#     #drawAvatar= ImageDraw.Draw(avatar)
#     #print(avatar.shape)
#     xSize  = avatar.shape[1]
#     ySize = avatar.shape[0]
#     #print(data_loader.data[0][0][0])
#     for i in range(data_loader.maxNumPeds):
#          #print(i)
#
#          y=int(data_loader.data[0][j][i][2])
#          x=int(data_loader.data[0][j][i][1])
#          if x!=0 and y!=0:
#              print(x, y)
#
#          cv2.rectangle(avatar, (x  - 2, y  - 2), (x  + 2, y + 2), green,thickness=-1)
#          #drawAvatar.rectangle([(x  - 2, y  - 2), (x  + 2, y + 2)], fill=(255, 100, 0))
#
#     #drawAvatar.rectangle([(466, 139), (91 + 466, 139 + 193.68)])
#     #avatar.show()
#     cv2.imshow("avatar", avatar)
#     cv2.waitKey(0)


'''Visualize Ground Truth (x,y)'''
# for j in range(len(data_loader.frameList[0])):
#
#     #sourceFileName = "/home/hesl/PycharmProjects/social-lstm-tf-HW/data/KITTI-17/img1/" + str(j + 1).zfill(6) + ".jpg"
#     #Visualize ETH/hotel
#     #sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/eth/hotel/frame-" + str(int(data_loader.frameList[0][j])).zfill(3) + ".jpg"
#     #Eth/eth
#     #sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/eth/eth/frame-" + str(int(data_loader.frameList[0][j])).zfill(3)+ ".jpeg"
#     #UCY/univ
#     #sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/ucy/univ/frame-" + str(int(data_loader.frameList[0][j])+1).zfill(3) + ".jpg"
#     # UCY/zara01
#     sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/ucy/zara01/frame-" + str(int(data_loader.frameList[0][j])+1).zfill(3) + ".jpg"
#
#     # UCY/zara02
#     #sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/ucy/zara02/frame-" + str(int(data_loader.frameList[0][j])).zfill(3) + ".jpg"
#
#     print(sourceFileName)
#
#     avatar= cv2.imread(sourceFileName)
#
#     xSize  = avatar.shape[1]
#     ySize = avatar.shape[0]
#
#     for i in range(data_loader.maxNumPeds):
#
#         pos = np.ones(3)
#         y=data_loader.data[0][j][i][2]
#         x=data_loader.data[0][j][i][1]
#
#         pos[0] = x
#         pos[1] = y
#
#         pos = np.dot(pos, np.linalg.inv(H.transpose()))
#
#         print(pos[0] ,pos[1] )
#         u=int(np.around(pos[0] / pos[2]))
#         v=int(np.around(pos[1] / pos[2]))
#
#         if data_loader.data[0][j][i][0]!=0:lengthy
#             print(u, v)
#             cv2.rectangle(avatar, (u - 2, v - 2), (u + 2, v + 2), green, thickness=-1)
#
#
#     cv2.imshow("avatar", avatar)
#     cv2.waitKey(0)
#

print(results[0][1][0][2])

'''Visualize result of image coordinate data'''
#Each Frame
# for k in range(int(len(data_loader.frameList[0])/(sample_args.obs_length+sample_args.pred_length))):
#     #Each
#     for j in range(sample_args.obs_length+sample_args.pred_length):
#
#         sourceFileName = "/media/hesl/OS/Documents and Settings/N1701420F/Desktop/video/eth/eth/frame-" + str(int(data_loader.frameList[0][j+k*(sample_args.obs_length+sample_args.pred_length)])).zfill(3) + ".jpeg"
#         print(sourceFileName)
#
#         avatar= cv2.imread(sourceFileName)
#
#
#         #width
#         xSize  = avatar.shape[1]
#         #height
#         ySize = avatar.shape[0]
#
#
#         for i in range(data_loader.maxNumPeds):
#             if results[k][1][j][i][0] != 0:
#                 # Predicted
#                 yp = int(np.round(results[k][1][j][i][2] * ySize))
#                 xp = int(np.round(results[k][1][j][i][1] * xSize))
#                 cv2.rectangle(avatar, (xp - 2, yp - 2), (xp + 2, yp + 2), red, thickness=-1)
#
#             if results[k][0][j][i][0]!=0:
#                 # GT
#                 y = int(np.round(results[k][0][j][i][2] * ySize))
#                 x = int(np.round(results[k][0][j][i][1] * xSize))
#                 cv2.rectangle(avatar, (x - 2, y - 2), (x + 2, y + 2), green, thickness=-1)
#
#                 print('x,y')
#                 print(x,y)
#             if results[k][1][j][i][0] != 0 and results[k][0][j][i][0]!=0 and results[k][0][j][i][0]==results[k][1][j][i][0]:
#                 cv2.line(avatar, (x, y), (xp, yp), (255,0,0),1)
#
#         cv2.imshow("avatar", avatar)
#         #imagename='/home/hesl/PycharmProjects/social-lstm-tf-HW/plot/visualize-'+str(int(data_loader.frameList[0][j+k*(sample_args.obs_length+sample_args.pred_length)])).zfill(3)+'.png'
#         #cv2.imwrite(imagename, avatar)
#         cv2.waitKey(0)


print(int((len(data_loader.frameList[0])/10)/(sample_args.obs_length+sample_args.pred_length)))


'''Visualize result of unnormalized world coordinate data 10frame-by-10frame'''

# #Each Trajectory
# for k in range(int((len(data_loader.frameList[0])/10)/(sample_args.obs_length+sample_args.pred_length))):
#     #Each frame
#     for j in range(sample_args.obs_length+sample_args.pred_length):
#
#         sourceFileName = videopath[sample_args.visual_dataset]+"frame-" + str(int(k*(10*(sample_args.obs_length+sample_args.pred_length))+j*10+data_loader.frameList[0][0])).zfill(3) + ".jpg"
#
#         avatar= cv2.imread(sourceFileName)
#
#
#         xSize  = avatar.shape[1]
#         ySize = avatar.shape[0]
#         print(sourceFileName)
#
#         for i in range(len(results[k][2][j])):
#
#             x=results[k][0][j][results[k][2][j][i]][0]
#             y=results[k][0][j][results[k][2][j][i]][1]
#
#             xp=results[k][1][j][results[k][2][j][i]][0]
#             yp=results[k][1][j][results[k][2][j][i]][1]
#
#             pos = np.ones(3)
#             posp = np.ones(3)
#
#
#             pos[0] = x
#             pos[1] = y
#
#             posp[0] = xp
#             posp[1] = yp
#
#             pos = np.dot(pos, np.linalg.inv(H.transpose()))
#
#             v = int(np.around(pos[1] / pos[2]))
#             u = int(np.around(pos[0] / pos[2]))
#
#             posp = np.dot(posp, np.linalg.inv(H.transpose()))
#
#             vp = int(np.around(posp[1] / posp[2]))
#             up = int(np.around(posp[0] / posp[2]))
#
#             cv2.rectangle(avatar, (up - 2, vp - 2), (up + 2, vp + 2), red, thickness=-1)
#             cv2.rectangle(avatar, (u - 2, v - 2), (u + 2, v + 2), green, thickness=-1)
#
#             # print('uv:')
#             # print(u, v)
#
#             #
#             # if results[k][1][j][i][0] != 0 and results[k][0][j][i][0]!=0 and results[k][0][j][i][0]==results[k][1][j][i][0]:
#             #     cv2.line(avatar, (u, v), (up, vp), (255,0,0),1)
#
#         cv2.imshow("avatar", avatar)
#         #imagename='/home/hesl/PycharmProjects/social-lstm-pytorch/plot/'+str(sample_args.visual_dataset)+'/visualize-'+ str(int(k*(10*(sample_args.obs_length+sample_args.pred_length))+j*10+data_loader.frameList[0][0])).zfill(3) +'.png'
#         imagename = '/home/hesl/PycharmProjects/social-lstm-pytorch/plot'+'/visualize-' + str(int(
#             k * (10 * (sample_args.obs_length + sample_args.pred_length)) + j * 10 + data_loader.frameList[0][
#                 0])).zfill(3) + '.png'
#
#         #print(imagename)
#
#         cv2.imwrite(imagename, avatar)
#         #print(t)
#         #cv2.waitKey(0)
#


'''Visualize result of normalized world coordinate data 10frame-by-10frame'''

#Each Trajectory
# for k in range(int((len(data_loader.frameList[0])/10)/(sample_args.obs_length+sample_args.pred_length))):
#     #Each frame
#     for j in range(sample_args.obs_length+sample_args.pred_length):
#
#         sourceFileName = videopath[sample_args.visual_dataset]+"frame-" + str(int(k*(10*(sample_args.obs_length+sample_args.pred_length))+j*10+data_loader.frameList[0][0])).zfill(3) + ".jpg"
#
#         avatar= cv2.imread(sourceFileName)
#
#
#         xSize  = avatar.shape[1]
#         ySize = avatar.shape[0]
#         print(sourceFileName)
#
#         for i in range(len(results[k][2][j])):
#
#             x=results[k][0][j][results[k][2][j][i]][0]
#             y=results[k][0][j][results[k][2][j][i]][1]
#
#             xp=results[k][1][j][results[k][2][j][i]][0]
#             yp=results[k][1][j][results[k][2][j][i]][1]
#
#             # y=(((y)*(11.8703)))-9.4262
#             # x=(((x)*(18.6887)))-10.8239
#             #
#             # yp = (((yp ) * (11.8703)) ) - 9.4262
#             # xp = (((xp ) * (18.6887)) ) - 10.8239
#
#             pos = np.ones(3)
#             posp = np.ones(3)
#
#
#             pos[0] = x
#             pos[1] = y
#
#             posp[0] = xp
#             posp[1] = yp
#
#             pos = np.dot(pos, np.linalg.inv(H.transpose()))
#
#             # v = int(np.around(pos[1] / pos[2]))
#             # u = int(np.around(pos[0] / pos[2]))
#
#             u = int(np.round((x+1)*(xSize /2)))
#             v =  int(np.round((y+1)*(ySize /2)))
#
#             posp = np.dot(posp, np.linalg.inv(H.transpose()))
#             #
#             # vp = int(np.around(posp[1] / posp[2]))
#             # up = int(np.around(posp[0] / posp[2]))
#
#             up = int(np.round((xp + 1) * (xSize / 2)))
#             vp = int(np.round((yp + 1) * (ySize / 2)))
#
#             cv2.rectangle(avatar, (up - 2, vp - 2), (up + 2, vp + 2), red, thickness=-1)
#             cv2.rectangle(avatar, (u - 2, v - 2), (u + 2, v + 2), green, thickness=-1)
#             cv2.line(avatar, (u, v), (up, vp), (255, 0, 0), 1)
#             print('uv:')
#             print(u, v)
#             print('upvp:')
#             print(up, vp)
#
#             #
#             # if results[k][1][j][i][0] != 0 and results[k][0][j][i][0]!=0 and results[k][0][j][i][0]==results[k][1][j][i][0]:
#             #     cv2.line(avatar, (u, v), (up, vp), (255,0,0),1)
#
#         cv2.imshow("avatar", avatar)
#         imagename='/home/hesl/PycharmProjects/social-lstm-pytorch/plot/'+str(sample_args.visual_dataset)+'/visualize-'+ str(int(k*(10*(sample_args.obs_length+sample_args.pred_length))+j*10+data_loader.frameList[0][0])).zfill(3) +'.png'
#         #print(imagename)
#
#         #cv2.imwrite(imagename, avatar)
#         #print(t)
#         cv2.waitKey(0)

'''Visualize result of normalized pixel coordinate data 10frame-by-10frame'''
count=0
#print(results[0][0])
#Each Trajectory (20 frames)
for k in range(int((len(data_loader.frameList[0])/skip_frame)/(sample_args.obs_length+sample_args.pred_length))):

    gt_trajectory=[]
    pred_trajectory=[]

    #Each frame
    Max_num_ped = max([val for sublist in results[k][2] for val in sublist])
    #print(Max_num_ped)

    for j in range(sample_args.obs_length+sample_args.pred_length):
        #print(str(int(data_loader.frameList[0][count])).zfill(3))
        print('j = {}'.format(j))
        print('k = {}'.format(k))
        sourceFileName = videopath[sample_args.visual_dataset]+"frame-" + str(int(data_loader.frameList[0][count+j*skip_frame+k*(sample_args.obs_length+sample_args.pred_length)*skip_frame])).zfill(3) + ".jpg"
        # print('k range:',int((len(data_loader.frameList[0])/10)/(sample_args.obs_length+sample_args.pred_length)))
        # print('j range:',sample_args.obs_length+sample_args.pred_length)
        avatar= cv2.imread(sourceFileName)

        width  = avatar.shape[1]
        height = avatar.shape[0]
        print(sourceFileName)

        current_frame_gt=[]
        current_frame_pred=[]


        print(len(results[k][2][j]))
        #Ped
        for i in range(Max_num_ped):

            u=results[k][0][j][i][0]
            v=results[k][0][j][i][1]

            up=results[k][1][j][i][0]
            vp=results[k][1][j][i][1]

            if u!=0 or v!=0:
                u = int(np.round((u + 1) * (width / 2)))
                v = int(np.round((v + 1) * (height / 2)))

            if up != 0 or vp != 0:
                up = int(np.round((up + 1) * (width / 2)))
                vp  = int(np.round((vp + 1) * (height / 2)))

            gt_tmp=[u,v]
            pred_tmp=[up,vp]

            current_frame_gt.append(gt_tmp)
            current_frame_pred.append(pred_tmp)

            # cv2.rectangle(avatar, (up - 2, vp - 2), (up + 2, vp + 2), red, thickness=-1)
            # cv2.rectangle(avatar, (u - 2, v - 2), (u + 2, v + 2), green, thickness=-1)
            # cv2.line(avatar, (u, v), (up, vp), (255, 0, 0), 1)
            # # print('uv:')
            # print(u, v)
            # print('upvp:')
            # print(up, vp)

            #
            # if results[k][1][j][i][0] != 0 and results[k][0][j][i][0]!=0 and results[k][0][j][i][0]==results[k][1][j][i][0]:
            #     cv2.line(avatar, (u, v), (up, vp), (255,0,0),1)

        gt_trajectory.append(current_frame_gt)
        pred_trajectory.append(current_frame_pred)

        #print('current length',len(gt_trajectory))
        for ped in range(Max_num_ped):
            random.seed(ped)
            current_color=(random.randint(0,256),random.randint(0,256),random.randint(0,256))
            #print(type(11))

            for instance in range(len(gt_trajectory)):
                if gt_trajectory[instance][ped][0]!=0 or gt_trajectory[instance][ped][1]!=0:

                    #cv2.drawMarker(avatar, (gt_trajectory[instance][ped][0], gt_trajectory[instance][ped][1]), (0, 0, 255),markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=1, line_type=cv2.LINE_AA)
                    cv2.rectangle(avatar, (gt_trajectory[instance][ped][0] - 1, gt_trajectory[instance][ped][1] - 1), (gt_trajectory[instance][ped][0] + 1, gt_trajectory[instance][ped][1] + 1), color=current_color, thickness=-1)
                    if instance !=0 and (gt_trajectory[instance-1][ped][0]!=0 or gt_trajectory[instance-1][ped][1]!=0):
                        cv2.line(avatar, (gt_trajectory[instance-1][ped][0], gt_trajectory[instance-1][ped][1]), (gt_trajectory[instance][ped][0], gt_trajectory[instance][ped][1]),color=current_color,thickness= 1)

                if instance >sample_args.obs_length-1 and (pred_trajectory[instance][ped][0]!=0 or pred_trajectory[instance][ped][1]!=0):
                    cv2.drawMarker(avatar, (pred_trajectory[instance][ped][0], pred_trajectory[instance][ped][1]), color=current_color,markerType=cv2.MARKER_TILTED_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_AA)

        #cv2.imshow("avatar", avatar)
        imagename='/home/hesl/PycharmProjects/srnn-pytorch/plot/FixedPixel_150epochs/'+str(sample_args.visual_dataset)+'/visualize-'+ str(int(k*(10*(sample_args.obs_length+sample_args.pred_length))+j*10+data_loader.frameList[0][0])).zfill(3) +'.jpg'
        #print(imagename)

        cv2.imwrite(imagename, avatar)
        #print(t)
        #cv2.waitKey(0)

print(len(results))