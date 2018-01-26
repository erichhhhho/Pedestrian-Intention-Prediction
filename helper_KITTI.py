import numpy as np
import torch
from torch.autograd import Variable

other=[[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[24.6152,-17.5380],[40.0037,3.4277]],[[14.2367,- 9.4110],[28.3571,2.6694]],[[0,0],[0,0]]]


def get_mean_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent,H,test_dataset):
    '''
    Computes average displacement error
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent : A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    Returns
    =======

    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length).cuda()
    counter = 0

    # if test_dataset == 0:
    #     width = 640
    #     height = 480
    # else:
    #     width = 720
    #     height = 576

    for tstep in range(pred_length):

        for nodeID in assumedNodesPresent:

            if nodeID not in trueNodesPresent[tstep]:
                continue

            pred_pos = ret_nodes[tstep, nodeID, :]
            true_pos = nodes[tstep, nodeID, :]

            print('pred_pos={}'.format(pred_pos))

            print('true_pos={}'.format(true_pos))

            true_pos_temp = torch.cuda.FloatTensor(2)
            pred_pos_temp = torch.cuda.FloatTensor(2)


            # Z
            pred_pos_temp[1] = (pred_pos[1] + 1) * (other[test_dataset][1][0] * 0.5) + other[test_dataset][1][1]
           # print('other[test_dataset][1][0]',other[test_dataset][1][0])
            # X
            pred_pos_temp[0] = (pred_pos[0] + 1) * (other[test_dataset][0][0] * 0.5)+other[test_dataset][0][1]

            # Z
            true_pos_temp[1] = (true_pos[1] + 1) * (other[test_dataset][1][0] * 0.5) + other[test_dataset][1][1]
            # X
            true_pos_temp[0] = (true_pos[0] + 1) * (other[test_dataset][0][0] * 0.5) + other[test_dataset][0][1]

            print('pred_pos_temp={}'.format(pred_pos_temp))
            print('true_pos_temp={}'.format(true_pos_temp))

            error[tstep] += torch.norm(pred_pos_temp - true_pos_temp, p=2)
            counter += 1

        if counter != 0:
            error[tstep] = error[tstep] / counter

    return torch.mean(error)


def get_final_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent,H,test_dataset):
    '''
    Computes final displacement error
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent : A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    Returns
    =======

    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = 0
    counter = 0

    # if test_dataset==0:
    #     width=640
    #     height=480
    # else:
    #     width=720
    #     height=576

    # Last time-step
    tstep = pred_length - 1
    for nodeID in assumedNodesPresent:

        if nodeID not in trueNodesPresent[tstep]:
            continue

        pred_pos = ret_nodes[tstep, nodeID, :]
        true_pos = nodes[tstep, nodeID, :]

        print('pred_pos={}'.format(pred_pos))
        print('true_pos={}'.format(true_pos))

        true_pos_temp = torch.cuda.FloatTensor(2)
        pred_pos_temp = torch.cuda.FloatTensor(2)

        # Z
        pred_pos_temp[1] = (pred_pos[1] + 1) * (other[test_dataset][1][0] * 0.5) + other[test_dataset][1][1]
        # X
        pred_pos_temp[0] = (pred_pos[0] + 1) * (other[test_dataset][0][0] * 0.5) + other[test_dataset][0][1]

        # Z
        true_pos_temp[1] = (true_pos[1] + 1) * (other[test_dataset][1][0] * 0.5) + other[test_dataset][1][1]
        # X
        true_pos_temp[0] = (true_pos[0] + 1) * (other[test_dataset][0][0] * 0.5) + other[test_dataset][0][1]

        print('pred_pos_temp={}'.format(pred_pos_temp))
        print('true_pos_temp={}'.format(true_pos_temp))

        error += torch.norm(pred_pos_temp - true_pos_temp, p=2)
        counter += 1

    if counter != 0:
        error = error / counter

    return error