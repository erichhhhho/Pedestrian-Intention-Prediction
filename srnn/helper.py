'''
Helper functions for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 3rd April 2017
'''
import numpy as np
import torch
from torch.autograd import Variable


def getVector(pos_list):
    '''
    Gets the vector pointing from second element to first element
    params:
    pos_list : A list of size two containing two (x, y) positions
    '''
    pos_i = pos_list[0]
    pos_j = pos_list[1]

    return np.array(pos_i) - np.array(pos_j)


def getMagnitudeAndDirection(*args):
    '''
    Gets the magnitude and direction of the vector corresponding to positions
    params:
    args: Can be a list of two positions or the two positions themselves (variable-length argument)
    '''
    if len(args) == 1:
        pos_list = args[0]
        pos_i = pos_list[0]
        pos_j = pos_list[1]

        vector = np.array(pos_i) - np.array(pos_j)
        magnitude = np.linalg.norm(vector)
        if abs(magnitude) > 1e-4:
            direction = vector / magnitude
        else:
            direction = vector
        return [magnitude] + direction.tolist()

    elif len(args) == 2:
        pos_i = args[0]
        pos_j = args[1]

        ret = torch.zeros(3)
        vector = pos_i - pos_j
        magnitude = torch.norm(vector)
        if abs(magnitude) > 1e-4:
            direction = vector / magnitude
        else:
            direction = vector

        ret[0] = magnitude
        ret[1:3] = direction
        return ret

    else:
        raise NotImplementedError('getMagnitudeAndDirection: Function signature incorrect')


def getCoef(outputs):
    '''
    Extracts the mean, standard deviation and correlation
    params:
    outputs : Output of the SRNN model
    '''
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]

    # Exponential to get a positive value for std dev
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    # tanh to get a value between [-1, 1] for correlation
    corr = torch.tanh(corr)

    return mux, muy, sx, sy, corr


def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent):
    '''
    Returns samples from 2D Gaussian defined by the parameters
    params:
    mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes
    Contains x-means, y-means, x-stds, y-stds and correlation
    nodesPresent : a list of nodeIDs present in the frame

    returns:
    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]

    numNodes = mux.size()[1]

    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    for node in range(numNodes):
        if node not in nodesPresent:
            continue
        mean = [o_mux[node], o_muy[node]]
        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]], [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y


def compute_edges(nodes, tstep, edgesPresent):
    '''
    Computes new edgeFeatures at test time
    params:
    nodes : A tensor of shape seq_length x numNodes x 2
    Contains the x, y positions of the nodes (might be incomplete for later time steps)
    tstep : The time-step at which we need to compute edges
    edgesPresent : A list of tuples
    Each tuple has the (nodeID_a, nodeID_b) pair that represents the edge
    (Will have both temporal and spatial edges)

    returns:
    edges : A tensor of shape numNodes x numNodes x 2
    Contains vectors representing the edges
    '''
    numNodes = nodes.size()[1]
    edges = (torch.zeros(numNodes * numNodes, 2)).cuda()
    for edgeID in edgesPresent:
        nodeID_a = edgeID[0]
        nodeID_b = edgeID[1]

        if nodeID_a == nodeID_b:
            # Temporal edge
            pos_a = nodes[tstep - 1, nodeID_a, :]
            pos_b = nodes[tstep, nodeID_b, :]

            edges[nodeID_a * numNodes + nodeID_b, :] = pos_a - pos_b
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)
        else:
            # Spatial edge
            pos_a = nodes[tstep, nodeID_a, :]
            pos_b = nodes[tstep, nodeID_b, :]

            edges[nodeID_a * numNodes + nodeID_b, :] = pos_a - pos_b
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)

    return edges


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

    if test_dataset == 0:
        width = 640
        height = 480
    else:
        width = 720
        height = 576

    for tstep in range(pred_length):

        for nodeID in assumedNodesPresent:

            if nodeID not in trueNodesPresent[tstep]:
                continue

            pred_pos = ret_nodes[tstep, nodeID, :]
            true_pos = nodes[tstep, nodeID, :]

            print('pred_pos={}'.format(pred_pos))

            print('true_pos={}'.format(true_pos))

            # transform pixel pos into meter
            pos = np.ones(3)
            posp = np.ones(3)

            # GT transformation
            u = int(round((true_pos[0] + 1) * width / 2))
            v = int(round((true_pos[1] + 1) * height / 2))

            print('u', u)
            print('v', v)

            if test_dataset == 0 or test_dataset == 1:
                pos[0] = v
                pos[1] = u
            else:
                pos[0] = u
                pos[1] = v

            true_pos_temp = torch.cuda.FloatTensor(np.dot(pos, H.transpose()))

            x = true_pos_temp[0] / true_pos_temp[2]
            y = true_pos_temp[1] / true_pos_temp[2]

            print('x', x)
            print('y', y)

            true_pos_temp[0] = x
            true_pos_temp[1] = y

            # Prediction Transform
            up = int(round((pred_pos[0] + 1) * width / 2))
            vp = int(round((pred_pos[1] + 1) * height / 2))

            if test_dataset == 0 or test_dataset == 1:
                posp[0] = vp
                posp[1] = up
            else:
                posp[0] = up
                posp[1] = vp

            pred_pos_temp = torch.cuda.FloatTensor(np.dot(posp, H.transpose()))

            xp = pred_pos_temp[0] / pred_pos_temp[2]
            yp = pred_pos_temp[1] / pred_pos_temp[2]

            pred_pos_temp[0] = xp
            pred_pos_temp[1] = yp

            print('up', up)
            print('vp', vp)

            print('xp', xp)
            print('yp', yp)

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

    if test_dataset==0:
        width=640
        height=480
    else:
        width=720
        height=576

    # Last time-step
    tstep = pred_length - 1
    for nodeID in assumedNodesPresent:

        if nodeID not in trueNodesPresent[tstep]:
            continue

        pred_pos = ret_nodes[tstep, nodeID, :]
        true_pos = nodes[tstep, nodeID, :]

        # transform pixel pos into meter
        pos = np.ones(3)
        posp = np.ones(3)

        # GT transformation
        u = int(round((true_pos[0] + 1) * width / 2))
        v = int(round((true_pos[1] + 1) * height / 2))

        if test_dataset == 0 or test_dataset == 1:
            pos[0] = v
            pos[1] = u
        else:
            pos[0] = u
            pos[1] = v

        true_pos_temp = torch.cuda.FloatTensor(np.dot(pos, H.transpose()))

        x = true_pos_temp[0] / true_pos_temp[2]
        y = true_pos_temp[1] / true_pos_temp[2]

        true_pos_temp[0] = x
        true_pos_temp[1] = y

        # Prediction Transform
        up = int(round((pred_pos[0] + 1) * width / 2))
        vp = int(round((pred_pos[1] + 1) * height / 2))

        if test_dataset == 0 or test_dataset == 1:
            posp[0] = vp
            posp[1] = up
        else:
            posp[0] = up
            posp[1] = vp

        pred_pos_temp = torch.cuda.FloatTensor(np.dot(posp, H.transpose()))

        xp = pred_pos_temp[0] / pred_pos_temp[2]
        yp = pred_pos_temp[1] / pred_pos_temp[2]

        pred_pos_temp[0] = xp
        pred_pos_temp[1] = yp

        error += torch.norm(pred_pos_temp - true_pos_temp, p=2)
        counter += 1

    if counter != 0:
        error = error / counter

    return error


def sample_gaussian_2d_batch(outputs, nodesPresent, edgesPresent, nodes_prev_tstep):
    mux, muy, sx, sy, corr = getCoef_train(outputs)

    next_x, next_y = sample_gaussian_2d_train(mux.data, muy.data, sx.data, sy.data, corr.data, nodesPresent)

    nodes = torch.zeros(outputs.size()[0], 2)
    nodes[:, 0] = next_x
    nodes[:, 1] = next_y

    nodes = Variable(nodes.cuda())

    edges = compute_edges_train(nodes, edgesPresent, nodes_prev_tstep)

    return nodes, edges


def compute_edges_train(nodes, edgesPresent, nodes_prev_tstep):
    numNodes = nodes.size()[0]
    edges = Variable((torch.zeros(numNodes * numNodes, 2)).cuda())
    for edgeID in edgesPresent:
        nodeID_a = edgeID[0]
        nodeID_b = edgeID[1]

        if nodeID_a == nodeID_b:
            # Temporal edge
            pos_a = nodes_prev_tstep[nodeID_a, :]
            pos_b = nodes[nodeID_b, :]

            edges[nodeID_a * numNodes + nodeID_b, :] = pos_a - pos_b
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)
        else:
            # Spatial edge
            pos_a = nodes[nodeID_a, :]
            pos_b = nodes[nodeID_b, :]

            edges[nodeID_a * numNodes + nodeID_b, :] = pos_a - pos_b
            # edges[nodeID_a * numNodes + nodeID_b, :] = getMagnitudeAndDirection(pos_a, pos_b)

    return edges


def getCoef_train(outputs):
    mux, muy, sx, sy, corr = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3], outputs[:, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr


def sample_gaussian_2d_train(mux, muy, sx, sy, corr, nodesPresent):
    o_mux, o_muy, o_sx, o_sy, o_corr = mux, muy, sx, sy, corr

    numNodes = mux.size()[0]

    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    for node in range(numNodes):
        if node not in nodesPresent:
            continue
        mean = [o_mux[node], o_muy[node]]

        cov = [[o_sx[node]*o_sx[node], o_corr[node]*o_sx[node]*o_sy[node]], [o_corr[node]*o_sx[node]*o_sy[node], o_sy[node]*o_sy[node]]]

        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y
