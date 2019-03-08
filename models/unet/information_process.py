"""
Calculate the information in the network

CREDIT: 
    https://github.com/makezur/information_bottleneck_pytorch
"""

from multiprocessing import cpu_count
from joblib import Parallel, delayed

import training_metadata as tm

import warnings
import numpy as np
import numba

NUM_CORES = cpu_count()
warnings.filterwarnings("ignore")

@numba.jit
def entropy(probs):
    #print("returning en")
    return -np.sum(probs * np.ma.log2(probs))

@numba.jit
def joint_entropy(unique_inverse_x, unique_inverse_y, bins_x, bins_y):

    joint_distribution = np.zeros((bins_x, bins_y))
    np.add.at(joint_distribution, (unique_inverse_x, unique_inverse_y), 1)
    joint_distribution /= np.sum(joint_distribution)

    #print("returning je")
    return entropy(joint_distribution)

@numba.jit
def layer_information(layer_output, bins, py, px, unique_inverse_x, unique_inverse_y):
    ws_epoch_layer_bins = bins[np.digitize(layer_output, bins) - 1]
    ws_epoch_layer_bins = ws_epoch_layer_bins.reshape(len(layer_output), -1)

    unique_t, unique_inverse_t, unique_counts_t = np.unique(
        ws_epoch_layer_bins, axis=0,
        return_index=False, return_inverse=True, return_counts=True
    )

    pt = unique_counts_t / np.sum(unique_counts_t)

    # # I(X, Y) = H(Y) - H(Y|X)
    # # H(Y|X) = H(X, Y) - H(X)

    x_entropy = entropy(px)
    y_entropy = entropy(py)
    t_entropy = entropy(pt)

    x_t_joint_entropy = joint_entropy(unique_inverse_x, unique_inverse_t, px.shape[0], layer_output.shape[0])
    y_t_joint_entropy = joint_entropy(unique_inverse_y, unique_inverse_t, py.shape[0], layer_output.shape[0])

    print("returning layer info")

    return {
        'local_IXT': t_entropy + x_entropy - x_t_joint_entropy,
        'local_ITY': y_entropy + t_entropy - y_t_joint_entropy
    }

@numba.jit
def calc_information_for_epoch(epoch_number, ws_epoch, bins, unique_inverse_x, unique_inverse_y, pxs, pys):
    """Calculate the information for all the layers for specific epoch"""
    information_epoch = []

    for i in range(len(ws_epoch)):
        information_epoch_layer = layer_information(
            layer_output=ws_epoch[i],
            bins=bins,
            unique_inverse_x=unique_inverse_x,
            unique_inverse_y=unique_inverse_y,
            px=pxs, py=pys
        )
        information_epoch.append(information_epoch_layer)
    information_epoch = np.array(information_epoch)

    # print('Processed epoch {}'.format(epoch_number))
    print("Returning cife")
    return information_epoch

@numba.jit
def extract_probs(label, x):
    """calculate the probabilities of the given data and labels p(x), p(y) and (y|x)"""
    
    # Probability across each of the present (one-hot) classes
    print("label", label.shape)
    if len(label.shape) == 4:
        pys = np.sum(label, axis=(0,2,3)) / float(label.shape[0] * label.shape[2] * label.shape[3])
    elif len(label.shape) == 3:
        pys = np.sum(label, axis=(0,2)) / float(label.shape[0] * label.shape[2])
    elif len(label.shape == 2):
        pys = np.sum(label, axis=0) / float(label.shape[0])
    
    assert pys.shape[0] == label.shape[1]
    assert len(pys.shape) == 1

    P = 0.
    for p in pys:
        assert p < 1
        P += p
    assert P == 1.

    # The number of unique x values present in each sample

    # x used to be (len_train_set, positions) and 1, 0 depending on whether position is filled
    # Different for segmentation, I suppose?
    unique_x, unique_x_indices, unique_inverse_x, unique_x_counts =\
        np.unique(x, axis=0, return_index=True, return_inverse=True, return_counts=True)

    # the probability of each value of x 
    pxs = unique_x_counts / np.sum(unique_x_counts)

    print("pxs", pxs.shape)

    unique_array_y, unique_y_indices, unique_inverse_y, unique_y_counts =\
        np.unique(label, axis=0, return_index=True, return_inverse=True, return_counts=True)

    return pys, None, unique_x, unique_inverse_x, unique_inverse_y, pxs

def get_information(ws, x, label, num_of_bins, every_n=1, return_matrices=False):
    """
    Calculate the information for the network for all the epochs and all the layers
    layers one hot
    ws.shape =  [n_epoch, n_layers, n_params]
    ws --- outputs of all layers for all epochs
    n_params are not in the numpy array because the number of channels varies per layer
    """

    print('Start calculating the information...')

    if x.shape[1] == 1:
        x = np.squeeze(x, axis=1)

    # TODO - could cast last 3 layers (channels, x, y) into 1d to match the original?
    # Would do for x and y in the same manner
    ## TEMP - try it
    print("BEFORE")
    print("x", x.shape)
    print("lab", label.shape)
    
    new_x = np.reshape(x, (x.shape[0], -1))
    label = np.reshape(label, (label.shape[0], label.shape[1], -1))

    assert new_x.shape[0] == x.shape[0] and new_x.size == x.size
    assert new_x.shape[0] == label.shape[0]
    assert new_x.shape[1] == label.shape[2]
    del x
    
    print("AFTER")
    print("x", new_x.shape)
    print("lab", label.shape)

    # TODO - check this was the case to maintain an extra channel in label in the original code

    # Number of bins for inputs
    bins = np.linspace(-1, 1, num_of_bins)
    label = label.astype(np.float)#.numpy().astype(np.float)
    pys, _, unique_x, unique_inverse_x, unique_inverse_y, pxs = extract_probs(label, new_x)

    print("unique x", unique_x.shape)
    print("unique_inverse_x", unique_inverse_x.shape)
    print("unique_inverse_y", unique_inverse_y.shape)
    print("pxs", pxs.shape)

    sys.exit()

    # NON PARALLEL - TEMP
    """
    for i, epoch_output in enumerate(ws):
        if i % every_n != 0: 
            continue
        
        ## TODO - first dimension should be the length of the training sample ##
        ## Potentially reshape to (size(0), -1)
        for j in range(len(epoch_output)):
            print("items shape", epoch_output[j].shape)

        print("bins", bins.shape)
        print("u in x", unique_inverse_x.shape)
        print("u in y", unique_inverse_y.shape)
        print("pxs", pxs.shape)
        print("pys", pys.shape)

        calc_information_for_epoch(i, epoch_output, bins, unique_inverse_x, unique_inverse_y, pxs, pys)
    
    """

    print("Starting parallel")

    # PARALLEL
    with Parallel(n_jobs=NUM_CORES, prefer='threads') as parallel:
        information_total = parallel(
            delayed(calc_information_for_epoch)(
                i, epoch_output, bins, unique_inverse_x, unique_inverse_y, pxs, pys
            ) for i, epoch_output in enumerate(ws) if i % every_n == 0
        )

    print("Finished parallel")

    if not return_matrices:
        return information_total
    
    else:

        ixt_matrix = np.zeros((len(information_total), len(ws[0])))
        ity_matrix = np.zeros((len(information_total), len(ws[0])))

        for epoch, layer_info in enumerate(information_total):
            for layer, info in enumerate(layer_info):
                ixt_matrix[epoch][layer] = info['local_IXT']
                ity_matrix[epoch][layer] = info['local_ITY']

        return ixt_matrix, ity_matrix