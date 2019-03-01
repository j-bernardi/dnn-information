import torch, sys
import torch.nn.functional as F
import numpy as np

# Epochs should be 160000 iterations total - have 88 data?
# other location "data/input_tensors/sample_scans/"
# TEMP batch_size = 8
# TEMP epochs = 160000 // 88
params = {
    "epochs" : 1,
    "lr_0" : 0.0001,
    "batch_size" : 1,
    "workers" : 4,
    "voxel_size" : 9,
    "label_smoothing" : 0.1,
    "validation_split" : 0.2,
    "device" : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "lr_idxs_array": np.array([0, 0.1, 0.2, 0.5, 0.7, 0.9, 0.95]),
    "lr_array": np.array([1, 0.5, 0.25, 0.125, 0.015625, 0.00390625, 0.001953125]),
    "save_location": "models/unet/saved_models",
    "scan_location": "data/input_tensors/segmentation_data/datasets/"
}

# TODO - transforms - handle the dataset...

def calc_loss(pred, gold, smoothing=0, one_hot=True):
    """
    Calc CEL and apply label smoothing.
    Came from:
        https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py
    Inputs:
        pred - b,C,X,Y,(Z) tensor of floats - indicating predicted class probabilities
        gold - b,X,Y,(Z) tensor of integers indicating labelled class
    """

    gold = gold.long()

    if one_hot:
        
        #gold = gold.contiguous().view(-1)
    
        eps = 0.1
        n_class = pred.size(1)

        if len(gold.shape) == 3:
            one_hot = make_one_hot(gold.view(gold.size(0), 1, gold.size(1), gold.size(2)))
        else:
            raise NotImplementedError("Only implemented for batch x X x Y (3D)\nGot %s." %\
                (len(gold.shape)))
    
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0) # NOT SURE WHAT DOES - JUST COPIED
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        # loss from pred and gold, not one-hot
        loss = F.cross_entropy(pred, gold, ignore_index=0, reduction='sum')

    return loss

def calc_adj(class_tensor):
    """
    Input: 
        Am (X x Y) matrix of n class integers.

    Output:
        A matrix of dimension nxn 
            signalling adjacency of class: row to class: col
    """
    def reduce(tens, axis):
        """Reduces 2d tensor in 0 and 1 dims by 1."""
        if axis == 0:
            return tens[:tens.shape[0]-1, :tens.shape[1]]
        elif axis == 1:
            return tens[:tens.shape[0], :tens.shape[1]-1]
        elif axis == (0,1):
            return tens[:tens.shape[0]-1, :tens.shape[1]-1]
        else:
            raise NotImplementedError

    if len(class_tensor.shape) != 2:
        raise NotImplementedError

    class_tensor = class_tensor.astype(int)
    n = np.max(class_tensor) + 1
    
    # Init
    adj_matrix = np.zeros((n, n))

    # A matrix of the classes under, right, diag of corresponding class
    class_under = reduce(np.roll(class_tensor, -1, axis=0), axis=0)
    class_right = reduce(np.roll(class_tensor, -1, axis=1), axis=1)
    class_diag = reduce(np.roll(class_tensor, (-1,-1), axis=(0,1)), axis=(0,1))

    # Reduce as don't count adjacency to under/right final row/col
    reduced_class_tensor_under = reduce(class_tensor, axis=0)
    reduced_class_tensor_right = reduce(class_tensor, axis=1)
    reduced_class_tensor_diag = reduce(class_tensor, axis=(0,1))

    # For each class in the class tensor
    for c in range(n):

        # find where equal, all the adjacencies are in under, right, diag. Count dictionary returns
        under_unique, under_counts = np.unique(np.where(reduced_class_tensor_under==c, class_under, -1), return_counts=True)
        under_counts = dict(zip(under_unique, under_counts))

        right_unique, right_counts = np.unique(np.where(reduced_class_tensor_right==c, class_right, -1), return_counts=True)
        right_counts = dict(zip(right_unique, right_counts))
        
        diag_unique, diag_counts = np.unique(np.where(reduced_class_tensor_diag==c, class_diag, -1), return_counts=True)
        diag_counts = dict(zip(diag_unique, diag_counts))
        
        # Count occurences per class 
        for a in range(n):
            if a in under_counts:
                adj_matrix[c, a] += under_counts[a]
            if a in right_counts:
                adj_matrix[c, a] += right_counts[a]
            if a in diag_counts:
                adj_matrix[c, a] += diag_counts[a]

    return adj_matrix

def make_one_hot(tens, C=9):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x C x (D) x H x W, where N is batch size, depth optional. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x D x H x W, where C is class number. One-hot encoded.
    '''
    if len(tens.shape) == 5:
        one_hot = torch.cuda.FloatTensor(tens.size(0), C, tens.size(2), tens.size(3), tens.size(4)).zero_()
    elif len(tens.shape) == 4:
        one_hot = torch.cuda.FloatTensor(tens.size(0), C, tens.size(2), tens.size(3)).zero_()
    elif tens.size(1) != 1:
        raise Exception("Got tens class shape {}\nExpected 1.".format(tens.size(1)))
    else:
        print("Got tensor of length", tens.shape)
        print("Expected 4 or 5 dimensions (N x C x (D) x H x W)")
        raise NotImplementedError

    target = one_hot.scatter_(1, tens.data, 1)
    
    # target = Variable(target)
        
    return target

def make_one_hot_mine(tens, like):

    #one_hot_labels = gold.view(-1, 1).type(torch.long) # OLD
    #print("gold", gold.shape)
    one_hot_labels = tens.type(torch.long)
    #print("one hot labels", one_hot_labels.shape)
    one_hot = torch.zeros_like(like).scatter(1, one_hot_labels, 1)
    #print("one hot", one_hot.shape)
    return one_hot

if __name__ == "__main__":

    """
    USAGE :
    for i, data in enumerate(trainloader, 0):

        inputs, labels, _ = data
            
        inputs, labels = inputs.float().to(params["device"]), labels.to(params["device"])

        adj = calc_adj(labels[0, :, :].cpu().numpy().astype(int))
    """

    tst = np.genfromtxt('data/tests/test_matrix.csv', delimiter=',')
    print(tst.shape)
    print(tst)
    adj = calc_adj(tst)

    print(adj)