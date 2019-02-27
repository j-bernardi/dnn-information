import torch
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
 # OLD
def cal_loss_depracated(pred, gold, smoothing=0, one_hot=True):
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
        
        n_class = pred.size(1)

        # Make predicted one-hot
        print("pred", pred.shape)
        one_hot_pred = make_one_hot(pred)

        # Get classes of highest probability
        _, pred_classes = torch.max(pred.data, 1, keepdim=True)

        # Make labels one-hot
        one_hot = make_one_hot(gold)

        # applhy label smoothing
        eps = smoothing

        # TODO - use the good one hot funtion

        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        # loss from pred and gold, not one-hot
        loss = F.cross_entropy(pred, gold, ignore_index=0, reduction='sum')

    return loss

## ORIGINAL
def cal_loss_og(pred, gold, batch_size, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)
    print("gold shape", gold.shape)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        
        print(one_hot.shape)

        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss

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
        An (X x Y) matrix of n class integers.

    Output:
        A matrix of dimension nxn signalling adjacency.
        DECIDE:
            Weightings or true / false?
            Maybe one function for each?
    """
    class_tensor = class_tensor.int()

    n = torch.max(class_tensor).item() + 1
    print("Got", n, "classes.")

    adj_matrix = torch.zeros((n, n))

    class_diag[i,j] = class_tensor[i+1, j+1]
    class_to_right[i,j] = class_tensor[i, j+1]
    class_under[i,j] = class_tensor[i+1, j]

    class_diag = make_one_hot(class_diag)
    class_to_right = make_one_hot(class_to_right)
    class_under = make_one_hot(class_under)



def calc_adj_old(class_tensor):
    """
    Input: 
        An (X x Y) matrix of n class integers.

    Output:
        A matrix of dimension nxn signalling adjacency.
        DECIDE:
            Weightings or true / false?
            Maybe one function for each?
    """

    class_tensor = class_tensor.int()
    
    # For now 
    assert len(class_tensor.shape) == 2

    n = torch.max(class_tensor).item() + 1
    print("Got", n, "classes.")

    adj_matrix = torch.zeros((n, n))

    ## check only the one to the right or below (sweep once)

    ## TODO - make numpy
    max_siz = (class_tensor.size(0) -1) * (class_tensor.size(1) -1)
    for r in range(class_tensor.size(0) - 1):
        for c in range(class_tensor.size(1) - 1):
            if r*c % (max_siz // 100) == 0 and r*c != 0:
                print(100 * r*c // max_siz, "%% of the way there")
            this_class = class_tensor[r, c].item()

            right_class = class_tensor[r, c+1].item()
            down_class = class_tensor[r+1, c].item()
            diag_class = class_tensor[r+1, c+1].item()

            adj_matrix[this_class, right_class] += 1
            adj_matrix[this_class, down_class] += 1
            adj_matrix[this_class, diag_class] += 1

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