import torch, os
import torch.nn.functional as F
import numpy as np

# Epochs should be 160000 iterations total - have 88 data?
# other location "data/input_tensors/sample_scans/"
# TEMP batch_size = 8
# TEMP epochs = 160000 // 88
# TEMP every_n was 10
params = {
    "epochs" : 2,
    "lr_0" : 0.0001,
    "batch_size" : 1,
    "workers" : 4,
    "voxel_size" : 9,
    "smoothing_type": "weighted_vary_eps",
    "label_smoothing" : 0.1,
    "validation_split" : 0.2,
    "device" : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "lr_idxs_array": np.array([0, 0.1, 0.2, 0.5, 0.7, 0.9, 0.95]),
    "lr_array": np.array([1, 0.5, 0.25, 0.125, 0.015625, 0.00390625, 0.001953125]),
    "save_location": "models/unet/saved_models",
    "scan_location": "data/input_tensors/segmentation_data/datasets/",
    "save_run": True,
    "save_to_dir": "data/training_data/",
    "information": True,
    "every_n": 1,
    "num_of_bins": 40
}

# TODO - transforms - handle the dataset...

def construct_file():
    """Produces file with headers."""

    file_name = params["smoothing_type"] + "_" + str(params["batch_size"]) + "_" + str(params["epochs"]) + "/"
    
    if not os.path.exists(params["save_to_dir"] + file_name):
        os.makedirs(params["save_to_dir"] + file_name)

    with open(params["save_to_dir"] + file_name +"DETAILS.txt", 'w') as file:
        for key in params:
            file.write(key + ": " + str(params[key]) + "\n")

    return params["save_to_dir"] + file_name + ".txt"

def calc_loss(pred, gold, one_hot=True, smoothing_type="uniform", smoothing=0):
    """
    Calc CEL and apply various label smoothings.
    Based on uniform label smoothing here:
        https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py
    
    Inputs:
        pred - b,C,X,Y,(Z) tensor of floats - indicating predicted class probabilities
        gold - b,X,Y,(Z) tensor of integers indicating labelled class
    
    Args:
        one_hot: 
            if False, just return "standard" cross entropy loss
            else apply smoothing of type smoothing_type:
        smoothing_type: 
            "uniform" 
                Applies uniform smoothing, fixed magnitude
            "uniform_vary_eps"
                Applies uniform smoothing, vary magnitude depending on self-adjacency
            "weighted_fixed_eps"
                Applies weighted smoothing of fixed mangnitude
                E.g. weights into the classes depending on class adjacency
                ignores self-adjacency
            "weighted_vary_eps"
                Applies weighted smoothing into adjacent classes only
                Accounts for self-adj - varies magnitude of smoothing
        smoothing: 
            (initial) magnitude of label smoothing 
                (e.g. one-hot 1 -> 1 - smoothing)
            Acts as the base for varying eps
    """

    # Cleanse class input
    gold = gold.long()

    # If one hot encoding
    if one_hot:

        n_batch = pred.size(0)
        n_class = pred.size(1)

        ## CHECK INPUT for 3D and RESHAPE for one-hot ##
        if len(gold.shape) == 3:
            reshaped_gold = gold.view(gold.size(0), 1, gold.size(1), gold.size(2))
        else:
            raise NotImplementedError("Only implemented for batch x X x Y (3D)\nGot %s." %\
                (len(gold.shape)))
        
        ## MAKE ONE HOT ##
        one_hot = make_one_hot(reshaped_gold, n_class)

        ## APPLY VARIOUS SMOOTHING TYPES ##
        if smoothing_type.startswith("uniform"):

            if smoothing_type == "uniform-vary-eps":
                
                # Tensor of (batch, class_name, class_name_adj_to) = count
                adj_matrix = calc_adj_batch(gold.cpu().numpy())

                # Base smoothing on the class-class adjacency
                eps = np.zeros((n_batch, n_class)) + smoothing

                # Adjacency magnitude per batch per class
                adjacency_mag = np.zeros((n_batch, n_class))

                # Weight eps depending on self-adjacency
                for c in range(adj_matrix.shape[0]):
                    
                    if c == adj_matrix.shape[1] - 1:
                        sum_others = adj_matrix[:, c, :c].sum(axis=2)
                    else:
                        sum_others = (adj_matrix[:, c, :c].sum(axis=2) + adj_matrix[:, c, (c+1):].sum(axis=2))

                    # Shape batch, class
                    adjacency_mag[:, c] = sum_others[:, c] / adj_matrix[:, c, c]

                # Add on the adjacency at eps. If large, it increases eps, as desired
                eps[:, :] += adjacency_mag[:, :]

                eps = torch.from_numpy(eps).to(params["device"]).float()

                # Across batches, class
                smooth_one = torch.mm(one_hot.float(), (1.-eps))
                assert smooth_one.shape == one_hot.shape
                smooth_others = torch.mm((1.-eps), one_hot) / (n_class -1)
                assert smooth_others.shape == one_hot.shape

                # per row 
                for i in range(smooth_one.shape[0]):
                    assert smooth_one[i].sum() + smooth_others[i].sum() == 1.

                one_hot =  smooth_one + smooth_others

            else:

                # Apply uniform epsilon smoothing        
                eps = smoothing
                one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)


        
        else:
            # Get the weighted epsilons based on class adjacency

            # Tensor of (batch, class_name, class_name_adj_to) = count
            adj_matrix = calc_adj_batch(gold.cpu().numpy())
            #print(adj_matrix.astype(int))

            # Base smoothing on the class-class adjacency
            eps = np.zeros((n_batch, n_class)) + smoothing

            # The smoothing into class(2) for each class(1) in the batch(0)
            eps_tens = np.zeros((n_batch, n_class, n_class))

            # Get the class-class epsilon - fixed or variable - if required, and add to base
            if smoothing_type == "weighted_fixed_eps":
                # E.g. fixed epsilon magnitude - don't change this
                pass

            elif smoothing_type == "weighted_vary_eps": 
                
                # Vary epsilon depending on adjacencies
                for c in range(adj_matrix.shape[1]):
                    
                    if c == adj_matrix.shape[1] - 1:
                        sum_others = adj_matrix[:, c, :c].sum()
                    else:
                        sum_others = (adj_matrix[:, c, :c].sum(axis=(2)) + adj_matrix[:, c, (c+1):].sum())
                    
                    eps[:, c] +=  (sum_others / adj_matrix[:, c, c])[0]

            # Now build the up tensor and do weighting
            for i in range(eps_tens.shape[1]):
                
                # Set the inter-class error
                eps_tens[:, i, i] = eps[:, i]

                # Get the row that isn't the c-c error
                if i == eps_tens.shape[1] - 1:
                    rest = adj_matrix[:,i,:i]
                else:
                    rest = np.concatenate((adj_matrix[:, i, :i], adj_matrix[:, i, (i+1):]), axis=1)

                assert len(rest.shape) == 2
                
                # Normalise so adds to the size of the error (eps[i])
                for b in range(rest.shape[0]):

                    rest[b] = eps[b, i] * (rest[b] / rest[b].sum())

                    assert (rest[b].sum() - eps[b, i]) + 1. > 0.95
                    assert (rest[b].sum() - eps[b, i]) + 1. < 1.05
                    assert (rest[b].sum() - eps[b, i]) + 1. == 1.

                # Put it back in
                eps_tens[:, i, :i] = rest[:, :i]
                if i < eps_tens.shape[1] - 1:
                    eps_tens[:, i, (i+1):] = rest[:, i:]

                # Now set one hot
                adj = (1. - eps_tens[:, gold[:,:,:].cpu().numpy(), i])
                
                smooth_one = one_hot[:,i,:,:].float() * torch.from_numpy(adj).to(params["device"]).float()#[:,:,:]
                adj2 = eps_tens[:, gold[:,:,:].cpu().numpy(), i]
                smooth_others = (1. - one_hot[:,i,:,:].float()) * torch.from_numpy(adj2).to(params["device"]).float()#[:,:,:]
                one_hot[:,i,:,:] =  smooth_one + smooth_others #/ (n_class -1)

        else:
            raise NotImplementedError

        ## CALC LOSS ## TODO - verify
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0) # NOT SURE WHAT DOES - JUST COPIED
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    
    # loss from pred and gold, not one-hot    
    else:

        loss = F.cross_entropy(pred, gold, ignore_index=0, reduction='sum')

    return loss

def calc_adj_batch(class_batch):
    """Performs calc_adj on each matrix of classes in a batch."""

    adj_batch = np.array([calc_adj(class_batch[i,:,:]) for i in range(class_batch.shape[0])])

    #print(adj_batch.shape)

    return adj_batch

def calc_adj(class_tensor):
    """
    Input: 
        Am (X x Y) matrix of n class integers.

    Output:
        A matrix of dimension nxn 
            signalling adjacency of class: row to class: col
    """
    def reduce(tens, axis, by):
        """
        Reduces 2d tensor in 0 and 1 dims.
        
        Integer by - decides 
            n from end if by = +n
            n from start if by = -n
        """
        if axis == 0:
            # up / down axis
            if by > 0:
                # drop the final row
                return tens[:tens.shape[0]-by, :tens.shape[1]]
            elif by < 0:
                return tens[(-by):tens.shape[0], :tens.shape[1]]
            else:
                return tens
        elif axis == 1:
            if by > 0:
                return tens[:tens.shape[0], :tens.shape[1]-by]
            elif by < 0:
                return tens[:tens.shape[0], (-by):tens.shape[1]]
            else:
                return tens
        elif axis == (0,1):
            if by > 0:
                return tens[:tens.shape[0]-by, :tens.shape[1]-by]
            elif by < 0:
                return tens[(-by):tens.shape[0], (-by):tens.shape[1]]
            else:
                return tens
        else:
            raise NotImplementedError("Not implemented for axis %s" % str(axis))

    if len(class_tensor.shape) != 2:
        raise NotImplementedError

    class_tensor = class_tensor.astype(int)
    n = np.max(class_tensor) + 1
    
    # Init
    adj_matrix = np.zeros((n, n))

    # A matrix of the classes under, over, right, left, of corresponding class
    class_under = reduce(np.roll(class_tensor, -1, axis=0), axis=0, by=1)
    class_up = reduce(np.roll(class_tensor, 1, axis=0), axis=0, by=-1)

    class_right = reduce(np.roll(class_tensor, -1, axis=1), axis=1, by=1)
    class_left = reduce(np.roll(class_tensor, 1, axis=1), axis=1, by=-1)
    
    # class_diag_br = reduce(np.roll(class_tensor, (-1,-1), axis=(0,1)), axis=(0,1))

    # Reduce as don't count adjacency to under/right final row/col
    reduced_class_tensor_under = reduce(class_tensor, axis=0, by=1)
    reduced_class_tensor_up = reduce(class_tensor, axis=0, by=-1)
    
    reduced_class_tensor_right = reduce(class_tensor, axis=1, by=1)
    reduced_class_tensor_left = reduce(class_tensor, axis=1, by=-1)
    
    #reduced_class_tensor_diag = reduce(class_tensor, axis=(0,1))

    # For each class in the class tensor
    for c in range(n):

        # find where equal, all the adjacencies are in under, right, diag. Count dictionary returns
        under_unique, under_counts = np.unique(np.where(reduced_class_tensor_under==c, class_under, -1), return_counts=True)
        under_counts = dict(zip(under_unique, under_counts))

        up_unique, up_counts = np.unique(np.where(reduced_class_tensor_up==c, class_up, -1), return_counts=True)
        up_counts = dict(zip(up_unique, up_counts))

        right_unique, right_counts = np.unique(np.where(reduced_class_tensor_right==c, class_right, -1), return_counts=True)
        right_counts = dict(zip(right_unique, right_counts))

        left_unique, left_counts = np.unique(np.where(reduced_class_tensor_left==c, class_left, -1), return_counts=True)
        left_counts = dict(zip(left_unique, left_counts))
        
        #diag_unique, diag_counts = np.unique(np.where(reduced_class_tensor_diag==c, class_diag, -1), return_counts=True)
        #diag_counts = dict(zip(diag_unique, diag_counts))
        
        # Count occurences per class 
        for a in range(n):
            if a in under_counts:
                adj_matrix[c, a] += under_counts[a]
            if a in up_counts:
                adj_matrix[c,a] += up_counts[a]
            if a in right_counts:
                adj_matrix[c, a] += right_counts[a]
            if a in left_counts:
                adj_matrix[c,a] += left_counts[a]
            #if a in diag_counts:
            #    adj_matrix[c, a] += diag_counts[a]

    return adj_matrix

def make_one_hot(tens, C=9):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x (D) x H x W, where N is batch size, depth optional. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x D x H x W, where C is class number. One-hot encoded.
    '''
    
    # Clean input
    tens = torch.tensor(tens, device=params["device"], dtype=torch.long)

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

def get_aligned_representations(representations, order):

    #print("reps", len(representations))
    #print(order)

    for epoch in range(len(representations)):
        
        #print("epoch", epoch)
        #print("order", order[epoch])

        for layer in range(len(representations[0])):

            #print("layer", layer)
            #print("\tbefore", representations[epoch][layer].shape)
            
            # TODO - check logic here
            """
            if len(representations[epoch][layer].shape) == 4:
                representations[epoch][layer][order[epoch], :, :, :] = representations[epoch][layer][:,:,:,:]
            
            elif len(representations[epoch][layer].shape) == 5:
                representations[epoch][layer][order[epoch], :, :, :, :] = representations[epoch][layer][:,:,:,:,:]
            else:
                raise NotImplementedError("Expected len of reps[epoch][layer] = 5, got %s" \
                    % str(len(representations[epoch][layer])))
            """

            representations[epoch][layer] = representations[epoch][layer][np.argsort(order[epoch]), :]
            #print("\tafter", representations[epoch][layer].shape)

    # [epochs, hidden-layers, reps(b,c,x,y)]
    return representations

if __name__ == "__main__":

    """
    USAGE of calc_adj:
    for i, data in enumerate(trainloader, 0):

        inputs, labels, _ = data
            
        inputs, labels = inputs.float().to(params["device"]), labels.to(params["device"])

        adj = calc_adj(labels[0, :, :].cpu().numpy().astype(int))
    """

    tst = np.genfromtxt('data/tests/test_matrix.csv', delimiter=',')
    tst = np.array([tst, tst])
    print(tst.shape)
    print(tst)
    adj = calc_adj_batch(tst)

    print(adj)