import torch, os, sys #TEMP
import torch.nn.functional as F
import numpy as np

# Epochs should be 160000 iterations total - have 88 data?
# other location "data/input_tensors/sample_scans/"
# TEMP batch_size = 8
# TEMP epochs = 160000 // 88
# TEMP every_n was 10

def get_params():

    params = {
        "scan_location": "data/input_tensors/segmentation_data/datasets/",
        "torch_data": False,
        "epochs" : 1,
        "lr_0" : 0.0005,
        "batch_size" : 1,
        "clean" : "loss", # 'loss' (0 loss on broken bit), or 'fill' (random dist into image)
        "chop" : True,
        "shuffle": False, # same order for good experiment
        "one_hot": True,
        "smoothing_type": "uniform_fixed_eps",
        "label_smoothing" : 0.1,
        "validation_split" : 0.2,
        "device" : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "lr_idxs_array": np.array([0, 0.1, 0.2, 0.5, 0.7, 0.9, 0.95]),
        "lr_array": np.array([1, 0.5, 0.25, 0.125, 0.015625, 0.00390625, 0.001953125]),
        "workers" : 4,
        "voxel_size" : 9,
        "information": False,
        "every_n": 1,
        "num_of_bins": 40
    }

    ## SAVE THE MODEL ##
    params["save_model"] = False#True

    ## Save the training and test info ##
    params["save_run"] = True

    if "torch" in params["scan_location"]:
        params["torch"] = True
    else:
        params["torch"] = False

    return params

def construct_file(params, direct, ignore=True, append=""):
    """Produces file with headers. Add to gitignore"""

    if direct == "no":
        return "no"

    file_name = params["smoothing_type"] + "_lr" + str(params["lr_0"]).split(".")[-1] +\
                "_ep" + str(params["epochs"]) +\
                "_bs" + str(params["batch_size"]) + append + "/"

    if not os.path.exists(direct + file_name):
        os.makedirs(direct + file_name)

    with open(direct + file_name + "DETAILS.txt", 'w') as file:
        for key in params:
            file.write(key + ": " + str(params[key]) + "\n")

    if ignore:
        with open(direct + "/.gitignore", "a+") as gi:
            gi.write(file_name + "*\n") # remove the last /

    return direct + file_name

def calc_loss(inp, pred, gold, one_hot_flag=True, smoothing_type="uniform_fixed_eps", smoothing=0, clean="loss", prnt=False):
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
            0) "non" - no label smoothing, just one-hot
            1) "uniform_fixed_eps" 
                Applies uniform smoothing, fixed magnitude
            2) "uniform_vary_eps"
                Applies uniform smoothing, vary magnitude depending on self-adjacency
            3) "weighted_fixed_eps"
                Applies weighted smoothing of fixed mangnitude
                E.g. weights into the classes depending on class adjacency
                ignores self-adjacency
            4) "weighted_vary_eps"
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
    if one_hot_flag:

        ## INITIALISE ##
        n_batch = pred.size(0)
        n_class = pred.size(1)

        ## CHECK INPUT for 3D and RESHAPE for one-hot ##
        if len(gold.shape) == 3:
            reshaped_gold = gold.view(gold.size(0), 1, gold.size(1), gold.size(2))
        else:
            raise NotImplementedError("Only implemented for batch x X x Y (3D)\nGot %s." %\
                (len(gold.shape)))
        
        ## MAKE LABEL ONE HOT ##
        one_hot = make_one_hot(reshaped_gold, reshaped_gold.device, n_class)

        ## APPLY VARIOUS SMOOTHING TYPES to one_hot ##
        if smoothing_type == "uniform_fixed_eps":

            if prnt:
                print("Smoothing by uniform fixed eps")

            # Apply uniform smoothing only
            eps = smoothing

            # For testing
            old_one_hot = one_hot.clone()

            old_max_indices = torch.argmax(old_one_hot, dim=1, keepdim=False)
            
            # Fix one hot
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

            # Check still the same label, with some smoothing
            new_max_indices = torch.argmax(one_hot, dim=1, keepdim=False)

            # Assert that structure has been maintained!
            assert (old_max_indices == new_max_indices).all()
            assert (one_hot != 0.).all()
            
            # Do some view-testing (despite asserting structure is maintained)
            if prnt:
                print("APPLIED uniform fixed epsilon.")
                for j in np.arange(150, 350, 25):
                    print("argmax (256,", j, ")", old_max_indices[0,256,j])
                    print("old_hot (256,", j, ")", old_one_hot[0,:,256,j])
                    print("one_hot (256,", j, ")", one_hot[0,:,256,j])
            
            del old_one_hot, old_max_indices, new_max_indices

        elif smoothing_type in ["weighted_vary_eps", "uniform_vary_eps", "weighted_fixed_eps"]:

            # Tensor of (batch, class_name, class_name_adj_to) = count
            adj_matrix = calc_adj_batch(gold.cpu().numpy())

            # Base smoothing on the class-class adjacency
            eps = np.zeros((n_batch, n_class)) + smoothing
            
            # Tensor indicating smoothing of class[1] into class[2] (batch[0])
            smooth_tens = np.zeros((n_batch, n_class, n_class)).astype(float)

            ## VARY EPS magnitude, if required ##
            if smoothing_type in ["weighted_vary_eps", "uniform_vary_eps"]:

                if prnt:
                    print("Smoothing by varying eps")

                for c in range(adj_matrix.shape[1]):

                    # if looking at the last class, just sum up to the end
                    if c + 1 == adj_matrix.shape[1]:
                    
                        assert len(adj_matrix[:, c, :c].shape) == 2 # batch and sum
                        sum_others = adj_matrix[:, c, :c].sum(axis=(1))
                    
                    else:

                        # Get the first and last bits
                        first_bit = adj_matrix[:, c, :c]
                        end_bit = adj_matrix[:, c, (c+1):]
                        
                        # End bit
                        if len(end_bit.shape) == 2:
                            sum_others = adj_matrix[:, c, (c+1):].sum(axis=(1))
                        elif len(end_bit.shape) == 1:
                            sum_others = adj_matrix[:, c, (c+1):]

                        # First bit
                        if len(first_bit.shape) == 2:
                            sum_others += first_bit.sum(axis=(1))
                        elif len(first_bit.shape) == 1:
                            sum_others += first_bit

                    ## Update eps weights. Shape: batch, class ##

                    assert sum_others.shape == adj_matrix[:, c, c].shape
                    
                    # Batch-wise division
                    new_eps = np.divide(sum_others, adj_matrix[:, c, c], dtype=np.dtype('float64'))
                    
                    # add pm the new eps in proportion to adjacency
                    assert eps[:,c].shape == new_eps.shape
                    eps[:, c] += new_eps 
                    
            if prnt:
                print("Resulting Epsilon:\n", eps)
            
            ## WEIGHT THE SMOOTHING - based off eps size ##
            for i in range(n_class):
                
                # Set the class-class smoothing values
                smooth_tens[:, i, i] = 1. - eps[:, i]
                
                # Set the non-self smoothing magnitudes
                if smoothing_type == "uniform_vary_eps":
                    
                    if prnt:
                        print("Smoothing by uniform vary eps")

                    rest = np.zeros((n_batch, n_class - 1))

                    # push repaining epsilon equally into other classes
                    for b in range(n_batch):
                        
                        rest[b] += eps[b, i] / (n_class - 1)

                        # Margins for error
                        assert rest[b].sum() > eps[b, i] - 0.0001
                        assert rest[b].sum() < eps[b, i] + 0.0001

                        # Put the rest back in to the smoothing tensor
                        smooth_tens[b, i, :i] = rest[b, :i]
                        
                        if i + 1 < smooth_tens.shape[1]:
                            smooth_tens[b, i, (i+1):] = rest[b, i:]

                elif smoothing_type in ["weighted_vary_eps", "weighted_fixed_eps"]:

                    if prnt:
                        print("Weighted smoothing")
                    
                    # Do the weighting on the non-self-adjacency if weighted
                    if i == smooth_tens.shape[1] - 1:
                        rest = adj_matrix[:,i,:i]
                    else:
                        rest = np.concatenate((adj_matrix[:, i, :i], adj_matrix[:, i, (i+1):]), axis=(-1))
                    
                    assert len(rest.shape) == 2

                    # Per batch
                    for b in range(n_batch):
                        
                        # normalise rest to size of error
                        rest[b] = eps[b, i] * (rest[b] / rest[b].sum())
                        
                        # Margins for error
                        assert rest[b].sum() > eps[b, i] - 0.0001
                        assert rest[b].sum() < eps[b, i] + 0.0001
                        # assert (rest[b].sum() - eps[b, i]) + 1. == 1. # too fine?

                        # Put the rest back in to the smoothing tensor
                        smooth_tens[b, i, :i] = rest[b, :i]
                        
                        if i + 1 < smooth_tens.shape[1]:
                            smooth_tens[b, i, (i+1):] = rest[b, i:]

                #if prnt:
                #    print("Smoothing was varied eps into all (other) classes for class", i, "\n", rest)

            if prnt:
                for b in range(n_batch):
                    print("Reconstructed smoothing tensors batch", b, "\n", smooth_tens[b])


            # Check the smooth matrix for (approx) normalisation
            # COULD normalise but it doesn't work very consistently
            assert (smooth_tens.sum(axis=(-1)) < 1.0001).all()
            assert (smooth_tens.sum(axis=(-1)) > 0.9999).all()

            ## BUILD THE ONE-HOT LABEL UP ##

            """
            Using:
                ONE_HOT - the currently one-hot labels to be smoothed
                GOLD - 1 is tensor of one-hot indices
                SMOOTH_TENS[b,gold,:] - the values that one_hot[i,:,k,l] should take
            """

            # for testing
            old_one_hot = one_hot.clone()
            old_max_indices = torch.argmax(old_one_hot, dim=1, keepdim=False)

            ########### CHANGING AROUND ##############
            # Use the smoothing matrix and gold
            # TODO - check sampling from batches correctly
            for b in range(n_batch):

                # (9,9) matrix of class:smoothing
                this_image_smooth = smooth_tens[b, :, :]

                # the one hot tensor per cell (9, 512, 512) is this_image_smooth(class, :), flipped
                np_tens = np.moveaxis(this_image_smooth[gold[b,:,:].cpu().numpy(), :], 2, 0)
                
                #print(np_tens[:, 256, 256])

                one_hot[b, :, :, :] = torch.from_numpy(np_tens).to(gold.device).float()

            assert one_hot.shape == old_one_hot.shape

            """
            # Works for batch size 1 ut casts np tens to (8,8,512,512)
            for c in range(one_hot.shape[1]):
                
                print("gold shape", gold.shape)
                print("one hot [c] shape", one_hot[:,c,:,:].shape)
                
                np_tens = smooth_tens[:, gold[:,:,:].cpu().numpy(), c]

                print("np tens", np_tens.shape)
                
                one_hot[:,c,:,:] = torch.from_numpy(np_tens).to(gold.device).float()
            """
            ########################################

            new_max_indices = torch.argmax(one_hot, dim=1, keepdim=False)

            # Do some view-testing (before asserting structure is maintained)
            if prnt:
                print("Testing old and new one hots")

                for j in np.arange(150, 350, 25):
                    print("argmax (256,", j, ")", old_max_indices[0,256,j])
                    print("old_hot (256,", j, ")", old_one_hot[0,:,256,j])
                    print("one_hot (256,", j, ")", one_hot[0,:,256,j])
                
            # Assert that structure has been maintained!
            assert old_max_indices.shape == new_max_indices.shape
            assert (old_max_indices[0] == new_max_indices[0]).all()
            assert (old_max_indices == new_max_indices).all()

            # no longer needed
            del old_one_hot, old_max_indices, new_max_indices

            # Check that each one-hot abel still sums to (approx) 1.
            assert (one_hot.sum(dim=1, keepdim=True) < 1.0001).all()
            assert (one_hot.sum(dim=1, keepdim=True) > 0.9999).all()

        elif smoothing_type == "none":
            if prnt:
                print("No smoothing applied. One hot:")
                print(one_hot[0,:,0,0])
            pass
        
        else:
            # If none of the 4 smoothing regimes
            raise NotImplementedError("Smoothing type" + smoothing_type + "not implemented")

        ## NOW CALC LOSS ##
        
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0)

        if prnt:
            try:
                print("Using one hot\n", one_hot[0,:,0,0])
            except:
                print("Failed indexing - untested")

        # Either log prob is 0 (e.g. log of 1) or one_hot is 0s
        # E.g. to symbolise classification was exactly right
        loss = -(one_hot * log_prb).sum(dim=1)

        # SET loss to 0 everywhere that input is 1. and gold is class 0
        # EG this is where the input was broken
        if clean == "loss":
            
            if prnt:
                print("ENTERING loss cleaning.")

            loss = torch.where(((gold == 0) & (inp == 1.)) | ((gold == 8) & (inp == 1.)), 
                               torch.tensor(0., device=gold.device, dtype=torch.float), 
                               loss)

        # LOSS
        loss = loss.masked_select(non_pad_mask).sum()

    else:

        # loss from pred and gold, NOT one-hot
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

def make_one_hot(tens, to_device, C=9):
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
    
    # TODO - if class is -1 
    # Set the one hot to 0s so that there's no loss

    # Clean input -CHECK works on torch tensor
    if type(tens).__name__ == "ndarray":
        
        tens = torch.from_numpy(tens).to(to_device).long()
    
    elif type(tens).__name__ == "Tensor":
        
        tens = tens.to(to_device).long()

    else:

        raise NotImplementedError("Only implemented for Tensor or ndarray type\nGot " + type(tens).__name__)

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

    #target = torch.where(tens.data > 0, one_hot.scatter_(1, tens.data, 1), one_hot.scatter_(1, 0, 1))
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
    params = get_params()

    tst = np.genfromtxt('data/tests/test_matrix.csv', delimiter=',')
    tst = np.array([tst, tst])
    print(tst.shape)
    print(tst)
    adj = calc_adj_batch(tst)

    print(adj)