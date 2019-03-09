"""
The old loss function implementations.
"""

## VERBOSE
def calc_loss2(pred, gold, one_hot=True, smoothing_type="uniform", smoothing=0):
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
            1) "uniform" 
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

            if smoothing_type == "uniform":

                # Apply uniform smoothing only
                eps = smoothing
                one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            
            if smoothing_type == "uniform_vary_eps":

                # Tensor of (batch, class_name, class_name_adj_to) = count
                adj_matrix = calc_adj_batch(gold.cpu().numpy())

                # Base smoothing on the class-class adjacency
                eps = np.zeros((n_batch, n_class)) + smoothing
                
                # Tensor of batch, class of one-hot, value for other classes
                smooth_tens = np.zeros((n_batch, n_class, n_class))

                # Set adjacency magnitudes
                for c in range(adj_matrix.shape[1]):

                    if c == adj_matrix.shape[1] - 1:
                        sum_others = adj_matrix[:, c, :c].sum()
                    else:
                        sum_others = (adj_matrix[:, c, :c].sum() + adj_matrix[:, c, (c+1):].sum())

                    # Shape batch, class
                    eps[:, c] += (sum_others / adj_matrix[:, c, c])[0]
                
                #TODO - bring this under the if, else as it's the same
                
                # Now build the up tensor and do weighting
                for i in range(smooth_tens.shape[1]):
                    
                    # Set the inter-class error
                    smooth_tens[:, i, i] = 1. - eps[:, i]
                
                    # Get the row that isn't the c-c error - uniform in this case
                    rest_row = np.zeros((n_batch, n_class - 1)) + eps[:, i] / (n_class - 1)

                    assert len(rest_row.shape) == 2

                    # Put it back in
                    smooth_tens[:, i, :i] = rest_row[:, :i]
                    if i + 1 < smooth_tens.shape[1]:
                        smooth_tens[:, i, (i+1):] = rest_row[:, i:]


                # Check the smooth matrix for normalisation
                assert (smooth_tens.sum(axis=2, keepdims=True) < 1.0001).all()
                assert (smooth_tens.sum(axis=2, keepdims=True) > 0.999).all()
                
                # final assertion FAILS quite often - leaving out for now
                """
                # Normalise if not
                row_sums = smooth_tens.sum(axis=2)
                smooth_tens = smooth_tens / row_sums[:, np.newaxis]
                
                # Final assertion
                assert (smooth_tens.sum(axis=2, keepdims=True) == 1.).all()
                """

                # Now set one hot as the smoothed matrix
                    # ONE_HOT - the one hot labels to be smoothed
                    # GOLD - 1 is tensor of one-hot indices
                    # SMOOTH_TENS[b,gold,:] - the values that one_hot[i,:,k,l] should take
                old_one_hot = one_hot.detach() # for testing

                for c in range(one_hot.shape[1]):
                    one_hot[:,c,:,:] = torch.from_numpy(
                        smooth_tens[:,gold[:,:,:].cpu().numpy(), c]).to(
                            params["device"]).float()

                # Assert that structure has been maintained
                assert (np.argmax(old_one_hot.cpu().numpy()) == np.argmax(one_hot.cpu().numpy())).all()
                del old_one_hot # no longer needed

                assert (one_hot.sum(dim=1, keepdim=True) < 1.0001).all()
                assert (one_hot.sum(dim=1, keepdim=True) > 0.9999).all()

        elif smoothing_type.startswith("weighted"):
            # Get the weighted epsilons based on class adjacency

            # Tensor of (batch, class_name, class_name_adj_to) = count
            adj_matrix = calc_adj_batch(gold.cpu().numpy())
            print(adj_matrix.astype(int))

            # Base smoothing on the class-class adjacency
            eps = np.zeros((n_batch, n_class)) + smoothing

            # Get the class-class epsilon - fixed or variable - if required, and add to base
            if smoothing_type == "weighted_fixed_eps":
                # E.g. fixed epsilon magnitude - don't change this
                pass

            elif smoothing_type == "weighted_vary_eps": 
                # E.g. need to calculate the weight of each class

                # Vary epsilon depending on adjacencies
                for c in range(adj_matrix.shape[1]):
                    
                    if c == adj_matrix.shape[1] - 1:
                        sum_others = adj_matrix[:, c, :c].sum()
                    else:
                        #print("others", adj_matrix[:, c, :c], "+", adj_matrix[:, c, (c+1):])
                        #print("Sum", (adj_matrix[:, c, :c].sum() + adj_matrix[:, c, (c+1):].sum()))
                        sum_others = (adj_matrix[:, c, :c].sum() + adj_matrix[:, c, (c+1):].sum())
                    
                    eps[:, c] +=  (sum_others / adj_matrix[:, c, c])[0]

            # Now build the up tensor and do weighting
            
            # The smoothing into class(2) for each class(1) in the batch(0)
            smooth_tens = np.zeros((n_batch, n_class, n_class))

            for i in range(smooth_tens.shape[1]):
                
                # Set the inter-class error
                smooth_tens[:, i, i] = 1.-eps[:, i]

                # Get the row that isn't the c-c error
                if i == smooth_tens.shape[1] - 1:
                    rest = adj_matrix[:,i,:i]
                else:
                    rest = np.concatenate((adj_matrix[:, i, :i], adj_matrix[:, i, (i+1):]), axis=1)

                assert len(rest.shape) == 2
                
                # Normalise so adds to the size of the error (eps[i])
                for b in range(rest.shape[0]):
                    rest[b] = eps[b, i] * (rest[b] / rest[b].sum())
                    assert (rest[b].sum() - eps[b, i]) + 1. > 0.999
                    assert (rest[b].sum() - eps[b, i]) + 1. < 1.001
                    assert (rest[b].sum() - eps[b, i]) + 1. == 1.

                # Put it back in
                smooth_tens[:, i, :i] = rest[:, :i]
                if i < smooth_tens.shape[1] - 1:
                    smooth_tens[:, i, (i+1):] = rest[:, i:]

                """
                # OLD one hot setting
                adj = (1. - eps_tens[:, gold[:,:,:].cpu().numpy(), i])
                
                smooth_one = one_hot[:,i,:,:].float() * torch.from_numpy(adj).to(params["device"]).float()#[:,:,:]
                adj2 = eps_tens[:, gold[:,:,:].cpu().numpy(), i]
                smooth_others = (1. - one_hot[:,i,:,:].float()) * torch.from_numpy(adj2).to(params["device"]).float()#[:,:,:]
                
                assert smooth_one.sum() + smooth_others.sum() == 1.
                
                one_hot[:,i,:,:] =  smooth_one + smooth_others #/ (n_class -1)
                assert (one_hot[:,i,:,:].sum(dim=1, keepdim=True) == 1.).all()
                """

            print(smooth_tens)

            # Now set one hot
            # Check the smooth matrix for normalisation
            assert (smooth_tens.sum(axis=2, keepdims=True) < 1.0001).all()
            assert (smooth_tens.sum(axis=2, keepdims=True) > 0.999).all()
            
            # final assertion FAILS quite often - leaving out for now
            """
            # Normalise if not
            row_sums = smooth_tens.sum(axis=2)
            smooth_tens = smooth_tens / row_sums[:, np.newaxis]
            
            # Final assertion
            assert (smooth_tens.sum(axis=2, keepdims=True) == 1.).all()
            """

            # Now set one hot as the smoothed matrix
                # ONE_HOT - the one hot labels to be smoothed
                # GOLD - 1 is tensor of one-hot indices
                # SMOOTH_TENS[b,gold,:] - the values that one_hot[i,:,k,l] should take
            old_one_hot = one_hot.detach() # for testing

            for c in range(one_hot.shape[1]):
                one_hot[:,c,:,:] = torch.from_numpy(
                    smooth_tens[:,gold[:,:,:].cpu().numpy(), c]).to(
                        params["device"]).float()

            for j in np.arange(0, 500, 50):
                print("one hot 256",j, one_hot[0,:,j,256])

            # Assert that structure has been maintained
            assert (np.argmax(old_one_hot.cpu().numpy()) == np.argmax(one_hot.cpu().numpy())).all()
            del old_one_hot # no longer needed

            assert (one_hot.sum(dim=1, keepdim=True) < 1.0001).all()
            assert (one_hot.sum(dim=1, keepdim=True) > 0.9999).all()
        
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

## ORIGINAL - only uniform
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

# OLD
def make_one_hot_mine(tens, like):

    #one_hot_labels = gold.view(-1, 1).type(torch.long) # OLD
    #print("gold", gold.shape)
    one_hot_labels = tens.type(torch.long)
    #print("one hot labels", one_hot_labels.shape)
    one_hot = torch.zeros_like(like).scatter(1, one_hot_labels, 1)
    #print("one hot", one_hot.shape)
    return one_hot