"""
The old loss function implementations.
"""

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

# OLD
def make_one_hot_mine(tens, like):

    #one_hot_labels = gold.view(-1, 1).type(torch.long) # OLD
    #print("gold", gold.shape)
    one_hot_labels = tens.type(torch.long)
    #print("one hot labels", one_hot_labels.shape)
    one_hot = torch.zeros_like(like).scatter(1, one_hot_labels, 1)
    #print("one hot", one_hot.shape)
    return one_hot