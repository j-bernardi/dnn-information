import torch, torchvision, os, sys, time
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Import the model
from unet_models.unet_model2d import UNet2D
from training_metadata import calc_loss, make_one_hot

import training_metadata as tm

params = tm.params 

# Where to save the model
params["save"] = True
params["save_location"] += "/unet2d.pth"

# limit number for testing
number_samples = 10 # 0 for all

# TODO - transforms - handle the dataset...

def load_data(params):

    # Data handler 
    print("Appending", os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]))
    sys.path.append(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]))
    from data.data_utils import get_imdb_data

    # Load in
    print("Loading images")
    (trainloader, testloader), classes = get_imdb_data(
        params["scan_location"], val_split=params["validation_split"], 
        num=number_samples, workers=params["workers"], 
        batch_size=params["batch_size"])

    # Report
    print("Loaded.")
    print("len trainset", len(trainloader))
    print("len testset", len(testloader))

    return trainloader, testloader, classes

def load_model(params):

    unet = UNet2D()
    unet.float()
    print("Moving model to", params["device"])
    unet = unet.to(params["device"])

    return unet

def train(params, fake=False):
    """Perform training."""

    # Learning rate and time to change #
    idxs = np.floor(params["epochs"] * len(trainloader) * params["lr_idxs_array"]).astype(int)
    lrs = params["lr_0"] * params["lr_array"]
    next_idx = 0

    # Adam optimizer - https://arxiv.org/abs/1412.6980 #
    optimizer = optim.Adam(unet.parameters(), lr=params["lr_0"])

    print("Starting training.")
    
    # epochs #
    for epoch in range(params["epochs"]):
        
        running_loss, correct = 0.0, 0.0
        epoch_loss = 0.0
        total_el = 0
        total_num = 0
        
        for i, data in enumerate(trainloader, 0):

            # inputs, labels, set?? TODO - check 3rd 
            inputs, labels, _ = data
            
            # Set up input images and labels
            inputs, labels = inputs.float().to(params["device"]), labels.to(params["device"])
            #data['image'].float().to(params["device"]), data['classes'].to(params["device"])

            optimizer.zero_grad()

            # adjust lr over iterations
            if i * epoch == idxs[next_idx]:
                # update the learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lrs[next_idx]
                # update the next one to look at
                if next_idx + 1 < len(idxs):
                    next_idx += 1
                else:
                    next_idx = 0

            # Get this batch's outputs - probabilities over 9 classes
            outputs = unet(inputs)

            # Skip the actual training if desired [for testing]
            if fake:
                return outputs.shape, outputs.numel()

            # Calc loss
            loss = calc_loss(outputs, labels, smoothing_type=params["smoothing_type"], smoothing=params["label_smoothing"])

            # calc accuracy
            _, pred_classes = torch.max(outputs.data, 1, keepdim=True)
            shaped_labels = labels.view(labels.size(0), 1, labels.size(1), labels.size(2))
            
            correct += torch.eq(pred_classes, shaped_labels.long()).float().sum()
            total_el += outputs.numel()
            total_num += outputs.size(0)

            loss.backward()
            optimizer.step()
                
            running_loss += loss.item()
            epoch_loss += loss.item()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if i % (len(trainloader) // 5) == (len(trainloader) // 5) - 1:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
                
                if params["save_run"]:
                    with open(params["experiment_file"].replace(".txt", "TRAIN.txt"), 'a+') as ef:
                        ef.write("%d,%d,%.3f\n" % (epoch + 1, i + 1, running_loss))


                running_loss = 0.0

        # Track losses per epoch
        accuracy = 100*correct/total_el
        
        print('[Epoch %d complete] loss: %.3f, accuracy %.3f %%' %
              (epoch, epoch_loss, accuracy))
        if params["save_run"]:
            with open(params["experiment_file"].replace(".txt", "TRAIN.txt"), 'a+') as ef:
                ef.write("EPOCH%d,%.3f,%.3f\n\n" % (epoch + 1, epoch_loss, accuracy))

        epoch_loss = 0.0

    return outputs.shape, outputs.numel()

def test(params, shape, numel, classes):
    """Test the network."""

    print("Testing")
    
    # Total cells classified, total correct, the number of batches seen
    total, total_all_classes, correct_count, num_batches = 0, 0, 0, 0

    # Total correct of this class in the prediction
    correct = torch.zeros(shape, device=params["device"], dtype=torch.uint8)
    false_positive = torch.zeros(shape, device=params["device"], dtype=torch.uint8)
    false_negative = torch.zeros(shape, device=params["device"], dtype=torch.uint8)
    correct_not_labelled = torch.zeros(shape, device=params["device"], dtype=torch.uint8)

    total_el_per_batch = numel
    total_el_per_batched_class = total_el_per_batch // 9

    # The sum of the confidences of prediction per class[0], test[1]
    confs = np.zeros((len(classes), len(testloader)))

    # The count of the number predicted per class[0], test[1]
    pred_counts = np.zeros((len(classes), len(testloader)))
    
    # totals per test [0]
    total_conf = [] 
    total_pred_count = []

    # predicted class tots
    class_tots = torch.tensor([0] * len(classes), device=params["device"])
    
    # labelled class tots
    label_tots = torch.tensor([0] * len(classes), device=params["device"])
    #class_tots_all_class = torch.tensor([0] * len(classes), device=params["device"])

    d_count = 0
    with torch.no_grad():
        for data in testloader:
            

            ## GET OUTPUTS ## 
            inputs, labels, _ = data
            inputs, labels = inputs.float().to(params["device"]), labels.to(params["device"])
            #data['image'].float().to(params["device"]), data['classes'].to(params["device"])
        
            outputs = unet(inputs)

            ## CALCULATE CONFIDENCES and LABELS ## 
            # predicted is the predicted class for each position - (bs,1,z,x,y)

            max_conf_matrix, predicted = torch.max(outputs.data, 1, keepdim=True)

            one_hot_predicted = make_one_hot(predicted, len(classes)).byte()

            one_hot_labels = make_one_hot(labels.long().view(labels.size(0), 1, labels.size(1), labels.size(2)), len(classes)).byte()

            # The confidences of predicted classes (0s elsewhere - sparse, one-hot rep)
            this_confs = torch.where(one_hot_predicted == 1, outputs, torch.tensor(0., device=params["device"]))
            
            # The total confidence for this image
            total_conf.append(this_confs.sum())
            total_pred_count.append(one_hot_predicted.sum())

            for c in range(len(classes)):
                confs[c, d_count] = torch.sum(this_confs[:,c,:,:]).item()
                pred_counts[c, d_count] = torch.sum(one_hot_predicted[:,c,:,:]).item()

            ## CALCULATE CORRECTNESS ##

            # Indicators of correctness across the 9 classes
            correct_indicators = one_hot_predicted + one_hot_labels

            # 2 if correct positive label
            correct += correct_indicators.eq(2)#.sum().item()
            # 1 if incorrect - false positive if predicted not labelled
            false_positive += torch.eq(correct_indicators.eq(1), one_hot_predicted)
            # 1 if incorrect - false negative if labelled not predicted
            false_negative += torch.eq(correct_indicators.eq(1), one_hot_labels)
            # 0 if correctly not labelled (negative) 
            correct_not_labelled += correct_indicators.eq(0)

            ## INCREASE TOTALS ##
            total += outputs[:,0,:,:].numel() # number of cells in this batch
            total_all_classes += outputs.numel()
            num_batches += inputs.size(0) # number of batches
            #print("adding to tots", torch.sum(one_hot_predicted, (0,2,3), keepdim=False))
            class_tots += torch.sum(one_hot_predicted, (0,2,3), keepdim=False) # number of cells in this class
        
            d_count += 1

            for i in range(len(classes)):
                label_tots[i] += (labels.int() == i).sum().item()

    ## CALCULATE ACCURACIES PER CLASS ##

    # Checking got reporting correct
    assert len(total_conf) == len(total_pred_count)
    assert len(total_conf) == len(testloader)
    assert len(confs) == len(classes)
    assert len(pred_counts) == len(classes)
    assert len(confs[0]) == len(testloader)
    assert len(pred_counts[0]) == len(testloader)

    # Per-class info
    class_correct = torch.sum(correct, (0,2,3), keepdim=False)
    class_false_positive = torch.sum(false_positive, (0,2,3), keepdim=False)
    class_false_negative = torch.sum(false_negative, (0,2,3), keepdim=False)
    class_correct_not_labelled = torch.sum(correct_not_labelled, (0,2,3), keepdim=False)
    
    ## my_test ##
    print("one hot predicted sum", one_hot_predicted.sum())
    assert one_hot_predicted.sum() == total_el_per_batched_class

    # calc overall accuracy TODO - check using correct totals
    overall_accuracy = 100. * (correct.sum().item() / total)
    overall_false_positive = false_positive.sum()
    overall_false_negative = false_negative.sum()
    overall_corr_not_label = correct_not_labelled.sum() # / total_all_classes
    
    # Calc average overall confidence
    average_confidences = []
    for i in range(len(total_conf)):

        # Account for training issues
        try:
            this_avg = total_conf[i] / total_pred_count[i]
        except ZeroDivisionError:
            print("err")
            this_avg = -1
        average_confidences.append(this_avg)

    average_confidence = 100 * sum(average_confidences)/len(average_confidences)

    print("\n*REPORT*\n")
    print("Trained on", len(testloader), "test images.")
    print("Acc: %d%%, AvgConf: %d FPos: %d  FNeg: %d CorrUnlabel: %d\n" %\
        (overall_accuracy, average_confidence, overall_false_positive, 
            overall_false_negative, overall_corr_not_label))

    if params["save_run"]:
        with open(params["experiment_file"].replace(".txt", "TEST.txt"), 'a+') as ef:
            ef.write("%.3f,%.3f,%d,%d,%d\n" % \
                (overall_accuracy, average_confidence, overall_false_positive, 
                 overall_false_negative, overall_corr_not_label))
    
    ## PRINT PER CLASS ## 
    # test t
    for c in range(len(classes)):

        # Try to get accuracy TODO - check using correct sums
        try:
            this_acc = 100*class_correct[c].item()/class_tots[c].item()
            this_false_pos = class_false_positive[c].item()
            this_false_neg = class_false_negative[c].item()
            this_correct_not_label = class_correct_not_labelled[c].item()#/class_tots.sum().item()
        except ZeroDivisionError:
            print("err")
            this_acc, this_false_pos, this_false_neg, this_correct_not_label = -1, -1, -1, -1
        
        # Try to get confidence
        this_avgs = []
        for t in range(len(confs[c])):
            try:
                this_conf_avg = confs[c][t] / pred_counts[c][t]
            except ZeroDivisionError:
                print("err")
                this_conf = -1
            this_avgs.append(this_conf_avg)
        
        this_conf = 100 * sum(this_avgs) / len(this_avgs)        
        
        print("%5s Acc: %d%% Conf: %.3f%% FPos: %d FNeg: %d CorrUnLab: %d on %d predicted labels (%d actual)." %\
            (classes[c], this_acc, this_conf, this_false_pos, 
                this_false_neg, this_correct_not_label, class_tots[c].item(), label_tots[c]))

        if params["save_run"]:
            with open(params["experiment_file"].replace(".txt", "TEST.txt"), 'a+') as ef:
                ef.write("CLASS%s,%.3f,%.3f,%d,%d,%d,%d,%d\n" % \
                    (classes[c], this_acc, this_conf, this_false_pos, 
                    this_false_neg, this_correct_not_label, class_tots[c].item(), label_tots[c]))


if __name__ == "__main__":
    # We have neither used dropout nor weight decay
    

    if params["save_run"]:
        params["experiment_file"] = tm.construct_file()

    ## LOAD DATA ##
    trainloader, testloader, classes = load_data(params)

    # LOAD MODEL #
    unet = load_model(params)

    ## TRAIN ##

    shape, numel = train(params, fake=False)

    print("Training complete")
    
    ## TEST ##
    test(params, shape, numel, classes)

    if params["save"]:
        print("\nSaving model to", params["save_location"])
        torch.save(unet.state_dict(), params["save_location"])
