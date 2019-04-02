import torch, os, sys, datetime, time, random#, torchvision
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import training_metadata as tm
# import plot_information, information_process # TEMP - not needed for now

# Import the model
from unet_models.unet_model2d import UNet2D
from sklearn.metrics import confusion_matrix as get_confusion_matrix

def load_h5_data(params, number_samples=-1):

    # Data handler 
    print("Appending", os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]))
    sys.path.append(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]))
    from data.data_utils import get_imdb_data

    # Load in
    print("Loading images")
    (trainloader, testloader), (train_id, test_id), classes = get_imdb_data(
        params["scan_location"], val_split=params["validation_split"], 
        num=number_samples, workers=params["workers"], 
        batch_size=params["batch_size"], shuffle=params["shuffle"],
        chop=params["chop"], clean=params["clean"])

    # Report
    print("Loaded.")
    print("len trainset", len(trainloader))
    print("len testset", len(testloader))

    return trainloader, testloader, train_id, test_id, classes

def load_torch_data(params, number_samples=-1):

    # Data handler 
    print("Appending", os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]))
    sys.path.append(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]))
    from data.data_utils import get_torch_segmentation_data

    # Load in
    print("Loading images")
    (trainloader, testloader), classes = get_torch_segmentation_data(
        params["scan_location"], val_split=params["validation_split"], 
        num=number_samples, workers=params["workers"], 
        batch_size=params["batch_size"])

    # Report
    print("Loaded.")
    print("len trainset", len(trainloader))
    print("len testset", len(testloader))

    return trainloader, testloader, classes

def load_model(params, experiment_folder="no", save_reps=False):

    unet = UNet2D(experiment_folder=experiment_folder, save_reps=save_reps)
    unet.float()

    # Multi GPU usage
    if torch.cuda.device_count() > 1:
        
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        unet = torch.nn.DataParallel(unet)

    print("Moving model to", params["device"])
    unet = unet.to(params["device"])

    return unet

def train(unet, trainloader, params, fake=False, experiment_folder="no", total_number_images=88):
    """Perform training."""

    # Set and create the reporting directory
    reporting_file = experiment_folder #+ "reporting/"
    if not os.path.exists(reporting_file):
        os.makedirs(reporting_file)


    # loss and accuracy per epoch
    loss_list = []
    epoch_mean_loss = []
    accuracy_mean_val = []
    central_accuracy_mean_val = []
    frst = True
    global_images_seen = 0

    # retains the order of original training images
    train_shuffles = []

    # Which images to change learning rate on:
    idxs = np.floor(params["epochs"] * total_number_images * params["lr_idxs_array"]).astype(int)
    print("Updating at img", idxs)

    # Update learning rate to:
    lrs = params["lr_0"] * params["lr_array"]
    print("Updating to lrs", lrs)

    # Next index to update learning rate on
    next_idx = 0

    # Adam optimizer - https://arxiv.org/abs/1412.6980 #
    optimizer = optim.Adam(unet.parameters(), lr=params["lr_0"])

    if torch.cuda.device_count() > 1:
        unet.module.reset()
    else:
        unet.reset()

    print("Starting training.")
    
    # epochs #
    for epoch in range(params["epochs"]):

        # Iteration trackers
        running_loss, running_number_images = 0.0, 0
        running_cells_seen = 0
        
        # Epoch trackers
        epoch_loss  = 0.0
        
        total_cells_seen, total_cells_correct = 0, 0
        total_central_cells_correct, total_central_cells_seen = 0, 0
        total_batches_seen, total_images_seen = 0, 0

        train_shuffles.append([])

        # Reset for the start of this epoch
        if torch.cuda.device_count() > 1:
            unet.module.reset()
        else:
            unet.reset()
        
        # ITERATE DATA
        for i, data in enumerate(trainloader, 0):

            # Load the tensors properly
            if params["torch_data"]:
                
                inputs, labels, original_index = data[0]["image"], data[0]["classes"], data[1]
                #print("inputs, labels\n", inputs.shape, labels.shape)

            else:

                inputs, labels, weight, original_index = data
                #print("inputs, labels, weight\n", inputs.shape, labels.shape, weight.shape)
            
            # Set up input images and labels
            inputs, labels = inputs.float().to(params["device"]), labels.to(params["device"])

            # Record the ordering
            train_shuffles[epoch].extend(original_index.numpy().tolist())

            optimizer.zero_grad()

            # adjust lr over iterations
            
            # If got to end (e.g. next_idx is outside range for idxs), don't update
            if next_idx < len(idxs):

                # If time to update, update
                if global_images_seen >= idxs[next_idx]:
                    
                    print("UPDATING LR", lrs[next_idx])

                    # update the learning rate
                    for g in optimizer.param_groups:
                        g['lr'] = lrs[next_idx]

                    # update the next one to look at
                    next_idx += 1

            # Get this batch's outputs - probabilities over 9 classes
            outputs = unet(inputs)

            if frst:
                print("Outputs sum to 1? (E.g. is output a probability)\n", (
                     ((outputs.sum(dim=1) > 0.999).all()) & 
                     ((outputs.sum(dim=1) < 1.001).all()))
                     )
                print("If true (e.g. 1) - then have a problem with double softmax application")

            # Skip the actual training if desired [for testing]
            if fake:
                return outputs.shape, [],[],[],[]

            # Calc loss
            loss = tm.calc_loss(inputs, outputs, labels, 
                                one_hot_flag=params["one_hot"], 
                                smoothing_type=params["smoothing_type"], 
                                smoothing=params["label_smoothing"],
                                clean=params["clean"], 
                                prnt=frst)

            # Get a tensor of the predicted classes
            pred_classes = torch.argmax(outputs.data, dim=1, keepdim=True)
    
            # Put the labels in the same shape            
            shaped_labels = labels.view(labels.size(0), 1, labels.size(1), labels.size(2))

            ## RECORD ##

            # Basic accuracy
            cells_correct = torch.eq(pred_classes, shaped_labels.long())
            cells_seen = outputs.size(0) * outputs.size(2) * outputs.size(3)

            # Remove from correct counts
            central_cells_correct = torch.where(shaped_labels == 0, 
                                        torch.tensor(0, device=inputs.device, dtype=torch.uint8),
                                        cells_correct)
            central_cells_correct = torch.where(shaped_labels == 8, 
                                        torch.tensor(0, device=inputs.device, dtype=torch.uint8),
                                        central_cells_correct)

            # Remove non-central cells
            central_cells_seen = outputs.size(0) * outputs.size(2) * outputs.size(3)\
                                - (shaped_labels == 0).sum().item()\
                                - (shaped_labels == 8).sum().item()
            if frst:
                print("Cells seen", cells_seen)
            
            # Remove (white and of class 0 or 8) from reporting metrics
            if params["clean"] == "loss":
                if frst:
                    print("Removing ignored loss cells from the training accuracy calculations")
                
                # Class 0 - remove white from correct count
                class0_remove = ((inputs == 1.) & (shaped_labels == 0))#.view((inputs.size(0), inputs.size(1), inputs.size(2)))
                
                # Class 8 - remove white from correct count
                class8_remove = ((inputs == 1.) & (shaped_labels == 8))#.view((inputs.size(0), inputs.size(1), inputs.size(2)))

                # Remove from correct counts
                cells_correct = torch.where(class0_remove == 1, 
                                            torch.tensor(0, device=inputs.device, dtype=torch.uint8),
                                            cells_correct)
                
                cells_correct = torch.where(class8_remove == 1,  
                                            torch.tensor(0, device=inputs.device, dtype=torch.uint8),
                                            cells_correct)

                # remove white from seen count
                cells_seen -= (class0_remove).sum().item()
                cells_seen -= (class8_remove).sum().item()
            
            if frst:
                print("New cells seen", cells_seen)

            total_cells_correct += cells_correct.sum().item()
            total_central_cells_correct += central_cells_correct.sum().item()
            
            total_cells_seen += cells_seen
            running_cells_seen += cells_seen
            total_central_cells_seen += central_cells_seen
            
            # Totals
            global_images_seen += outputs.size(0) # all epochs
            total_images_seen += outputs.size(0) # current epoch
            running_number_images += outputs.size(0) # current report iter
            total_batches_seen += 1

            ## UPDATE WEIGHTS ##
            loss.backward()
            optimizer.step()

            # Add losses
            running_loss += loss.item()
            epoch_loss += loss.item()

            # May as well
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            ## RECORD ITERATION INFO ##

            # Do we print this it?
            print_it = False
            if len(trainloader) >= 5:
                if i % (len(trainloader) // 5) == (len(trainloader) // 5) - 1:
                    print_it = True
            else:
                print_it = True
            
            # Do the printing
            if print_it:
                # Print iteration info
                print('[%d, %5d] loss / pixel: %.5f' %
                  (epoch + 1, i + 1, running_loss / running_cells_seen))
                
                # Write iteration info to file
                if reporting_file != "no":
                    with open(reporting_file + "TRAIN.txt", 'a+') as ef:
                        ef.write("%d,%d,%.5f\n" % (epoch + 1, i + 1, running_loss / running_cells_seen))

                # Reset info
                running_loss = 0.0
                running_number_images = 0
                running_cells_seen = 0

            # Not the first iteration anymore
            frst = False

        ## EPOCH COMPLETE ##

        # Record next epoch
        if torch.cuda.device_count() > 1:
            unet.module.next_epoch()
        else:   
            unet.next_epoch()

        ## Track losses / accuracy per epoch ##
        
        # Accuracy
        accuracy = 100. * total_cells_correct / total_cells_seen
        central_accuracy = 100. * total_central_cells_correct / total_central_cells_seen

        # Losses - per image
        epoch_mean_loss.append(epoch_loss / total_cells_seen)
        accuracy_mean_val.append(accuracy)
        central_accuracy_mean_val.append(central_accuracy)

        # Print
        print('[Epoch %d complete] mean loss / pixel: %.3f, accuracy %.3f%%, dt %s' %
              (epoch + 1, epoch_loss / total_cells_seen, accuracy, datetime.datetime.now()))

        # Save to file
        if reporting_file != "no":
            with open(reporting_file + "TRAIN.txt", 'a+') as ef:
                ef.write("EPOCH%d,%.3f,%.3f\n\n" % (epoch + 1, epoch_loss / total_cells_seen, accuracy))
    
    return outputs.shape, epoch_mean_loss, accuracy_mean_val, central_accuracy_mean_val, train_shuffles

def do_info(unet, training_order, trainloader, params):
    """
    Do the information analysis.
    NOTE - currently only implemented for reps saved as list
    Not to file - need to reconstruct reps list from files if taking this approach
    TOOD ^
    """
    if torch.cuda.device_count() > 1:
        ws = tm.get_aligned_representations(unet.module.representations_per_epochs, training_order)
    else:
        ws = tm.get_aligned_representations(unet.representations_per_epochs, training_order)
    # TODO - have unet store the order in which the training data came, too (e.g. X_train)
    X_train, y_one_hot = get_original_order(trainloader)
    
    IXT_array, ITY_array = information_process.get_information(ws, X_train, y_one_hot, 
                                                               params["num_of_bins"], every_n=params["every_n"], return_matrices=True)

    return IXT_array, ITY_array

def get_original_order(trainloader):

    ## TEMP HACK ##
    len_tl = 0
    ## TODO - find X_train, y_train from trainloader
    
    print("Creating X_train")
    """
    for i, _ in enumerate(trainloader):
        len_tl += 1
    """ 
    X_train = None #np.array([None for _ in range(len_tl)])
    y_train = None #np.array([None for _ in range(len_tl)])

    # A list of the order in which X_train appears in terms of original index
    # E.g. [2, 1, 3, 0] means shuffle order was as such
    og_idxs = []

    for _, data in enumerate(trainloader):
        # for item in the trainingset
        x = data[0].cpu().numpy() # X data
        #print("expecting data to be in batches")
        #print(x.shape)
        y = data[1].cpu().numpy() # y data
        #print("idxs of batch")
        #print(data[-1].cpu().numpy().tolist())
        og_idxs += data[-1].cpu().numpy().tolist()

        if X_train is None:
            X_train = x
            y_train = y
        else:
            X_train = np.concatenate((X_train, x))
            y_train = np.concatenate((y_train, y))

    # reorder them
    # idx lines up with order of X, so X[0] is at idx.index(0))
    # X[i] = X[idx[i]]
    X_train[:] = X_train[np.array(og_idxs)[:]]
    y_train[:] = y_train[np.array(og_idxs)[:]]

    y_train = np.expand_dims(y_train, axis=1)

    # y_one_hot = np.concatenate([y_train, 1 - y_train], axis=1) - WAS making it one hot for binary data
    y_one_hot = tm.make_one_hot(y_train, "cpu").cpu().numpy()

    return X_train, y_one_hot

def test(unet, testloader, params, shape, classes, experiment_folder="no", save_graph=False, return_confusion=False):
    """Test the network."""

    print("Testing")
    img_idx_to_print = random.randint(0, len(testloader)-1)

    # Create file if it doesn't exist yet 
    reporting_file = experiment_folder + "test_results/"
    if not os.path.exists(reporting_file):
        os.makedirs(reporting_file)
    
    # OVER ALL BATCHES: cells classified, correct, the number of batches seen
    total_cells_seen = 0
    total_batches_seen, total_images_seen = 0, 0

    # Accuracy trackers - total
    correct, false_positive, false_negative, correct_not_labelled = 0, 0, 0, 0
    
    # Per-class accuracy trackers
    class_correct = torch.zeros((shape[1]), device=params["device"], dtype=torch.long)
    class_false_positive = torch.zeros((shape[1]), device=params["device"], dtype=torch.long)
    class_false_negative = torch.zeros((shape[1]), device=params["device"], dtype=torch.long)
    class_correct_not_labelled = torch.zeros((shape[1]), device=params["device"], dtype=torch.long)

    # Total labels for each class - count
    class_tots = torch.tensor([0] * len(classes), device=params["device"])
    
    # Total predicted for each class - count
    predicted_tots = torch.tensor([0] * len(classes), device=params["device"])

    ## CONFIDENCE TRACKING ##

    # totals per test [0]
    total_confs, total_pred_counts = [], []

    # The sum of the confidences of prediction per class[0], test[1]
    class_confs = np.zeros((len(classes), len(testloader)))

    ## ACCURACY MATRIX ##
    confusion_mat = np.zeros((len(classes), len(classes)), dtype=np.long)

    ## TESTING ##
    with torch.no_grad():
        
        first = True

        for data in testloader:

            ## GET OUTPUTS ## 
            if params["torch_data"]:
                inputs, labels, og_idx = data[0]["image"], data[0]["classes"], data[1]
            else:
                inputs, labels, _, og_idx = data

            inputs, labels = inputs.float().to(params["device"]), labels.to(params["device"])

            total_images_seen += inputs.size(0)

            # Get outputs - converted to a probability in loss function
            outputs = unet(inputs)

            ## NEW LINE ## - assert probability-like
            if first:
                print("Example output tensor", outputs[0, :, 256, 256])

            ## CALCULATE CONFIDENCES and LABELS ## 
            _, predicted = torch.max(outputs, 1, keepdim=True)

            # Put the labels in the same shape
            shaped_labels = labels.view(labels.size(0), 1, labels.size(1), labels.size(2))

            # Make the predictions and labels one hot
            one_hot_predicted = tm.make_one_hot(predicted, params["device"], C=len(classes)).byte()
            one_hot_labels = tm.make_one_hot(labels.long().view(labels.size(0), 1, labels.size(1), labels.size(2)), params["device"], C=len(classes)).byte()

            # Number of cells actually in this class
            class_tots += torch.sum(one_hot_labels, (0,2,3), keepdim=False)
            predicted_tots += torch.sum(one_hot_predicted, (0,2,3), keepdim=False)

            total_pred_counts.append(one_hot_predicted.sum())

            ## CONFIDENCE ## 

            # Convert whole one hot reps to probabilities
            probability_matrix = F.softmax(outputs, dim=1)

            # The confidences of predicted classes (0s elsewhere - sparse, prob-hot rep)
            this_confs = torch.where(one_hot_predicted == 1, probability_matrix, torch.tensor(0., device=params["device"]))

            # The total confidence for this image batch
            total_confs.append(this_confs.sum())

            for c in range(len(classes)):

                class_confs[c, total_batches_seen] = torch.sum(this_confs[:,c,:,:]).item()

            ## CALCULATE CORRECTNESS ##

            # Basic accuracy
            cells_correct = torch.eq(predicted, shaped_labels.long())
            cells_seen = outputs.size(0) * outputs.size(2) * outputs.size(3)

            if first:
                print("Cells seen", cells_seen)

            # Remove the cells that are white and of class 0 or 8
            if params["clean"] == "loss":

                class0_remove = ((inputs == 1.) & (shaped_labels == 0))
                class8_remove = ((inputs == 1.) & (shaped_labels == 8))

                # Class 0 - remove white from correct count
                cells_correct = torch.where(class0_remove == 1,
                                            torch.tensor(0, device=inputs.device, dtype=torch.uint8),
                                            cells_correct)

                # Class 8 - remove white from correct count
                cells_correct = torch.where(class8_remove == 1,
                                            torch.tensor(0, device=inputs.device, dtype=torch.uint8),
                                            cells_correct)

                # remove white from seen count
                cells_seen -= (class0_remove).sum().item()
                cells_seen -= (class8_remove).sum().item()

                # Class 0 - remove blank cells from predictions indicators by setting to 3
                # was one_hot_labels[:,0,:,:] == 1) & (inputs[:,0,:,:] == 1.)
                assert class0_remove.size(1) == 1
                one_hot_predicted[:,0,:,:] = torch.where(class0_remove.view(class0_remove.size(0), class0_remove.size(2), class0_remove.size(3)) == 1, 
                                                           torch.tensor(3, device=inputs.device, dtype=torch.uint8), 
                                                           one_hot_predicted[:,0,:,:])
            
                # Class 8 - remove blank cells from predictions indicators by setting to 3
                # was (one_hot_labels[:,8,:,:] == 1) & (inputs[:,0,:,:] == 1.)
                assert class8_remove.size(1) == 1
                one_hot_predicted[:,8,:,:] = torch.where(class8_remove.view(class8_remove.size(0), class8_remove.size(2), class8_remove.size(3)) == 1, 
                                                           torch.tensor(3, device=inputs.device, dtype=torch.uint8), 
                                                           one_hot_predicted[:,8,:,:])

            if first:
                print("New cells seen", cells_seen)

            # Indicators of correctness across the 9 classes - 3 won't be caught below
            correct_indicators = one_hot_predicted + one_hot_labels

            # 2 if correct positive label
            correct_mat = correct_indicators.eq(2)
            # 1 if incorrect - false positive if predicted not labelled
            false_positive_mat = torch.eq(correct_indicators.eq(1), one_hot_predicted)
            # 1 if incorrect - false negative if labelled not predicted
            false_negative_mat = torch.eq(correct_indicators.eq(1), one_hot_labels)
            # 0 if correctly not labelled (negative) 
            correct_not_labelled_mat = correct_indicators.eq(0)

            ## CALCULATE ACCURACIES PER CLASS ##
            try:
                assert correct_mat.sum().item() == cells_correct.sum().item()
            except:
                print("***Assertion error***\n" +\
                      "Correct matrix =", correct_mat.sum().item(),
                      ", cells correct =", cells_correct.sum().item(),
                      "*********************")
                # Continue instead of reassert:
                #assert correct_mat.sum().item() == cells_correct.sum().item()

            # Totals
            correct += cells_correct.sum().item()#correct_mat.sum().item()
            false_positive += false_positive_mat.sum().item()
            false_negative += false_negative_mat.sum().item()
            correct_not_labelled += correct_not_labelled_mat.sum().item()

            # Per-class info
            class_correct += torch.sum(correct_mat, (0,2,3), keepdim=False)
            class_false_positive += torch.sum(false_positive_mat, (0,2,3), keepdim=False)
            class_false_negative += torch.sum(false_negative_mat, (0,2,3), keepdim=False)
            class_correct_not_labelled += torch.sum(correct_not_labelled_mat, (0,2,3), keepdim=False)

            ## INCREASE TOTALS ##

            # number of cells in this batch item
            total_cells_seen += cells_seen
            total_images_seen += outputs.size(0)
            total_batches_seen += 1

            # Set the faulty labels to be correct for the confusion matrix and saving the image
            if params["clean"] == "loss":
                fixed_predictions = torch.where(((shaped_labels == 0) & (inputs == 1.)) | ((shaped_labels == 8) & (inputs == 1.)),
                                                shaped_labels.long(), predicted.long())
            else:
                fixed_predictions = predicted

            # Get confusion matrix - use fixed predictions
            this_confusion_mat =\
               get_confusion_matrix(labels.view(-1).cpu().numpy(), 
                                    fixed_predictions.view(-1).cpu().numpy())

            ## SAVE first graph for visualising ##

            if total_batches_seen - 1 == img_idx_to_print:
                if save_graph:
                    save_graphs(inputs, labels, fixed_predictions, og_idx, reporting_file)
            
            if first:
                confusion_mat = this_confusion_mat
            else:
                confusion_mat += this_confusion_mat

            # Labelling first
            first = False

    ##### ANALYSIS #####

    ## CONFUSION MATRIX ##

    # Normalise
    norm_confusion_mat = confusion_mat / confusion_mat.sum(axis=1)[:, np.newaxis]
    
    # Plot 
    save_confusion(confusion_mat, classes, reporting_file + "confusion_matrix")
    save_confusion(norm_confusion_mat, classes, reporting_file + "norm_confusion_matrix")

    ## OVERALL ACCURACY - ALL CLASSES ##
    
    # Total, conventional accuracy
    overall_accuracy = 100. * (correct / total_cells_seen)

    # Accuracy excluding edge classes
    central_accuracy = 100. * class_correct[1:(len(class_correct)-1)].sum().item()\
                              / class_tots[1:(len(class_tots)-1)].sum().item()

    try:
        # Calc overall confidence of the classification
        average_confidence = sum(total_confs) / sum(total_pred_counts)
    except ZeroDivisionError:
        print("Zero division error in total confidence")
        average_confidence = -1

    # Print to command
    print("\n*OVERALL REPORT*\n")
    print("Trained on", len(testloader), "test images.")
    print("Acc: %d%%, CentralAcc: %d%% AvgConf: %.3f FPos: %d  FNeg: %d CorrUnlabel: %d\n" %\
        (overall_accuracy, central_accuracy, average_confidence, false_positive, 
            false_negative, correct_not_labelled))

    # Print to file
    if reporting_file != "no":
        with open(reporting_file + "TEST.txt", 'a+') as ef:
            ef.write("total,%.3f,%.3f,%.3f,%d,%d,%d\n" % \
                (overall_accuracy, central_accuracy, average_confidence, false_positive, 
                 false_negative, correct_not_labelled))

    ## PRINT ACCURACY PER CLASS ## 
    print("\n*PER CLASS REPORT*\n")
    accuracies, confs = [], []
    for c in range(len(classes)):

        # Accuracy per class
        try:
            this_acc = 100.* class_correct[c].item() / class_tots[c].item()
            this_false_pos = class_false_positive[c].item()
            this_false_neg = class_false_negative[c].item()
            this_correct_not_label = class_correct_not_labelled[c].item()
        except ZeroDivisionError:
            print("Zero division error in accuracy")
            this_acc, this_false_pos, this_false_neg, this_correct_not_label = -1, -1, -1, -1

        # Confidence per class
        try:
            this_conf = sum(class_confs[c]) / predicted_tots[c].item()
        except ZeroDivisionError:
            print("Zero division error in confidence")
            this_conf = -1

        # Keep a list of the accuracies per class
        accuracies.append(this_acc)
        confs.append(this_conf)
        
        print("%5s Acc: %d%% Conf: %.3f Corr: %d FPos: %d FNeg: %d CorrUnLab: %d on %d predicted labels (%d actual)." %\
            (classes[c], this_acc, this_conf, class_correct[c], this_false_pos, 
                this_false_neg, this_correct_not_label, predicted_tots[c].item(), class_tots[c]))

        if reporting_file != "no":
            with open(reporting_file + "TEST.txt", 'a+') as ef:
                ef.write("%s,%.3f,%.3f,%d,%d,%d,%d,%d,%d\n" % \
                    (classes[c], this_acc, this_conf, class_correct[c], this_false_pos, 
                    this_false_neg, this_correct_not_label, predicted_tots[c].item(), class_tots[c]))

    ## PRINT EQUAL CONTRIBUTION DATA ##
    print("\n*EQUAL WEIGHT REPORT*\n")
    print("Accuracies (equal contribution) %d%%" % (sum(accuracies) / len(accuracies)))
    print("Confidence (equal contribution) %.3f" % (sum(confs) / len(confs)))

    if reporting_file != "no":
        with open(reporting_file + "TEST.txt", 'a+') as ef:
                ef.write("\nEqually weighted accuracies: %.d%%" % (sum(accuracies) / len(accuracies)))
                ef.write("\nEqually weighted confidence: %.3f\n" % (sum(confs) / len(confs)))

    if return_confusion:
        return overall_accuracy, central_accuracy , confusion_mat, norm_confusion_mat
    else:
        return overall_accuracy, central_accuracy

def save_graphs(inputs, labels, predicted, og_idx, reporting_file):
    """Save the graphs to visualise how well we've done."""
    try:
        print("Image index", og_idx[0])
        plt.figure()
        plt.imshow(torch.squeeze(inputs, dim=1)[0,:,:].cpu().numpy(), cmap="gray")
        plt.savefig(reporting_file + "original_image" + str(og_idx[0].item()) + ".png")
        plt.close()
    except:
        print("Failed to visualise inputs", inputs.shape)

    # Save label
    try:
        plt.figure()
        plt.imshow(labels[0,:,:].cpu().numpy(), cmap="gray")
        plt.savefig(reporting_file + "labels" + str(og_idx[0].item()) + ".png")
        plt.close()
    except:
        print("Failed to visualise inputs", labels.shape)

    # save output
    try:
        plt.figure()
        plt.imshow(torch.squeeze(predicted, dim=1)[0,:,:].cpu().numpy(), cmap="gray")
        plt.savefig(reporting_file + "output" + str(og_idx[0].item()) + ".png")
        plt.close()
    except:
        print("Failed to visualise predicted", predicted.shape)

    return

def save_confusion(mat, classes, to_file):
    """Saves matrix plot to file and dumps."""

    mat.dump(to_file + ".pkl")

    if "norm" in to_file:
        fmt   = '.2f'
        title = "Normalised Confusion Matrix"
        clim  = (0. , 1.)
        thresh = 0.9
        col_list = [(0., 'white'), (0.01, 'lightcoral'), (0.1, 'indianred'), (0.8, 'lightgreen'), (1., 'green')]
        #col_list = [(0., 'white'), (0.01, 'r'), (0.05, 'coral'), (0.1, 'gold'), (0.2, 'g'), (1., 'b')]
        cmap = colors.LinearSegmentedColormap.from_list('show_confusion', col_list, N=256, gamma=1.0)
    else:
        clim  = (0. , mat.max())
        fmt = 'd'
        title = "Confusion Matrix"
        cmap = plt.cm.Blues
        thresh = mat.max() / 2.

    fig, ax = plt.subplots()

    im = ax.imshow(mat, interpolation='nearest', cmap=cmap, clim=clim)
    
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(mat.shape[1]),
           yticks=np.arange(mat.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Label',
           xlabel='Prediction')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(mat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black")

    fig.tight_layout()

    fig.savefig(to_file + ".png")

    plt.close(fig)

    return

if __name__ == "__main__":

    # We have neither used dropout nor weight decay
    
    # FIX UP FILES #
    classes = ["class" + str(i) for i in range(9)]
    
    experiment = "uniform_fixed_eps_lr001_ep120_bs4"
    fl = "data/2019-03-23-initialising_loss/" + experiment + "/reporting/norm_confusion_matrix"
    
    mat = np.load( fl + ".pkl")
    
    save_confusion(mat, classes, fl)
    
    sys.exit()

    ##

    # limit number for testing- 0 for all
    number_samples = 5
    
    params = tm.get_params()

    if params["save_run"]:
        experiment_folder = tm.construct_file(params, "data/training_data/")
    else:
        experiment_folder = "no"

    ## LOAD DATA ##
    trainloader, testloader, train_id, test_id, classes = load_h5_data(params, number_samples)

    # LOAD MODEL #
    unet = load_model(params, experiment_folder=experiment_folder)

    ## TRAIN ##
    shape, epoch_mean_loss, accuracy_mean_val, central_accuracy, training_order =\
        train(unet, trainloader, params, fake=False, experiment_folder=experiment_folder)

    if params["save_model"]:
        print("\nSaving model to", experiment_folder + "unet2d_TRAINED.pth")
        torch.save(unet.state_dict(), experiment_folder + "unet2d_TRAINED.pth")

    print("Training complete")

    # can plot epoch_mean_loss on epoch for loss change:
    # plt.plot(np.arange(len(epoch_mean_loss)), epoch_mean_loss)
    
    # Plot the accuracy on epoch of the validation set
    # plt.plot(np.arange(len(accuracy_mean_val)), accuracy_mean_val)

    ## INFO ANALYSIS ##
    if params["information"]:

        IXT_array, ITY_array = do_info(unet, training_order, trainloader, params)

        print("IXT shape", IXT_array.shape)
        print(IXT_array)

        print("ITY shape", ITY_array.shape)
        print(ITY_array)

        import importlib
        importlib.reload(plot_information)

        plot_information.plot_information_plane(IXT_array, ITY_array, num_epochs=params["epochs"], every_n=params["every_n"])

    ## TEST ##
    acc, cent_acc = test(unet, testloader, params, shape, classes, experiment_folder=experiment_folder)

    if params["save_model"]:
        print("\nSaving model to", experiment_folder + "unet2d_TRAINED.pth")
        torch.save(unet.state_dict(), experiment_folder + "unet2d_TRAINED.pth")
