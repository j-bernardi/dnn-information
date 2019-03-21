import torch, os, sys#, torchvision
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import training_metadata as tm
# import plot_information, information_process # TEMP - not needed for now

# Import the model
from unet_models.unet_model2d import UNet2D
from sklearn.metrics import confusion_matrix as get_confusion_matrix

# TODO - transforms - handle the dataset...

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
        batch_size=params["batch_size"])

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

def train(unet, trainloader, params, fake=False, experiment_folder="no", torch_data=False):
    """Perform training."""

    # Set and create the reporting directory
    reporting_file = experiment_folder + "reporting/"
    if not os.path.exists(reporting_file):
        os.makedirs(reporting_file)


    # loss and accuracy per epoch
    loss_list = []
    epoch_mean_loss = []
    accuracy_mean_val = []
    
    # retains the order of original training images
    train_shuffles = []

    # Learning rate and time to change #
    idxs = np.floor(params["epochs"] * len(trainloader) * params["lr_idxs_array"]).astype(int)
    lrs = params["lr_0"] * params["lr_array"]
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
            if torch_data:
                
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
            loss = tm.calc_loss(outputs, labels, one_hot=params["one_hot"], 
                                                 smoothing_type=params["smoothing_type"], 
                                                 smoothing=params["label_smoothing"])

            # Get a tensor of the predicted classes
            pred_classes = torch.argmax(outputs.data, dim=1, keepdim=True)
    
            # Put the labels in the same shape            
            shaped_labels = labels.view(labels.size(0), 1, labels.size(1), labels.size(2))

            ## RECORD ##

            # Basic accuracy
            total_cells_correct += torch.eq(pred_classes, shaped_labels.long()).sum().item() # was + .float() instead of item before sum
            
            total_cells_seen += outputs.size(0) * outputs.size(2) * outputs.size(3)
            running_cells_seen += outputs.size(0) * outputs.size(2) * outputs.size(3)
            
            # Totals
            total_images_seen += outputs.size(0)
            running_number_images += outputs.size(0)
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
                print('[%d, %5d] loss / pixel: %.3f' %
                  (epoch + 1, i + 1, running_loss / running_cells_seen))
                
                # Write iteration info to file
                if reporting_file != "no":
                    with open(reporting_file + "TRAIN.txt", 'a+') as ef:
                        ef.write("%d,%d,%.3f\n" % (epoch + 1, i + 1, running_loss / running_cells_seen))

                # Reset info
                running_loss = 0.0
                running_number_images = 0

        ## EPOCH COMPLETE ##

        # Record next epoch
        if torch.cuda.device_count() > 1:
            unet.module.next_epoch()
        else:   
            unet.next_epoch()

        ## Track losses / accuracy per epoch ##
        
        # Accuracy
        accuracy = 100. * total_cells_correct / total_cells_seen

        # Losses - per image
        epoch_mean_loss.append(epoch_loss / total_cells_seen)
        accuracy_mean_val.append(accuracy)

        # Print
        print('[Epoch %d complete] mean loss / pixel: %.3f, accuracy %.3f %%' %
              (epoch + 1, epoch_loss / total_cells_seen, accuracy))
        
        # Save to file
        if reporting_file != "no":
            with open(reporting_file + "TRAIN.txt", 'a+') as ef:
                ef.write("EPOCH%d,%.3f,%.3f\n\n" % (epoch + 1, epoch_loss / total_cells_seen, accuracy))

    return outputs.shape, epoch_mean_loss, accuracy_mean_val, train_shuffles

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

def test(unet, testloader, params, shape, classes, experiment_folder="no", torch_data=False, save_graph=False):
    """Test the network."""

    print("Testing")

    # Create file if it doesn't exist yet 
    reporting_file = experiment_folder + "reporting/"
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

    # The count of the number predicted per class[0], test[1]
    class_pred_counts = np.zeros((len(classes), len(testloader)))

    ## ACCURACY MATRIX ##
    confusion_mat = np.zeros((len(classes), len(classes)), dtype=np.long)

    with torch.no_grad():
        
        first = True

        for data in testloader:

            ## GET OUTPUTS ## 
            if torch_data:
                inputs, labels, og_idx = data[0]["image"], data[0]["classes"], data[1]
            else:
                inputs, labels, _, og_idx = data

            inputs, labels = inputs.float().to(params["device"]), labels.to(params["device"])

            total_images_seen += inputs.size(0)

            # Get outputs 
            outputs = unet(inputs)

            ## CALCULATE CONFIDENCES and LABELS ## 
            max_conf_matrix, predicted = torch.max(outputs, 1, keepdim=True)

            ## SAVE first graph for visualising ##
            if save_graph and first:

                save_graphs(inputs, labels, predicted, og_idx, reporting_file)

            # Make the predictions and labels one hot
            one_hot_predicted = tm.make_one_hot(predicted, params["device"], C=len(classes)).byte()
            one_hot_labels = tm.make_one_hot(labels.long().view(labels.size(0), 1, labels.size(1), labels.size(2)), params["device"], C=len(classes)).byte()

            # Number of cells actually in this class
            class_tots += torch.sum(one_hot_labels, (0,2,3), keepdim=False)
            predicted_tots += torch.sum(one_hot_predicted, (0,2,3), keepdim=False)

            # The confidences of predicted classes (0s elsewhere - sparse, one-hot rep)
            this_confs = torch.where(one_hot_predicted == 1, outputs, torch.tensor(0., device=params["device"]))
            
            # The total confidence for this image batch
            total_confs.append(this_confs.sum())
            total_pred_counts.append(one_hot_predicted.sum())

            for c in range(len(classes)):
                class_confs[c, total_batches_seen] = torch.sum(this_confs[:,c,:,:]).item()
                class_pred_counts[c, total_batches_seen] = torch.sum(one_hot_predicted[:,c,:,:]).item()

            ## CALCULATE CORRECTNESS ##

            # Indicators of correctness across the 9 classes
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

            # Totals
            correct += correct_mat.sum().item()
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
            total_cells_seen += outputs.size(0) * outputs.size(2) * outputs.size(3)
            total_images_seen += outputs.size(0)
            total_batches_seen += 1

            # Get the confusion matrix
            this_confusion_mat =\
               get_confusion_matrix(labels.view(-1).cpu().numpy(), 
                                    predicted.view(-1).cpu().numpy())

            #print(this_confusion_mat)

            if first:
                confusion_mat = this_confusion_mat
            else:
                confusion_mat += this_confusion_mat

            # Labelling first
            first = False

    # calc overall accuracy
    overall_accuracy = 100. * (correct / total_cells_seen)

    # Normalise matrix accuracy
    row_sums = confusion_mat.sum(axis=1)
    norm_confusion_mat = confusion_mat / row_sums[:, np.newaxis]

    ## PLOT confusion ##
    print("Confusion matrix\n", confusion_mat)

    save_confusion(confusion_mat, classes, reporting_file + "confusion_matrix")
    save_confusion(norm_confusion_mat, classes, reporting_file + "norm_confusion_matrix")

    # Calc average overall confidence
    average_confidences = []
    for i in range(len(total_confs)):
        
        try:
            # Account for training issue
            this_avg = total_confs[i] / total_pred_counts[i]
        
        except ZeroDivisionError:
            print("err")
            this_avg = -1
        
        average_confidences.append(this_avg)

    average_confidence = 100 * sum(average_confidences)/len(average_confidences)

    print("\n*REPORT*\n")
    print("Trained on", len(testloader), "test images.")
    print("Acc: %d%%, AvgConf: %d FPos: %d  FNeg: %d CorrUnlabel: %d\n" %\
        (overall_accuracy, average_confidence, false_positive, 
            false_negative, correct_not_labelled))

    if reporting_file != "no":
        with open(reporting_file + "TEST.txt", 'a+') as ef:
            ef.write("total,%.3f,%.3f,%d,%d,%d\n" % \
                (overall_accuracy, average_confidence, false_positive, 
                 false_negative, correct_not_labelled))
    
    ## PRINT PER CLASS ## 
    for c in range(len(classes)):

        # Try to get accuracy TODO - check using correct sums
        try:
            this_acc = 100.* class_correct[c].item() / class_tots[c].item()
            this_false_pos = class_false_positive[c].item()
            this_false_neg = class_false_negative[c].item()
            this_correct_not_label = class_correct_not_labelled[c].item()
        except ZeroDivisionError:
            print("err")
            this_acc, this_false_pos, this_false_neg, this_correct_not_label = -1, -1, -1, -1
        
        # Try to get confidence
        this_avgs = []
        for t in range(len(class_confs[c])):
            
            try:
                this_conf_avg = class_confs[c][t] / class_pred_counts[c][t]
            
            except ZeroDivisionError:
                print("err")
                this_conf = -1
            
            this_avgs.append(this_conf_avg)
        
        this_conf = 100 * sum(this_avgs) / len(this_avgs)
        
        print("%5s Acc: %d%% Conf: %.3f%% FPos: %d FNeg: %d CorrUnLab: %d on %d predicted labels (%d actual)." %\
            (classes[c], this_acc, this_conf, this_false_pos, 
                this_false_neg, this_correct_not_label, predicted_tots[c].item(), class_tots[c]))

        if reporting_file != "no":
            with open(reporting_file + "TEST.txt", 'a+') as ef:
                ef.write("%s,%.3f,%.3f,%d,%d,%d,%d,%d\n" % \
                    (classes[c], this_acc, this_conf, this_false_pos, 
                    this_false_neg, this_correct_not_label, predicted_tots[c].item(), class_tots[c]))

    return overall_accuracy

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

    fig, ax = plt.subplots()
    
    im = ax.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
    
    ax.figure.colorbar(im, ax=ax)

    if "norm" in to_file:
        fmt = '.2f'
        title = "Normalised Confusion Matrix"
    else:
        fmt = 'd'
        title = "Confusion Matrix"

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
    thresh = mat.max() / 2.

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
    shape, epoch_mean_loss, accuracy_mean_val, training_order =\
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
    acc = test(unet, testloader, params, shape, classes, experiment_folder=experiment_folder)

    if params["save_model"]:
        print("\nSaving model to", experiment_folder + "unet2d_TRAINED.pth")
        torch.save(unet.state_dict(), experiment_folder + "unet2d_TRAINED.pth")
