"""
NOTE: 
    Script is out-dated. Work on this script ceased after data availability changed.
    Needs to be updated to operate similarly to train_segment2d.py
"""

import torch, torchvision, os, sys, math
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Import the model
from unet_models.unet_model3d import UNet3D
from training_metadata import calc_loss
from training_metadata import make_one_hot_mine as make_one_hot

import training_metadata as tm

## TODO - use training metadata for below vars ##

# for 160,000 iterations
epochs = 2# for data: 100
# The initial learning rate was 0.0001
learn_rate_0 = 0.0001
# batch sizes - 8
batch_size = 1 #8
# workers - on 8 graphics processing units (GPUs)
workers = 1 #2# Cuda count gpus? not sure
voxel_size = 9
# Label smoothing - 0.1
label_smoothing = 0.1
number_samples = 5
validation_split = 0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# with dataset 1 in Supplementary Table 3
location = "data/input_tensors/sample_scans/"

save_location = "models/unet/saved_models/unet3d.pth"
save = True
# TODO - transforms - handle the dataset...

if __name__ == "__main__":
    # We have neither used dropout nor weight decay
    
    ## LOAD DATA ##
    
    # Import data handler
    print("Appending", os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]))
    sys.path.append(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-3]))
    from data.data_utils import get_voxels_data
    
    # Load in images
    print("Loading images")
    trainloader, testloader, classes = get_voxels_data(location, batch_size,
                                                 workers, validation_split, num=number_samples)
    # Report
    print("Loaded.")
    print("len trainset", len(trainloader))
    print("len testset", len(testloader))

    # LOAD MODEL #
    unet = UNet3D()
    unet.float()
    print("Moving model to", device)
    unet = unet.to(device)

    # Learning rate and time to change #
    idxs = epochs * np.array([0, 0.1, 0.2, 0.5, 0.7, 0.9, 0.95])
    lrs = learn_rate_0 * np.array([1, 0.5, 0.25, 0.125, 0.015625,
                                  0.00390625, 0.001953125])
    next_idx = 1

    # Adam optimizer - https://arxiv.org/abs/1412.6980 #
    optimizer = optim.Adam(unet.parameters(), lr=learn_rate_0)

    # The input for each of the 128 slices is a 448 × 512 × 9 voxels image

    ## TRAIN ##
    print("Starting training.")
    # epochs #
    for epoch in range(epochs):
        #print("Epoch", epoch)

        accuracy = 0
        total_loss = 0

        # Iterate through data #
        for i, data in enumerate(trainloader, 0):

            running_loss = 0
            total_num = 0
            correct = 0

            if i == idxs[next_idx]:
                # update the learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lrs[next_idx]
                # update the next one to look at
                if next_idx + 1 < len(idxs):
                    next_idx += 1
                else:
                    next_idx = 0

            # batch inputs and classes per pixel
            inputs, labels = data['image'].float(), data['classes']
            inputs, labels = inputs.to(device), labels.to(device)
            #print("inputs shape", inputs.shape, "labels shape", labels.shape)
            
            # For each slice in the input image
            print("i", i)
            for z in range(inputs.size(2)):

                # find the z-4 and z+4 slices for a 9 voxel size
                strt, lst = z - voxel_size // 2, z + voxel_size // 2
                
                # Skip the outer voxels
                if strt < 0 or lst >= inputs.size(2):
                    continue

                optimizer.zero_grad()

                # Indices of the slices to pass to the network
                indices = torch.from_numpy(np.array(range(strt, lst+1))).long().to(device)

                # Get the 9 slices for the central slice
                this_voxels = torch.index_select(inputs, 2, indices)

                # Get the xy labels for the central slice
                these_labels = torch.index_select(labels, 2, torch.tensor([z]).to(device))
                
                # Get the classification from the model - just for the central slice
                #this_class = unet(this_voxels)
                
                # Get the one-hot classification from the model - just for the central slice
                this_means, this_stds, this_class = unet(this_voxels)
                
                # OLD: used to output 9-deep, now just outputting 1
                #this_class = torch.index_select(unet(this_voxels), 2, torch.tensor([voxel_size // 2]).to(device))
                #print("outputs shape", this_class.shape)

                # Per-voxel x-entropy, with 0.1 label-smoothing regularization
                # TODO: Is this what it means to be per-voxel?
                
                loss = calc_loss(this_class, these_labels, batch_size, smoothing=label_smoothing)
                
                # CHECK sum index - e.g. might have an extra dimension
                #print("mns stds", this_means.shape, this_stds.shape)
                this_info_loss = -0.5*(1+2*this_stds.log()-this_means.pow(2)-this_stds.pow(2)).mean(dim=0).div(math.log(2))
                #print("thi il shape", this_info_loss.shape)
                if running_loss == 0:
                    # TODO - dimensionality is wrong
                    izy_bounds = math.log(this_class.size(1), 2) - loss.div(math.log(2))
                    izx_bounds = this_info_loss
                else:
                    izy_bounds += math.log(this_class.size(1), 2) - loss.div(math.log(2))
                    izx_bounds += this_info_loss

                correct += torch.eq(this_class, make_one_hot(these_labels, this_class)).float().sum()
                total_num += this_class.numel()
                cnt = this_class.size(0)

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Track losses per epoch
        accuracy = correct/total_num
        izy_bounds /= cnt
        izx_bounds /= cnt
        running_loss /= total_num
        
        print('[Epoch %d complete] loss: %.3f, accuracy %.3f' %
              (epoch +1, running_loss, accuracy))
        running_loss = 0.0
        print("izy", izy_bounds)
        print("izx", izx_bounds)
    

    print("Training complete")
    print("Testing")
    
    class_correct, class_total = list(0. for i in range(len(classes))), list(0. for i in range(len(classes)))
    class_confs = list(0. for i in range(len(classes)))
    total, correct = 0, 0
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data['image'].float().to(device), data['classes'].to(device)

            first = True
            for z in range(inputs.size(2)):
                
                # find the z-4 and z+4 slices for a 9 voxel size
                strt, lst = z - voxel_size // 2, z + voxel_size // 2
                
                # Skip the outer voxels
                if strt < 0 or lst >= inputs.size(2):
                    continue

                indices = torch.from_numpy(np.array(range(strt, lst+1))).long().to(device)

                this_voxels = torch.index_select(inputs, 2, indices)
                
                # take only the middle slice of the output
                _, _, this_class = unet(this_voxels)
                # OLD: used to output 9-deep, now just outputting 1
                #this_class = torch.index_select(unet(this_voxels), 2, torch.tensor([voxel_size // 2]).to(device))
                
                if first:
                    outputs = this_class
                    first=False
                else:
                    outputs = torch.cat((outputs, this_class), dim=2)

            # Find the indexes that have tissue maps
            label_idxs = torch.from_numpy(np.array(range(voxel_size//2, labels.size(2)-voxel_size //2))).long().to(device)
            
            # IGNORE the outer labels that don't have segmentation maps
            labels = torch.index_select(labels, 2, label_idxs)

            # Scatter labels into one-hot format
            labels = torch.zeros_like(outputs).scatter(1, labels, 1)
            labels = labels.type(torch.long)

            # predicted is the predicted class for each position - (bs,1,z,x,y)
            _, predicted = torch.max(outputs.data, 1)
            
            # conf should be the confidence of each of the predicted classes per xyz           
            # TODO for each x,y,z position, keep the max class, set all others to 0
            
            # c is the 1, 0 matrix of correct predictions (positive or negative) in each position for all in the batch
            c = (predicted == labels)

            # sum the number of correct predictions against the labels
            correct += c.sum().item()
            total += labels.numel()

            # Iterate through batches
            for i in range(len(labels)):

                #label_matrix = labels[i] # -> (15, z, x, y)
                
                # sum the correct predictions for each label
                for l in range(len(classes)):
                    
                    class_correct[l] += c[i][l].sum() # total correct for class l (e.g. is or isn't)
                    class_total[l] += labels[i][l].numel() # total for class l, wrong and right
                    
                    #class_confs[l] += outputs[i][l] # e.g. sum only on the axes where 
    
    print("Accuracy of network on", len(testloader), "test images: %d %%" % (100*correct/total))
    
    #class_confs.div_(torch,norm(class_confs,2))
    
    for i in range(len(classes)):
        print("accuracy of %5s : %.3f %%" % (classes[i], 100*class_correct[i]/class_total[i]))
        #print("\tconfidence %.3f" % class_confs[i]/class_total[i])

    if save:
        print("Saving model to", save_location)
        torch.save(unet.state_dict(), save_location)
