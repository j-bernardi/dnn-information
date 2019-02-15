"""" map a segmentation map to the four referral decisions and the ten additional diagnoses (see Supplementary Fig. 16)."""
import torch, torchvision, os, sys
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from classify_model import CNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# for 160,000 iterations
epochs = 5# for data: 160000
# The initial learning rate was 0.02
learn_rate_0 = 0.02
# batch sizes - 8
batch_size = 1 #8
number_samples = 0 # 0 for all
# workers - on 8 graphics processing units (GPUs)
# spread across 8 GPUs with 1 sample per GPU with dataset 3 in Supplementary Table 3.
workers = 1 #2# Cuda count gpus? not sure
# Label smoothing - 0.05
label_smoothing = 0.05

# TODO ^^
weight_decay = 0.00001

validation_split = 0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# with dataset 1 in Supplementary Table 3
index_prefix = ""
location = "data/segmented_tensors/dummy_half_slice_sample_scans/"

save_location = "models/unet/saved_models/classification.pth"
save = True

# TODO - and added some (1 × 10−5) weight decay.

# TODO - transforms - handle the dataset...
class MapDataset(Dataset):
    """
    Retinal scan dataset.
    Just stores the file location and creates objects for each image in the dataset...
    """

    def __init__(self, root_dir, index_prefix, transform=None):
        """
        Args:
            csv_file (string) path to csv file with annot
            root_dir (string) dir with images
            transform (callable, optional): apply trans to sample
        NOTES:
            # and augmented by random three-dimensional affine and elastic transformations [14]

        """
        # TODO: self.labels = pd.get_dummies(self.scan_frame[-1]).as_matrix
        self.root_dir = root_dir
        self.index_prefix = index_prefix
        self.transform = transform

    def __len__(self):
        """Gets length of data."""
        return len([f for f in os.listdir(self.root_dir) 
                    if f.endswith(".pt") and f.startswith(self.index_prefix + "map")])

    def __getitem__(self, index):
        """
        Returns a single dataframe representing an image file
        with a given index.
        """
        img_name = os.path.join(self.root_dir, self.index_prefix) + "map" + str(index) + ".pt"
        class_name = os.path.join(self.root_dir, self.index_prefix) + "class" + str(index) + ".pt"

        #print("getting")
        img_tensor = torch.load(img_name).cpu()
        class_tensor = torch.load(class_name).cpu() # note - took off sqr brackets

        #img_tensor = np.concatenate([it[np.newaxis] for it in img_tensor])
        #class_tensor = np.concatenate([it[np.newaxis] for it in class_tensor])

        #print("Got item", index)
        #print(img_tensor.shape)
        #print(class_tensor.shape)

        sample = {'image': img_tensor, 'classes': class_tensor}

        if self.transform:
            sample = self.transform(sample)
        #print("Got item", index)
        #print("returning index", index)
        #print(img_tensor.shape)
        #print("Classes", class_tensor)
        return sample


def load_data(map_dataset, batch_size, workers, val_split, shuffle=True, num=0):
    """
    Loads the data from location specified.
    num = number of batches to load
        0 for all
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        map_dataset.transform
    ])

    # Creating data indices for training and validation splits:
    if num > 0 :
        dataset_size = num
    else:
        dataset_size = len(map_dataset)
    
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    trainloader = torch.utils.data.DataLoader(map_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=workers)
    testloader = torch.utils.data.DataLoader(map_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=workers)

    classes = ('r0', 'r1', 'r2', 'r3', 'd0', 'd1', 'd2', 'd3', 'd4',
                   'd5', 'd6', 'd7', 'd8', 'd9')

    return trainloader, testloader, classes

def calc_loss(pred, target, batch_size, smoothing=0):

    #print("pred size", pred.size())
    #print("target size", target.size())

    if smoothing > 0:
        eps = smoothing
        n_class = pred.size(1)

        decision_indices = torch.from_numpy(np.array(range(4))).long().to(device)
        diagnosis_indices = torch.from_numpy(np.array(range(4, 14))).long().to(device)

        one_hot_labels = target.type(torch.long)

        #print("size one_hots", one_hot_labels.size())
        one_hot_labels = torch.zeros_like(pred).scatter(1, one_hot_labels, 1)
        #one_hot_diagnosis_labels = torch.zeros_like(pred_diagnosis).scatter(1, target_diagnosis, 1)

        one_hot_labels = one_hot_labels * (1-eps) + (1-one_hot_labels) * eps / (n_class-1)
        #one_hot_decision_labels = one_hot_decision_labels * (1-eps) + (1-one_hot_decision_labels) * eps / (n_class-1)
        #one_hot_diagnosis_labels = one_hot_diagnosis_labels * (1-eps) + (1-one_hot_diagnosis_labels) * eps / (n_class-1)

        # sum of the softmax cross entropy loss for the first four components

        pred_decision, pred_diagnosis = \
            torch.index_select(pred, 1, decision_indices), torch.index_select(pred, 1, diagnosis_indices) 
        target_diagnosis, target_decision = \
            torch.index_select(target, 1, decision_indices), torch.index_select(target, 1, diagnosis_indices) 

        one_hot_decision_labels, one_hot_diagnosis_labels = \
            torch.index_select(one_hot_labels, 1, decision_indices), torch.index_select(one_hot_labels, 1, diagnosis_indices)
        #print("size pred_decision", pred_decision.size())
        #print("size pred diagnosis", pred_diagnosis.size())

        log_decision_prb = F.log_softmax(pred_decision, dim=1)
        non_pad_mask_dec = target_decision.ne(0)

        # the sigmoid cross entropy losses for the remaining ten components (additional diagnoses labels)    
        log_diagnosis_prb = F.logsigmoid(pred_diagnosis) # QUESTION - no axis , dim=1
        non_pad_mask_diag = target_diagnosis.ne(0)

        decision_loss = -(one_hot_decision_labels * log_decision_prb).sum(dim=1)
        decision_loss = decision_loss.masked_select(non_pad_mask_dec).sum() # QUESTION - Needed?

        diagnosis_loss = -(one_hot_diagnosis_labels * log_diagnosis_prb).sum(dim=1)
        diagnosis_loss = diagnosis_loss.masked_select(non_pad_mask_diag).sum() # QUESTION - needed?        

    else:

        pred_decision, pred_diagnosis = pred[:4], pred[4:]
        target_decision, target_diagnosis = target[:4], target[4:]

        # sum of the softmax cross entropy loss for the first four components
        decision_loss = F.cross_entropy(pred_decision, target_decision, ignore_index=0, reduction='sum')
    
        # the sigmoid cross entropy losses for the remaining ten components (additional diagnoses labels)
        # QUESTION - need sum?
        diagnosis_loss = F.binary_cross_entropy_with_logits(pred_diagnosis, target_decision, ignore_index=0)

    return decision_loss + diagnosis_loss


if __name__ == "__main__":
    
    ## LOAD DATA ##
    print("Loading images")
    map_dataset = MapDataset(location, index_prefix)
    trainloader, testloader, classes = load_data(map_dataset, batch_size,
                                                 workers, validation_split, num=number_samples)
    print("Loaded.")
    print("len trainloader", len(trainloader))
    print("len testloader", len(testloader))


    # LOAD MODEL #
    ## TODO - check batch size QUESTION 
    net = CNet(in_channels=15, drop_rate=0, n_classes=14) # bn_size=batch_size
    net.float()
    print("Moving model to", device)
    net = net.to(device)

    # Learning rate and time to change #
    idxs = epochs * np.array([0, 0.2, 0.5, 0.7, 0.9, 0.95])
    lrs = learn_rate_0 * np.array([1, 0.25, 0.125, 0.015625,
                                  0.00390625, 0.001953125])
    next_idx = 1

    # Adam optimizer - https://arxiv.org/abs/1412.6980 #
    optimizer = optim.Adam(net.parameters(), lr=learn_rate_0, weight_decay=weight_decay)

    # Data: 300x350x43 subsampling
    
    ## TRAIN ##
    print("Starting training.")
    
    # epochs #
    for epoch in range(epochs):
        #print("Epoch", epoch)
        running_loss = 0

        # Iterate through data #
        for i, data in enumerate(trainloader, 0):
            #print("i in data", i)
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

            optimizer.zero_grad()

            #print("input shape", inputs.shape)
            outputs = net(inputs)

            loss = calc_loss(outputs, labels, batch_size, smoothing=label_smoothing)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # helps?
            torch.cuda.empty_cache()

        # Track loss per epoch
        print('[Epoch %d complete] loss: %.3f' %
              (epoch +1, running_loss / len(trainloader)))
        running_loss = 0.0
    print("Training complete")
    
    ## TEST ##
    print("Testing")
    class_correct, class_total = list(0. for i in range(len(classes))), list(0. for i in range(len(classes)))
    class_confs = list(0. for i in range(len(classes)))
    total, correct = 0, 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data['image'].float(), data['classes']
            images, labels = images.to(device), labels.long().to(device)

            # outputs is probabilities over the 15 classes
            outputs = net(images)
            
            # Scatter labels into one-hot format
            labels = torch.zeros_like(outputs).scatter(1, labels, 1)
            
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
        torch.save(net.state_dict(), save_location)

    # TODO does this show images?
    # imshow(torchvision.utils.make_grid(images[0]))
