import torch, torchvision, os, sys
import torch.optim as optim
import numpy as np
from unet_model import UNet3D
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# for 160,000 iterations
epochs = 50# for data: 160000
# The initial learning rate was 0.0001
learn_rate_0 = 0.0001
# batch sizes - 8
batch_size = 1 #8
# workers - on 8 graphics processing units (GPUs)
workers = 1 #2# Cuda count gpus? not sure
# Label smoothing - 0.1
label_smoothing = 0.1

validation_split = 0.2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# with dataset 1 in Supplementary Table 3
index_prefix = "dummy_half_slice_sample_scans_"
location = "data/tensors"

# TODO - transforms - handle the dataset...
class VoxelsDataset(Dataset):
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
            # we augmented the data by applying
            #   affine and
            #   elastic transformations
            # jointly over the inputs and ground-truth segmentations
            # Intensity transformations over the inputs were also applied.

        """
        # TODO: self.labels = pd.get_dummies(self.scan_frame[-1]).as_matrix
        self.root_dir = root_dir
        self.index_prefix = index_prefix
        self.transform = transform

    def __len__(self):
        return len([f for f in os.listdir(self.root_dir) if f.endswith(".pt") and f.startswith(self.index_prefix + "image")])

    def __getitem__(self, index):
        """
        Returns a single dataframe representing an image file
        with a given index.
        """
        img_name = os.path.join(self.root_dir, self.index_prefix) + "image" + str(index) + ".pt"
        class_name = os.path.join(self.root_dir, self.index_prefix) + "class" + str(index) + ".pt"

        img_tensor = [torch.load(img_name)]
        class_tensor = [torch.load(class_name)]

        img_tensor = np.concatenate([it[np.newaxis] for it in img_tensor])
        class_tensor = np.concatenate([it[np.newaxis] for it in class_tensor])

        sample = {'image': img_tensor, 'classes': class_tensor}

        if self.transform:
            sample = self.transform(sample)
        print("Got item", index)
        return sample


def load_data(scan_dataset, batch_size, workers, val_split, shuffle=True):
    """
    Loads the data from location specified.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        scan_dataset.transform
    ])

    # Creating data indices for training and validation splits:
    dataset_size = len(scan_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    trainloader = torch.utils.data.DataLoader(scan_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=workers)
    testloader = torch.utils.data.DataLoader(scan_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=workers)

    classes = ('s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
                   's10', 's11', 's12', 's13', 's14', 's15')

    return trainloader, testloader, classes

def calc_loss(pred, gold, smoothing=0):
    """
    Calc CEL and apply label smoothing.
    TODO - verify
    Came from:
        https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/train.py
    """
    gold = gold.contiguous().view(-1)

    if smoothing > 0:
        eps = smoothing
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=0, reduction='sum')

    return loss

if __name__ == "__main__":
    # We have neither used dropout nor weight decay
    ## LOAD DATA ##
    print("Loading images")
    scan_dataset = VoxelsDataset(location, index_prefix)
    trainloader, testloader, classes = load_data(scan_dataset, batch_size,
                                                 workers, validation_split)
    print("Loaded.")

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

    # Data: 448x512x128 image
    # The input for each of the 128 slices is a 448 × 512 × 9 voxels image
    #   TODO: load in the 4 slices either side to pass the 448x512x9 to the network
    # no padding

    ## TRAIN ##
    print("Starting training.")
    # epochs #
    for epoch in range(epochs):
        #print("Epoch", epoch)
        running_loss = 0

        # Iterate through data #
        for i, data in enumerate(trainloader, 0):

            if i == idxs[next_idx]:
                # update the learning rate
                for g in optim.param_groups:
                    g['lr'] = lrs[next_idx]
                # update the next one to look at
                if next_idx + 1 < len(idxs):
                    next_idx += 1
                else:
                    next_idx = 0

            # batch inputs and classes per pixel
            inputs, labels = data['image'].float(), data['classes'].float()
            inputs, labels = inputs.to(device), labels.to(device)
            #print("inputs shape", inputs.shape, "labels shape", labels.shape)

            # TODO - for each plane, pass the 4 either side
            # e.g. 9 slices per slice to the network to train on - outputs 3d image still

            optimizer.zero_grad()

            outputs = unet(inputs)

            # Per-voxel x-entropy, with 0.1 label-smoothing regularization
            # TODO - verify calc_loss
            # TODO: What does it mean to be per-voxel?
            # Voxel is one of the 9 slices passed as input
            # Some sort of "for z-layer in input (average(loss))?"
            loss = calc_loss(outputs, labels, smoothing=label_smoothing)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Track loss per epoch
        print('[Epoch %d complete] loss: %.3f' %
              (epoch +1, running_loss / len(trainloader)))
        running_loss = 0.0
    print("Training complete")

    print("Loading test data")
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    print("Testing")
    # TODO - check this works for looking over every pixel - this was written for a 2d image classifier
    # the output is an estimated probability over the 15 classes, for each of the 448 × 512 × 1 output voxels
    #   https://arxiv.org/abs/1512.00567
    outputs = net(images)
    class_corrct, class_total = list(0. for i in range(len(classes))), list(0. for i in range(len(classes)))
    class_confs = list(0. for i in range(len(classes)))
    total, correct = 0, 0
    with torch.no_grad():
        for data in testloader:
            total += labels.size(0)
            images, labes = data
            outputs = net(images)
            conf, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                class_confs[label] += conf[i].item()
    print("Accuracy of network on", len(testloader), "test images: %d %%" % (100*correct/total))
    conf.div_(torch,norm(conf,2))
    for i in range(len(classes)):
        print("accuracy of %5s : %2d %%" % (classes[i], 100*class_correct[i]/class_total[i]))

    # TODO does this show images?
    imshow(torchvision.utils.make_grid(images))
