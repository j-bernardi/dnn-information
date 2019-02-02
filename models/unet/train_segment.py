import torch, torchvision
import torch.optim as optim
from unet_model import UNet3D

# for 160,000 iterations
epochs = 160000
# The initial learning rate was 0.0001
learn_rate_0 = 0.0001
# batch sizes - 8
batch_size = 8
# workers - on 8 graphics processing units (GPUs)
workers = 2 # Cuda count gpus? not sure


class VoxelsDataset(Dataset):
    """Retinal scan dataset. Takes in a csv file as a scan"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string) path to csv file with annot
            root_dir (string) dir with images
            transform (callable, optional): apply trans to sample
        """
        self.scan = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.scan)

    def __getitem__(self, index):

        pixels = self.scan.iloc[index, 1:].as_matrix()
        pixels = scan.astype('float').reshape(-1, 2) # check dimensions
        sample ={'image':image}

        if self.transform:
            sample = self.transform(sample)
        return sample


# TODO - transforms
def load_data(location, loader_object, batch_size, workers):
    """
    Loads the data from location specified.
    NOTES:
        # we augmented the data by applying
        #   affine and
        #   elastic transformations
        # jointly over the inputs and ground-truth segmentations
        # Intensity transformations over the inputs were also applied.
    """
    # TODO - finish
    transform = transforms.Compose(
        [transforms.ToTensor(),
        ])

    trainset = loader_object(root=location, train=True, download=True,
                             transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=workers)

    testset = loader_object(root=location, train=False, download=True,
                             transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=workers)

    classes = ('s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
                   's10', 's11', 's12', 's13', 's14', 's15')

    return trainloader, testloader, classes

if __name__ == "__main__":

    # TODO
    # with dataset 1 in Supplementary Table 3
    location = "images"
    csv_file = location + '/scans.csv'

    print("Loading images")
    scan = pd.read_csv(csv_file)

    scan_dataset = VoxelsDataset(csv_file=csv_file, root_dir=location)

    trainloader, testloader, classes = load_data(location, scan_dataset,
                                                 batch_size, workers)

    # The input is therefore a 448 × 512 × 9 voxels image
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    unet = UNet3D()

    # We used per-voxel cross entropy as the loss function, with 0.1 label-smoothing regularization
    # TODO: What does it mean to be per-voxel?
    # TODO: add label smoothing
    # We have neither used dropout nor weight decay
    """
    ADD LABEL SMOOTHING:
        For label_smoothing, you cat look at the implementation of NJUNMT-pytorch
        In the class NMTCritierion
        https://github.com/whr94621/NJUNMT-pytorch/blob/aff968c0da9273dc42eabbb8ac4e459f9195f6e4/src/modules/criterions.py#L131
    """
    criterion = nn.CrossEntropyLoss()

    # Learning rate and time to change
    idxs = epochs * np.array([0, 0.1, 0.2, 0.5, 0.7, 0.9, 0.95])
    lrs = learn_rate_0 * n.array([1, 0.5, 0.25, 0.125, 0.015625, 0.00390625, 0.001953125])
    next_idx = 1

    # Adam optimizer - https://arxiv.org/abs/1412.6980
    optimizer = optim.Adam(net.parameters(), lr=learn_rate_0)

    for epoch in range(epochs):

        running_loss = 0

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

            inputs, labels = data
            optimizer.zero_grad()

            outputs = unet(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Track loss per epoch
        print('[Epoch %d complete] loss: %.3f' %
              (epoch +1,running_loss / len(trainloader)))
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
