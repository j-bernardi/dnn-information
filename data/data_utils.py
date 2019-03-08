"""Data utility functions."""
import pickle, torch, os, sys, random, math, h5py
import numpy as np
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#print("appending", os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2]))
#sys.path.append(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2]))
from models.unet import UNet3D, UNet2D
from models.unet.training_metadata import make_one_hot

## DATA LOADING ##

class ImdbData(data.Dataset):
    def __init__(self, X, y, w):
        self.X = X
        self.y = y
        self.w = w

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]
        weight = self.w[index]

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        weight = torch.from_numpy(weight)
        
        return img, label, weight, index

    def __len__(self):
        return len(self.y)

def get_imdb_data(fn, batch_size=8, val_split=0.8, num=0, shuffle=True, workers=4, NumClass=9):

    classes = ["class" + str(i) for i in range(9)]

    # Load DATA
    Data = h5py.File(fn + "Data.h5", 'r')
    a_group_key = list(Data.keys())[0]
    Data = list(Data[a_group_key])
    Data = np.squeeze(np.asarray(Data))
    
    # labels
    Label = h5py.File(fn + "label.h5", 'r')
    a_group_key = list(Label.keys())[0]
    Label = list(Label[a_group_key])
    Label = np.squeeze(np.asarray(Label))
    
    # indexes
    set = h5py.File(fn + "set.h5", 'r')
    a_group_key = list(set.keys())[0]
    set = list(set[a_group_key])
    set = np.squeeze(np.asarray(set))
    
    sz = Data.shape
    print("og sz", sz)

    # Add gray channel
    Data = Data.reshape([sz[0], 1, sz[1], sz[2]])
    
    print("Data shape", Data.shape)
    Data = Data[:, :, 61:573, :] # WAS: Data = Data[:, :, 61:573, :]?
    
    print("Label shape", Label.shape)
    weights = Label[:, 1, 61:573, :]
    Label = Label[:, 0, 61:573, :]
    
    sz = Label.shape
    Label = Label.reshape([sz[0], 1, sz[1], sz[2]])
    weights = weights.reshape([sz[0], 1, sz[1], sz[2]])
    train_id = set == 1
    test_id = set == 3
    
    dataset_size = Data.shape[0]
    indices = list(range(dataset_size))

    # reduce size if desired
    if num > 0:
        dataset_size = num
        indices = indices[:num]
    
    split = int(np.floor(val_split * dataset_size))
    
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)
    
    train_id, test_id = indices[split:], indices[:split]

    Tr_Dat = Data[train_id, :, :, :]
    Tr_Label = np.squeeze(Label[train_id, :, :, :]) - 1 # Index from [0-(NumClass-1)]
    #print(Tr_Label.shape)
    Tr_weights = weights[train_id, :, :, :]
    Tr_weights = np.tile(Tr_weights, [1, NumClass, 1, 1])

    Te_Dat = Data[test_id, :, :, :]
    # TODO - don't squeeze in first axis (when test size is 1)
    Te_Label = np.squeeze(Label[test_id, :, :, :]) - 1
    #print(Te_Label.shape)
    Te_weights = weights[test_id, :, :, :]
    Te_weights = np.tile(Te_weights, [1, NumClass, 1, 1])

    train_loader = torch.utils.data.DataLoader(ImdbData(Tr_Dat, Tr_Label, Tr_weights), batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = torch.utils.data.DataLoader(ImdbData(Te_Dat, Te_Label, Te_weights), batch_size=batch_size, shuffle=False, num_workers=workers)

    return (train_loader, val_loader), classes


class VoxelsDataset(Dataset):
    """
    Retinal scan dataset.
    Just stores the file location and creates objects for each image in the dataset...
    """

    def __init__(self, root_dir, transform=None):
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
        self.transform = transform

    def __len__(self):
        """Length of object."""

        return len([f for f in os.listdir(self.root_dir) if f.endswith(".pt") and f.startswith("image")])

    def __getitem__(self, index):
        """
        Returns a single dataframe representing an image file
        with a given index.
        """
        img_name = self.root_dir + "image" + str(index) + ".pt"
        class_name = self.root_dir + "class" + str(index) + ".pt"

        img_tensor = [torch.load(img_name)]
        class_tensor = [torch.load(class_name)]

        img_tensor = np.concatenate([it[np.newaxis] for it in img_tensor])
        class_tensor = np.concatenate([it[np.newaxis] for it in class_tensor])

        sample = {'image': img_tensor, 'classes': class_tensor}

        #print("Got item", index)
        #print(img_tensor.shape)
        #print(class_tensor.shape)
        
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_voxels_data(location, batch_size, workers, val_split, shuffle=True, num=0):
    """
    Loads the data from location specified.
    """

    scan_dataset = VoxelsDataset(location)

    transform = transforms.Compose([
        transforms.ToTensor(),
        scan_dataset.transform
    ])

    # Creating data indices for training and validation splits:
    if num > 0:
        dataset_size = num
    else:
        dataset_size = len(scan_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    trainloader = torch.utils.data.DataLoader(scan_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=workers)
    testloader = torch.utils.data.DataLoader(scan_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=workers)

    classes = ('s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
                   's10', 's11', 's12', 's13', 's14')

    return trainloader, testloader, classes

## RANDOM GENERATION ##

def generate_classification(this_input_dir="sample_scans/", x=int(448/4), y=int(512/4), z=9, in_classes=15, to_generate=10):

    save_root_dir = "data/segmented_tensors/"

    save_loc = save_root_dir + this_input_dir
    # 1. load in the input tensors from location
    print("Found", len([f for f in os.listdir(save_loc) if f.endswith(".pt") and f.startswith("map")]), "files to segment.")

    with torch.no_grad():
        for f in os.listdir(save_loc):
            
            if not f.endswith(".pt") or not f.startswith("map"):
                continue

            this_input_nm = save_loc + f

            # (14) with a 1 somewhere in the first 4 elements, and a 1 somewhere in the next 10     
            ref = torch.zeros(4)
            ref[random.randint(0, len(ref)-1)] = 1
            dis = torch.zeros(10)
            dis[random.randint(0, len(dis)-1)] = 1
            
            classification = torch.cat((ref, dis))
            
            nm = this_input_nm.split(os.sep)[-1].replace("map", "class")

            # save
            torch.save(classification, nm)
            #print(classification)
        print("Done.")

def generate_csv(this_input_dir="sample_scans/", x=int(448), y=int(512), z=9, classes=15, to_generate=5):
    """Old csv generator"""
    save_loc = "data/csvs/slice_sample_scans/"

    for i in range(to_generate):
        with open(save_loc + str(i) + ".csv", 'w+') as ss:
            pass

        lines = []

        for row in range(x):
            if row % int(math.ceil(x/25)) == 0 :
                print("row", row, "/", x)
            for col in range(y):
                for plane in range(z):
                    r = random.uniform(0, 1)
                    c = random.randint(0, classes-1)
                    lines.append("%d, %d, %d, %.3f, %d\n" % (row, col, plane, r, c))

        with open(save_loc + str(i) + ".csv", 'w+') as ss:
            so_far = 0

            #ss.write("plane,x,y,scale,class\n")
            ss.write("x,y,z,scale,class\n")

            for line in lines:
                ss.write(line)
                if so_far < 10:
                    print(line)
                    so_far += 1

def generate_input_tensor(this_dir="sample_scans/", x=int(448/4), y=int(512/4), z=16, classes=15, to_generate=10):

    root_dir = "data/input_tensors/"

    save_loc = root_dir + this_dir

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    for n in range(to_generate):

        image_vector = torch.rand(z, x, y)
        class_vector = torch.randint(low=0, high=(classes-1), size=(z, x, y))

        torch.save(image_vector, save_loc + "image" + str(n) + ".pt")
        torch.save(class_vector, save_loc + "class" + str(n) + ".pt")

    with open(save_loc + "details.txt", "w+") as f:
        f.write("z, x, y:\n" + str(z) + "," + str(x) + "," + str(y) + "\nclasses=" + str(classes))

def segment_tensors(this_input_dir="sample_scans/", classes=15, to_generate=10, voxel_size=9):


    ## METAVARIABLES ##
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_loc = "models/unet/saved_models/unet.pth"

    input_root_dir = "data/input_tensors/"
    save_root_dir = "data/segmented_tensors/"

    # Make the locations
    input_loc = input_root_dir + this_input_dir
    save_loc = save_root_dir + this_input_dir

    # make save location
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    unet = make_model(model_loc, device)

    # 1. load in the input tensors from location
    print("Found", len([f for f in os.listdir(input_root_dir + this_input_dir) if f.endswith(".pt") and f.startswith("image")]), "files to segment.")

    with torch.no_grad():
        for f in os.listdir(input_root_dir + this_input_dir):
            if not f.endswith(".pt") or not f.startswith("image"):
                continue
    
            # load in the input image - all cells
            this_input = torch.load(input_root_dir + this_input_dir + f)
            this_input = this_input.reshape(1, 1, this_input.size(0), this_input.size(1), this_input.size(2)).to(device)

            ## NEW INPUT ##

            # for each slice, find the 4 either side and classify these
            first = True
            for z in range(this_input.size(2)):
                #print("z", z, end=" - ")
                # find the z-4 and z+4 slices for a 9 voxel size
                strt, lst = z - voxel_size // 2, z + voxel_size // 2
                
                # Skip the outer voxels
                if strt < 0 or lst >= this_input.size(2):
                    continue

                indices = torch.from_numpy(np.array(range(strt, lst+1))).long().to(device)

                # TODO - make this output just 1 layer
                this_voxels = torch.index_select(this_input, 2, indices)
                #print("voxels shape")
                
                this_class = unet(this_voxels)
                # tale only the middle
                this_class = torch.index_select(this_class, 2, torch.tensor([this_class.size(2) // 2]).to(device))
                #print("outputs shape", this_class.shape)

                if first:
                    #print("Creating outputs", this_class.shape)
                    outputs = this_class
                    first=False
                else:
                    #print("appending outputs + this_class", outputs.shape, "+", this_class.shape)
                    #print("should be bx15x1xXxY + bx15x(z-4)xXxY")
                    outputs = torch.cat((outputs, this_class), dim=2)

            ## TO HERE ##

            #this_output = unet(this_input)

            # get the indexes of all the maxima
            _, max_matrix = torch.max(outputs.data, 1)
            
            # make class encoding one-hot
            one_hot = make_one_hot(max_matrix.unsqueeze(0))

            # save the prediction
            nm = f.split(os.sep)[-1]
            # TEMP: Added [0] here
            torch.save(one_hot[0], (save_loc + nm).replace("image", "map"))

    # Create a file with some details
    print("Saving output", one_hot[0].shape)
    print("To", save_loc)
    with open(save_loc + "details.txt", "w+") as f:
        f.write(str(len([f for f in os.listdir(input_root_dir + this_input_dir) if f.endswith(".pt") and f.startswith("image")])) + " files" 
                + "\nclasses=" + str(classes))

def make_model(model_loc, device, dim=2):
    if "2" in model_loc:
        unet = UNet2D()
    elif "3" in model_loc:
        unet = UNet3D()
    else:
        raise Exception(
            "Expected '2' or '3' in model location to indicate dimension.\nModel location: %s" % (model))
    
    unet.load_state_dict(torch.load(model_loc))
    
    # unet.eval()
    print("Moving model to", device)
    unet.to(device)

    return unet
