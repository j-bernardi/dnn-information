## IMPORT ##
import pickle, torch, os, sys
import numpy as np
print("appending", os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2]))
sys.path.append(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2]))
from models.unet import UNet3D


## METAVARIABLES ##
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes = 15
to_generate = 10

model_loc = "models/unet/saved_models/unet.pth"

input_root_dir = "data/input_tensors/"
save_root_dir = "data/segmented_tensors/"
this_input_dir = "dummy_half_slice_sample_scans/"

def make_one_hot(labels, C=15):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x D x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x D x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3), labels.size(4)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    
    #target = Variable(target)
        
    return target

def make_model(model_loc, device):
    unet = UNet3D()
    unet.load_state_dict(torch.load(model_loc))
    # unet.eval()
    print("Moving model to", device)
    unet.to(device)

    return unet

if __name__ == "__main__":
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
    
            # load in
            this_input = torch.load(input_root_dir + this_input_dir + f)
            this_input = this_input.reshape(1, 1, this_input.size(0), this_input.size(1), this_input.size(2))
            this_input = this_input.to(device)

            # evaluate TODO - make sure these are 1-hot encoded
            this_output = unet(this_input)

            # get the indexes of all the maxima
            _, max_matrix = torch.max(this_output.data, 1)
            
            # make class encoding one-hot
            one_hot = make_one_hot(max_matrix.unsqueeze(0))

            # save the prediction
            nm = f.split(os.sep)[-1]
            # TEMP: Added [0] here
            torch.save(one_hot[0], (save_loc + nm).replace("image", "map"))

    # Create a file with some details
    with open(save_loc + "details.txt", "w+") as f:
        f.write(str(len([f for f in os.listdir(input_root_dir + this_input_dir) if f.endswith(".pt") and f.startswith("image")])) + " files" 
                + "\nclasses=" + str(classes))