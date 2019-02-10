import pickle, torch, os, sys
import numpy as np
#print("Path to file", os.path.realpath(__file__))
print("appending", os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2]))
sys.path.append(os.sep.join(os.path.realpath(__file__).split(os.sep)[:-2]))
from models.unet import UNet3D

classes = 15
to_generate = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_loc = "models/unet/saved_models/unet.pth"

input_root_dir = "data/input_tensors/"
save_root_dir = "data/segmented_tensors/"
this_input_dir = "dummy_half_slice_sample_scans/"

input_loc = input_root_dir + this_input_dir
save_loc = save_root_dir + this_input_dir

# make save location
if not os.path.exists(save_loc):
    os.makedirs(save_loc)

# 0. load in the unet model
unet = UNet3D()
unet.load_state_dict(torch.load(model_loc))
#unet.eval()
print("Moving model to", device)
unet.to(device)

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
        _, this_output = torch.max(this_output.data, 1)

        # one hot these indices
        one_hot = torch.zeros_like(this_output).scatter(1, this_output, 1)
        print(one_hot.shape)
        #print(this_output)

        # save
        nm = f.split(os.sep)[-1]
        torch.save(one_hot, save_loc + nm)

with open(save_loc + "details.txt", "w+") as f:
    f.write(str(len([f for f in os.listdir(input_root_dir + this_input_dir) if f.endswith(".pt") and f.startswith("image")])) + " files" 
            + "\nclasses=" + str(classes))
