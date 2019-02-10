import pickle, torch, os

x = int(448/4)#28#*16#= 448
y = int(512/4)#32#*16#= 512
z = 9#8 #*16#= 128
classes = 15
to_generate = 10

root_dir = "data/input_tensors/"
this_dir = "dummy_half_slice_sample_scans/"

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
