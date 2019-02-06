import pickle, torch

x = int(448/2)#28#*16#= 448
y = int(512/2)#32#*16#= 512
z = 9#8 #*16#= 128
classes = 15
to_generate = 5

save_loc = "data/tensors/dummy_slice_sample_scans_"

for n in range(to_generate):

    image_vector = torch.rand(z, x, y)
    class_vector = torch.randint(low=1, high=classes, size=(z, x, y))

    torch.save(image_vector, save_loc + "image" + str(n) + ".pt")
    torch.save(class_vector, save_loc + "class" + str(n) + ".pt")

with open(save_loc + "details.txt", "w+") as f:
	f.write("z, x, y:\n" + str(z) + "," + str(x) + "," + str(y) + "\nclasses=" + str(classes))
