import pickle, torch

x = int(448/4)#28#*16#= 448
y = int(512/4)#32#*16#= 512
z = 9#8 #*16#= 128
classes = 15
to_generate = 5

save_loc = "data/tensors/small_slice_sample_scans_"

for n in range(to_generate):

    image = torch.rand(z, x, y)
    classes = torch.randint(low=1, high=15, size=(z, x, y))

    torch.save(image, save_loc + "image" + str(n) + ".pt")
    torch.save(image, save_loc + "class" + str(n) + ".pt")
