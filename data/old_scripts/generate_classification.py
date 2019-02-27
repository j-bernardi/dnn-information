import pickle, torch, os, random

x = int(448/4)#28#*16#= 448
y = int(512/4)#32#*16#= 512
z = 9#8 #*16#= 128

in_classes = 15
to_generate = 10

save_root_dir = "data/segmented_tensors/"
this_input_dir = "sample_scans/"

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