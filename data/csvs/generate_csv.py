import random, math
x = 448#28#*16#= 448
y = 512#32#*16#= 512
z = 9#8 #*16#= 128
classes = 15

to_generate = 5
save_loc = "data/slice_sample_scans_"

if __name__ == "__main__":
    for i in range(to_generate):
        with open(save_loc + str(i) + ".csv", 'w+') as ss:
            pass

        lines = []
        """
        for plane in range(z):
            if plane % int(math.ceil(z/25)) == 0 :
                print("plane", plane, "/", z)
            for row in range(x):
                for col in range(y):
                    r = random.uniform(0, 1)
                    c = random.randint(1, classes)
                    lines.append("%d, %d, %d, %.3f, %d\n" % (plane, row, col, r, c))
        """

        for row in range(x):
            if row % int(math.ceil(x/25)) == 0 :
                print("row", row, "/", x)
            for col in range(y):
                for plane in range(z):
                    r = random.uniform(0, 1)
                    c = random.randint(1, classes)
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
