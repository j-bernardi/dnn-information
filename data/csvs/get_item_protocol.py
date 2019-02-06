img_name = os.path.join(self.root_dir, self.index_prefix) + str(index) + ".csv"
df = pd.read_csv(img_name)

# Coords
coords = df.iloc[:,0:3]
# Greyscale at each coord
scales = df.iloc[:,3]
# class labels per coordinate
classes = df.iloc[:,-1]

# TODO - just iterate once?
scale_matrix = self.to_matrix(coords, scales)
class_matrix = self.to_matrix(coords, classes)

sample = {'image': scale_matrix, 'classes': class_matrix}

if self.transform:
    sample = self.transform(sample)
print("Got item", index)
return sample


def to_matrix(self, coord_df, values):

    """UPDATE if structure changes"""
    x_dim = max(coord_df.iloc[:,0]) + 1
    y_dim = max(coord_df.iloc[:,1]) + 1
    z_dim = max(coord_df.iloc[:,2]) + 1

    # 1 colour channel, 3 coord channels
    output = [np.zeros((x_dim, y_dim, z_dim))]

    for i in range(len(coord_df)):
        row = coord_df.iloc[i,:]
        output[0][row[0], row[1], row[2]] = values.iloc[i]
    return np.concatenate([out[np.newaxis] for out in output])
