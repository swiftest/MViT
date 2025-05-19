import numpy as np


def choose_all_pixels(groundtruth):
    pos_forward = np.argwhere(groundtruth != 0)
    pos_backward = np.argwhere(groundtruth == 0)
    return pos_forward, pos_backward


def choose_pretrain(groundtruth, pretrain_proportion=0.2, seed=42):
    rs = np.random.RandomState(seed)
    total_pos_pretrain = np.argwhere(groundtruth != 0)  # Returns an array of positional indices for label 1, 2, ...

    total_samples = total_pos_pretrain.shape[0]
    ids = np.arange(total_samples)
    rs.shuffle(ids)
    ids = ids[:int(total_samples * pretrain_proportion)]
    
    total_pos_pretrain = total_pos_pretrain[ids]
    y_pretrain = np.array([groundtruth[x, y] for x, y in zip(total_pos_pretrain[:, 0], total_pos_pretrain[:, 1])])
    y_pretrain = y_pretrain - 1
    return total_pos_pretrain, y_pretrain


def choose_train_and_test(groundtruth, num_train_per_class=10, seed=42):  # divide dataset into train, test and validation datasets
    rs = np.random.RandomState(seed)   # If you want to use the fixed training samples, keep lines 26 and 39 and comment out line 40.
    
    num_classes = np.max(groundtruth)
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_valid = []
    pos_valid = {}
    
    for i in range(num_classes):
        each_class = np.argwhere(groundtruth == i+1)  # Returns an array of positional indices for label i.
        total_samples = each_class.shape[0]
        rs.shuffle(each_class)  # If you want to use the fixed training samples, keep lines 26 and 39 and comment out line 40.
        #np.random.shuffle(each_class)  # If you want to use the random training samples, keep line 40 and comment out lines 26 and 39.
        
        pos_train[i] = each_class[:num_train_per_class]
        number_train.append(num_train_per_class)  # The number of training samples for each class.
        
        pos_test[i] = each_class[num_train_per_class:]
        number_test.append(pos_test[i].shape[0])  # The number of testing samples for each class.
        
        pos_valid[i] = each_class[-100:]
        number_valid.append(100)
    
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
    total_pos_train = total_pos_train.astype(int)
    
    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_test = total_pos_test.astype(int)
    
    total_pos_valid = pos_valid[0]
    for i in range(1, num_classes):
        total_pos_valid = np.r_[total_pos_valid, pos_valid[i]]
    total_pos_valid = total_pos_valid.astype(int)
    return total_pos_train, total_pos_test, total_pos_valid, number_train, number_test, number_valid


# Boundary Extension: Mirroring
def mirror_hsi(height, width, band, data, patch_size=15):
    padding = patch_size // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    # Central Zone
    mirror_hsi[padding:(padding+height), padding:(padding+width), :] = data
    # Left Mirror
    for i in range(padding):
        mirror_hsi[padding:(padding+height), i, :] = data[:, padding-i-1, :]
    # Right Mirror
    for i in range(padding):
        mirror_hsi[padding:(height+padding), width+padding+i, :] = data[:, width-1-i, :]
    # Upper Mirror
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding*2-i-1, :, :]
    # Bottom Mirror
    for i in range(padding):
        mirror_hsi[height+padding+i, :, :] = mirror_hsi[height+padding-1-i, :, :]
    print("*******************************************************")
    print("patch_size : {}".format(patch_size))
    print("mirror_data shape : [{0}, {1}, {2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("*******************************************************")
    return mirror_hsi


# Obtain image data for patch
def gain_neighborhood_pixel(mirror_data, pos, i, patch_size):
    x = pos[i, 0]
    y = pos[i, 1]
    temp_image = mirror_data[x:(x+patch_size), y:(y+patch_size), :]
    return temp_image


def all_data(mirror_data, band, pos_forward, pos_backward, patch_size=15):
    x_forward = np.zeros((pos_forward.shape[0], patch_size, patch_size, band), dtype=float)  # (695, 9, 9, 176)
    x_backward = np.zeros((pos_backward.shape[0], patch_size, patch_size, band), dtype=float)
    for i in range(pos_forward.shape[0]):
        x_forward[i] = gain_neighborhood_pixel(mirror_data, pos_forward, i, patch_size)
    for j in range(pos_backward.shape[0]):
        x_backward[j] = gain_neighborhood_pixel(mirror_data, pos_backward, j, patch_size)
    print("*******************************************************")
    print("x_forward shape = {}, type = {}".format(x_forward.shape, x_forward.dtype))
    print("x_backward  shape = {}, type = {}".format(x_backward.shape, x_forward.dtype))
    print("*******************************************************")
    y_forward = np.zeros(pos_forward.shape[0])
    y_backward = np.zeros(pos_backward.shape[0])
    return x_forward, x_backward, y_forward, y_backward


def pretrain_data(mirror_data, band, pretrain_pos, patch_size=15):
    x_pretrain = np.zeros((pretrain_pos.shape[0], patch_size, patch_size, band), dtype=float)  
    for i in range(pretrain_pos.shape[0]):
        x_pretrain[i] = gain_neighborhood_pixel(mirror_data, pretrain_pos, i, patch_size)
    print("x_pretrain shape : {}, type = {}".format(x_pretrain.shape, x_pretrain.dtype))
    print("*******************************************************")
    return x_pretrain


# Summarize training and testing data
def train_and_test_data(mirror_data, band, train_pos, test_pos, valid_pos, patch_size=15):
    x_train = np.zeros((train_pos.shape[0], patch_size, patch_size, band), dtype=float)  # (695, 9, 9, 176)
    x_test = np.zeros((test_pos.shape[0], patch_size, patch_size, band), dtype=float)
    x_valid = np.zeros((valid_pos.shape[0], patch_size, patch_size, band), dtype=float)
    for i in range(train_pos.shape[0]):
        x_train[i] = gain_neighborhood_pixel(mirror_data, train_pos, i, patch_size)
    for j in range(test_pos.shape[0]):
        x_test[j] = gain_neighborhood_pixel(mirror_data, test_pos, j, patch_size)
    for k in range(valid_pos.shape[0]):
        x_valid[k] = gain_neighborhood_pixel(mirror_data, valid_pos, k, patch_size)
    print("*******************************************************")
    print("x_train shape = {}, type = {}".format(x_train.shape, x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape, x_test.dtype))
    print("x_valid  shape = {}, type = {}".format(x_valid.shape, x_valid.dtype))
    print("*******************************************************")
    return x_train, x_test, x_valid


def train_and_test_label(number_train, number_test, number_valid, num_classes):
    y_train = []
    y_test = []
    y_valid = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
        for n in range(number_valid[i]):
            y_valid.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_valid = np.array(y_valid)
    print("y_train: shape = {}, type = {}".format(y_train.shape, y_train.dtype))
    print("y_test: shape = {}, type = {}".format(y_test.shape, y_test.dtype))
    print("y_valid: shape = {}, type = {}".format(y_valid.shape, y_valid.dtype))
    print("*******************************************************")
    return y_train, y_test, y_valid