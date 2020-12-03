import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from copy import deepcopy


def resize_image(img, factor):
    '''
    
    :param img: 
    :param factor: 
    :return: 
    '''
    img2 = np.zeros(np.array(img.shape) * factor)

    for a in range(0, img.shape[0]):
        for b in range(0, img.shape[1]):
            img2[a * factor:(a + 1) * factor, b * factor:(b + 1) * factor] = img[a, b]
    return img2


import matplotlib.pyplot as plt

cur = 1


# Function to plot images;



def plot(img, title, g=True):
    global cur, fig
    p = fig.add_subplot(10, 10, cur)
    p.set_title(title)
    cur += 1
    if g:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)


def visualizeTensor(t, path):
    global cur, fig
    cur = 1
    # Function to plot images;
    fig = plt.figure(figsize=(10, 10))
    for a in t:
        img = a.cpu().numpy()
        img = np.swapaxes(img, 0, 2)
        imgMin = np.min(img)

        # img = img-np.min(img)
        # img = img/np.max(img)
        plot(img, str(cur), False)
    plt.savefig(path)
    plt.gcf().clear()
    plt.close()

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    print("finally change to best model")
    return

def get_bias_layer(bias_correction_layer):
    return deepcopy(bias_correction_layer)

def set_bias_layer_(bias_correction_layer, best_bias_correction):
    bias_correction_layer = best_bias_correction
    print("finally change to best bias layer")
    return
