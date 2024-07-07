
import os
import csv
import json
import random

import numpy as np
import matplotlib.pyplot as plt
import torch

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE


##### For Training #####
def set_seed(seed):
    ''' set random seeds '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

def save_model(model, optimizer, scheduler, path):
    # print(f'Saving model to {path}...')
    state = {'state_dict':model.state_dict()}
            # 'optimizer':optimizer.state_dict(),
            # 'scheduler':scheduler.state_dict()}
    torch.save(state, path)
    # print('End of saving model!!!')

def load_parameters(model, path, device):
    # print(f'Loading model parameters from {path}...')
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['state_dict'])
    # print("End of loading!!!")

def visualize_pca(sec_last_layers, labels, num_components=2):
    # Standardize the data
    mean = np.mean(sec_last_layers, axis=0)
    std_dev = np.std(sec_last_layers, axis=0)
    sec_last_layers_standardized = (sec_last_layers - mean) / std_dev

    # Perform PCA
    pca = PCA(n_components=num_components)
    pca_result = pca.fit_transform(sec_last_layers_standardized)

    # Create a scatter plot
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Visualization of Second Last Layer Activations')
    plt.colorbar(label='Labels')
    # plt.savefig('./output/vis_PCA_epoch20')
    plt.savefig('./output/vis_PCA_best')
    plt.show()

def visualize_tsne(sec_last_layers, labels, num_components=2):
    # Perform t-SNE
    tsne = TSNE(n_components=num_components)
    tsne_result = tsne.fit_transform(sec_last_layers)

    # Create a scatter plot
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Second Last Layer Activations')
    plt.colorbar(label='Labels')
    # plt.savefig('./output/vis_tsne_epoch20')
    plt.savefig('./output/vis_tsne_best')
    plt.show()