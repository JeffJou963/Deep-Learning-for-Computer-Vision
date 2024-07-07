from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms

from pb_dataloader import pb3
from pb3_model import DANNResNeSt

# load dataset
fig_path = Path('./')
fig_path.mkdir(parents=True, exist_ok=True)
batch_size = 256


source_val_set = pb3(f"./hw2_data/digits/mnistm", 'val.csv', 'mnsist')
svhn_val_set = pb3(f"./hw2_data/digits/svhn", 'val.csv', 'svhn')
usps_val_set = pb3(f"./hw2_data/digits/usps", 'val.csv', 'usps')

source_val_loader = DataLoader(source_val_set, batch_size, shuffle=False, num_workers=6)
SVHN_val_loader = DataLoader(svhn_val_set, batch_size, shuffle=False, num_workers=6)
USPS_val_loader = DataLoader(usps_val_set, batch_size, shuffle=False, num_workers=6)

# SVHN
device = 'cuda'
net = DANNResNeSt().to(device)
net.load_state_dict(torch.load('./pb3_output/svhn_ep88_acc0.4344.pth', map_location=device))
net.eval()
all_feature = []
all_label = []
all_domains = []
for x, y in source_val_loader:
    x = x.to(device)
    with torch.no_grad():
        f = net.get_features(x)
    all_feature.append(f.cpu().numpy())
    all_label.append(y.cpu().numpy())
    all_domains.append(np.zeros((len(x),), dtype=np.int32))

for x, y in SVHN_val_loader:
    x = x.to(device)
    with torch.no_grad():
        f = net.get_features(x)
    all_feature.append(f.cpu().numpy())
    all_label.append(y.cpu().numpy())
    all_domains.append(np.ones((len(x),), dtype=np.int32))

all_feature = np.concatenate(all_feature, axis=0)
all_label = np.concatenate(all_label, axis=0)
all_domains = np.concatenate(all_domains, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
all_feature = TSNE(
    2, init='pca', learning_rate='auto').fit_transform(all_feature)
scatter = axes[0].scatter(
    all_feature[..., 0], all_feature[..., 1],
    c=all_label, alpha=0.5, s=10
)
axes[0].legend(*scatter.legend_elements(), title='Digits')
axes[0].set_title("Colored by Different Classes")

scatter = axes[1].scatter(
    all_feature[..., 0], all_feature[..., 1],
    c=all_domains, alpha=0.5, s=10
)
axes[1].legend(handles=scatter.legend_elements()[0], labels=[
    'Source', 'Target'], title='Domains')
axes[1].set_title("Colored by Different Domains")
fig.savefig(fig_path / 'tsne_svhn')


# USPS
net = DANNResNeSt().to(device)
net.load_state_dict(torch.load('./pb3_output/usps_ep93_acc0.8575.pth', map_location=device))

all_feature = []
all_label = []
all_domains = []
for x, y in source_val_loader:
    x = x.to(device)
    with torch.no_grad():
        f = net.get_features(x)
    all_feature.append(f.cpu().numpy())
    all_label.append(y.cpu().numpy())
    all_domains.append(np.zeros((len(x),), dtype=np.int32))

for x, y in USPS_val_loader:
    x = x.to(device)
    with torch.no_grad():
        f = net.get_features(x)
    all_feature.append(f.cpu().numpy())
    all_label.append(y.cpu().numpy())
    all_domains.append(np.ones((len(x),), dtype=np.int32))

all_feature = np.concatenate(all_feature, axis=0)
all_label = np.concatenate(all_label, axis=0)
all_domains = np.concatenate(all_domains, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
all_feature = TSNE(
    2, init='pca', learning_rate='auto').fit_transform(all_feature)
scatter = axes[0].scatter(
    all_feature[..., 0], all_feature[..., 1],
    c=all_label, alpha=0.5, s=10
)
axes[0].legend(*scatter.legend_elements(), title='Digits')
axes[0].set_title("Colored by Digit Classes(0-9)")

scatter = axes[1].scatter(
    all_feature[..., 0], all_feature[..., 1],
    c=all_domains, alpha=0.5, s=10
)
axes[1].legend(handles=scatter.legend_elements()[0], labels=[
    'Source', 'Target'], title='Domains')
axes[1].set_title("Colored by Domains")
fig.savefig(fig_path / 'tsne_usps')
