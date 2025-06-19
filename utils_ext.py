import os
import random
import time

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from thop import profile
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from IPython import display
import torch

import import_ipynb
from utils import get_dataset, get_network, get_default_convnet_setting
from networks import ConvNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = torch.stack([i[0] for i in data]).to(device)
        self.targets = torch.tensor([i[1] for i in data], device=device)
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def get_dataset_ext(dataset_name: str, batch_size: int=256, data_path: str = './data'):
    if dataset_name.upper()=="MNIST":
        _, _, _, _, _, _, train_dataset, test_dataset, test_loader = get_dataset("MNIST", data_path)
        num_classes = 10
        train_dataset = MyDataset(train_dataset)
        test_dataset = MyDataset(test_dataset)
        
    elif dataset_name.upper().startswith("MHIST"):
        df = pd.read_csv(os.path.join(data_path,'mhist_dataset','annotations.csv'))
        df['Label'] = LabelEncoder().fit_transform(df['Majority Vote Label'])
        num_classes = df['Label'].nunique()
        img_folder = os.path.join(data_path,'mhist_dataset','images')
        train_imgs, train_labels, test_imgs, test_labels = [], [], [], []
        
        if dataset_name=="MHIST_FULL":
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224 // 2, 224 // 2)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])

        for _, row in df.iterrows():
            img_path = os.path.join(img_folder, row['Image Name'])
            try:
                # Load the image
                img = Image.open(img_path)
                img = transform(img)
            except FileNotFoundError:
                print(f"Warning: Image {img_path} not found.")
                continue
            
            if row['Partition'] == 'train':        
                train_labels.append(row['Label'])
                train_imgs.append(img)
            elif row['Partition'] == 'test':        
                test_labels.append(row['Label'])
                test_imgs.append(img)
                
        train_labels = torch.tensor(train_labels, dtype=torch.int64)
        test_labels = torch.tensor(test_labels, dtype=torch.int64)
        train_imgs = torch.stack(train_imgs)
        test_imgs = torch.stack(test_imgs)
        train_dataset = TensorDataset(train_imgs, train_labels)
        test_dataset = TensorDataset(test_imgs, test_labels)

    else:
        raise ValueError(f"Invalid dataset name '{dataset_name}'. Please choose 'MNIST' or 'MHIST'.")
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_no_shuf = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, train_loader_no_shuf, test_loader, train_dataset, num_classes

def get_network_ext(model, num_classes, dataset=None, dataloader=None):
    if dataset is not None:
        im_shape = dataset[0][0].shape
    elif dataloader is not None:
        imgs, _ = next(iter(dataloader))
        im_shape = imgs.shape[1:]
        
    if model.startswith("ConvNet-"):
        net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()
        net_depth = int(model.split("-")[-1])
        net = ConvNet(channel=im_shape[0], num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_shape[1:])
        return net.to(device)
    else:
        return get_network(model, im_shape[0], num_classes, im_size=im_shape[1:]).to(device)

def calculate_flops(net, test_loader):
    total_flop = 0
    total_time = 0
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        torch.cuda.synchronize()  
        start_time = time.time() 
        
        with torch.no_grad():
            _ = net(imgs)
        
        torch.cuda.synchronize() 
        end_time = time.time()
        
        macs, _ = profile(net, (imgs,))
        total_flop += macs * 2 
        
        total_time += (end_time - start_time)
    
    gflops_per_second = total_flop / total_time / 1e9
    display.clear_output(wait=False)
    print(f"Total FLOP: {total_flop/1e9:.2f} GFLOP")
    print(f"Time taken: {total_time:.4f} seconds")
    print(f"FLOPS: {gflops_per_second:.2f} GFLOP/s")
    
def get_syn_dataloader(imgs: torch.Tensor, num_classes=10, ipc=10, batch_size=256, shuffle=True):
    labels = torch.arange(num_classes).repeat_interleave(ipc).to(device)
    data_loader = DataLoader(TensorDataset(imgs, labels), batch_size=batch_size, shuffle=shuffle)
    return data_loader

def plot_images(folder_path, dataset='mnist'):
    # Set parameters based on the dataset type
    if dataset == 'mnist':
        num_classes = 10
        plot_per_classes = 2
        grid_rows, grid_cols = 10, 10
        ipc = 2
        filenames = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    elif dataset == 'mhist':
        num_classes = 2
        plot_per_classes = 5
        grid_rows, grid_cols = 10, 10
        ipc = 5
        filenames = [f for f in os.listdir(folder_path) if not f.startswith('0') and int(f.split('_')[0]) % 1000 == 0]
        sorted_filenames = sorted(filenames, key=lambda x: int(x.split('_')[0]))
        filenames = ['0.pt'] + sorted_filenames[:8] + sorted_filenames[-1:]
    
    # Create plot grid
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(10, 10), gridspec_kw={'wspace': 0, 'hspace': 0})

    for i, filename in enumerate(filenames):
        file_path = os.path.join(folder_path, filename)
        tensor = torch.load(file_path)
        t = filename.split('_')[0].split('.')[0]
        
        for c in range(num_classes):
            for n in range(plot_per_classes):
                img = tensor[c * num_classes + n].cpu().detach().numpy().squeeze()
                if dataset == 'mnist':
                    ax = axes[c, i + 5 * n]
                else:  # MHIST
                    ax = axes[n + c * 5, i]
                
                ax.imshow(img, cmap='gray')
                ax.axis('off')

                # Add label on the first column
                if i == 0:
                    ax.text(0, 0.5, f'C{c}-{n + 1}', ha='left', va='center', transform=ax.transAxes, 
                            fontsize=8, color='black', zorder=10,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    
    plt.tight_layout()
    plt.show()                
                
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
    
    def __len__(self):
        return len(self.buffer)
    
    def add(self, data, labels):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((data, labels))
    
    def sample(self, num_samples):
        num_samples = min(num_samples, len(self.buffer))
        
        if num_samples <= 0:
            return torch.Tensor([]).to(device), torch.Tensor([]).to(torch.int).to(device)
        
        batch = random.sample(self.buffer, num_samples)
        data_batch, label_batch = zip(*batch)
        return torch.stack(data_batch), torch.stack(label_batch)