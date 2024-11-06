import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython import display
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils_ext import get_network_ext, get_syn_dataloader, ReplayBuffer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_classification(net, train_loader, test_loader, num_epochs:int=20, lr:float=0.01, plot=True, early_stop: bool=False):
    def get_accuracy(dataloader):
        net.eval()
        correct = 0
        total = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = net(imgs)
            _, pred = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        return correct / total
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_accs=[]
    test_accs=[]
    
    # training
    for epoch in range(num_epochs):
        net.train()
        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            out = net(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        
        # compute accuracy
        train_acc = get_accuracy(train_loader)
        test_acc = get_accuracy(test_loader)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # plot training accuracy every 1% progress
        if plot and epoch%max(num_epochs // 100, 1)==0:
            plt.clf()
            plt.plot(range(1, epoch+2), train_accs)
            plt.plot(range(1, epoch+2), test_accs)
            plt.title('Training vs. Testing Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            display.clear_output(wait=True)
            display.display(plt.gcf())
        
        if early_stop and train_acc >= 0.99 and epoch>200:
            return test_accs[-1], epoch
        
        print(f"Epochs: {epoch+1}/{num_epochs} Train Acc: {round(train_acc,3)}  Test Acc: {round(test_acc,3)}")
    display.clear_output(wait=False)
    print(f"Epochs: {epoch+1}/{num_epochs} Train Acc: {round(train_acc,3)}  Test Acc: {round(test_acc,3)}")
    return test_accs[-1]
    
def compute_el2n(net, dataloader):
    scores = []
    labels_all = []
    net.eval()
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = net(imgs)
            prob = torch.softmax(out, dim=1)
            labels_one_hot = F.one_hot(labels, num_classes=out.shape[1]).float()
            score = torch.norm(prob - labels_one_hot, p=2, dim=1)
            scores.extend(score.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    return np.array(scores), np.array(labels_all)

def sorted_diff(scores: np.ndarray, labels: np.ndarray, num_classes=10):
    sorted_indices = {}
    for c in range(num_classes):
        c_indices = np.where(labels==c)[0]
        sorted_indices[c] = c_indices[np.argsort(scores[c_indices])]
    return sorted_indices


def get_sc_images(dataset, indices_dict, ep, max_ep, p_remove=0.5, n=10, num_classes=10, remove_samples='easy'):
    if ep < max_ep // 2:
        p = (ep + 1) / (max_ep // 2)
        class_images = {c: indices_dict[c][:max(int(p * len(indices_dict[c])), n)] for c in range(num_classes)}
    else:
        if remove_samples=='easy':
            class_images = {c: indices_dict[c][int(p_remove * len(indices_dict[c])):] for c in range(num_classes)}
            
        else:
            class_images = {c: indices_dict[c][:int((1-p_remove) * len(indices_dict[c]))] for c in range(num_classes)}
            
    
    indices = np.array([np.random.permutation(class_images[c])[:n] for c in range(num_classes)]).flatten()
    return dataset[indices]
    
def train_synthetic_dataset(
    train_dataset, model, tdict=None, ipc=10, num_classes=10, lr=0.1, K=100, T=10, batch_size=256, task_balance=0.01, init="real", img_syn=None, alpha=0.0, p_remove=0.5, remove_samples='hard', save_path=None, save_intervals=1000,
    ):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # images functions
    class_images = {class_n: [] for class_n in range(10)}
    for idx, (imgs, labels) in enumerate(train_dataset):
        class_images[labels.item()].append(idx)
        
    def get_images(dataset, n=10, nc=10):
        indices = np.array([np.random.permutation(class_images[class_n])[:n] for class_n in range(nc)]).flatten()
        # n: number of images per class, nc: number of class
        return dataset[indices]
    
    def maskout_params(params, alpha=0.5):
        return params[:, int(params.shape[1]*alpha):]
    
    # network related functions
    activations = {}
    def get_activation(name):
        def hook_fn(module, input, output):
            activations[name] = output.clone()
        return hook_fn

    def attach_hooks(net):
        return [layer.register_forward_hook(get_activation(name)) 
                for name, layer in net.named_modules() 
                if isinstance(layer, nn.ReLU)]
        
    def remove_hooks(hooks):
        for hook in hooks:
            hook.remove()
            
    def get_attention(feature):
        A = torch.sum(torch.abs(feature), dim=1)
        return F.normalize(A.flatten(start_dim=1))

    def error(real, syn):
        syn = syn.to(device)
        return torch.sum(torch.square(
            (torch.mean(real.reshape(num_classes, batch_size, -1), dim=1) - 
             torch.mean(syn.reshape(num_classes, ipc, -1), dim=1))
        ))

    # main (training)
    if img_syn is None:
        if init=="real":
            img_syn, label_syn = get_images(train_dataset, ipc, num_classes)
            img_syn.requires_grad_(True)
            print(img_syn.shape)
        elif init=="noise":
            img_syn = torch.randn((ipc*num_classes, *tuple(train_dataset[0][0].shape)), requires_grad=True, device=device)
        else:
            raise ValueError(f"Invalid value for 'init': {init}. Expected 'real' or 'noise'.")
        if save_path is not None:
            torch.save(img_syn, os.path.join(save_path, '0.pt'))
            
        
    optimizer = torch.optim.SGD([img_syn], lr=lr)
    
   # initialize random net K times, each used to train syn img T times
    for k in range(K):
        net = get_network_ext(model, num_classes, train_dataset)
                    
        for param in net.parameters():
            param.requires_grad = False
            
        for it in range(T):
            hooks = attach_hooks(net)
            if tdict is None:   
                img_real, label_real = get_images(train_dataset, batch_size, num_classes)
            else:             
                img_real, label_real = get_sc_images(train_dataset, tdict, k*T+it, T*K,n=batch_size, num_classes=num_classes, p_remove=p_remove, remove_samples=remove_samples)
            img_real = img_real.to(device)
            out_real = net.embed(img_real)
            act_real, activations = list(activations.values()), {}
            
            out_syn = net.embed(img_syn.to(device))
            act_syn, activations = list(activations.values()), {}
                
            remove_hooks(hooks)
            
            loss = 0
            
            att_real = torch.cat([get_attention(act_real[layer]) for layer in range(len(act_real))], dim=1)
            att_syn = torch.cat([get_attention(act_syn[layer]) for layer in range(len(act_syn))], dim=1)
            
            att_real = maskout_params(att_real, alpha=alpha)
            att_syn = maskout_params(att_syn, alpha=alpha)
            loss += error(att_real, att_syn)
            
            loss += task_balance*error(out_real, out_syn)
            loss *= 100
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"K: {k+1}/{K}, it:{it+1}/{T}, loss: {loss.item()}")
                
        if k%10==0:
            plt.clf()
            img_per_row = 4
            img_ids = [i + ipc * cid for cid in range(num_classes) for i in [0, 1]]
            syn_imgs = [img_syn[i].clone().squeeze().cpu().detach().numpy() for i in img_ids]
            num_rows = int(np.ceil(len(img_ids)/img_per_row))
            scale = 2
            fig, axes = plt.subplots(num_rows, img_per_row, figsize=(scale*img_per_row, scale*num_rows), gridspec_kw={'wspace': 0, 'hspace': 0})
            
            for ax, img in zip(axes.flat, syn_imgs):
                if img_syn.shape[1]==1:
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img.transpose(1, 2, 0))
                ax.axis('off')

            for ax in axes.flat[len(syn_imgs):]:
                ax.axis('off')
                
            display.clear_output(wait=True)
            display.display(plt.gcf())
        
        if (k+1)%save_intervals==0:
            torch.save(img_syn, os.path.join(save_path, f"{k+1}_"+time.strftime("%m%d_%H%M")+".pt"))

    return img_syn

def cross_generalization(models, syn_train_loader, test_loader, num_epochs=100, num_classes=10):
    accs = {}
    confusion_matrix = {}
    if isinstance(num_epochs, int):
        num_epochs = [num_epochs]
    for model in models:
        net_base = get_network_ext(model, num_classes, dataloader=test_loader)
        for epochs in num_epochs:
            net = copy.deepcopy(net_base)
            try:
                acc = train_classification(net, syn_train_loader, test_loader, num_epochs=epochs, early_stop=True)
            except:                
                dummy_input = next(iter(syn_train_loader))[0].cuda()
                x = net.features(dummy_input)
                in_feat = x.view(x.size(0), -1).shape[1]
                if model=='LeNet':
                    net.fc_1 = nn.Linear(in_feat, net.fc_1.out_features)
                elif model=='VGG11':
                    net.classifier = nn.Linear(in_feat, net.classifier.out_features)
                elif model=='AlexNet':
                    net.fc = nn.Linear(in_feat, net.fc.out_features)
                net.cuda()
                acc = train_classification(net, syn_train_loader, test_loader, num_epochs=epochs, early_stop=True)
            accs[(model, epochs)] = acc
            confusion_matrix[(model, epochs)] = get_accuracy_and_confusion_matrix(test_loader, num_classes, net)[1]
    return accs, confusion_matrix

def continual_learning(
    img_syn, test_loader, buffer_size=1000, batch_size=256, num_epochs=20, buffer_sample_size=None, num_classes=10,
    subtasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    ):
    ipc = img_syn.size(0)//num_classes
    if buffer_sample_size is None:
        buffer_sample_size=ipc*2
    def get_subtask_loader(subtasks, test_images, test_labels, batch_size=256):
        all_images = torch.cat([test_images[c] for c in subtasks], dim=0)
        all_labels = torch.cat([test_labels[c] for c in subtasks], dim=0)

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(all_images, all_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return loader

    test_images = {class_n: [] for class_n in range(10)}
    for idx, (imgs, labels) in enumerate(test_loader):
        for i in range(imgs.size(0)):
            label = int(labels[i].cpu())
            test_images[label].append(imgs[i])
    for c in range(10):
        test_images[c] = torch.stack(test_images[c])
    test_labels = {c: torch.full((len(test_images[c]),), c, dtype=torch.long) for c in range(10)}
    
    syn_train_loader = get_syn_dataloader(img_syn, shuffle=False, ipc=ipc)
    data, labels = zip(*[(dat, lab) for dat, lab in syn_train_loader])
    data = torch.cat(data, dim=0)
    labels = torch.cat(labels, dim=0)
    
    net = get_network_ext('ConvNet', 10, dataloader=test_loader)
    
    replay_buffer = ReplayBuffer(buffer_size)
    accs = []
    classes = []
    
    for i, sublabels in enumerate(subtasks):
        indices = ipc*len(sublabels)
        current_data, current_labels = data[:indices], labels[:indices]
        data, labels = data[indices:], labels[indices:]
        
        data_sample, labels_sample = replay_buffer.sample(buffer_sample_size)
        dataset = TensorDataset(torch.cat((data_sample, current_data)), torch.cat((labels_sample, current_labels)))
        subtrain_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        testset_labels = [task for current_task in subtasks[:i+1] for task in current_task]
        subtest_loader = get_subtask_loader(testset_labels, test_images, test_labels)
        
        acc = train_classification(net, subtrain_loader, subtest_loader, num_epochs=num_epochs)
        accs.append(acc)
        classes.append(len(testset_labels))
        for i in range(current_data.size(0)):
            replay_buffer.add(current_data[i], current_labels[i])
    
    return classes, accs


def get_accuracy_and_confusion_matrix(dataloader, num_classes, net):
    net.eval()
    correct = 0
    total = 0
        
    # Initialize confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = net(imgs)
            _, pred = torch.max(out.data, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

            # Update confusion matrix
            for t, p in zip(labels.view(-1), pred.view(-1)):
                confusion_matrix[t.item(), p.item()] += 1

    accuracy = correct / total
    return accuracy, confusion_matrix
