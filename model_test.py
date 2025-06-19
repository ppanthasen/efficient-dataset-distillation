import torch
import torch.nn as nn
import time
from itertools import count
from IPython import display
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils_ext import get_network_ext
import pandas as pd
import numpy as np
from model_train import train_classification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_result(epochs, train_accs, test_accs):
    plt.clf()
    plt.plot(epochs, train_accs)
    plt.plot(epochs, test_accs)
    
    plt.title('Training vs. Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(["Training Accuracy", "Testing Accuracy"])
    display.display(plt.gcf())

# def train_classification(net, train_loader, test_loader, lr:float=0.01, timelimit=300):
#     def get_accuracy(dataloader):
#         net.eval()
#         correct = 0
#         total = 0
#         for imgs, labels in dataloader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             out = net(imgs)
#             _, pred = torch.max(out.data, 1)
#             total += labels.size(0)
#             correct += (pred == labels).sum().item()
#         return correct / total
        
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
#     train_accs=[]
#     test_accs=[]
#     times = []
    
#     # training
#     start_time = time.time()
#     for epoch in count(start=1):
#         net.train()
#         for imgs, labels in train_loader:
#             out = net(imgs)
#             loss = criterion(out, labels)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#         scheduler.step()
        
#         # compute accuracy
#         train_acc = get_accuracy(train_loader)
#         test_acc = get_accuracy(test_loader)
        
#         train_accs.append(train_acc)
#         test_accs.append(test_acc)
        
#         times.append(time.time() - start_time)
        
#         print(f"Time elasped: {int(times[-1])}/{timelimit} Epoch: {epoch}",
#               f"Train Acc:{train_acc:.4f} Test Acc:{test_acc:.4f}")
        
#         if time.time() - start_time > timelimit:
#             break
        
#     display.clear_output(wait=False)
    
#     print(f"Time elasped: {int(times[-1])}/{timelimit} Epoch: {epoch}",
#           f"Train Acc:{train_acc:.4f} Test Acc:{test_acc:.4f}")
#     plot_result(range(1, epoch+1), train_accs, test_accs)
    
#     return epoch, test_accs, times


# def compare_training_time(models, train_loader, syn_train_loader, test_loader, num_classes, timelimit=300):
#     result_dict = {}
    
#     for model in models:
#         net = get_network_ext(model, num_classes, dataloader=train_loader)
#         net1 = copy.deepcopy(net)
#         net2 = copy.deepcopy(net)
        
#         print(f"Evaluating {model} - base")
#         epoch_base, test_accs_base, t_base = train_classification(net1, train_loader, test_loader, timelimit=timelimit)
        
#         print(f"Evaluating {model} - syn")
#         epoch_syn, test_accs_syn, t_syn = train_classification(net2, syn_train_loader, test_loader, timelimit=t_base[-1])
        
#         result_dict[model] = {
#             'train_time_base': t_base,
#             'epoch_base': epoch_base,
#             'test_accs_base': test_accs_base,
#             'best_test_acc_base': max(test_accs_base),
#             'train_time_syn': t_syn,
#             'epoch_syn': epoch_syn,
#             'test_accs_syn': test_accs_syn,
#             'best_test_acc_syn': max(test_accs_syn),
#         }
        
#     plt.figure(figsize=(6.5, 4))
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


#     for i, (k, v) in enumerate(result_dict.items()):
#         plt.plot(v['train_time_base'], [a*100 for a in v['test_accs_base']], '-', color=colors[i], label=f'{k} (base)')
#         plt.plot(v['train_time_syn'], [a*100 for a in v['test_accs_syn']], '--', color=colors[i], label=f'{k} (syn)')
#     plt.legend()
#     plt.xlabel('Time (s)')
#     plt.ylabel('Testing Accuracy (%)')
#     plt.show()
    
    
#     data = []
#     for k, results in result_dict.items():
#         maxid_base = np.argmax(results['test_accs_base'])
#         maxid_syn = np.argmax(results['test_accs_syn'])
#         data.append({
#             'Model': k,
#             'Test Acc Base': results['best_test_acc_base']*100,
#             'Test Acc Syn': results['best_test_acc_syn']*100,
#             'Time Base': int(results['train_time_base'][maxid_base]),
#             'Time Syn': int(results['train_time_syn'][maxid_syn]),
#             'Epoch Base': results['epoch_base'],
#             'Epoch Syn': results['epoch_syn'],
#         })

#     # Create DataFrame
#     results_df = pd.DataFrame(data)

#     # Display the DataFrame
#     print(results_df)
        
#     return result_dict
    