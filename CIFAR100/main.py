import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import svm_losses
import utils  # Ensure util.py is correctly imported
from model import Model
from termcolor import cprint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))

############################
# Helper functions
############################
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

############################
# Training function
############################
def train(net, data_loader, train_optimizer, crit, args, epoch, epochs, batch_size):
    net.train()
    total_loss, total_num = 0.0, 0
    kxz_losses, kyz_losses = 0.0, 0.0
    train_bar = tqdm(data_loader)

    for iii, (pos_1, pos_2, target, index) in enumerate(train_bar):
        pos_1, pos_2 = pos_1.to(device, non_blocking=True), pos_2.to(device, non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        features = torch.cat([out_1.unsqueeze(1), out_2.unsqueeze(1)], dim=1)
        kxz_loss, kyz_loss = crit(features) 
        loss = kxz_loss + kyz_loss
        kxz_losses += kxz_loss.item() * batch_size
        kyz_losses += kyz_loss.item() * batch_size

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description(f'Train Epoch: [{epoch}/{epochs}] Loss: {total_loss / total_num:.4f}')

    metrics = {
        'total_loss': total_loss / total_num,
        'kxz_loss': kxz_losses / total_num,
        'kyz_loss': kyz_losses / total_num,
    }

    return metrics

############################
# Testing function
############################
def test(net, memory_data_loader, test_data_loader, k, c, epoch, epochs, dataset_name):
    net.eval()
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    feature_bank = []
    temperature = 0.5

    with torch.no_grad():
        # Generate feature bank
        for data, _, target, _ in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = (
            torch.tensor(memory_data_loader.dataset.dataset.targets, device=feature_bank.device)
            if isinstance(memory_data_loader.dataset, torch.utils.data.Subset)
            else torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        )

        # Loop through test data
        test_bar = tqdm(test_data_loader)
        for data, _, target, _ in test_bar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            feature, out = net(data)
            total_num += data.size(0)

            # Compute similarity
            sim_matrix = torch.mm(feature, feature_bank)
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # Compute classification scores
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            one_hot_label.scatter_(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            pred_scores = torch.sum(one_hot_label.view(data.size(0), k, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            
            test_bar.set_description(
                f'KNN Test Epoch: [{epoch}/{epochs}] Acc@1:{total_top1 / total_num * 100:.2f}% Acc@5:{total_top5 / total_num * 100:.2f}%'
            )

    return total_top1 / total_num * 100, total_top5 / total_num * 100

############################
# Main execution
############################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR on Tiny ImageNet')
    parser.add_argument('--feature_dim', default=128, type=int)
    parser.add_argument('--k', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)  # Reduced for memory efficiency
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--dataset_name', default='tiny_imagenet', type=str)
    parser.add_argument('--criterion_to_use', default='mmcl_pgd', type=str)
    parser.add_argument('--val_freq', default=1, type=int)  # Validate every epoch
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--dataset_location', default='/kaggle/input/tiny-image-net/tiny-imagenet-200', type=str)
    parser.add_argument('--num_workers', default=4, type=int)

    args = parser.parse_args()
    feature_dim, k = args.feature_dim, args.k
    batch_size, epochs = args.batch_size, args.epochs
    dataset_name = args.dataset_name

    # Load subsampled datasets
    train_data, memory_data, test_data = utils.get_dataset(dataset_name, args.dataset_location)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model setup and optimizer config
    model = Model(feature_dim).to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)

    # Number of classes
    c = len(memory_data.dataset.dataset.classes) if isinstance(memory_data, torch.utils.data.Subset) else len(memory_data.classes)
    print(f'# Classes: {c}')

    epoch_start = 1
    crit = svm_losses.MMCL_pgd(sigma=args.k, batch_size=args.batch_size, anchor_count=2, C=100.0, solver_type='nesterov', use_norm='false')

    # Training loop
    for epoch in range(epoch_start, epochs + 1):
        metrics = train(model, train_loader, optimizer, crit, args, epoch, epochs, batch_size)
        metrics['epoch'] = epoch
        metrics['lr'] = get_lr(optimizer)

        if epoch % args.val_freq == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, k, c, epoch, epochs, dataset_name)
            torch.save(model.state_dict(), f'../results/{dataset_name}/MMCL/model_{epoch}.pth')
            metrics['top1'] = test_acc_1
            metrics['top5'] = test_acc_5
