import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import svm_losses
from model import Model
from termcolor import cprint

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Home device: {}'.format(device))

############################
# Utility: GaussianBlur with kernel fix
############################
class GaussianBlur(object):
    """
    Implements Gaussian blur as described in the SimCLR paper.
    Ensures that the kernel size is positive and odd.
    """
    def __init__(self, kernel_size, min=0.1, max=2.0):
        # Ensure kernel size is odd. If even, add 1.
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.min = min
        self.max = max

    def __call__(self, sample):
        sample = np.array(sample)
        # Apply blur with a 50% chance.
        if np.random.random_sample() < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(sample)

############################
# Dataset loaders for Tiny ImageNet
############################

class TinyImageNetPair_true_label(ImageFolder):
    """
    Data loader that randomly samples two different images from the same class.
    """
    def __init__(self, root='/kaggle/input/tiny-image-net/tiny-imagenet-200/train', transform=None):
        super().__init__(root=root, transform=transform)
        self.label_index = self._get_label_indices()

    def _get_label_indices(self):
        label_dict = {}
        for idx, (path, target) in enumerate(self.samples):
            if target not in label_dict:
                label_dict[target] = []
            label_dict[target].append(idx)
        return label_dict

    def __getitem__(self, index):
        path1, target = self.samples[index]
        img1 = Image.open(path1).convert("RGB")
        
        # Randomly sample a different image from the same class.
        index_example = np.random.choice(self.label_index[target])
        path2, _ = self.samples[index_example]
        img2 = Image.open(path2).convert("RGB")
        
        if self.transform is not None:
            pos_1 = self.transform(img1)
            pos_2 = self.transform(img2)
        else:
            pos_1, pos_2 = img1, img2
        return pos_1, pos_2, target, index

class TinyImageNetPair(ImageFolder):
    """
    Data loader that returns two augmented versions of the same image.
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
        else:
            pos_1, pos_2 = img, img
        return pos_1, pos_2, target, index

############################
# Transforms for Tiny ImageNet:
# Note that Tiny ImageNet images are 64x64.
############################
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    # 0.1 * 64 = 6.4 -> int(6.4)=6, which is even, so GaussianBlur will add 1 and use kernel size 7.
    GaussianBlur(kernel_size=int(0.1 * 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_dataset(dataset_name, dataset_location, pair=True):
    """
    Returns the dataset splits (train, memory, test) for Tiny ImageNet.
    The expected structure is:
       dataset_location/
           train/
           val/
    """
    if pair:
        if dataset_name == 'tiny_imagenet':
            train_data = TinyImageNetPair(root=os.path.join(dataset_location, "train"), transform=train_transform)
            memory_data = TinyImageNetPair(root=os.path.join(dataset_location, "train"), transform=test_transform)
            test_data = TinyImageNetPair(root=os.path.join(dataset_location, "val"), transform=test_transform)
        elif dataset_name == 'tiny_imagenet_true_label':
            train_data = TinyImageNetPair_true_label(root=os.path.join(dataset_location, "train"), transform=train_transform)
            memory_data = TinyImageNetPair_true_label(root=os.path.join(dataset_location, "train"), transform=test_transform)
            test_data = TinyImageNetPair_true_label(root=os.path.join(dataset_location, "val"), transform=test_transform)
        else:
            raise Exception('Invalid dataset name')
    else:
        if dataset_name in ['tiny_imagenet', 'tiny_imagenet_true_label']:
            train_data = ImageFolder(root=os.path.join(dataset_location, "train"), transform=train_transform)
            memory_data = ImageFolder(root=os.path.join(dataset_location, "train"), transform=test_transform)
            test_data = ImageFolder(root=os.path.join(dataset_location, "val"), transform=test_transform)
        else:
            raise Exception('Invalid dataset name')
    return train_data, memory_data, test_data

############################
# Training and Testing functions
############################

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(net, data_loader, train_optimizer, crit, args, epoch, epochs, batch_size):
    net.train()
    total_loss, total_num = 0.0, 0
    kxz_losses, kyz_losses = 0.0, 0.0
    train_bar = tqdm(data_loader)
    for iii, (pos_1, pos_2, target, index) in enumerate(train_bar):
        pos_1 = pos_1.to(device, non_blocking=True)
        pos_2 = pos_2.to(device, non_blocking=True)
        # Forward pass.
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        features = torch.cat([out_1.unsqueeze(1), out_2.unsqueeze(1)], dim=1)
        
        # Compute loss.
        kxz_loss, kyz_loss = crit(features)
        loss = kxz_loss + kyz_loss
        kxz_losses += kxz_loss.item() * batch_size
        kyz_losses += kyz_loss.item() * batch_size
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        
        total_num += batch_size
        total_loss += loss.item() * batch_size
        
        train_bar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num)
        )
    
    metrics = {
        'total_loss': total_loss / total_num,
        'kxz_loss': kxz_losses / total_num,
        'kyz_loss': kyz_losses / total_num,
    }
    return metrics

def test(net, memory_data_loader, test_data_loader, k, c, epoch, epochs, dataset_name):
    net.eval()
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    feature_bank = []
    temperature = 0.5
    with torch.no_grad():
        # Generate feature bank.
        for data, _, target, _ in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.to(device, non_blocking=True))
            feature_bank.append(feature)
        # Shape: [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # Get labels.
        if 'cifar' in dataset_name:
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        else:
            feature_labels = torch.tensor([s[1] for s in memory_data_loader.dataset.samples],
                                          device=feature_bank.device)
        
        test_bar = tqdm(test_data_loader)
        for data, _, target, _ in test_bar:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            feature, out = net(data)
            total_num += data.size(0)
            # Compute cosine similarity.
            sim_matrix = torch.mm(feature, feature_bank)  # [B, N]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)  # [B, k]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()
            # kNN scores.
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            pred_scores = torch.sum(one_hot_label.view(data.size(0), k, c) * sim_weight.unsqueeze(dim=-1), dim=1)
            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.long().unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description(
                'KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'.format(
                    epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100
                )
            )
    return total_top1 / total_num * 100, total_top5 / total_num * 100

############################
# Main training loop
############################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR on Tiny ImageNet')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=400, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset_name', default='tiny_imagenet', type=str, help='Dataset name to use')
    parser.add_argument('--criterion_to_use', default='default', type=str, help='Choose loss function')
    parser.add_argument('--val_freq', default=25, type=int, help='Frequency (in epochs) of validation')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--dataset_location', default='/kaggle/input/tiny-image-net/tiny-imagenet-200', type=str, help='Location of the dataset')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loader workers')
    
    # MMCL-specific arguments.
    parser.add_argument('--C_reg', type=float, default=100.0)
    parser.add_argument('--topK', type=int, default=128)
    parser.add_argument('--gamma', type=str, default="50")
    parser.add_argument('--kernel_type', type=str, default="rbf")
    parser.add_argument('--drop_sigma', type=str, default='75,125')
    parser.add_argument('--reg', type=float, default=0.1)
    parser.add_argument('--num_iter', type=int, default=1000)
    parser.add_argument('--eta', type=float, default=1E-3)
    parser.add_argument('--stop_condition', type=float, default=0.01)
    parser.add_argument('--solver_type', type=str, default='nesterov')
    parser.add_argument('--use_norm', type=str, default='false')
    parser.add_argument('--run_name', type=str, default='MMCL')
    
    args = parser.parse_args()
    feature_dim, k = args.feature_dim, args.k
    batch_size, epochs = args.batch_size, args.epochs
    dataset_name = args.dataset_name

    if args.drop_sigma != 'no':
        sigma_drop_epochs = args.drop_sigma.split(',')
        sigma_drop_epochs = [int(val) for val in sigma_drop_epochs]
    else:
        sigma_drop_epochs = []

    run_name = args.run_name

    # Data preparation.
    train_data, memory_data, test_data = get_dataset(dataset_name, args.dataset_location)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Model and optimizer.
    model = Model(feature_dim).to(device)
    model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    
    # Number of classes.
    c = len(memory_data.classes)
    print('# Classes: {}'.format(c))
    
    epoch_start = 1
    if args.criterion_to_use == 'mmcl_inv':
        crit = svm_losses.MMCL_inv(sigma=args.gamma, batch_size=args.batch_size, anchor_count=2, C=args.C_reg)
    elif args.criterion_to_use == 'mmcl_pgd':
        crit = svm_losses.MMCL_pgd(sigma=args.gamma, batch_size=args.batch_size, anchor_count=2, C=args.C_reg,
                                   solver_type=args.solver_type, use_norm=args.use_norm)
    else:
        crit = svm_losses.MMCL_inv(sigma=args.gamma, batch_size=args.batch_size, anchor_count=2, C=args.C_reg)
    
    # Ensure results directories exist.
    result_dir = os.path.join('..', 'results', dataset_name, run_name)
    if not os.path.exists(os.path.join('..', 'results')):
        os.mkdir(os.path.join('..', 'results'))
    if not os.path.exists(os.path.join('..', 'results', dataset_name)):
        os.mkdir(os.path.join('..', 'results', dataset_name))
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    # Training loop.
    for epoch in range(epoch_start, epochs + 1):
        
        if epoch in sigma_drop_epochs:
            cprint(f'Sigma right now {crit.sigma}', 'red')
            crit.sigma = str(float(crit.sigma) / 10.0)
            cprint(f'Sigma after drop {crit.sigma}', 'green')
        
        metrics = train(model, train_loader, optimizer, crit, args, epoch, epochs, batch_size)
        metrics['epoch'] = epoch
        metrics['lr'] = get_lr(optimizer)
        
        if epoch % args.val_freq == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, k, c, epoch, epochs, dataset_name)
            model_path = os.path.join('..', 'results', dataset_name, run_name, f'model_{epoch}.pth')
            torch.save(model.state_dict(), model_path)
            metrics['top1'] = test_acc_1
            metrics['top5'] = test_acc_5
