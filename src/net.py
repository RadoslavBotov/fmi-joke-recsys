import os
import pickle
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchinfo import summary
from tqdm import tqdm

from src.metrics import get_mae, get_mrr, get_rmse
from src.utils import get_mean_for_users


def check_gpu(verbose: bool = True) -> torch.device:
    """
    Checks for CUDA availability and returns the appropriate torch device.
    """
    if (verbose is True):
        print('CUDA:', torch.cuda.is_available())
        print('Device count:', torch.cuda.device_count())
        print('Current device:', torch.cuda.current_device())
        print('GPU:', torch.cuda.get_device_name(0))
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    """
    PyTorch Dataset wrapper for Jester joke rating data.
    """
    def __init__(self, df: pd.DataFrame, device: torch.device = torch.device('cpu')):
        self.users = torch.tensor(df['user_id'].to_numpy(), device=device)
        self.items = torch.tensor(df['joke_id'].to_numpy(), device=device)
        self.ratings = torch.tensor(df['rating'].to_numpy(), device=device)
        self.length = len(df)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.items[idx], self.ratings[idx]

class RecommenderNet(nn.Module):
    """
    Neural Collaborative Filtering model using embedding layers and a deep dense architecture.
    """
    def __init__(self, user_vocab_size: int, item_vocab_size: int, embedding_dim: int, hidden_size: int) -> None:
        super(RecommenderNet, self).__init__()
        # Embedding layers to map categorical IDs to dense vectors
        self.user_embedding = nn.Embedding(num_embeddings=user_vocab_size, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=item_vocab_size, embedding_dim=embedding_dim)

        # Dense layers (input = 2 * embedding_dim because we concatenate user + item)
        self.dense1 = nn.Linear(embedding_dim * 2, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

        # Activations
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        user_emb = self.user_embedding(user_indices).squeeze(1)
        item_emb = self.item_embedding(item_indices).squeeze(1)

        # Concatenate user and item features
        x = torch.cat([user_emb, item_emb], dim=1)

        # Deep architecture pass
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.relu(self.dense3(x))

        # Tanh normalizes output to [-1, 1], then scaled to Jester range [-10, 10]
        x = self.tanh(self.output_layer(x))
        return x * 10

def train_step(model: nn.Module,
               optimizer: optim.Optimizer,
               criterion: nn.Module, 
               user_indices: torch.Tensor,
               item_indices: torch.Tensor, 
               target_ratings: torch.Tensor) -> float:
    """
    Performs a single training iteration.
    """
    model.train()
    optimizer.zero_grad()

     # (batch_size, 1) -> (batch_size, )
    output = model(user_indices, item_indices).squeeze(1)
    loss = criterion(output, target_ratings)

    loss.backward()
    optimizer.step()
    
    return loss.item()

def net_train(data_path: str):
    """
    Orchestrates the training process and saves model artifacts.
    """
    device = check_gpu(verbose=False)
    jester_data_svd_df = pd.read_csv(data_path)

    # Hyperparameters
    user_embedding_dim = len(jester_data_svd_df['user_id'].unique())
    joke_embedding_dim = len(jester_data_svd_df['joke_id'].unique())
    embedding_dim = 128
    hidden_features = 256
    epochs = 100
    batch_size = 1024
    g = torch.Generator()
    g.manual_seed(42)

    # Split data
    train, _ = train_test_split(jester_data_svd_df, random_state=42, stratify=jester_data_svd_df['user_id'])
    
    # Format data for easy use with pytorch
    train_set = CustomDataset(train, device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)

    # Instantiate model
    model = RecommenderNet(
        user_vocab_size=user_embedding_dim,
        item_vocab_size=joke_embedding_dim,
        embedding_dim=embedding_dim,
        hidden_size=hidden_features
    ).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    # print(model)
    summary(model, input_size=[(batch_size, 1), (batch_size, 1)], dtypes=[torch.int64, torch.int64])

    # Training loop
    loss_logs = []
    for epoch in tqdm(range(epochs), desc='Epoch'):
        for batch_idx, (u_idx, j_idx, ratings) in enumerate(train_loader):
            current_loss = train_step(model, optimizer, criterion, u_idx, j_idx, ratings)
            if batch_idx % 10 == 0:
                loss_logs.append((epoch, batch_idx, current_loss))

    # Save losses
    loss_path = os.path.join('losses', 'loss_' + data_path.split(os.sep)[1].split('.')[0] + '.pkl')
    with open(loss_path, 'wb') as f:
        pickle.dump(loss_logs, f)

    # Save pretrained model
    model_path = os.path.join('models', 'model_' + data_path.split(os.sep)[1].split('.')[0] + '.pt')
    torch.save(model.state_dict(), model_path)

def test_step(model: nn.Module,
              criterion: nn.Module, 
              user_indices: torch.Tensor,
              item_indices: torch.Tensor, 
              target_ratings: torch.Tensor) -> Tuple[float, torch.Tensor]:
    """
    Performs an evaluation pass on a batch of data without calculating gradients.
    """
    model.eval()

    # Calculate loss without tracking gradients
    with torch.no_grad():
        output = model(user_indices, item_indices)
        output = output.squeeze(1) # Flatten (batch_size, 1) -> (batch_size, )
        loss = criterion(output, target_ratings)

    return loss.item(), output

def net_test(data_path: str) -> None:
    """
    Performs inference on the test set and calculates metrics.
    """
    device = check_gpu(verbose=False)
    jester_data_svd_df = pd.read_csv(data_path)

    # Hyperparameters
    user_embedding_dim = len(jester_data_svd_df['user_id'].unique())
    joke_embedding_dim = len(jester_data_svd_df['joke_id'].unique())
    embedding_dim = 128
    hidden_features = 256
    batch_size = 1024
    g = torch.Generator()
    g.manual_seed(42)

    # Split data
    train, test = train_test_split(jester_data_svd_df, test_size=0.2, random_state=42, stratify=jester_data_svd_df['user_id'])
    test_set = CustomDataset(test, device)
    test_loader = DataLoader(test_set, batch_size=batch_size, generator=g)
    user_means = get_mean_for_users(train)

    # Load pretrained model for inference
    model = RecommenderNet(
        user_vocab_size=user_embedding_dim,
        item_vocab_size=joke_embedding_dim,
        embedding_dim=embedding_dim,
        hidden_size=hidden_features
    )
    model_path = os.path.join('models', 'model_' + data_path.split(os.sep)[1].split('.')[0] + '.pt')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    criterion = nn.MSELoss()

    # predictions: list[(user_id, actual, pred, user_mean)]
    predictions = []
    test_loss = 0
    for u_idx, j_idx, ratings in tqdm(test_loader, total=len(test_loader)):
        batch_loss, batch_pred = test_step(model, criterion, u_idx, j_idx, ratings)
        test_loss += batch_loss * u_idx.size(0)
        for u, a, p in zip(u_idx, ratings, batch_pred):
            predictions.append((int(u), float(a), float(p), user_means[int(u)]))

    # metrics
    print(f'> Metrics for \'{data_path}\'')
    print(f'LOSS: {test_loss / len(test_set)}')
    print(f'RMSE: {get_rmse(predictions)}')
    print(f'MAE: {get_mae(predictions)}')
    print(f'MRR: {get_mrr(predictions)}')

def net_calcs(args_dict: Dict[str, int|str]) -> None:
    if (args_dict['nn_mode'] == 'train'):
        print('>> Training nn ...')
        net_train(args_dict['data'])
    else:
        print('>> Testing nn ...')
        net_test(args_dict['data'])


# CUDA: True
# Device count: 1
# Current device: 0
# GPU: NVIDIA GeForce GTX 1650
# RecommenderNet(
#   (user_embedding): Embedding(users_count, 128)
#   (item_embedding): Embedding(100, 128)
#   (dense1): Linear(in_features=256, out_features=256, bias=True)
#   (dense2): Linear(in_features=256, out_features=256, bias=True)
#   (dense3): Linear(in_features=256, out_features=256, bias=True)
#   (output_layer): Linear(in_features=256, out_features=1, bias=True)
#   (relu): ReLU()
#   (tanh): Tanh()
# )


# > Metrics for 'data_svd\jester_data_1_svd.csv'
# LOSS: 29.3261
# RMSE:  5.4153
# MAE:   4.0842
# MRR:   0.8128
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# RecommenderNet                           [1024, 1]                 --
# ├─Embedding: 1-1                         [1024, 1, 128]            6,205,824
# ├─Embedding: 1-2                         [1024, 1, 128]            12,800
# ├─Linear: 1-3                            [1024, 256]               65,792
# ├─ReLU: 1-4                              [1024, 256]               --
# ├─Linear: 1-5                            [1024, 256]               65,792
# ├─ReLU: 1-6                              [1024, 256]               --
# ├─Linear: 1-7                            [1024, 256]               65,792
# ├─ReLU: 1-8                              [1024, 256]               --
# ├─Linear: 1-9                            [1024, 1]                 257
# ├─Tanh: 1-10                             [1024, 1]                 --
# ==========================================================================================
# Total params: 6,416,257
# Trainable params: 6,416,257
# Non-trainable params: 0
# Total mult-adds (Units.GIGABYTES): 6.57
# ==========================================================================================
# Input size (MB): 0.02
# Forward/backward pass size (MB): 8.40
# Params size (MB): 25.67
# Estimated Total Size (MB): 34.08
# ==========================================================================================


# > Metrics for 'data_svd\jester_data_2_svd.csv'
# LOSS: 29.0257
# RMSE:  5.3875
# MAE:   4.1300
# MRR:   0.7094
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# RecommenderNet                           [1024, 1]                 --
# ├─Embedding: 1-1                         [1024, 1, 128]            3,192,064
# ├─Embedding: 1-2                         [1024, 1, 128]            12,800
# ├─Linear: 1-3                            [1024, 256]               65,792
# ├─ReLU: 1-4                              [1024, 256]               --
# ├─Linear: 1-5                            [1024, 256]               65,792
# ├─ReLU: 1-6                              [1024, 256]               --
# ├─Linear: 1-7                            [1024, 256]               65,792
# ├─ReLU: 1-8                              [1024, 256]               --
# ├─Linear: 1-9                            [1024, 1]                 257
# ├─Tanh: 1-10                             [1024, 1]                 --
# ==========================================================================================
# Total params: 3,402,497
# Trainable params: 3,402,497
# Non-trainable params: 0
# Total mult-adds (Units.GIGABYTES): 3.48
# ==========================================================================================
# Input size (MB): 0.02
# Forward/backward pass size (MB): 8.40
# Params size (MB): 13.61
# Estimated Total Size (MB): 22.02
# ==========================================================================================


# > Metrics for 'data_svd\jester_data_comb_svd.csv'
# LOSS: 29.8239
# RMSE:  5.4611
# MAE:   4.1077
# MRR:   0.7774
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# RecommenderNet                           [1024, 1]                 --
# ├─Embedding: 1-1                         [1024, 1, 128]            9,397,888
# ├─Embedding: 1-2                         [1024, 1, 128]            12,800
# ├─Linear: 1-3                            [1024, 256]               65,792
# ├─ReLU: 1-4                              [1024, 256]               --
# ├─Linear: 1-5                            [1024, 256]               65,792
# ├─ReLU: 1-6                              [1024, 256]               --
# ├─Linear: 1-7                            [1024, 256]               65,792
# ├─ReLU: 1-8                              [1024, 256]               --
# ├─Linear: 1-9                            [1024, 1]                 257
# ├─Tanh: 1-10                             [1024, 1]                 --
# ==========================================================================================
# Total params: 9,608,321
# Trainable params: 9,608,321
# Non-trainable params: 0
# Total mult-adds (Units.GIGABYTES): 9.84
# ==========================================================================================
# Input size (MB): 0.02
# Forward/backward pass size (MB): 8.40
# Params size (MB): 38.43
# Estimated Total Size (MB): 46.85
# ==========================================================================================
