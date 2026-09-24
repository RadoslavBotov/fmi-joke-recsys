import unittest
from unittest.mock import patch

import pandas as pd
import torch

from src.net import CustomDataset, RecommenderNet, check_gpu


class TestCheckGpu(unittest.TestCase):

    @patch('torch.device')
    @patch('torch.cuda.is_available')
    def test_gpu_available(self, mock_cuda, mock_device):
        # Arrange:
        mock_cuda.return_value = True
        
        # Act
        device = check_gpu(verbose=False)
        
        # Assert
        mock_cuda.assert_called_once()
        mock_device.assert_called_once_with('cuda')

    @patch('torch.device')
    @patch('torch.cuda.is_available')
    def test_gpu_unavailable(self, mock_cuda, mock_device):
        # Arrange:
        mock_cuda.return_value = False
        
        # Act
        device = check_gpu(verbose=False)
        
        # Assert
        mock_cuda.assert_called_once()
        mock_device.assert_called_once_with('cpu')


class TestCustomDataset(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'user_id': [0, 1, 2],
            'joke_id': [0, 1, 1],
            'rating': [5.0, -2.0, 8.0]
        })
        self.dataset = CustomDataset(self.df)

    def test_dataset_length(self):
        # Assert
        self.assertEqual(len(self.dataset), 3)

    def test_getitem_shapes_and_values(self):
        # Act and Assert
        for idx, (user, item, rating) in enumerate(self.dataset):
            self.assertEqual(user.item(), self.df['user_id'][idx])
            self.assertEqual(item.item(), self.df['joke_id'][idx])
            self.assertEqual(rating.item(), self.df['rating'][idx])

    def test_types_and(self):
        # Assert
        self.assertEqual(self.dataset.users.device.type, 'cpu')
        self.assertEqual(self.dataset.items.device.type, 'cpu')
        self.assertEqual(self.dataset.ratings.device.type, 'cpu')
        self.assertEqual(self.dataset.users.shape, (3,))
        self.assertEqual(self.dataset.items.shape, (3,))
        self.assertEqual(self.dataset.ratings.shape, (3,))
        self.assertIsInstance(self.dataset.users, torch.Tensor)
        self.assertIsInstance(self.dataset.items, torch.Tensor)
        self.assertIsInstance(self.dataset.ratings, torch.Tensor)


class TestRecommenderNet(unittest.TestCase):

    def setUp(self):
        self.user_vocab = 100
        self.item_vocab = 20
        self.emb_dim = 16
        self.hidden_dim = 32
        self.model = RecommenderNet(self.user_vocab, self.item_vocab, self.emb_dim, self.hidden_dim)
        self.model.eval()

    def test_layer_shapes(self):
        # Assert:
        self.assertEqual(self.model.user_embedding.weight.shape, (self.user_vocab, self.emb_dim))
        self.assertEqual(self.model.item_embedding.weight.shape, (self.item_vocab, self.emb_dim))
        self.assertEqual(self.model.dense1.weight.t().shape, (self.emb_dim * 2, self.hidden_dim))
        self.assertEqual(self.model.dense2.weight.t().shape, (self.hidden_dim, self.hidden_dim))
        self.assertEqual(self.model.dense3.weight.t().shape, (self.hidden_dim, self.hidden_dim))
        self.assertEqual(self.model.output_layer.weight.t().shape, (self.hidden_dim, 1))

    def test_forward_pass_output_shape(self):
        # Arrange:
        batch_size = 8
        u_idx = torch.randint(0, self.user_vocab, (batch_size, 1))
        j_idx = torch.randint(0, self.item_vocab, (batch_size, 1))
        
        # Act
        output = self.model(u_idx, j_idx)
        
        # Assert:
        self.assertEqual(output.shape, (batch_size, 1))

    def test_output_range_constraints(self):
        # Arrange:
        batch_size = 1000
        u_idx = torch.randint(0, self.user_vocab, (batch_size, 1))
        j_idx = torch.randint(0, self.item_vocab, (batch_size, 1))
        
        # Act
        output = self.model(u_idx, j_idx)
        
        # Assert:
        self.assertTrue(torch.all(output >= -10.0))
        self.assertTrue(torch.all(output <= 10.0))

    def test_invalid_input_handling(self):
        # Arrange:
        u_idx = torch.tensor([[999]]) # user_vocab goes up to 100
        j_idx = torch.tensor([[1]])
        
        # Act and Assert
        with self.assertRaises(IndexError):
            self.model(u_idx, j_idx)
    
