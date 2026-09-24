import os
import unittest
from unittest.mock import MagicMock, patch

from main import execute_algo, get_parser_args, init_parser, parser_data_to_path, run_main


class TestInitParser(unittest.TestCase):

    def setUp(self):
        self.parser = init_parser()

    def test_parser_defaults(self):
        # Arrange: only required arguments
        test_args = ['--algo', 'tfidf', '--data', '2']
        
        # Act
        args = self.parser.parse_args(test_args)
        
        # Assert:
        self.assertEqual(args.algo, 'tfidf')
        self.assertEqual(args.data, 2)
        self.assertEqual(args.nn_mode, 'train')
        self.assertEqual(args.tfidf_mode, 'tf')

    def test_parser_with_valid_arguments(self):
        # Arrange: valid command line input
        test_args = ['--algo', 'nn', '--data', '1', '--nn_mode', 'test']
        
        # Act
        args = self.parser.parse_args(test_args)
        
        # Assert
        self.assertEqual(args.algo, 'nn')
        self.assertEqual(args.data, 1)
        self.assertEqual(args.nn_mode, 'test')

    def test_parser_invalid_algorithm_name(self):
        # Arrange:
        test_args = ['--algo', 'invalid_algo', '--data', '1']
        
        # Act & Assert:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(test_args)
    
    def test_parser_invalid_data_name(self):
        # Arrange:
        test_args = ['--algo', 'invalid_algo', '--data', '10']
        
        # Act & Assert:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(test_args)
    
    def test_parser_invalid_nn_mode(self):
        # Arrange:
        test_args = ['--algo', 'nn', '--data', '1', '--nn_mode', 'val']
        
        # Act & Assert:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(test_args)

    def test_parser_invalid_tfidf_mode_int(self):
        # Arrange:
        test_args = ['--algo', 'tfidf', '--data', '1', '--tfidf_mode', 'count']
        
        # Act & Assert:
        with self.assertRaises(SystemExit):
            self.parser.parse_args(test_args)


class TestGetParserArgs(unittest.TestCase):

    def test_get_parser_args_default(self):
        # Arrange:
        parser = init_parser()
        test_args = ['main.py', '--algo', 'nn', '--data', '3']
        
        # Act: Use patch to replace sys.argv with our mock_argv
        with patch('sys.argv', test_args):
            result = get_parser_args(parser)
        
        # Assert:
        self.assertIsInstance(result, dict)
        self.assertEqual(result['algo'], 'nn')
        self.assertEqual(result['data'], 3)
        self.assertEqual(result['nn_mode'], 'train')
        self.assertEqual(result['tfidf_mode'], 'tf')

    def test_get_parser_args_custom(self):
        # Arrange:
        parser = init_parser()
        test_args = ['main.py', '--algo', 'tfidf', '--data', '2', '--nn_mode', 'test', '--tfidf_mode', 'bow']
        
        # Act: Use patch to replace sys.argv with our mock_argv
        with patch('sys.argv', test_args):
            result = get_parser_args(parser)
        
        # Assert:
        self.assertIsInstance(result, dict)
        self.assertEqual(result['algo'], 'tfidf')
        self.assertEqual(result['data'], 2)
        self.assertEqual(result['nn_mode'], 'test')
        self.assertEqual(result['tfidf_mode'], 'bow')


class TestParserDataToPath(unittest.TestCase):

    def test_data_mapping_1(self):
        # Arrange
        args_dict = {'data': 1}
        expected_path = os.path.join('data_svd', 'jester_data_1_svd.csv')
        
        # Act
        parser_data_to_path(args_dict)
        
        # Assert
        self.assertEqual(args_dict['data'], expected_path)

    def test_data_mapping_2(self):
        # Arrange
        args_dict = {'data': 2}
        expected_path = os.path.join('data_svd', 'jester_data_2_svd.csv')
        
        # Act
        parser_data_to_path(args_dict)
        
        # Assert
        self.assertEqual(args_dict['data'], expected_path)

    def test_data_mapping_3(self):
        # Arrange
        args_dict = {'data': 3}
        expected_path = os.path.join('data_svd', 'jester_data_comb_svd.csv')
        
        # Act
        parser_data_to_path(args_dict)
        
        # Assert
        self.assertEqual(args_dict['data'], expected_path)


class TestExecuteAlgo(unittest.TestCase):

    @patch('main.tfidf_calcs')
    def test_execute_algo_tfidf(self, mock_tfidf):
        # Arrange
        args = {'algo': 'tfidf', 'data': 'some/path.csv', 'tfidf_mode': 'bow'}
        
        # Act
        execute_algo(args)
        
        # Assert:
        mock_tfidf.assert_called_once_with(args)

    @patch('main.svd_calcs')
    def test_execute_algo_svd(self, mock_svd):
        # Arrange
        args = {'algo': 'svd', 'data': 'some/path.csv'}
        
        # Act
        execute_algo(args)
        
        # Assert
        mock_svd.assert_called_once_with(args)

    @patch('main.net_calcs')
    def test_execute_algo_nn(self, mock_nn):
        # Arrange
        args = {'algo': 'nn', 'data': 'some/path.csv', 'nn_mode': 'test'}
        
        # Act
        execute_algo(args)
        
        # Assert
        mock_nn.assert_called_once_with(args)


class TestPipeline(unittest.TestCase):

    @patch('main.parser_data_to_path')
    @patch('main.get_parser_args')
    @patch('main.execute_algo')
    @patch('main.init_parser')
    def test_run_main_pipeline(self, mock_init, mock_execute, mock_get_args, mock_data_path):
        # Arrange:
        mock_parser = MagicMock()
        mock_init.return_value = mock_parser
        dummy_args = {'algo': 'tfidf', 'data': 2, 'tfidf_mode': 'bow'}
        mock_get_args.return_value = dummy_args
        
        # Act: Run the pipeline
        run_main()
        
        # Assert:
        mock_init.assert_called_once()
        mock_execute.assert_called_once_with(dummy_args)
        mock_get_args.assert_called_once_with(mock_parser)
        mock_data_path.assert_called_once_with(dummy_args)

