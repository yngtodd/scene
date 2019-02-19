import argparse


def parse_args():
    """
    Parse Arguments for our LSTM example.
    Returns:
    -------
    * `args`: [argparse object]
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Highway ConvNet!')
    parser.add_argument('-d','--datapath', metavar='DIR',
                        default='/home/ygx/kaggle/scene/data/splits/small_val/csv',
                        help='path to dataset')
    parser.add_argument('--savepath', type=str,
                        default='/home/ygx/kaggle/scene/saves',
                        help='path to save checkpoints')
    parser.add_argument('--serialization_dir', type=str,
                        default='/home/ygx/kaggle/scene/examples/convolutional_net/saves',
                        help='path to save serialized models')
    parser.add_argument('--options_file', type=str,
                        default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/' \
                                '2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json',
                        help='Elmo embeddings option file')
    parser.add_argument('--weight_file', type=str,
                        default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/' \
                                '2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',
                        help='Elmo embedding weights')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--num_epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args
