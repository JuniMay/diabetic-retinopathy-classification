import argparse


class Config:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description='DDR Classifier')
        self.parser.add_argument('--timestamp',
                                 default=None,
                                 type=str,
                                 help='timestamp of current task')
        self.parser.add_argument('--valid',
                                 action='store_true',
                                 help='switch to valid mode')
        self.parser.add_argument('--num-classes',
                                 default=5,
                                 type=int,
                                 help='number of classes')
        self.parser.add_argument('--num-workers',
                                 default=4,
                                 type=int,
                                 help='number of workers for dataloader')
        self.parser.add_argument('--num-epochs',
                                 default=20,
                                 type=int,
                                 help='number of epochs')
        self.parser.add_argument(
            '--num-local-epochs',
            default=5,
            type=int,
            help='number of epochs in each local iteration(for clustering)')
        self.parser.add_argument('--batch-size',
                                 default=1024,
                                 type=int,
                                 help='batch size')
        self.parser.add_argument(
            '--local-batch-size',
            default=128,
            type=int,
            help='batch size of local iterations(for clustering)')
        self.parser.add_argument('--dataset-dir',
                                 default='./data/DDR',
                                 type=str,
                                 help='directory of dataset')
        self.parser.add_argument('--train-data-dir',
                                 default=None,
                                 type=str,
                                 help='directory of train data')
        self.parser.add_argument('--valid-data-dir',
                                 default=None,
                                 type=str,
                                 help='directory of valid data')
        self.parser.add_argument(
            '--log-dir',
            default='./log',
            type=str,
            help='directory to store logs and checkpoints')
        self.parser.add_argument('--no-cuda',
                                 action='store_true',
                                 help='do not use cuda')
        self.parser.add_argument(
            '--method',
            default='classification',
            type=str,
            help='mathod used for training and validating')

        self.parser.add_argument('--lr',
                                 default=0.001,
                                 type=float,
                                 help='learning rate')
        self.parser.add_argument('--gamma',
                                 default=0.9,
                                 type=float,
                                 help='gamma for learning rate scheduler')
        self.parser.add_argument('--label-smoothing',
                                 default=0.3,
                                 type=float,
                                 help='label smoothing')
        self.parser.add_argument('--input-size',
                                 default=512,
                                 type=int,
                                 help='image size to input')
        self.parser.add_argument('--dim',
                                 default=128,
                                 type=int,
                                 help='dimension of the model')
        self.parser.add_argument('--depth',
                                 default=8,
                                 type=int,
                                 help='depth of the model')
        self.parser.add_argument('--kernel-size',
                                 default=9,
                                 type=int,
                                 help='kernel size of the model')
        self.parser.add_argument('--patch-size',
                                 default=8,
                                 type=int,
                                 help='patch size of the model')
        self.parser.add_argument('--drop',
                                 default=0.2,
                                 type=float,
                                 help='dropout rate of the model')

        self.config = dict()
        self.args = None

    def configurate(self):
        self.args = self.parser.parse_args()
        self.config = vars(self.args)
        return self.config


if __name__ == '__main__':
    config = Config()
    metadata = config.configurate()
    print(metadata)
