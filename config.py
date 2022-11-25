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
        self.parser.add_argument('--batch-size',
                                 default=64,
                                 type=int,
                                 help='batch size')
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

        self.parser.add_argument('--lr',
                                 default=0.001,
                                 type=float,
                                 help='learning rate')
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
                                 help='dimension of the model'
                                 ' (works only when using patch as backbone)')
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
                                 help='patch size of the model'
                                 ' (works only when using patch as backbone)')
        self.parser.add_argument('--backbone',
                                 default='patch',
                                 type=str,
                                 help='backbone of the model')
        self.parser.add_argument('--pretrained',
                                 action='store_true',
                                 help='using pretrained backbone')
        self.parser.add_argument(
            '--only-fc',
            action='store_true',
            help='only use fully connected layer after backbone.')
        self.parser.add_argument('--freeze-backbone',
                                 action='store_true',
                                 help='freeze backbone in training '
                                 '(works only when --only-fc is disabled)')
        self.parser.add_argument(
            '--backbone-lr',
            default=None,
            type=float,
            help='separate learning rate for backbone'
            ' (works only when --freeze-backbone and --only-fc are disabled)')
        self.parser.add_argument('--cabnet',
                                 action='store_true',
                                 help='use CabNet')
        self.parser.add_argument('--cabnet-k',
                                 default=5,
                                 type=int,
                                 help='k of CabNet')

    def configurate(self):
        args = self.parser.parse_args()
        config = vars(args)
        return config


if __name__ == '__main__':
    config = Config()
    metadata = config.configurate()
    print(metadata)
