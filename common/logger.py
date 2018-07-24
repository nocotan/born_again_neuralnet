# -*- coding: utf-8 -*-


class Logger(object):
    def __init__(self, args):
        self.args = args

    def print_args(self):
        print("weight: ", self.args.weight)
        print("lr: ", self.args.lr)
        print("n_epoch: ", self.n_epoch)
        print("batch_size: ", self.batch_size)
        print("n_gen: ", self.n_gen)
        print("dataset: ", self.dataset)
        print("outdir: ", self.outdir)
        print("print_interval: ", self.print_interval)

    def print_log(self, epoch, it, train_loss, val_loss):
        print("epoch: {}, iter: {}, train_loss: {}, val_loss: {}".format(
            epoch, it, train_loss, val_loss,
        ))
