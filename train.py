import pytorch_lightning as pl
from argparse import ArgumentParser
from utils.mvpDataset import MvpDataModule
from utils.train_utils import weights_init, setup_seed
from model.SCNet import SCNet

def main(args):
    setup_seed(0)
    data = MvpDataModule(npoints=args.num_points, batch_size=args.batchSize, novel_input=True)
    model = SCNet(num_coarse=args.num_coarse, num_dense=args.num_points, lrate=args.lrate)
    model.apply(weights_init)

    trainer = pl.Trainer(gpus=args.gpus,
                         logger= not args.debug,
                         fast_dev_run=args.debug,
                         terminate_on_nan=True,
                         max_epochs=args.epochs,
                         accumulate_grad_batches= args.accumulate_batches,
                         check_val_every_n_epoch = 10,
                         accelerator='ddp' if args.gpus >1 else None,
                         )
    trainer.fit(model, datamodule=data)
    trainer.save_checkpoint('logs/SCNet.ckpt')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1, help='gpus')
    parser.add_argument('--lrate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='train epoch')
    parser.add_argument('--batchSize',type=int, default=32,help='batch size')
    parser.add_argument('--num_coarse', type=int, default=1024, help='the number of coarse point cloud')
    parser.add_argument('--num_points', type=int, default=2048,help='the number of input and output')
    parser.add_argument('--accumulate_batches', type=int, default=1, help='accumulate_grad_batches size')
    parser.add_argument('--debug', type=bool, default=False, help='fast_dev_run')
    args = parser.parse_args()

    main(args)
