from tqdm import tqdm
from argparse import ArgumentParser
import torch
from utils.model_utils import calc_cd,calc_emd
from utils.train_utils import setup_seed, AverageValueMeter
from torch.utils.data import DataLoader
from utils.mvpDataset import ShapeNetH5
from model.SCNet import SCNet

def load_model(args):
    logs = 'logs/SCNet.ckpt'

    model = SCNet(num_coarse=args.num_coarse, num_dense=args.num_points, lrate=args.lrate)
    model = model.load_from_checkpoint(logs, **dict(model.hparams))
    print('Successfully load:', logs)

    model.cuda()
    model.freeze()
    model.eval()
    return model

def main(args):
    model = load_model(args)
    setup_seed(0)
    dataset = ShapeNetH5(train=False, npoints=args.num_points, npose=1, novel_input=True, novel_input_only=False)
    dataloader = DataLoader(dataset, args.batchSize, shuffle=False, num_workers=8)

    cat_name = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel',
                'bed', 'bench', 'bookshelf', 'bus', 'guitar', 'motorbike', 'pistol', 'skateboard']
    loss = {i: [AverageValueMeter() for j in range(4)] for i in range(len(cat_name))}

    with tqdm(total=len(dataloader), desc=f"Processing ", leave=False) as pbar:
        with torch.no_grad():
            for x,labels,y in dataloader:
                pbar.update(1)
                fine = model(x.cuda())[-1]

                cd_p, cd_t, f1= calc_cd(fine, y.cuda(),calc_f1=True)
                emd = calc_emd(fine, y)
                for i in range(x.shape[0]):
                    k = int(labels[i])
                    loss[k][0].update(cd_p[i] * 10000)
                    loss[k][1].update(cd_t[i] * 10000)
                    loss[k][2].update(emd[i]  * 10000)
                    loss[k][3].update(f1[i])

    loss_avg = [AverageValueMeter() for j in range(4)]
    print('\r{:^10}\t{:^8}\t{:^8}\t{:^8}\t{:^8}'.format('Categorie', 'cd_p', 'cd_t', 'EMD', 'F1'))
    for i, name in enumerate(cat_name):
        loss_avg[0].update(loss[i][0].avg)
        loss_avg[1].update(loss[i][1].avg)
        loss_avg[2].update(loss[i][2].avg)
        loss_avg[3].update(loss[i][3].avg)
        print("\r{:^10}\t{:8.4f}\t{:8.4f}\t{:8.4f}\t{:8.4f}".format(name, loss[i][0].avg, loss[i][1].avg, loss[i][2].avg, loss[i][3].avg))
    print('\r{:^10}\t{:8.4f}\t{:8.4f}\t{:8.4f}\t{:8.4f}'.format("avg", loss_avg[0].avg, loss_avg[1].avg, loss_avg[2].avg, loss_avg[3].avg))

    for k, name in enumerate(cat_name):
        print("{:4.2f}".format(loss[k][1].avg), end=' & ')
    print("{:4.2f}".format(loss_avg[1].avg))

    for k, name in enumerate(cat_name):
        print("{:4.2f}".format(loss[k][2].avg/100), end=' & ')
    print("{:4.2f}".format(loss_avg[2].avg/100))

    for k, name in enumerate(cat_name):
        print("{:4.2f}".format(loss[k][3].avg), end=' & ')
    print("{:4.2f}".format(loss_avg[3].avg))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lrate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batchSize',type=int, default=32,help='batch size')
    parser.add_argument('--num_coarse', type=int, default=1024, help='the number of coarse point cloud')
    parser.add_argument('--num_points', type=int, default=2048,help='the number of input and output')
    args = parser.parse_args()

    main(args)