import argparse, os
import glob
from tqdm import tqdm
from multiprocessing import Pool

import torch
from torch.utils.data import DataLoader

from utils import *
from dataloader import *
from modules import mean_max

def copy_file(args):
    source, dest, feat, save_feat = args
    os.system('cp {} {}'.format(source, dest))
    if save_feat:
        dest = dest.replace('.jpg', '.pt')
        torch.save(feat.clone(), dest)

class WsiPathUtil():
    def __init__(self, wsi_root, output_root, num_parallel, is_ngc):
        NGC_SUB_PATHS = [
            'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-NILM',
            'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-POS',
            'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-NILM',
            'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-POS'
        ]
        GC_SUB_PATHS = [
            'NILM',
            'POS'
        ]
        if is_ngc:
            sub_paths = NGC_SUB_PATHS
        else:
            sub_paths = GC_SUB_PATHS    
        self.wsi_root = wsi_root
        self.output_root = output_root
        self.num_parallel = num_parallel
        self.wsi_dict = {}
        data_roots = list(map(lambda x: os.path.join(wsi_root, x), sub_paths)) 
        for data_root in data_roots: 
            for subdir in os.listdir(data_root):
                self.wsi_dict[subdir] = os.path.join(data_root, subdir)

    def saveSubimages(self, wsi_name, topk_dices, bag):
        wsi_path = self.wsi_dict[wsi_name]
        num_k = len(topk_dices)
        patch_files = glob.glob(os.path.join(wsi_path, '*.jpg'))
        patch_files = sorted(patch_files, key=lambda x: (int(os.path.basename(x).split(".")[0].split("_")[0]), 
        int(os.path.basename(x).split(".")[0].split("_")[1])))
        processes = []
        process_count = self.num_parallel  
        pool = Pool(process_count)
        args_list = []
        dest_wsi_path = os.path.join(self.output_root, 
                                     os.path.relpath(wsi_path, self.wsi_root))
        if os.path.exists(dest_wsi_path):
            if len(glob.glob(os.path.join(dest_wsi_path, '*.jpg'))) == num_k:
                return
            else:
                os.rmdir(dest_wsi_path)
        os.makedirs(dest_wsi_path)

        for dice in topk_dices:
            source_path = patch_files[dice]
            rel_path = os.path.relpath(source_path, self.wsi_root)
            dest_path = os.path.join(self.output_root, rel_path)
            save_feat = args.save_feat
            args_list.append((source_path, dest_path, bag[dice], save_feat))
        
        pool.map(copy_file, args_list)
        pool.close()
        pool.join()

def topk(tensor, k):
    if tensor.size()[0] < k:
        k = tensor.size()[0]
    values, indices = torch.topk(tensor, k, dim=0, largest=True, sorted=False)
    return indices

def main(args):
    # set seed
    seed_torch(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    topk_num = args.topk_num
    # get dataset
    wsi_root = args.wsi_root
    feature_root = args.feature_root
    train_label = args.train_label
    output_root = args.output_root
    num_parallel = args.num_parallel
    if args.datasets.lower() == 'ngc' or 'gc':
        dataset_p, dataset_l = [], []
        with open(train_label, 'r') as f:
            lines = f.readlines()
            for line in lines:
                dataset_p.append(line.split(',')[0])
                dataset_l.append(line.split(',')[1])
            f.close()
        dataset_p = np.array(dataset_p)
        dataset_l = np.array(dataset_l)
        dataset = GCDatasetInfer(dataset_p,dataset_l,feature_root)
        print(len(dataset))
        
    # build model
    model_type = args.model
    model_args = {
        'n_classes': args.n_classes,
        'dropout': args.dropout,
        'act': args.act,
        'input_dim': args.input_dim
    }
    if model_type == 'meanmil':
        model = mean_max.MeanMIL(test=True, **model_args).to(device)
    elif model_type == 'maxmil':
        model = mean_max.MaxMIL(test=True, **model_args).to(device)
    
    # load checkpoint
    ckp_path = args.ckp_path
    ckp = torch.load(ckp_path)
    if 'model' in ckp:
        ckp = ckp['model']
    model.load_state_dict(ckp)

    # inference
    is_ngc = False
    if args.datasets.lower() == 'ngc':
        is_ngc = True
    wsi_util = WsiPathUtil(wsi_root, output_root, num_parallel, is_ngc)
    loader_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
    }
    loader = DataLoader(dataset, shuffle=False, **loader_args)
    
    with torch.no_grad():
        for data in tqdm(loader):
            bag = data[0].to(device)
            label = data[1].to(device)
            wsi_name = data[2][0]
            test_logits = model(bag)
            test_logits = test_logits.squeeze()
            test_logits = test_logits.cpu()[:, 1]
            bag=bag[0].cpu()
            topk_dices = topk(test_logits, topk_num)
            wsi_util.saveSubimages(wsi_name=wsi_name, topk_dices=topk_dices, bag=bag)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter high risk patches for gc')
    
    # Dataset
    parser.add_argument('--datasets', default='gc', type=str, help='[camelyon16, gc]')
    parser.add_argument('--feature_root', default='', type=str, help='wsi feature root path')
    parser.add_argument('--train_label', default='',type=str, help='the path of train wsi list')
    parser.add_argument('--wsi_root', default='', type=str, help='Dataset root path')
    # Model
    parser.add_argument('--model', default='meanmil', type=str, help='[pure, mhim, meanmil, maxmil]')
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--input_dim', default=1024, type=int, help='The dimention of patch feature')
    # Inference
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--ckp_path', default='', type=str)
    # Output
    parser.add_argument('--output_root', default='', type=str, help='output path')
    parser.add_argument('--topk_num', default=50, type=int, help='topk_num')
    parser.add_argument('--save_feat', action='store_true', help='save feature or images')
    # parallel
    parser.add_argument('--num_parallel', default=16, type=int)
    
    args = parser.parse_args()
    main(args)
    