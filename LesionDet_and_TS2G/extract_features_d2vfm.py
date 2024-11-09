import torch
from PIL import Image
import numpy as np
import os
import time
from torch.utils.data import DataLoader, Dataset
from datasets import transforms as T
import argparse
from models.r50_d2vfm import R50_D2VFM
from models.backbone import build_backbone
from util.file_utils import save_hdf5
from PIL import Image
import h5py
import glob


#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# patchs in one bag 
class Whole_Slide_Patchs(Dataset):
    def __init__(self,
                 wsi_path):
        # for resnet50
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.preprocess = T.Compose([
            #T.RandomResize([800], max_size=1333),
            normalize,
        ])
        
        self.wsi_path = wsi_path
        patch_files = glob.glob(os.path.join(wsi_path, '*.jpg')) + glob.glob(os.path.join(wsi_path, '*.png'))
        # sort to ensure reproducibility
        try:
            self.patch_files = sorted(patch_files, key=lambda x: (int(os.path.basename(x).split(".")[0].split("_")[0]), 
        int(os.path.basename(x).split(".")[0].split("_")[1])))
        except Exception as e:
            print(e)
            
    def __getitem__(self, idx):
        img = Image.open(self.patch_files[idx]).convert('RGB')
        img, _ = self.preprocess(img, None)
        return img
    
    def __len__(self):
        return len(self.patch_files)

    def __str__(self) -> str:
        return f'the length of patchs in {self.wsi_path} is {self.__len__()}'

def compute_w_loader(wsi_dir, 
                      output_path, 
                      model,
                      args):
    # set parameters
    batch_size = args.batch_size
    num_workers = args.num_workers
    verbose = args.verbose
    print_every = args.print_every
    dataset = Whole_Slide_Patchs(wsi_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if verbose > 0:
        print('processing {}: total of {} batches'.format(wsi_dir,len(loader)))
    
    mode = 'w'
    for i, batch in enumerate(loader):
        with torch.no_grad():
            if i % print_every == 0:
                print('batch {}/{}, {} files processed'.format(i, len(loader), i * batch_size))
            # print(batch.shape, type(batch), type(batch[0]))
            # exit()
            batch = batch.to(args.device)
            features = model.extract_adapted_feat(batch)
            if isinstance(features, tuple):
                features = features[0]
            features = features.cpu().numpy()
            asset_dict = {'features': features}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'
    return output_path
    
def main():
    # set argsrget_patch
    parser = argparse.ArgumentParser(description='TCT D2VFM Feature Extraction')
    parser.add_argument('--wsi_root', type=str, default='/home1/ligaojie/TCT_smear_lgj')
    parser.add_argument('--output_dir', type=str, default='/home1/ligaojie/gc-all-features/distill')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=20)
    # inference options 
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=12)  
    parser.add_argument('--num_workers', type=int, default=16)  
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    # model options
    parser.add_argument('--ckp_dir', type=str, default='/home/ligaojie/LungCancer/Deformable-DETR-main/exps/tct_d2vfm/biomedclip_MSE_gmpa_ep4')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # * Backbone
    parser.add_argument('--lr_backbone', default=0, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    

    #segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    #new
     #adapter adapter_type
    parser.add_argument('--needle', default=1, type=int)
    

    args = parser.parse_args()
    
    if args.multi_gpu:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
    
    # get wsi paths
    wsi_root = args.wsi_root
    sub_paths = [
        'NILM',
        'POS'
    ]
    wsi_dirs = []
    for sub_path in sub_paths:
        wsi_dirs.extend([os.path.join(wsi_root, sub_path, wsi_name) for wsi_name in os.listdir(os.path.join(wsi_root, sub_path))])
            
    # get output path
    output_dir = args.output_dir
    if not args.ckp_dir or not os.path.exists(args.ckp_dir):
        print('Error! check_point should not be none and path should be right.')
        exit()
    ckp_dirname = os.path.basename(args.ckp_dir).split('_')#biomedclip_MSE_gmpa_ep4
    
    distill_dataset = ckp_dirname[0]
    if distill_dataset in ['biomedclip', 'dinov2', 'mae']:
        out_feat_dim = 768
    elif distill_dataset in ['clip', 'plip']:
        out_feat_dim = 512
    elif distill_dataset == 'gigapath':
        out_feat_dim = 1536
    else:
        raise ValueError(f'Invalid value for arg distill_dataset:{distill_dataset}')
    
    adapter_type = ckp_dirname[2][1:]
    epoch_num = int(ckp_dirname[-1][2:])
    
    ckp_path = os.path.join(args.ckp_dir, f'checkpoint000{epoch_num-1}.pth' if epoch_num-1 < 10 else f'checkpoint00{epoch_num-1}.pth')
    output_path = os.path.join(output_dir, os.path.basename(args.ckp_dir))
    output_path_pt = os.path.join(output_path, 'pt')
    output_path_h5 = os.path.join(output_path, 'h5_files')
    os.makedirs(output_path_pt, exist_ok=True)
    os.makedirs(output_path_h5, exist_ok=True)
    dest_files = os.listdir(output_path_pt)
    
    # build model
    torch.cuda.set_device(args.local_rank)
    print('building model.....')
    backbone = build_backbone(args)
    model = R50_D2VFM(
        backbone,
        num_feature_levels=args.num_feature_levels,
        needle=args.needle,
        out_feat_dim = out_feat_dim,
        adapter_type = adapter_type
    )
    print('building model successfully')
    #load model
    print('loading pretrained weights.....')
    
    checkpoint = torch.load(ckp_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    print('Successfully loading pretrained weights.')

    model.to(args.device)
    model.eval()

    total = len(wsi_dirs)    

    for idx in range(total):
        if idx % args.world_size != args.local_rank:
            continue
        
        wsi_dir = wsi_dirs[idx]
        wsi_name = os.path.basename(wsi_dir)
        print('\nprogress: {}/{}'.format(idx, total))
        print(wsi_name)
        
        if wsi_name+'.pt' in dest_files:
            print('skipped {}'.format(wsi_name))
            continue
        
        output_file_path = os.path.join(output_path_h5, wsi_name+'.h5')
        time_start = time.time()
        compute_w_loader(wsi_dir,
                         output_file_path,
                         model,
                         args,
                        )
        time_elapesd = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_path, time_elapesd))
        
        file = h5py.File(output_file_path, 'r')
        
        features = file['features'][:]
        print('features size: ', features.shape)
        features = torch.from_numpy(features)
        torch.save(features, os.path.join(output_path_pt, wsi_name+'.pt'))

if __name__ == '__main__':
    time_start = time.time()
    
    main()
    
    time_end = time.time()
    time_elapesd = time_end - time_start
    print('\n The program took {} h {} min {} s'.format(time_elapesd//3600,
                                                    time_elapesd%3600//60,
                                                    time_elapesd%60))