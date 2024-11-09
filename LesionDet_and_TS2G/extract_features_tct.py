import torch
from PIL import Image
import numpy as np
import os
import time
from torch.utils.data import DataLoader, Dataset
from datasets import transforms as T
import argparse
from models.deformable_detr_IF import DeformableDETR_IF
from models.deformable_detr_IF_v2 import DeformableDETR_IF_V2
from models.deformable_transformer import build_deforamble_transformer
from models.backbone import build_backbone
from util import misc as utils
from util.file_utils import save_hdf5
from PIL import Image
import h5py
import glob

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
            T.RandomResize([800], max_size=1333),
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
            batch = batch.to(device)
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
    parser = argparse.ArgumentParser(description='TCT dataset Feature Extraction')
    parser.add_argument('--dataset', type=str, default='tct', choices=['tct'])
    parser.add_argument('--wsi_root', type=str, default='/home1/lgj/TCT_smear_lgj')
    parser.add_argument('--output_path', type=str, default='/home1/lgj/TCT_2625/deformable_adapter_extracted_cosine/biomed')
    parser.add_argument('--feat_dir', type=str, default='adapter_v2')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=20)
    # inference options 
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=12)  
    parser.add_argument('--num_workers', type=int, default=16)  
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    # model options
    parser.add_argument('--ckp_path', type=str, default='exps/tct_IF/r50_deformable_detr_cosine_240526/checkpoint0003.pth')

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
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    #segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    #new
    parser.add_argument('--adapter_out_dim', default=768, type=int)
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
    output_path = args.output_path
    output_path = os.path.join(output_path, args.feat_dir)
    output_path_pt = os.path.join(output_path, 'pt')
    output_path_h5 = os.path.join(output_path, 'h5_files')
    os.makedirs(output_path_pt, exist_ok=True)
    os.makedirs(output_path_h5, exist_ok=True)
    dest_files = os.listdir(output_path_pt)
    
    # build model
    torch.cuda.set_device(args.local_rank)
    print('building model.....')
    num_classes = 10 + 1 #tct
    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)
    model = DeformableDETR_IF(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=False,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        adapter_out_dim=args.adapter_out_dim,
    )
    print('building model successfully')
    #load model
    print('loading pretrained weights.....')
    if not args.ckp_path or not os.path.exists(args.ckp_path):
        print('Error! check_point should not be none and path should be right.')
        exit()
    checkpoint = torch.load(args.ckp_path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
    print('Successfully loading pretrained weights.')

    model.to(device)
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