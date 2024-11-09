import torch
import torch.nn as nn
from math import floor
from PIL import Image
import open_clip
from torchvision import transforms
import os
import copy
import time
from torch.utils.data import DataLoader, Dataset
from models.resnet_custom import resnet50_baseline
from models.model_adapter import LinearAdapter
from models.model_backbone import resnet50_baseline, biomedCLIP_backbone
import argparse
from PIL import Image
from transformers import CLIPModel,CLIPProcessor
import glob
from transformers import AutoImageProcessor, ViTMAEModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# patchs in one bag 
class Whole_Slide_Patchs_Ngc(Dataset):
    def __init__(self,
                 wsi_path,
                 target_patch_size,
                 preprocess,
                 from_pilp=False):
        # for resnet50
        self.preprocess = transforms.Compose([
            transforms.Resize(target_patch_size),
            transforms.ToTensor(),
        ])
        self.from_plip = from_pilp
        # for biomedclip
        if preprocess != None:
            self.preprocess = preprocess
        self.wsi_path = wsi_path
        patch_files = glob.glob(os.path.join(wsi_path, '*.jpg')) + glob.glob(os.path.join(wsi_path, '*.png'))
        # sort to ensure reproducibility
        try:
            self.patch_files = sorted(patch_files, key=lambda x: (int(os.path.basename(x).split(".")[0].split("_")[0]), 
        int(os.path.basename(x).split(".")[0].split("_")[1])))
        except Exception as e:
            print(e)
            
    def __getitem__(self, idx):
        img = Image.open(self.patch_files[idx])
        if not self.from_plip:
            img = self.preprocess(img)
        else:
            img = self.preprocess(images=img, return_tensors='pt').data['pixel_values'].squeeze(0)
        return img, self.patch_files[idx]
    
    def __len__(self):
        return len(self.patch_files)

    def __str__(self) -> str:
        return f'the length of patchs in {self.wsi_path} is {self.__len__()}'

def compute_w_loader(wsi_dir, 
                      output_dir,
                      model,
                      preprocess_val, # for biomedclip pretrain
                      args):
    # set parameters
    batch_size = args.batch_size
    target_patch_size = args.target_patch_size
    num_workers = args.num_workers
    verbose = args.verbose
    print_every = args.print_every
    if args.base_model == 'plip_vit-b/32':
        dataset = Whole_Slide_Patchs_Ngc(wsi_dir, target_patch_size, preprocess_val, from_plip=True)
    else:
        dataset = Whole_Slide_Patchs_Ngc(wsi_dir, target_patch_size, preprocess_val)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if verbose > 0:
        print('processing {}: total of {} batches'.format(wsi_dir,len(loader)))
    
    
    base_dir = os.path.dirname(os.path.dirname(wsi_dir)) #only for TCT-gc dataset
    sufix = '.jpg'

    for i, (batch_imgs, batch_names)in enumerate(loader):
        with torch.no_grad():
            if i % print_every == 0:
                print('batch {}/{}, {} files processed'.format(i, len(loader), i * batch_size))
            batch_imgs = batch_imgs.to(device)
            if args.base_model == 'clip_vit-b/16':
                features = model.encode_image(batch_imgs)
            elif args.base_model == 'plip_vit-b/32':
                #batch = batch.squeeze(1)
                features = model.vision_model(batch_imgs)[1]
                features = model.visual_projection(features)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
            elif args.base_model == 'mae_vit-b':
                features = torch.mean(model(batch_imgs).last_hidden_state, dim=1)
            else:
                features = model(batch_imgs)
            if isinstance(features, tuple):
                features = features[0]
            features = features.cpu()
            assert features.shape[0] == len(batch_names), f'Error, length dose not match!'
            for i in range(len(batch_names)):
                img_save_path = (output_dir + '/' + os.path.relpath(batch_names[i], base_dir)).replace(sufix, '.pt')
                if os.path.exists(img_save_path):
                    continue
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                torch.save(features[i], img_save_path)
    return 
    
def main():
    # set argsrget_patch
    parser = argparse.ArgumentParser(description='GC dataset Feature Extraction')
    parser.add_argument('--dataset', type=str, default='gc', choices=['ngc', 'ubc', 'gc'])
    parser.add_argument('--wsi_root', type=str, default='/home1/lgj/TCT_smear_lgj')
    parser.add_argument('--output_dir', type=str, default='/home1/lgj/TCT_2625/patch_extracted/dinov2_vit-b-14_768')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=20)
    # inference options 
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=16)  
    parser.add_argument('--num_workers', type=int, default=16)  
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--target_patch_size', type=int, nargs='+', default=(224, 224))
    # model options
    parser.add_argument('--base_model', default='dinov2_vit-b/14', type=str, choices=['biomedclip', 'resnet50', 'clip_vit-b/16', 'plip_vit-b/32', 'dinov2_vit-b/14', 'mae_vit-b'])
    parser.add_argument('--with_adapter', action='store_true')
    parser.add_argument('--ckp_path', type=str, default=None)
    parser.add_argument('--without_head', action='store_true')
    args = parser.parse_args()
    
    if args.multi_gpu:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
    
    # get wsi paths
    wsi_root = args.wsi_root
    if args.dataset == 'ngc':
        sub_paths = [
            'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-NILM',
            'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-POS',
            'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-NILM',
            'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-POS'
        ]
        data_roots = list(map(lambda x: os.path.join(wsi_root, x), sub_paths)) 
        wsi_dirs = []

        for data_root in data_roots:
            wsi_dirs.extend([os.path.join(data_root, subdir) for subdir in os.listdir(data_root)])
    
    elif args.dataset == 'gc':
        sub_paths = [
            'NILM',
            'POS'
        ]
        wsi_dirs = []
        for sub_path in sub_paths:
            wsi_dirs.extend([os.path.join(wsi_root, sub_path, wsi_name) for wsi_name in os.listdir(os.path.join(wsi_root, sub_path))])
            
    elif args.dataset == 'ubc':
        wsi_dirs = [os.path.join(wsi_root, subdir) for subdir in os.listdir(wsi_root)]
    
    # get output path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # load model
    torch.cuda.set_device(args.local_rank)
    print('loading model')
    preprocess_val = None
    if args.base_model == 'resnet50':
        backbone = resnet50_baseline(pretrained=True)
        input_dim = 1024
    elif args.base_model == 'biomedclip':
        if args.without_head:
            backbone, preprocess_val = biomedCLIP_backbone(without_head=True)
        else:
            backbone, preprocess_val = biomedCLIP_backbone()
        input_dim = 512
    elif args.base_model == 'clip_vit-b/16':
        backbone, _ ,preprocess_val = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        input_dim = 512
    elif args.base_model == 'plip_vit-b/32':
        backbone = CLIPModel.from_pretrained("vinid/plip")
        preprocess_val = CLIPProcessor.from_pretrained("vinid/plip")
        input_dim = 512
    elif args.base_model == 'dinov2_vit-b/14':
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        backbone.head = nn.Identity()
        imput_dim = 768
    elif args.base_model == 'mae_vit-b':
        backbone = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        #preprocess_val = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        input_dim = 768
    print('load backbone successfully')
    
    if args.base_model in ['clip_vit-b/16', 'plip_vit-b/32', 'mae_vit-b']:
        model = backbone.to(device)
    else:
        model = nn.Sequential(backbone).to(device)

    model.eval()
    total = len(wsi_dirs)    

    for idx in range(total):
        if idx % args.world_size != args.local_rank:
            continue
        
        wsi_dir = wsi_dirs[idx]
        wsi_name = os.path.basename(wsi_dir)
        print('\nprogress: {}/{}'.format(idx, total))
        print(wsi_name)
        time_start = time.time()
        compute_w_loader(wsi_dir,
                         output_dir,
                         model,
                         preprocess_val,
                         args,
                        )
        time_elapesd = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(wsi_name, time_elapesd))
        
        
if __name__ == '__main__':
    time_start = time.time()
    
    main()
    
    time_end = time.time()
    time_elapesd = time_end - time_start
    print('\n The program took {} h {} min {} s'.format(time_elapesd//3600,
                                                    time_elapesd%3600//60,
                                                    time_elapesd%60))