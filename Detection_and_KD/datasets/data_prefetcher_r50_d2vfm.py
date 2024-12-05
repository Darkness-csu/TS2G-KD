# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

def to_cuda(samples, feats, device):
    #print(type(samples), type(feats), type(targets))
    #print(len())
    samples = samples.to(device, non_blocking=True)
    feats = [feat.to(device, non_blocking=True) for feat in feats]
    
    return samples, feats

class data_prefetcher_r50_d2vfm():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_feats = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_feats = None
            return
        
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_feats = to_cuda(self.next_samples, self.next_feats, self.device)

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            feats = self.next_feats
            
            if samples is not None:
                samples.record_stream(torch.cuda.current_stream())
            if feats is not None:
                for feat in feats:
                    feat.record_stream(torch.cuda.current_stream())
            
            self.preload()
        else:
            try:
                samples, feats = next(self.loader)
                samples, feats = to_cuda(samples, feats, self.device)
            except StopIteration:
                samples = None
                feats = None

        return samples, feats
