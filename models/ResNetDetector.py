import torch
from torch import nn

import torchvision

import numpy as np

def get_highpass(in_chan, out_chan):
    filter = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]).astype(np.float32)

    filter = filter.reshape((1, 1, 3, 3))
    filter = np.repeat(filter, in_chan, axis=1)
    filter = np.repeat(filter, out_chan, axis=0)

    filter = torch.from_numpy(filter)
    filter = nn.Parameter(filter, requires_grad=False)
    conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv.weight = filter
    
    network = nn.Sequential(
        conv,
        nn.Tanh() # we add this as the edge values are origianlly too close to zero
    )
    return network

class ResNetDetector(nn.Module):
    def __init__(self, highpass=True, crop_size=128, latent_dim=2048):
        super(ResNetDetector, self).__init__()

        self.highpass = highpass
        if (self.highpass):
            self.hp = get_highpass(1, 1)

        self.resnet_g = self.setup_resnet()
        self.resnet_r = self.setup_resnet()
        self.resnet_b = self.setup_resnet()

        fc_in_features = torchvision.models.resnet50(pretrained=False).fc.in_features

        self.final_g = nn.Sequential(
            nn.Linear(fc_in_features, latent_dim),
            nn.Dropout(0.5),
            nn.ReLU(inplace = True),
            nn.Linear(latent_dim, 1)
        )

        self.final_r = nn.Sequential(
            nn.Linear(fc_in_features, latent_dim),
            nn.Dropout(0.5),
            nn.ReLU(inplace = True),
            nn.Linear(latent_dim, 1)
        )

        self.final_b = nn.Sequential(
            nn.Linear(fc_in_features, latent_dim),
            nn.Dropout(0.5),
            nn.ReLU(inplace = True),
            nn.Linear(latent_dim, 1)
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(2)

    def setup_resnet(self):
        resnet = torchvision.models.resnet50(pretrained=False)

        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # remove the last layer of resnet (linear layer which is before avgpool layer)
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

        return resnet

    def forward_single(self, clean, noisy, pu, times, model):
        # high-pass filter
        if (self.highpass):
            clean = self.hp(clean)
            noisy = self.hp(noisy)

        # unshuffle
        for _ in range(times):
            clean = pu(clean) # B, C * 4, H // 2, W // 2
            noisy = pu(noisy) # B, C * 4, H // 2, W // 2

        clean = clean.permute(1, 0, 2, 3) # C * 4, B, H // 2, W // 2
        noisy = noisy.permute(1, 0, 2, 3) # C * 4, B, H // 2, W // 2

        feas = []

        for cp, np in zip(clean, noisy):
            fea = model(np.unsqueeze(1))
            fea = fea.view(fea.size()[0], -1)

            feas.append(fea)

        return feas

    def forward(self, clean, noisy):
        clean_r = clean[:, 0].unsqueeze(1)
        clean_g = clean[:, 1].unsqueeze(1)
        clean_b = clean[:, 2].unsqueeze(1)

        noisy_r = noisy[:, 0].unsqueeze(1)
        noisy_g = noisy[:, 1].unsqueeze(1)
        noisy_b = noisy[:, 2].unsqueeze(1)

        feas_g = self.forward_single(clean_g, noisy_g, self.pixel_unshuffle, 1, self.resnet_g)
        feas_r = self.forward_single(clean_r, noisy_r, self.pixel_unshuffle, 2, self.resnet_r)
        feas_b = self.forward_single(clean_b, noisy_b, self.pixel_unshuffle, 2, self.resnet_b)

        # concatenate both images' features
        g_i_indices = [0, 3] # diag
        g_o_indices = [1, 2] # anti-diag

        rb1_indices = [0, 1, 2, 3] # diag
        rb2_indices = [4, 5, 6, 7] # anti-diag
        rb3_indices = [8, 9, 10, 11] # anti-diag
        rb4_indices = [12, 13, 14, 15] # diag

        d_outs = []

        for ii in g_i_indices:
            for oi in g_o_indices:
                d_outs.append(self.final_g(torch.abs(feas_g[ii] - feas_g[oi])))

        for di in rb1_indices + rb4_indices:
            for adi in rb2_indices + rb3_indices:
                d_outs.append(self.final_r(torch.abs(feas_r[di] - feas_r[adi])))
                d_outs.append(self.final_b(torch.abs(feas_b[di] - feas_b[adi])))

        s_outs = []

        for i1 in range(len(g_i_indices)):
            for i2 in range(i1 + 1, len(g_i_indices)):
                s_outs.append(self.final_g(torch.abs(feas_g[g_i_indices[i1]] - feas_g[g_i_indices[i2]])))
        for o1 in range(len(g_o_indices)):
            for o2 in range(o1 + 1, len(g_o_indices)):
                s_outs.append(self.final_g(torch.abs(feas_g[g_o_indices[o1]] - feas_g[g_o_indices[o2]])))

        for l in [rb1_indices, rb2_indices, rb3_indices, rb4_indices]:
            for i1 in range(len(l)):
                for i2 in range(i1 + 1, len(l)):
                    s_outs.append(self.final_r(torch.abs(feas_r[l[i1]] - feas_r[l[i2]])))
                    s_outs.append(self.final_b(torch.abs(feas_b[l[i1]] - feas_b[l[i2]])))

        return d_outs, s_outs, \
            feas_r, feas_g, feas_b