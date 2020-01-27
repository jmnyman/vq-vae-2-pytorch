import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler

# my additions
import sys
sys.path.append('/home/nyman/')
from tmb_bot import utilities
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('/home/nyman/histopath-analysis')
from generic_vae import pca_simple_vis


# TODO add dev_loader param and chunk to loop code 
def train(epoch, loader, dev_loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss, enc_t, enc_b = model(img)
        #print('enc_t ', enc_t.shape)
        #print('enc_b ', enc_b.shape)
        
        #out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        # TODO possibly also add a classifier term here (would require adding module to model itself as well)
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        if i % 100 == 0:
            model.eval()
            
            sample = img[:sample_size]

            #with torch.no_grad():
            #    out, _ = model(sample)
            
            # my edit to visualize PRE gradient step decoding
            out = out[:sample_size].detach()

            utils.save_image(
                torch.cat([sample, out], 0),
                f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            # visualize pre-quantized encodings
            # plot 2 component PCA vis of latent encodings
            fig = pca_simple_vis(enc_t.detach().cpu().view(enc_t.shape[0], -1), label) # unrolled entirely
            # @ TODO modify / repeat labels to match the latent field unrolled size (ie, 32x32 unrolled)
            # fig = pca_simple_vis(enc_t.detach().cpu().view(-1, enc_t.shape[1]), label) # each latent field point separately
            fig.savefig(f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_enc_top_vis.png')
            plt.close()
            fig = pca_simple_vis(enc_b.detach().cpu().view(enc_b.shape[0], -1), label)
            fig.savefig(f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_enc_bot_vis.png')
            plt.close()

            # TODO dev_loader eval code 
            

            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    # manual hardcoding
    paths_df = pd.read_pickle('/home/nyman/20191025_dfci_full_rcc_paths_df_annotated_reexport.pkl') # tried exporting above in `rapids` conda env
    partial_indicator = pd.read_csv('/home/nyman/20191025_dfci_full_rcc_paths_df_annotated_jupyterexport_PARTIAL_TILE_INDICATOR.csv')
    # subset out tiles that aren't 512x512 
    paths_df = paths_df.loc[partial_indicator['is_512'].values]
    # subset and only take a few slides
    subset_ids = pd.Series(paths_df.index.unique()).sample(20)
    
    # grab a set of slides to evaluate during training 
    dev_subset = paths_df.drop(subset_ids.values)
    dev_subset_ids = pd.Series(dev_subset.index.unique()).sample(20)
    
    subset_ids.to_csv('./20200127_train_slide_ids.csv')
    dev_subset_ids.to_csv('./20200127_dev_slide_ids.csv')

    paths_df = paths_df.loc[subset_ids]
    print('Subset DF size: ', paths_df.shape)

    dataset = utilities.Dataset(paths_df.full_path.values, paths_df.index.values, transform)

    #dataset = datasets.ImageFolder(args.path, transform=transform)
    #loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=12)

    N_EMBED = 512
    print('Using K={} codebook size'.format(N_EMBED))
    model = nn.DataParallel(VQVAE(n_embed=N_EMBED)).to(device)
    #model = nn.DataParallel(VQVAE()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)
        torch.save(
            model.module.state_dict(), f'checkpoint/vqvae_{str(i + 1).zfill(3)}.pt'
        )
