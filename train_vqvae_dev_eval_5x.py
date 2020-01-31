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
import os
from glob import glob
sys.path.append('/home/nyman/')
from tmb_bot import utilities
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('/home/nyman/histopath-analysis')
from generic_vae import pca_simple_vis


def train(epoch, loader, model, optimizer, scheduler, device, log):
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
                f'train; '
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

            model.train()

    # TODO dev_loader eval code [at end of each epoch] 
    # TODO perhaps just make this a separate function to call...
    train_mse_mean_epoch = mse_sum / mse_n
    training_log.loc[(epoch, i),'train_loss'] = train_mse_mean_epoch


def evaluate_dataset(epoch, loader, model, device, log):
    """
    Just taking above train function and removing any gradient updates etc
    """
    model.eval()

    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0
    
    enc_t_track = []
    enc_b_track = []

    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            model.zero_grad()
            img = img.to(device)

            out, latent_loss, enc_t, enc_b = model(img)
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            # TODO possibly also add a classifier term here (would require adding module to model itself as well)
            loss = recon_loss + latent_loss_weight * latent_loss

            mse_sum += recon_loss.item() * img.shape[0]
            mse_n += img.shape[0]

            #enc_t_track.append(enc_t.detach().cpu())
            #enc_b_track.append(enc_b.detach().cpu())

            loader.set_description(
                (
                    f'eval; '
                    f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                    f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                )
            )
            if i % 100 == 0:
                
                sample = img[:sample_size]
                out = out[:sample_size].detach()

                utils.save_image(
                    torch.cat([sample, out], 0),
                    f'sample/eval_set_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )

                # visualize pre-quantized encodings
                # plot 2 component PCA vis of latent encodings
                
                # too large for now... 
                #temp_enc_b_agg = torch.cat(enc_b_track)
                #temp_enc_t_agg = torch.cat(enc_t_track)
                temp_enc_t_agg = enc_t.detach().cpu()
                temp_enc_b_agg = enc_b.detach().cpu()

                fig = pca_simple_vis(temp_enc_t_agg.view(enc_t.shape[0], -1), label) # unrolled entirely
                # @ TODO modify / repeat labels to match the latent field unrolled size (ie, 32x32 unrolled)
                # fig = pca_simple_vis(enc_t.detach().cpu().view(-1, enc_t.shape[1]), label) # each latent field point separately
                fig.savefig(f'sample/eval_set_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_enc_top_vis.png')
                plt.close()
                fig = pca_simple_vis(temp_enc_b_agg.view(enc_b.shape[0], -1), label)
                fig.savefig(f'sample/eval_set_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_enc_bot_vis.png')
                plt.close()

    model.train()
    eval_mse_mean_epoch = mse_sum / mse_n
    training_log.loc[(epoch, i),'eval_loss'] = eval_mse_mean_epoch




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)
     # set up logging
    checkpoint_dir = './checkpoint'
    sys.stderr = open(os.path.join(checkpoint_dir,'stderr.txt'), 'w')
    sys.stdout = open(os.path.join(checkpoint_dir,'stdout.txt'), 'w')

    device = 'cuda'

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    # TODO update if we do augmentation on train set...
    eval_transform = transform

    # 5x tiles 
    root_dir = '/mnt/disks/image_data2/5x/'
    paths = glob(os.path.join(root_dir, '*/*/*.jpeg'))
    ids = [x.split('/')[-3].split('_files')[0] for x in paths]
    paths_df = pd.DataFrame()
    paths_df['slide_id'] = ids
    paths_df['full_path'] = paths
    paths_df = paths_df.set_index('slide_id')

    NUM_TRAIN_SLIDES = 200
    NUM_DEV_SLIDES = 40
    #NUM_TRAIN_SLIDES = 100
    #NUM_DEV_SLIDES = 20
    # subset and only take a few slides
    subset_ids = pd.Series(paths_df.index.unique()).sample(NUM_TRAIN_SLIDES)
    
    # grab a set of slides to evaluate during training 
    dev_subset = paths_df.drop(subset_ids.values)
    dev_subset_ids = pd.Series(dev_subset.index.unique()).sample(NUM_DEV_SLIDES)
    
    subset_ids.to_csv('./20200127_train_slide_ids.csv')
    dev_subset_ids.to_csv('./20200127_dev_slide_ids.csv')
    dev_paths_df = paths_df.loc[dev_subset_ids]

    paths_df = paths_df.loc[subset_ids]
    print('Train Subset DF size: ', paths_df.shape)
    print('Dev Subset DF size: ', dev_paths_df.shape)

    # TODO add subsampling via controlling # tiles per slide sampled....

    dataset = utilities.Dataset(paths_df.full_path.values, paths_df.index.values, transform)
    # TODO give different transform to train and dev/eval datasets
    dev_dataset = utilities.Dataset(dev_paths_df.full_path.values, dev_paths_df.index.values, transform)
    BATCH_SIZE=140
    #BATCH_SIZE=50
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

    N_EMBED = 10
    print('Using K={} codebook size'.format(N_EMBED))
    model = nn.DataParallel(VQVAE(n_embed=N_EMBED)).to(device)
    #model = nn.DataParallel(VQVAE()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    training_log = pd.DataFrame(columns=[
        'epoch','batch','train_mse_loss','dev_mse_loss']).set_index(['epoch', 'batch'])


    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device, training_log)
        evaluate_dataset(i, dev_loader, model, device, training_log)
        torch.save(
            model.module.state_dict(), f'checkpoint/vqvae_{str(i + 1).zfill(3)}.pt'
        )
        training_log.to_csv(f'checkpoint/training_log_vqvae_{str(i + 1).zfill(3)}.csv')

