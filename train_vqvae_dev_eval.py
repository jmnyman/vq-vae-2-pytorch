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
sys.path.append('/home/nyman/')
from tmb_bot import utilities
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append('/home/nyman/histopath-analysis')
#from generic_vae import pca_simple_vis
import datetime
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np 

def get_timestamp():
    timestamp = '_'.join(str(datetime.datetime.utcnow()).replace(
        ':', '.').replace('.', '-').split(' '))
    return timestamp

def lmplot_custom(x,y,data,hue,alpha=0.15,s=1, **kwargs):
    g = sns.lmplot(x=x, y=y, data=data, hue=hue, fit_reg=False, scatter_kws={'alpha':alpha, 's':s}, **kwargs)
    for lh in g._legend.legendHandles: 
        lh.set_alpha(1)
        lh._sizes = [50]
    return g

def pca_simple_vis(z_sample_store, label_store, alpha=0.3, legend=True):
    pc_scores = PCA(n_components=2).fit_transform(z_sample_store)
    score_df = pd.DataFrame(pc_scores)
    score_df.columns = ['PC' + str(x + 1) for x in score_df.columns]
    score_df['label'] = label_store
    g = sns.lmplot(x='PC1', y='PC2', data=score_df, hue='label', fit_reg=False, legend=legend,
                   scatter_kws={'alpha': alpha})
    for lh in g._legend.legendHandles: 
        lh.set_alpha(1)
        lh._sizes = [50]
        
    return g

def umap_simple_vis(umap_scores, label_store, alpha=0.3, legend=True, **kwargs):
    umap_embed_df = pd.DataFrame(umap_scores)
    umap_embed_df['label'] = label_store
    umap_embed_df = umap_embed_df.rename(columns={0: 'umap1', 1: 'umap2'})
    g = sns.lmplot(x='umap1', y='umap2', data=umap_embed_df, hue='label', fit_reg=False, scatter_kws={'alpha': alpha}, **kwargs)
    for lh in g._legend.legendHandles: 
        lh.set_alpha(1)
        lh._sizes = [50]
    
    return g

def train(epoch, loader, model, optimizer, scheduler, device, log, expt_dir,latent_loss_weight=0.25, classifier_loss_weight=0.001):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    #latent_loss_weight = 0.25
    #classifier_loss_weight = 0.001 # no idea what to put as for now
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss, enc_t, enc_b, classifier_loss = model(img, label)
        
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        classifier_loss = classifier_loss.mean() # multi GPU 

        # TODO possibly also add a classifier term here (would require adding module to model itself as well)
        loss = recon_loss + (latent_loss_weight * latent_loss) + (classifier_loss_weight * classifier_loss)
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
                f'classifier: {classifier_loss_weight * classifier_loss.item():.3f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )
        
        training_log.loc[(epoch,i),'train_loss'] = loss.item()
        training_log.loc[(epoch,i),'train_recon_loss'] = recon_loss.item()
        training_log.loc[(epoch,i),'train_classifier_loss'] = classifier_loss_weight * classifier_loss.item()
        training_log.loc[(epoch,i),'train_latent_loss'] = latent_loss_weight * latent_loss.item()


        if i % 100 == 0:
            model.eval()
            
            sample = img[:sample_size]

            #with torch.no_grad():
            #    out, _ = model(sample)
            
            # my edit to visualize PRE gradient step decoding
            out = out[:sample_size].detach()

            utils.save_image(
                torch.cat([sample, out], 0),
                os.path.join(expt_dir, f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png'),
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            # visualize pre-quantized encodings
            # plot 2 component PCA vis of latent encodings
            fig = pca_simple_vis(enc_t.detach().cpu().view(enc_t.shape[0], -1), label) # unrolled entirely
            # @ TODO modify / repeat labels to match the latent field unrolled size (ie, 32x32 unrolled)
            # fig = pca_simple_vis(enc_t.detach().cpu().view(-1, enc_t.shape[1]), label) # each latent field point separately
            fig.savefig(os.path.join(expt_dir, f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_enc_top_vis.png'))
            plt.close()
            fig = pca_simple_vis(enc_b.detach().cpu().view(enc_b.shape[0], -1), label)
            fig.savefig(os.path.join(expt_dir, f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_enc_bot_vis.png'))
            plt.close()

            model.train()

    # TODO dev_loader eval code [at end of each epoch] 
    # TODO perhaps just make this a separate function to call...
    #train_mse_mean_epoch = mse_sum / mse_n
    #training_log.loc[(epoch, i),'train_loss'] = train_mse_mean_epoch

def evaluate_dataset(epoch, loader, model, device, log, expt_dir, latent_loss_weight=0.25, classifier_loss_weight=0.001):
    """
    Just taking above train function and removing any gradient updates etc
    """
    model.eval()

    loader = tqdm(loader)

    criterion = nn.MSELoss()

    #latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0
    classifier_sum = 0 
   

    enc_t_track = []
    enc_b_track = []
    label_track = []

    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            model.zero_grad()
            img = img.to(device)

            out, latent_loss, enc_t, enc_b, classifier_loss = model(img, label)
            recon_loss = criterion(out, img)
            latent_loss = latent_loss.mean()
            classifier_loss = classifier_loss.mean()
            # TODO possibly also add a classifier term here (would require adding module to model itself as well)
            loss = recon_loss + (latent_loss_weight * latent_loss) + (classifier_loss_weight * classifier_loss)

            mse_sum += recon_loss.item() * img.shape[0]
            mse_n += img.shape[0]

            classifier_sum += (classifier_loss_weight * classifier_loss.item())

            #enc_t_track.append(enc_t.detach().cpu())
            #enc_b_track.append(enc_b.detach().cpu())
            #label_track.append(label)

            loader.set_description(
                (
                    f'eval; '
                    f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                    f'classifier: {classifier_loss_weight * classifier_loss.item():.3f}; '
                    f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                )
            )

            training_log.loc[(epoch,i),'eval_loss'] = loss.item()
            training_log.loc[(epoch,i),'eval_recon_loss'] = recon_loss.item()
            training_log.loc[(epoch,i),'eval_classifier_loss'] = classifier_loss_weight * classifier_loss.item()
            training_log.loc[(epoch,i),'eval_latent_loss'] = latent_loss_weight * latent_loss.item()

            if i % 100 == 0:
                
                sample = img[:sample_size]
                out = out[:sample_size].detach()

                utils.save_image(
                    torch.cat([sample, out], 0),
                    os.path.join(expt_dir, f'sample/eval_set_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png'),
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
                fig.savefig(os.path.join(expt_dir, f'sample/eval_set_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_enc_top_vis.png'))
                plt.close()
                fig = pca_simple_vis(temp_enc_b_agg.view(enc_b.shape[0], -1), label)
                fig.savefig(os.path.join(expt_dir, f'sample/eval_set_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_enc_bot_vis.png'))
                plt.close()

    run_all_pca = False # for now since memory issues running as is
    if run_all_pca:
        # aggregate PCA 
        temp_enc_t_agg = torch.cat(enc_t_track).detach().cpu()
        temp_enc_b_agg = torch.cat(enc_b_track).detach().cpu()
        label_agg = np.concatenate(label_track)

        fig = pca_simple_vis(temp_enc_t_agg.view(temp_enc_t_agg.shape[0], -1), label_agg) # unrolled entirely
        # @ TODO modify / repeat labels to match the latent field unrolled size (ie, 32x32 unrolled)
        # fig = pca_simple_vis(enc_t.detach().cpu().view(-1, enc_t.shape[1]), label) # each latent field point separately
        fig.savefig(os.path.join(expt_dir, f'sample/eval_set_{str(epoch + 1).zfill(5)}_agg_enc_top_vis.png'))
        plt.close()
        #fig = pca_simple_vis(temp_enc_b_agg.view(temp_enc_b_agg.shape[0], -1), label_agg)
        #fig.savefig(os.path.join(expt_dir, f'sample/eval_set_{str(epoch + 1).zfill(5)}_agg_enc_bot_vis.png'))
        #plt.close()

    model.train()
    #eval_mse_mean_epoch = mse_sum / mse_n
    #training_log.loc[epoch,'eval_total_loss'] = eval_mse_mean_epoch
    #training_log.loc[epoch,'eval_classifier_loss'] = classifier_sum / mse_n




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('-bs','--batch_size', type=int, default=100)
    parser.add_argument('-nal','--n_additional_layers', type=int, default=0)
    parser.add_argument('--augment', type=bool, default=False) 
    parser.add_argument('--train_slides', type=int, default=10)
    parser.add_argument('--dev_slides', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--n_embed', help='vqvae2 `n_embed` param; codebook size', type=int, default=512) # 'n_embed'
    parser.add_argument('--embed_dim', help='vqvae2 `embed_dim` param; dimension/channels in pre-quantized latent field', type=int, default=64) # 'embed_dim'
    parser.add_argument('--channel', help='vqvae2 `channel` param; number of channels in the hidden \
            representation (prior to latent field representation/quantization)', type=int, default=128) # 'channel'
    parser.add_argument('-clw','--classifier_loss_weight', help='Classifier loss weight', type=float, default=0.001)
    parser.add_argument('-llw','--latent_loss_weight', help='Latent loss weight', type=float, default=0.25)
    parser.add_argument('--full_size',help='Uncropped input tile size', default=512)
    parser.add_argument('--seed', type=int, default=None, metavar='N', help='set a random seed for torch and numpy (default: None)')

    args = parser.parse_args()

    print(args)
    # set random seed if specified
    if type(args.seed) != None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print('using random seed {}'.format(args.seed))


    # create timestamped experiment output 
    timestamp = get_timestamp()
    output_dir = './experiment_outputs/'

    expt_dir = os.path.join(output_dir, timestamp)
    print(expt_dir)
    try:
        os.mkdir(expt_dir)
        os.mkdir(os.path.join(expt_dir, 'checkpoint'))
        os.mkdir(os.path.join(expt_dir, 'sample'))
    except:
        pass

     # set up logging
    checkpoint_dir = os.path.join(expt_dir, 'checkpoint')
   
    #LOG_OUT = False
    LOG_OUT = True # leaving this as hardcoded for now
    if LOG_OUT:
        sys.stderr = open(os.path.join(expt_dir,'stderr.txt'), 'w')
        sys.stdout = open(os.path.join(expt_dir,'stdout.txt'), 'w')

    print(args)
    
    device = 'cuda'

    AUGMENT = args.augment
    #AUGMENT=True
    if AUGMENT:
        print('Using augmentations (RHF, CJ [defaults])')
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.full_size),
                transforms.RandomCrop(args.size),
                #transforms.CenterCrop(args.size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=32./255.,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.1
                    ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize(args.full_size),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    else:
        print('Not using augmentations')
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.full_size),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        eval_transform = train_transform
    

    # manual hardcoding
    paths_df = pd.read_pickle('/home/nyman/20191025_dfci_full_rcc_paths_df_annotated_reexport.pkl') # tried exporting above in `rapids` conda env
    partial_indicator = pd.read_csv('/home/nyman/20191025_dfci_full_rcc_paths_df_annotated_jupyterexport_PARTIAL_TILE_INDICATOR.csv')
    # subset out tiles that aren't 512x512 
    paths_df = paths_df.loc[partial_indicator['is_512'].values]

    # subset by clearcell vs nonclearcell and sample evenly
    print('evenly splitting b/t kirc/nonkirc slides')
    paths_df.index.name = 'idx'
    data_anno = paths_df.reset_index().drop_duplicates('idx')
    all_ids_subset = data_anno.groupby('is_kirc').apply(lambda x: x.sample(int(2*args.train_slides))).reset_index(0, drop=True)
    all_ids_subset = all_ids_subset['idx'].values
    print(all_ids_subset.shape)
    
    # subset and only take a few slides
    train_ids = pd.Series(all_ids_subset).sample(args.train_slides)
    train_paths_df = paths_df.loc[train_ids]

    # grab a set of slides to evaluate during training 
    #dev_subset = paths_df.drop(subset_ids.values)
    dev_subset = [x for x in all_ids_subset if x not in train_ids.values]
    dev_subset_ids = pd.Series(dev_subset).sample(args.dev_slides)
    dev_paths_df = paths_df.loc[dev_subset_ids]
    
    train_ids.to_csv(os.path.join(expt_dir, 'train_slide_ids.csv'))
    dev_subset_ids.to_csv(os.path.join(expt_dir, 'dev_slide_ids.csv'))

    print('Train Subset DF size: ', train_paths_df.shape)
    print('Dev Subset DF size: ', dev_paths_df.shape)

    print('train balance: ', train_paths_df.is_kirc.mean())
    print('dev balance: ', dev_paths_df.is_kirc.mean())

    # TODO add subsampling via controlling # tiles per slide sampled....

    dataset = utilities.Dataset(train_paths_df.full_path.values, train_paths_df.is_kirc.values.astype(int), train_transform)
    dev_dataset = utilities.Dataset(dev_paths_df.full_path.values, dev_paths_df.is_kirc.values.astype(int), eval_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)

    print('Using K={} codebook size'.format(args.n_embed))
    
    model = nn.DataParallel(
            VQVAE(
                input_size=args.size,
                channel=args.channel,
                embed_dim=args.embed_dim,
                n_embed=args.n_embed, 
                n_additional_downsample_layers=args.n_additional_layers,
                n_additional_upsample_layers=args.n_additional_layers
                )
            ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    training_log = pd.DataFrame(columns=[
        'epoch','batch','train_mse_loss','dev_mse_loss']).set_index(['epoch', 'batch'])


    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device, training_log, expt_dir, args.latent_loss_weight, args.classifier_loss_weight)
        evaluate_dataset(i, dev_loader, model, device, training_log, expt_dir, args.classifier_loss_weight)
        torch.save(
            model.module.state_dict(), os.path.join(expt_dir, f'checkpoint/vqvae_{str(i + 1).zfill(3)}.pt')
        )
        training_log.to_csv(os.path.join(expt_dir, f'checkpoint/training_log_vqvae_{str(i + 1).zfill(3)}.csv'))

