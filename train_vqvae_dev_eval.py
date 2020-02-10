import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.utils import data
from tqdm import tqdm
from vqvae import VQVAE
from scheduler import CycleScheduler

# my additions
import sys
import os
sys.path.append('/home/nyman/')
from tmb_bot import utilities
from tmb_bot.utilities import class_balance_sampler, pil_loader
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

class SlideDataset(data.Dataset):
    """
    Modification of vanilla `tmb_bot.utilities.Dataset` class to facilitate having
    a label for classification as well as the slide name itself
    """

    def __init__(self, paths, slide_ids, labels, transform_compose):
        """
        Paths and labels should be array like
        """
        self.paths = paths
        self.slide_ids = slide_ids
        self.labels = labels
        self.transform = transform_compose

    def __len__(self):
        'Denotes the total number of samples'
        return self.paths.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        img_path = self.paths[index]
        pil_file = pil_loader(img_path)
        pil_file = self.transform(pil_file)
        slide_id = self.slide_ids[index]
        label = self.labels[index]

        return pil_file, label, slide_id


def train(epoch, loader, model, optimizer, scheduler, device, log, expt_dir, latent_loss_weight=0.25,
          classifier_loss_weight=0.001):
    sample_size = loader.batch_size
    loader = tqdm(loader)

    # criterion = nn.MSELoss() # TODO Remove once working within forward
    mse_sum = 0
    mse_n = 0

    for i, (img, label, slide_id) in enumerate(loader):
        model.zero_grad()

        # img = img.to(device)

        #out, latent_loss, enc_t, enc_b, classifier_loss, recon_loss = model(img, label)
        #out, latent_loss, enc_t, enc_b, classifier_loss, recon_loss = model(img.cuda(), label.cuda())
        #latent_loss, enc_t, enc_b, classifier_loss, recon_loss = model(img.cuda(), label.cuda())
        #latent_loss, _, _, classifier_loss, recon_loss = model(img.cuda(), label.cuda())
       
        # try just returning losses and latent code IDs for a lighter memory output (to ease up gather size for GPU0) 
        # latent_loss, classifier_loss, recon_loss, id_t, id_b = model(img.cuda(), label.cuda())

        # now see if letting model figure out cuda assignment helps / if we don't need to call `.cuda()`
        latent_loss, classifier_loss, recon_loss, id_t, id_b = model(img, label) # dont think it matters with dataparallel!

        loss = recon_loss + (latent_loss_weight * latent_loss) + (classifier_loss_weight * classifier_loss)
        loss = loss.mean()  # collapse multi GPU output format
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        with torch.no_grad():
            mse_sum += (recon_loss * img.shape[0]).mean().item()
            mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'train; '
                f'epoch: {epoch + 1}; mse: {recon_loss.mean().item():.5f}; '
                f'classifier: {(classifier_loss_weight * classifier_loss).mean().item():.3f}; '
                f'latent: {latent_loss.mean().item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        log.loc[(epoch, i), 'train_loss'] = loss.item()
        log.loc[(epoch, i), 'train_recon_loss'] = recon_loss.mean().item()
        log.loc[(epoch, i), 'train_classifier_loss'] = (classifier_loss_weight * classifier_loss).mean().item()
        log.loc[(epoch, i), 'train_latent_loss'] = (latent_loss_weight * latent_loss).mean().item()

        

        if i % 100 == 0:
        #if i is None: # block for now while troubleshooting
            model.eval()
                
            # my addition to try to just decode from category IDs instead
            with torch.no_grad():
                out = model.module.decode_code(id_t, id_b).cpu()

            sample = img[:sample_size]

            # with torch.no_grad():
            #    out, _ = model(sample)

            # my edit to visualize PRE gradient step decoding
            # out = out[:sample_size].cpu().detach()

            utils.save_image(
                torch.cat([sample, out], 0),
                os.path.join(expt_dir, f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png'),
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            DO_PCA = False 
            if DO_PCA:
                # visualize pre-quantized encodings
                # plot 2 component PCA vis of latent encodings
                fig = pca_simple_vis(enc_t.detach().cpu().view(enc_t.shape[0], -1), label)  # unrolled entirely
                fig.savefig(os.path.join(expt_dir, f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_enc_top_vis.png'))
                plt.close()
                fig = pca_simple_vis(enc_b.detach().cpu().view(enc_b.shape[0], -1), label)
                fig.savefig(os.path.join(expt_dir, f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_enc_bot_vis.png'))
                plt.close()

            model.train()


def evaluate_dataset(epoch, loader, model, device, log, expt_dir, latent_loss_weight=0.25, classifier_loss_weight=0.001):
    """
    Just taking above train function and removing any gradient updates etc
    """
    model.eval()
    sample_size = loader.batch_size
    loader = tqdm(loader)
    # criterion = nn.MSELoss()

    mse_sum = 0
    mse_n = 0
    classifier_sum = 0

    enc_t_track = []
    enc_b_track = []
    label_track = []

    with torch.no_grad():
        for i, (img, label, slide_id) in enumerate(loader):
            # img = img.to(device)

            #out, latent_loss, enc_t, enc_b, classifier_loss, recon_loss = model(img, label)
            #latent_loss, enc_t, enc_b, classifier_loss, recon_loss = model(img, label)
            #latent_loss, classifier_loss, recon_loss, id_t, id_b = model(img.cuda(), label.cuda())
            latent_loss, classifier_loss, recon_loss, id_t, id_b = model(img, label)

            # recon_loss = criterion(out, img)
            #latent_loss = latent_loss.mean()
            #classifier_loss = classifier_loss.mean()
            loss = recon_loss + (latent_loss_weight * latent_loss) + (classifier_loss_weight * classifier_loss)
            loss = loss.mean()

            mse_sum += (recon_loss * img.shape[0]).mean().item()
            mse_n += img.shape[0]

            classifier_sum += (classifier_loss_weight * classifier_loss).mean().item()

            #enc_t_track.append(enc_t.detach().cpu())
            #enc_b_track.append(enc_b.detach().cpu())
            #label_track.append(label)

            loader.set_description(
                (
                    f'eval; '
                    f'epoch: {epoch + 1}; mse: {recon_loss.mean().item():.5f}; '
                    f'classifier: {(classifier_loss_weight * classifier_loss).mean().item():.3f}; '
                    f'latent: {latent_loss.mean().item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                )
            )

            log.loc[(epoch,i),'eval_loss'] = loss.item()
            log.loc[(epoch,i),'eval_recon_loss'] = recon_loss.mean().item()
            log.loc[(epoch,i),'eval_classifier_loss'] = (classifier_loss_weight * classifier_loss).mean().item()
            log.loc[(epoch,i),'eval_latent_loss'] = (latent_loss_weight * latent_loss).mean().item()


            if i % 100 == 0:
                with torch.no_grad():
                    out = model.module.decode_code(id_t, id_b).cpu()

                sample = img[:sample_size]
                #out = out[:sample_size].detach()
                #out = out[:sample_size].cpu().detach()

                utils.save_image(
                    torch.cat([sample, out], 0),
                    os.path.join(expt_dir, f'sample/eval_set_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png'),
                    nrow=sample_size,
                    normalize=True,
                    range=(-1, 1),
                )
                DO_PCA = False 
                if DO_PCA:
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
        fig.savefig(os.path.join(expt_dir, f'sample/eval_set_{str(epoch + 1).zfill(5)}_agg_enc_top_vis.png'))
        plt.close()
        #fig = pca_simple_vis(temp_enc_b_agg.view(temp_enc_b_agg.shape[0], -1), label_agg)
        #fig.savefig(os.path.join(expt_dir, f'sample/eval_set_{str(epoch + 1).zfill(5)}_agg_enc_bot_vis.png'))
        #plt.close()

    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('-bs', '--batch_size', type=int, default=100)
    parser.add_argument('-nal', '--n_additional_layers', type=int, default=0)
    parser.add_argument('--augment', type=bool, default=False)
    # parser.add_argument('--train_slides', type=int, default=10)
    # parser.add_argument('--dev_slides', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--n_embed', help='vqvae2 `n_embed` param; codebook size', type=int, default=512)  # 'n_embed'
    parser.add_argument('--embed_dim',
                        help='vqvae2 `embed_dim` param; dimension/channels in pre-quantized latent field', type=int,
                        default=64)  # 'embed_dim'
    parser.add_argument('--channel', help='vqvae2 `channel` param; number of channels in the hidden \
            representation (prior to latent field representation/quantization)', type=int, default=128)  # 'channel'
    parser.add_argument('-clw', '--classifier_loss_weight', help='Classifier loss weight', type=float, default=0.001)
    parser.add_argument('-llw', '--latent_loss_weight', help='Latent loss weight', type=float, default=0.25)
    parser.add_argument('--full_size', type=int, help='Uncropped input tile size', default=512)
    parser.add_argument('--seed', type=int, default=None, metavar='N',
                        help='set a random seed for torch and numpy (default: None)')

    parser.add_argument('--paths_df', type=str, help='file path of dataframe with ids/tile paths')
    parser.add_argument('--train_ids', type=str, help='file path of list of IDs in train set')
    parser.add_argument('--dev_ids', type=str, help='file path of list of IDs in validation (dev) set')
    parser.add_argument('-tps', '--tiles_per_slide', type=int, default=50,
                        help='if specified, num. tiles to sample per slide')
    parser.add_argument('--balance_var', type=str, default='is_kirc',
                        help='paths_df column on which to groupby to evenly sample from')
    args = parser.parse_args()

    print(args)
    # set random seed if specified
    if args.seed is not None:
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
        dev_transform = transforms.Compose(
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
        dev_transform = train_transform
    

    try:
        paths_df = pd.read_pickle(args.paths_df)
    except:
        paths_df = pd.read_csv(args.paths_df)
        paths_df = paths_df.set_index(paths_df.columns[0])

    paths_df.index.name = 'idx'

    train_ids = pd.read_csv(args.train_ids, header=None).iloc[:,1].values
    dev_ids = pd.read_csv(args.dev_ids, header=None).iloc[:,1].values

    train_paths_df = paths_df.loc[train_ids]
    dev_paths_df = paths_df.loc[dev_ids]

    # TODO find a cleaner way to flag when we don't want to use this arg (instead of checking != -1)
    if args.tiles_per_slide != -1:
        num_true = args.tiles_per_slide
        num_false = args.tiles_per_slide
        pred_variable = args.balance_var

        train_paths_df = train_paths_df.reset_index().groupby('idx').apply(
                lambda x: class_balance_sampler(x, num_true, num_false, pred_variable))
        dev_paths_df = dev_paths_df.reset_index().groupby('idx').apply(
                lambda x: class_balance_sampler(x, num_true, num_false, pred_variable))

    train_dataset = SlideDataset(
        paths=train_paths_df.full_path.values,
        slide_ids=train_paths_df.index.values,
        labels=train_paths_df[args.balance_var].values,
        transform_compose=train_transform
    )
    dev_dataset = SlideDataset(
        paths=dev_paths_df.full_path.values,
        slide_ids=dev_paths_df.index.values,
        labels=dev_paths_df[args.balance_var].values,
        transform_compose=dev_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)

    print(len(train_dataset), len(dev_dataset))
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
            #) # not sure if `to(device)` is the issue

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(train_loader) * args.epoch, momentum=None
        )

    training_log = pd.DataFrame(columns=[
        'epoch','batch','train_mse_loss','dev_mse_loss']).set_index(['epoch', 'batch'])


    for i in range(args.epoch):
        train(i, train_loader, model, optimizer, scheduler, device, training_log, expt_dir, args.latent_loss_weight, args.classifier_loss_weight)
        evaluate_dataset(i, dev_loader, model, device, training_log, expt_dir, args.classifier_loss_weight)
        torch.save(
            model.module.state_dict(), os.path.join(expt_dir, f'checkpoint/vqvae_{str(i + 1).zfill(3)}.pt')
        )
        training_log.to_csv(os.path.join(expt_dir, f'checkpoint/training_log_vqvae_{str(i + 1).zfill(3)}.csv'))

