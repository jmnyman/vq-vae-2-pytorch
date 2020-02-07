import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_slides', type=int, default=10)
    parser.add_argument('--dev_slides', type=int, default=5)
    parser.add_argument('--seed', type=int, default=None, metavar='N', help='set a random seed for torch and numpy (default: None)')
    parser.add_argument('-df','--paths_df', type=str, help='file path of dataframe with ids/tile paths [pickle or csv]')
    parser.add_argument('--balance_var', type=str, default='is_kirc', help='paths_df column on which to groupby to evenly sample from')
    parser.add_argument('--prefix', type=str, default='20x_512px_', help='prefix to append to output ID csvs')
    # parser.add_argument('tps','--tiles_per_slide', type=int, default=None, help='if specified, num. tiles to sample per slide')
    args = parser.parse_args()

	

    # manual hardcoding
    try:
        paths_df = pd.read_pickle(args.paths_df)
    except:
        paths_df = pd.read_csv(args.paths_df)

    paths_df.index.name = 'idx'
    data_anno = paths_df.reset_index().drop_duplicates('idx')
    #if type(args.balance_var) != None:
    if args.balance_var != 'None':
        print('Balancing based on {}'.format(args.balance_var))
        all_ids_subset = data_anno.groupby(args.balance_var).apply(lambda x: x.sample(int(2*args.train_slides))).reset_index(0, drop=True)
        all_ids_subset = all_ids_subset['idx'].values
    else:
        all_ids_subset = data_anno.idx.values

    print(all_ids_subset.shape)

    # subset and only take a few slides
    train_ids = pd.Series(all_ids_subset).sample(args.train_slides)
    train_paths_df = paths_df.loc[train_ids]

    # grab a set of slides to evaluate during training 
    #dev_subset = paths_df.drop(subset_ids.values)
    dev_subset = [x for x in all_ids_subset if x not in train_ids.values]
    dev_subset_ids = pd.Series(dev_subset).sample(args.dev_slides)
    dev_paths_df = paths_df.loc[dev_subset_ids]

    train_ids.to_csv(args.prefix+'train_slide_ids.csv')
    dev_subset_ids.to_csv(args.prefix+'dev_slide_ids.csv')

