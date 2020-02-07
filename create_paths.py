import pandas as pd
from glob import glob
import os

root_dir = '/mnt/disks/image_data2/'
paths = glob(os.path.join(root_dir, '*/*/*.jpeg'))
ids = [x.split('/')[-3].split('_files')[0] for x in paths]
paths_df = pd.DataFrame()
paths_df['file_id'] = ids
paths_df['full_path'] = paths
paths_df = paths_df.set_index('file_id')

paths_df['x'] = paths_df.full_path.apply(lambda x: float(x.split('/')[-1].split('_')[-2])).astype(int)
paths_df['y'] = paths_df.full_path.apply(lambda x: float(x.split('/')[-1].split('_')[-1].split('.')[0])).astype(int)

data_anno = pd.read_csv('/home/jupyter/profile_rcc_updated_data_anno_20200130.csv').set_index('file_id')
data_anno = data_anno.drop(columns=['full_path','x','y'])
paths_df = paths_df.merge(data_anno, how='left', left_index=True, right_index=True)
paths_df['is_cc'] = paths_df.histology_types == 'Clear cell'
paths_df['is_kirc'] = paths_df.histology_types == 'Clear cell'
paths_df['is_kirp'] = paths_df.histology_types == 'Papillary'
paths_df['is_kich'] = paths_df.histology_types == 'Chromophobe'



paths_df.to_pickle('/home/nyman/20x_2048px_paths_df_profile_rcc.pkl')
