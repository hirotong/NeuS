#!/usr/bin/env python3
# Author: hiro.tong


import os
import subprocess

COLMAP = 'colmap'   # path to colmap executable

# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense

def run_colmap(basedir, image_dir='images', match_type='exhaustive_matcher'):
    
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')

    feature_extractor_args = [
        COLMAP, 'feature_extractor', 
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, image_dir),
        '--ImageReader.camera_model', 'SIMPLE_PINHOLE',
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_params', '1111.11,400,400',
    ]
    feat_output = (subprocess.check_output(feature_extractor_args, universal_newlines=True))
    logfile.write(feat_output)
    print('Features extracted')

    exhaustive_matcher_args = [
        COLMAP, match_type, 
        '--database_path', os.path.join(basedir, 'database.db'),
    ]
    
    match_output = (subprocess.check_output(exhaustive_matcher_args, universal_newlines=True))
    logfile.write(match_output)
    print('Features matched')

    p = os.path.join(basedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)
    
# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

    mapper_args = [
        COLMAP, 'mapper',
        '--database_path', os.path.join(basedir, 'database.db'),
        '--image_path', os.path.join(basedir, image_dir),
        '--output_path', p,
        '--Mapper.init_min_tri_angle', '4',
        '--Mapper.multiple_models', '1',
        '--Mapper.extract_colors', '0',
        '--Mapper.ba_refine_focal_length', '0',
        ]
    
    map_out = (subprocess.check_output(mapper_args, universal_newlines=True))
    logfile.write(map_out)
    logfile.close()
    print('Sparse map created')

    print("Finished running COLMAP, see {} for logs".format(logfile_name))
    
    
if __name__ == '__main__':
    run_colmap('/home/hiro/dataset/NeRF/data/nerf_synthetic/insect_asymmetric_1.2_0.8_0.5', 'images')