#!/bin/sh

DATASET_PATH=$1

if [ ! -d $DATASET_PATH/images ]; then
    mkdir $DATASET_PATH/images
fi
#ffmpeg -i $DATASET_PATH/input.mov -vf fps=5 $DATASET_PATH/images/output_%04d.png

echo "# feature_extractor (1/4) #"
colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images

echo "# exhaustive_matcher (2/4) #"
colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

if [ ! -d $DATASET_PATH/sparse ]; then
    mkdir $DATASET_PATH/sparse
fi

echo "# mapper (3/4) #"
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse \
    --Mapper.ba_global_function_tolerance=0.000001

if [ ! -d $DATASET_PATH/dense ]; then
    mkdir $DATASET_PATH/dense
fi

echo "# image_undistorter (4/4) #"
colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP

if [ ! -d $DATASET_PATH/to_opensplat ]; then
    mkdir $DATASET_PATH/to_opensplat
fi
ln -s ../dense/images $DATASET_PATH/to_opensplat
ln -s ../dense/sparse/cameras.bin $DATASET_PATH/to_opensplat
ln -s ../dense/sparse/images.bin $DATASET_PATH/to_opensplat
ln -s ../dense/sparse/points3D.bin $DATASET_PATH/to_opensplat
