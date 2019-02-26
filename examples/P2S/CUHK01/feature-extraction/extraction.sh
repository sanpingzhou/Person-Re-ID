#!/usr/bin/env sh
# args for EXTRACT_FEATURE
TOOL=./build/tools
MODEL=./examples/CUHK01/feature-extraction/model/CUHK01_p2s_iter_36000.caffemodel
PROTOTXT=./examples/CUHK01/feature-extraction/test.prototxt 
LAYER=concat_3
LEVELDB=./examples/CUHK01/feature-extraction/feature
BATCHNUM=1
BATCHSIZE=100
# args for LEVELDB to MAT
DIM=800	
OUT=./examples/CUHK01/feature-extraction/feature/CUHK01_featureTestProbe.mat
	
		                      
$TOOL/extract_features  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHNUM 'leveldb' GPU
python ./examples/CUHK01/feature-extraction/leveldb2mat.py $LEVELDB $BATCHNUM  $BATCHSIZE $DIM $OUT
rm $LEVELDB/*.ldb $LEVELDB/*.log $LEVELDB/LO* $LEVELDB/CURRENT $LEVELDB/MANIFEST*

