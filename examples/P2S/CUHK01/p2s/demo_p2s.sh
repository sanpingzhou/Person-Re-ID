./build/tools/caffe train \
       --solver=examples/CUHK01/p2s/solver_p2s.prototxt -gpu 8 \
       2>examples/CUHK01/log/CUHK01_margin1=0.1_margin2=0.5_margin3=1.2_weight=0.15.log

