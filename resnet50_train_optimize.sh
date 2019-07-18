#!/bin/bash sh
git clone -b cnn_tf_v1.12_compatible  https://github.com/tensorflow/benchmarks.git
cd ~/DL/benchmarks/scripts/tf_cnn_benchmarks
rm *.log # remove logs from any previous benchmark runs
#activate default environment with tensorflow (not intel optimized tensorflow)
source activate tf
export MKL_VERBOSE=1
export MKLDNN_VERBOSE=1
time python3 tf_cnn_benchmarks.py \
	 --device=CPU \
	 --data_name=imagenet \
	 --batch_size=128 \
	 --num_batches=30 \
	 --model=resnet50 \
	 --data_format=NHWC \
	 2>&1 | tee default.log 
source deactivate
#run with optimized tf.config_proto() and MKL=True in intel python environment
source activate IDP
#set environment settings
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export OMP_NUM_THREADS=18
export OMP_PROC_BIND=true
#run the python code with config update
time python3 tf_cnn_benchmarks.py \
	 --device=CPU \
	 --data_name=imagenet \
     --batch_size=128 \
	 --num_batches=30 \
	 --model=resnet50 \
	 --data_format=NHWC \
	 --mkl=true \
	 --num_inter_threads=2 \
	 --num_intra_threads=18 \
	 2>&1 | tee optimized1.log
#run with numactl optimzed
time numactl -p 1 python3 tf_cnn_benchmarks.py \
	 --device=CPU \
	 --data_name=imagenet \
	 --batch_size=128 \
	 --num_batches=30 \
	 --model=resnet50 \
	 --data_format=NHWC \
	 --mkl=true \
	 --num_inter_threads=2 \
	 --num_intra_threads=18 \
	 2>&1 | tee optimized2.log
#run with numact -m option	 
time numactl -m 1 python3 tf_cnn_benchmarks.py \
	 --device=CPU \
	 --data_name=imagenet \
	 --batch_size=128 \
	 --num_batches=30 \
	 --model=resnet50 \
	 --data_format=NHWC \
	 --mkl=true \
	 --num_inter_threads=2 \
	 --num_intra_threads=18 \
	 2>&1 | tee optimized3.log
#run with OMP_NUM_THREADS and num_inter_threads set to 1
export OMP_NUM_THREADS=1
time python3 tf_cnn_benchmarks.py \
	 --device=CPU \
	 --data_name=imagenet \
     --batch_size=128 \
	 --num_batches=30 \
	 --model=resnet50 \
	 --data_format=NHWC \
	 --mkl=true \
	 --num_inter_threads=2 \
	 --num_intra_threads=1 \
	 2>&1 | tee optimized4.log
#print summary and logs
echo $'\n'
echo "######### Executive Summary #########"
echo $'\n'
echo "Environment |  Network   | Batch Size | Images/Second"
echo "--------------------------------------------------------"
default_fps=$(grep  "total images/sec:"  default.log | cut -d ":" -f2 | xargs)
optimized1_fps=$(grep  "total images/sec:"  optimized1.log | cut -d ":" -f2 | xargs)
optimized2_fps=$(grep  "total images/sec:"  optimized2.log | cut -d ":" -f2 | xargs)
optimized3_fps=$(grep  "total images/sec:"  optimized3.log | cut -d ":" -f2 | xargs)
optimized4_fps=$(grep  "total images/sec:"  optimized4.log | cut -d ":" -f2 | xargs)
echo "Default     |  Resnet50  |     128    | $default_fps"
echo "Optimized1  |  Resnet50  |     128    | $optimized1_fps"
echo "Optimized2  |  Resnet50  |     128    | $optimized2_fps"
echo "Optimized3  |  Resnet50  |     128    | $optimized3_fps"
echo "Optimized4  |  Resnet50  |     128    | $optimized4_fps"
speedup1=$((${optimized1_fps%.*}/${default_fps%.*}))
speedup2=$((${optimized2_fps%.*}/${default_fps%.*}))
speedup3=$((${optimized3_fps%.*}/${default_fps%.*}))
speedup4=$((${optimized4_fps%.*}/${default_fps%.*}))
echo "speedup1=$speedup1"
echo "speedup2=$speedup2"
echo "speedup3=$speedup3"
echo "speedup4=$speedup4"