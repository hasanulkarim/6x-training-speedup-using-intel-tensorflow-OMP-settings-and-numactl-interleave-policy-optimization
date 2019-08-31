## 6x training speedup using intel-tensorflow, OMP-settings and numactl interleave policy optimization

This is a quick tryout to optimize training performance on CPU using intel tensorflow, KMP affinity settings and numactl policies on an AWS instance and compare speedup on training performance.

Below is a walk through of the steps to get the results and the full bash script can be found at esnet50_train_optimize.sh file.

## Connecting to AWS and installing the dependencies

#### Setup the AWS environment
1. Launch instances by following EC2 tutorial: [Launch a Linux Virtual Machine
with Amazon EC2](https://aws.amazon.com/getting-started/tutorials/launch-a-virtual-machine/#)

2. Confirm _htop_ is already installed in the ubuntu server (I found it already installed): ```$ htop```


3. Python 3.6.7 was already istalled

4. Numactl can be installed by:
   - ```$ Sudo apt install numactl```
   - Confirm installation: ```$ numactl```

5. Install anaconda:
   - ```$ wget http://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh```, it's a big file, might take a while depending on internet speed
   - install once download complete: ```$ sudo bash Anaconda3-4.1.1-Linux-x86_64.sh```
   - Further optimize the conda configurations to set default port for jupyter notebook etc. can be found in this nice article:
[Getting Spark, Python, and Jupyter Notebook running on Amazon EC2](https://medium.com/@josemarcialportilla/getting-spark-python-and-jupyter-notebook-running-on-amazon-ec2-dec599e1c297)

#### Install tensorflow without optimization

1. confirm python 3.6 is installed: ```$ python3```

2. Update all pkgs and environments: ```$ sudo apt update```

3. Install tensorflow in a new environment called _tf_ to make life easier later: ```$ conda create -n tf tensorflow```, this installs the default CPU version
   - confirm the version of tensorflow after installation:
     - activate the new environment: 
     ```sh
     $ source activate tf
     ```
     - check the tensorflow version: 
     ```sh
     $ python3 -c “import tensorflow as tf; print(tf.__version__)”
     ```

#### Install intel optimized tensorflow

Install intel distribution for python by: 
```shell
$conda create -n IDP intelpython3_full -c intel
```

#### Download the tf_cnn_benchmarks 

The code below can be used to get the tf_cnn_benchmarks files.
```

$ git clone -b cnn_tf_v1.12_compatible  https://github.com/tensorflow/benchmarks.git

``` 
More details can be found here: [tf_cnn_benchmarks: High performance benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks)   

## Configuration

A single node on an Intel® Xeon® CPU E5-2666 v3 @ 2.90GHz with 18 physical cores with 2 threads per core and 60 GB of RAM* was used. This node was accessed via a c4.8xLarge AWS instance with 2 Numa Nodes available. (Detailed CPU configuration can be found in the supplementary section at the end of the readme file) 
tf_cnn_benchmarks: High performance benchmarks was used for the test. ResNet-50 model with synthetic data and batch size of 128 and 30 batches was trained with different settings as detailed below. 

\*Note: the size of the RAM was not a constraint in this experiment

## Setup

The environmental and tensorflow settings can be setup using a bash script that activates the desired environment, sets the MPI settings and calls the _tf_cnn_benchmarks.py_ with the desired flags. This will ensure consistency and makes it easier to tweak parameters later.

First, Download the benchmark files as noted above. For simplicity, save the bash the script in the same directory as the _tf_cnn_benchmarks.py_ file. The _MKL_VERBOSE_ and _MKLDNN_VERBOSE_ are set to 1 in order to output when MKL is used for computation. This is one way of confirming that intel optimized Math Kernel Libraries are being used for computations.

```sh
$ export MKL_VERBOSE=1
$ export MKLDNN_VERBOSE=1
```

#### Setup 1 (Default environment - no optimization)

The code below will activate the unoptimized tensorflow environtment _tf_ and will call the _tf_cnn_benchmarks.py_ script with flags to run on CPU with synthetic imagenet data of batch_size 128 and 30 batches. Data format is set to NHWC (Channels_last)
``` shell
$ source activate tf
$ python3 tf_cnn_benchmarks.py \
	 --device=CPU \
	 --data_name=imagenet \
	 --batch_size=128 \
	 --num_batches=30 \
	 --model=resnet50 \
	 --data_format=NHWC \
	 2>&1 | tee default.log 
$ source deactivate
```

#### Setup 2 (optimized tf.config_proto() and MKL=True in intel python environment)

```sh
$ source activate IDP
#set environment settings (these are recommended settings based on some read, 
#resources mentioned in the resources section at the end)
$ export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
$ export KMP_BLOCKTIME=1
$ export KMP_SETTINGS=1
$ export OMP_NUM_THREADS=18
$ export OMP_PROC_BIND=true
#run the python code with config update
$ python3 tf_cnn_benchmarks.py \
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
```

#### Setup 3 (Applying numa interleave policy on top of setup 2)
```sh
$ numactl -i all python3 tf_cnn_benchmarks.py \
	 --device=CPU \
	 --data_name=imagenet \
     	 --batch_size=128 \
	 --num_batches=30 \
	 --model=resnet50 \
	 --data_format=NCHW \
	 --mkl=true \
	 --num_inter_threads=18 \
	 --num_intra_threads=18 \
	 2>&1 | tee optimized2.log #sends the stderror to where stdoutput is going and 
	 			   #then tee combines both and sends to display and the log file
```

## Results

htop was used to monitor the load distribution during the training and the difference can be seen on how many idle threads are present in the default settings against when the environment variables are optimized.

![alt text](./images/htop_setup1.png)  
**Fig1:** _(setup1:no optimization)_ – _htop_ (right) showing the load distribution on the threads of the CPU and on Memory, Red represents idle or ‘waiting’ thread, while training (left)

![alt text](./images/htop_setup2.png)
**Fig2:** _(setup2: intel optimized tensorflow and affinity settings optimized)_ - _htop_ showing the load distribution on the threads of the CPU and on Memory, Red represents idle or ‘waiting’ thread. MKLDNN_VERBOSE is displaying the operation performed (left)

![alt text](./images/htop_setup3.png)
**Fig3:** _(setup3: intel optimized tensorflow and affinity settings optimized with numactl interleave policy)_ - _htop_ showing the load distribution on the threads of the CPU and on Memory, Red represents idle or ‘waiting’ thread, MKLDNN_VERBOSE is displaying the operation performed (left)

