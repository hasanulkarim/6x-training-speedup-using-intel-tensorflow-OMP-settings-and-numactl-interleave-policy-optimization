## 6x training speedup using intel-tensorflow, OMP-settings and numactl interleave policy optimization

This is a quick tryout to optimize training performance on CPU using intel tensorflow, KMP affinity settings and numactl policies on an AWS instance and compare speedup on training performance.

Below is a walk through of the steps to get the results and the full bash script can be found at esnet50_train_optimize.sh file.

## Connecting to AWS and installing the dependencies

#### setup the AWS environment
1. Launch instances by following EC2 tutorial: [Launch a Linux Virtual Machine
with Amazon EC2](https://aws.amazon.com/getting-started/tutorials/launch-a-virtual-machine/#)

2. Confirm _htop_ is already installed in the ubuntu server (I found it already installed): ```$ htop```
it should start _htop_

3. Python 3.6.7 was already istalled

4. Numactl can be installed by:
   - ```$ Sudo apt install numactl```
   - Confirm installation: ```$ numactl```

5. Install anaconda:
   - ```$ wget http://repo.continuum.io/archive/Anaconda3-4.1.1-Linux-x86_64.sh```, it's a big file, might take a while depending on internet speed
   - install once download complete: ```$ sudo bash Anaconda3-4.1.1-Linux-x86_64.sh```
   - Further optimize the conda configurations to set default port for jupyter notebook etc. can be found in this nice article:
[Getting Spark, Python, and Jupyter Notebook running on Amazon EC2](https://medium.com/@josemarcialportilla/getting-spark-python-and-jupyter-notebook-running-on-amazon-ec2-dec599e1c297)

6. Install intel distribution for python by: 
```$conda create -n IDP intelpython3_full -c intel```
This installs python for intel and tensorflow 1.3 







