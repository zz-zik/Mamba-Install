前提条件，你已经成功安装 `cuda` 和 `cudnn` 根据最新的 `torch` 官方支持，仅支持 `cuda11.8 `、`cuda12.1 `、`cuda12.4 `、`cuda12.6 `，因此请勿安装其它版本的 `cuda`，这里我们以 `cuda12.1` 为例。

```bash
cuda12.1+cuDNN9.0+torch2.2+causal_conv1d-1.5.0.post7+mamba_ssm-2.2.4+selective_scan_cuda
```
## 一、CUDA 安装

### 1. 查看系统配置

#### 1.1 查看系统

使用以下命令可以查看 Linux 的系统架构

```bash
cat /etc/issue
```

会输出如下结果：

```bash
Debian GNU/Linux 10 \n \l
```

可以看到系统的版本为 `Debian10`
#### 1.2 查询显卡型号

命令行输入：

```bash
nvidia-smi
```

输出内容如下，可以查看显卡驱动（Driver Version）的版本号为 535.104.12，支持的 `CUDA` 版本为 12.2 以下

```bash
(base) root@7v0ut9u0b2vdl-0:/sxs# nvidia-smi
Fri Feb 14 09:47:08 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.12             Driver Version: 535.104.12   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A800 80GB PCIe          On  | 00000000:36:00.0 Off |                    0 |
| N/A   31C    P0              46W / 300W |      7MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
```

### 2. CUDA 安装

#### 2.1 安装依赖包

首先需要安装 Linux 环境缺少的依赖包文件

```bash
sudo apt-get update
sudo apt-get install gcc g++
sudo apt install kmod
sudo apt install libxml2
```

#### 2.2 下载 CUDA

访问 [CUDA Toolkit Archive | NVIDIA Developer](https://developer.nvidia.com/cuda-toolkit-archive) 官网，选择 `CUDA12.1.0` 版本，依次选择，`Linux->x 86_64->Debian->10->runfile (local)`

安装 CUDA 的方法有好几种，例如 `deb` 法，在这里我们主要使用 `runfile` 方法来安装 `runtime CUDA`

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
```

执行上述命令完成下载，如果出现 443，多执行几次，或者使用下述命令指定谷歌的 `DNS` 服务器。

```shell
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf > /dev/null
echo "nameserver 8.8.4.4" | sudo tee -a /etc/resolv.conf > /dev/null
```

#### 2.3 CUDA 安装

```bash
sudo sh cuda_12.1.0_530.30.02_linux.run
```

执行上述命令，键入 accept，如果已经安装显卡驱动，则取消掉“Driver”这个选项（在 Driver 位置键入 enter 取消），光标移至 Install，按 ENTER 键完成安装。


> [!error] Driver CUDA 冲突
> 问题描述
> ```bash
> # sudo cat /var/log/cuda-installer. Log 
> [INFO]: Driver not installed. 
> [INFO]: Checking compiler version... 
> [INFO]: gcc location: /usr/bin/gcc 
> [INFO]: gcc version: gcc version 8.3.0 (Debian 8.3.0-6) 
> [INFO]: Initializing menu 
> [INFO]: nvidia-fs.SetKOVersion (2.16.1) 
> [INFO]: Setup complete 
> [INFO]: Installing: Driver 
> [INFO]: Installing: 535.54.03 
> [INFO]: Executing NVIDIA-Linux-x 86_64-535.54.03. Run --ui=none --no-questions --accept-license --disable-nouveau --no-cc-version-check --install-libglvnd 2>&1 
> [INFO]: Finished with code: 256 
> [ERROR]: Install of driver component failed. Consult the driver log at /var/log/nvidia-installer. Log for more details. 
> [ERROR]: Install of 535.54.03 failed, quitting
> ```
> 解决方案
> 此时打开一个**新的终端窗口**，输入
> ```bash
> Nvidia-smi
> ```
>  如果会显示的话，说明你的电脑已经安装过了 driver CUDA，需要在安装选择界面里面取消掉“Driver”这个选项，光标移至 Install，按 ENTER 键完成安装。

到这里，你的电脑里就安装了两个 CUDA：第一个 driver CUDA 就是当你输入指令 nvidia-smi 后显示的信息。第二个 CUDA 就是我们通过 runfile 方法下载的 runtime CUDA 12.2.0，此时下载成功的 CUDA 11.0 的位置在/usr/local/cuda-12.2 目录下 (注意：此时只是说你下载了 CUDA 12.2，但是并不能说你的深度学习实验就可以用它来加速，因此需要配置环境变量，这也是最后一步)

#### 2.4 配置 CUDA 环境变量

终端输入

```bash
vim ~/.bashrc
```

键入 “i” 进入 INSERT 模式，可以进行回车操作，复制以下变量，键入“ESC”退出 INSERT 模式。

```bash
#在安装成功后的显示中可以看到cuda的安装位置
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

按 ESC 退出写入，最后键入: wq 即可保存并退出。

终端输入以下命令，使环境变量生效

```bash
source ~/.bashrc
```

#### 2.5 验证安装

在终端输入：
```bash
nvcc -V
```

即可显示如下：  (如果未显示 `cuda12.1`，则说明刚才的指令 `source ~/.bashrc` 没有立即生效，**此时重启电脑，再输入 `nvcc -V` ，若显示如下则成功！！！！**)

```bash
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Feb__7_19:32:13_PST_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0
```
### 3. cuDNN 安装

根据查找 `CUDA12.1` 需要 `cuDNN9.0` 版本。

#### 3.1 CuDNN 下载

进入 [cudnn Downloads | NVIDIA Developer](https://developer.nvidia.com/cudnn-9-0-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_network) 官网，依次选择 `Linux->x86_64->Debian->11->deb(network)`

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb
```

#### 3.2 CuDNN 安装

我们继续执行下述安装命令进行安装：

```bash
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo add-apt-repository contrib
sudo apt-get updatesudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12
```

#### 3.3 验证安装  

运行以下命令检查 cuDNN 版本：

```bash
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
```

您应该看到类似以下的输出：

```c
#define CUDNN_MAJOR 9
#define CUDNN_MINOR 1
#define CUDNN_PATCHLEVEL 1
--
#define CUDNN_VERSION (CUDNN_MAJOR * 10000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

/* cannot use constexpr here since this is a C-only file */
```

## 二、配置 conda环境

### 1. 配置 conda 虚拟环境

```bash
conda create -n vmamba python=3.10
conda activate vmamba
```
### 2. 安装 PyTorch

根据 [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba) 提供的方法安装

```bash
pip install torch==2.2 torchvision torchaudio triton pytest chardet yacs termcolor fvcore seaborn packaging ninja einops numpy==1.24.4 timm==0.4.12
```
## 三、mamba 环境

由官方推荐的安装方式可知，`Mamba` 的使用需要安装两个相关依赖：

- [causal_conv1d](https://github.com/Dao-AILab/causal-conv1d/releases/)
- [mamba_ssm](https://github.com/state-spaces/mamba/releases/tag/v2.2.2)

Mamba 模型介绍不在这里多说，此文主要讲 Mamba 环境的搭建。简单来说，其核心在于通过输入依赖的方式调整SSM参数，允许模型根据当前的数据选择性地传递或遗忘信息，从而解决了以前模型在处理离散和信息密集型数据（如文本）时的不足。这种改进使得Mamba在处理长序列时展现出更高的效率和性能，并与 `Transformer` 可以打平手的情况下，比 `Transformer` 复杂度更低。

如果使用官方安装的方式，则基本上都会出现问题。问题较为多样，但是归根到底是 `CUDA` 版本或者网络的问题，所以不推荐使用 `pip install` 进行，即使使用国内镜像源也会出错，因为在后面会有校验操作，结果还是安装不上。

那么直接进入正题，这边强烈推荐使用 `.whl` 文件进行离线安装，可以很好地解决网络原因导致的安装问题。如果 `whl` 包不能很好的解决问题，则需要使用方法三重头编译安装（编译安装时间比较久，取决于你的网络）。
### 1. 方法一：在线安装

找到相应的版本，复制安装地址，然后直接在服务器安装，这种方法适用于网络条件较好的情况。

```bash
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post7/causal_conv1d-1.5.0.post7+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

### 2. 方法二：离线安装

如果服务器网络并不能直接访问 `github`，可以选择手动下载到本地后上传服务器，执行安装。

```bash
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post7/causal_conv1d-1.5.0.post7+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
wget https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

将 `whl` 包手动上传到服务器中，切换到包所在路径，然后执行下述安装命令。

```bash
cd /home/
chmod +x causal_conv1d-1.5.0.post7+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install --force-reinstall causal_conv1d-1.5.0.post7+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
chmod +x mamba_ssm-2.2.4+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install --force-reinstall mamba_ssm-2.2.4+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

注意：不要听 ai 的在安装命令中添加--user, 这样会安装到 root 目录下，而不是 conda 目录下。

看到下述即代表安装成功

```text
Collecting causal-conv1d==1.5.0.post7+cu12torch2.2cxx11abitrue
```

可以查看安装路径是否正确

```bash
# 查看安装路径是否正确
pip show causal_conv1d
pip show mamba_ssm
```


上述安装完成后，虽然安装并未报错，可能还会出现 `selective_scan_cuda` 找不到的情况，因此我们需要重新编译安装。

### 3. 方法三：编译安装

#### 3.1 编译 `causal-conv1d`

从 [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) 拉取镜像到服务器。

```bash
git clone https://github.com/Dao-AILab/causal-conv1d.git
```

之后修改源码文件夹中 `setup.py` 文件，将

```bash
# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("CAUSAL_CONV1D_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("CAUSAL_CONV1D_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("CAUSAL_CONV1D_FORCE_CXX11_ABI", "FALSE") == "TRUE"
```

修改为:

```bash
FORCE_BUILD = True
SKIP_CUDA_BUILD = False
FORCE_CXX11_ABI = False
```

再将

```python
cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
```

修改为

```python
cmdclass={"bdist_wheel": CachedWheelsCommand, 'build_ext': BuildExtension.with_options(use_ninja=False)}
```

最后切换到 `causal-conv1d` 源码目录下，通过以下命令进行编译安装：

```shell
pip install .
```

安装完成后出现

```bash
Successfully installed causal_conv1d-1.5.0.post8
```

#### 3.2 编译 `mamba-ssm`

从 [state-spaces/mamba: Mamba SSM architecture](https://github.com/state-spaces/mamba)拉取镜像到服务器。

```bash
git clone https://github.com/state-spaces/mamba.git
```

之后修改源码文件夹中 `setup.py` 文件，将

```python
# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("MAMBA_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("MAMBA_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("MAMBA_FORCE_CXX11_ABI", "FALSE") == "TRUE"
```

修改为

```python
FORCE_BUILD = True
SKIP_CUDA_BUILD = False
FORCE_CXX11_ABI = False
```

再将

```python
cmdclass={"bdist_wheel": CachedWheelsCommand, "build_ext": BuildExtension}
```

修改为

```python
cmdclass={"bdist_wheel": CachedWheelsCommand, 'build_ext': BuildExtension.with_options(use_ninja=False)}
```

最后切换到`mamba-ssm`源码目录下，通过以下命令进行编译安装：

```shell
pip install .
```

安装完成后出现

```bash
Successfully installed mamba_ssm-2.2.4
```

### 4. 验证安装

为了验证是否安装成功，我们还需要依次执行下述命令：

```bash
python
import torch
import causal_conv1d_cuda
```

没有报错即为成功。输入 `exit()` 即可退出 `python`

## 三、安装其余依赖

现在可以根据你的项目，安装其余的依赖了，这里注意如果 `requirements.txt` 中有 `torch`、`torchvision `、`numpy` 和 `transformers` 指定包版本的情况，这里要把版本号或者包名删除，以免覆盖。

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

我们需要测试 torch 是否能够成功调用 GPU，以免出现 GPU 未调用的情况。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"
```


> [!error] subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.
> 
> 出现问题：
> ```
> Traceback (most recent call last):
>   File "/opt/conda/envs/moe/lib/python3.11/site-packages/torch/utils/cpp_extension.py", line 2506, in _run_ninja_build
>     subprocess.run(
>   File "/opt/conda/envs/moe/lib/python3.11/subprocess.py", line 571, in run
>     raise CalledProcessError(retcode, process.args,
> subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.
> ```
> 解决方案：
> 
> "/opt/conda/envs/moe/lib/python 3.11/site-packages/torch/utils/cpp_extension. Py"中将['ninja','-v']改成['ninja','--v'] 或者['ninja','--version']
> 


> [!error] ERROR: Failed building wheel for selective_scan
> 出现问题：
> ```bash
>       error: command 'g++' failed: No such file or directory
>       [end of output]
>   
>   note: This error originates from a subprocess, and is likely not a problem with pip.
>   ERROR: Failed building wheel for selective_scan
>   Running setup.py clean for selective_scan
> Failed to build selective_scan
> ERROR: Failed to build installable wheels for some pyproject.toml based projects (selective_scan)
> ```
> 解决方案：
> ```bash
> sudo apt-get update
> sudo apt-get install build-essential g++ python3-dev
> ```

### 参考文章

[Linux 环境下 Mamba 配置 - Tanxy](https://tanxy.club/2024/mamba_setup)
[Linux 下安装 mamba-ssm 踩过的坑 | Vanilla_chan](https://vanilla-chan.cn/blog/2025/05/24/Linux%E4%B8%8B%E5%AE%89%E8%A3%85mamba-ssm%E8%B8%A9%E8%BF%87%E7%9A%84%E5%9D%91/)
[AlwaysFHao/Mamba-Install: 本仓库旨在介绍如何通过源码编译的方法成功安装mamba，可解决selective_scan_cuda和本地cuda环境冲突的问题](https://github.com/AlwaysFHao/Mamba-Install)
