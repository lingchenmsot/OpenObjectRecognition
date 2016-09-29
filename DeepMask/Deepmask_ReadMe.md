# Deepmask环境搭建
deepmask github网址：
>https://github.com/facebookresearch/deepmask

---

## 系统要求
- linux 或者 macos操作系统
  - 本人采用的是ubuntu 14.04 LTS系统   
![image](https://github.com/lingchenmsot/OpenObjectRecognition/blob/master/DeepMask/Markdown_images/OS_GPU.png?raw=true)

---

## 硬件要求
+ NVIDIA GPU，并且compute capacity > 3.5
   - 本机使用的是NVIDIA QUADRO K1200，compute capacity 5.0 （见上图）
   - GPU性能参考网址：[CUDA-GPU官网链接](https://developer.nvidia.com/cuda-gpus)
+ 显卡驱动安装：

>1. 显卡驱动下载：[GPU驱动官网链接](http://www.nvidia.com.tw/Download/index.aspx?lang=cn)

>2. 选择好对应的GPU，下载驱动程序。（Tips：我当初下载的时候选择中文有问题，需要下载英文的）。得到的文件名字类似于：NVIDIA-Linux-*.run

>3. chmod 777 NVIDIA-Linux-*.run　　#　修改权限，是的具有可执行权限

>4. ctrl + alt + f1　　　#　进入命令行模式

>5. sudo service lightdm stop　　#　关闭x-server

>6. ./NVIDIA-*.run　　#　安装驱动，需要root权限

---

## 环境安装：
1. 先安装CUDA
   - 参考网址：[安装CUDA](https://github.com/facebook/fbcunn/blob/master/INSTALL.md)
   
   > - CUDA下载网址：https://developer.nvidia.com/cuda-downloads  
   
   > - 下载得到类似cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb的文件名
   
   > - sudo apt-get install build-essential
   
   > - sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
   
   > - sudo apt-get update
   
   > - sudo apt-get install cuda
   
   > - echo "export PATH=/usr/local/cuda/bin/:\\$PATH; export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:\\$LD_LIBRARY_PATH; " >>~/.bashrc && source ~/.bashrc
   
   > - 重启电脑
   
   - 本人安装的是CUDA 7.5版本。　使用nvcc -v：测试版本
   - 完成后测试是否安装成功，编译Sample：
   
   > - cd usr/local/cuda-7.5/samples
   
   > - sudo make
   
   > - 之后进入 samples/bin/x86_64/linux/release
   
   > - sudo ./deviceQuery 　　#　显示NVIDIA显卡的相关信息
   
   > - 如下图，说明安装成功  
   
   > ![image](https://github.com/lingchenmsot/OpenObjectRecognition/blob/master/DeepMask/Markdown_images/deviceQuery.png?raw=true)
   
2. Torch安装
   - 参考网址：[安装torch](http://torch.ch/docs/getting-started.html#_)
      - 执行的是./install.sh命令
      - 并没有尝试安装TORCH_LUA_VERSION=LUA52 ./install.sh这个命令
   - 测试是否安装成功：   
    ![image](https://github.com/lingchenmsot/OpenObjectRecognition/blob/master/DeepMask/Markdown_images/th.png?raw=true)
   - 查看已经安装的包的命令：
      > luarocks list

3. Deepmask依赖的torch的包
   - image：安装torch的时候已经安装好
   - nnx：安装torch的时候已经安装好
   - optim：安装torch的时候已经安装好
   - tds:
      - 安装命令：luarocks install tds
   - cjson：
      - 安装命令：luarocks install json
   - cutorch：
      - 安装命令：luarocks install cutorch
   - cudnn：
   
   > - 下载地址：https://developer.nvidia.com/cudnn 这个网址下载cudnn的库，需要对应CUDA的版本。
   
   > - 我下载的是：Download cuDNN v5.1 (August 10,2016), for CUDA 7.5 -> cuDNN v5.1 Library for Linux这个链接的内容，下载得到的文件名称是：cudnn-7.5-linux-x64-v5.1.tgz
   
   > - tar -zxvf cudnn-7.5-linux-x64-v5.1.tgz
   
   > - cd cuda #进入解压之后得到的目录，该目录下有两个文件夹include和lib64
   
   > - sudo cp include/cudnn.h /usr/local/cuda/include/      #复制头文件
   
   > - sudo cp lib64/li* /usr/local/cuda/lib64/    #复制lib文件到cuda安装路径下的lib
   
   > - cd lib64，使用ls -al查看软连接
   
   > - cd /usr/local/cuda/lib64/
   
   > - sudo rm -rf libcudnn.so libcudnn.so.5    #删除原有的软链接
   
   > - sudo ln -s libcudnn.5.1.3 libcudnn.so.5
   
   > - sudo ln -s libcudnn.so.5 libcudnn.so
   
   > - 至此，能够在torch里面使用require "cudnn"不报错了
   
   - cunn：
      - 一开始一直报错，后来重新采用如下方式安装：
         - luarocks install nn
         - luarocks install cutorch
         - luatocks install cunn　　#　安装的时候会出一堆warning：removing non-existent dependency file。 ignored it
   - inn：
      - luarocks install inn
   - COCO API：
      - 未完待续
