# Deepmask train手册
前文阅读：[Deepmask环境搭建](https://github.com/lingchenmsot/OpenObjectRecognition/blob/deepmask/DeepMask/Deepmask_ReadMe.md)

本文参考链接：[Deepmask Train常见问题](https://gist.github.com/ryanfb/13bd5cf3d89d6b5e8acbd553256507f2)

***
## 基于COCO数据集训练
先介绍基于COCO数据集进行Train的步骤，以及运行过程中遇到的问题和解决办法。
### 准备工作
1. 参考[Deepmask Github网址](https://github.com/facebookresearch/deepmask)上Training Your Own Model模块下的步骤
2. 需要对下载得到的instances_train-val2014.zip、train2014.zip、val2014.zip 这三个文件进行解压。

### 遇到问题：
- 运行th train.lua遇到问题如下：
```
-- ignore option rundir
-- ignore option dm
-- ignore option reload
-- ignore option gpu
-- ignore option datadir
| running in directory /....../deepmask/exps/deepmask/exp
| number of paramaters trunk: 15198016
| number of paramaters mask branch: 1608768
| number of paramaters score branch: 526337
| number of paramaters total: 17333121
convert: data//annotations/instances_train2014.json --> .t7 [please be patient]
FATAL THREAD PANIC: (write) not enough memory
```
这时候进行的工作是将 $DEEPMASK/data/annotations/文件夹下的 instances_train2014.json 以及 instances_val2014.json 转换为t7格式的文件。
   - **REASON**：
   这是由于在安装Torch的时候安装的LuaJit，LuaJit有memory limitation。
   - **SOLUTION**：
   参考链接：https://github.com/facebookresearch/deepmask/issues/14
      1. 重新安装torch，安装lua5.2，([参考Torch安装的网址](http://torch.ch/docs/getting-started.html#_))，命令如下：
      ```shell
      # clean old torch installation
      ./clean.sh
      # optional clean command (for older torch versions)
      # curl -s https://raw.githubusercontent.com/torch/ezinstall/master/clean-old.sh | bash

      # https://github.com/torch/distro : set env to use lua
      TORCH_LUA_VERSION=LUA52 ./install.sh
      ```
      2. 重新torch之后，需要重新安装一些包：
      ```shell
      # reinstall four packages: coco API, tds, cjson, inn
      
      # install tds
      luarocks install tds
      
      # install cjson
      luarocks install json
      
      # install inn
      luarocks install inn
      
      # install coco API
      luarocks make LuaAPI/rocks/coco-scm-1.rockspec # run this command under coco/
      ```
      
      3. bring up an interactive torch shell with th and load the annotations to do the .t7 conversion outside the training process:
      ```shell
      th> coco = require 'coco'
                                                                      [0.0000s]
      th> coco.CocoApi("data/annotations/instances_train2014.json")
      convert: data/annotations/instances_train2014.json --> .t7 [please be patient]
      converting: annotations
      converting: categories
      converting: images
      convert: building indices
      convert: complete [57.22 s]
      CocoApi
                                                                      [58.0662s]
      th> coco.CocoApi("data/annotations/instances_val2014.json")
      convert: data/annotations/instances_val2014.json --> .t7 [please be patient]
      converting: annotations
      converting: categories
      converting: images
      convert: building indices
      convert: complete [26.07 s]
      CocoApi
                                                                      [26.3127s]
      th>
      ```
      这一步已经完成了将.json文件转换为.t7的工作，可以在$DEEPMASK/data/annotations/文件夹下面看到两个新的文件：instances_train2014.t7和instances_val2014.t7。有了这两个，再运行th train.lua的时候就不会在执行.json -> .t7的转换了。
      4. 基于lua5.2的torch运行th train.lua之后，又会出现一些问题。我的解决办法是重新切换为原来的LuaJit的torch。按照上面的方法重新安装torch。

- 解决上述问题之后，运行th train.lua遇到如下问题:
```
   -- ignore option rundir
   -- ignore option dm
   -- ignore option reload
   -- ignore option gpu
   -- ignore option datadir

   | running in directory /....../deepmask/exps/deepmask/exp
   | number of paramaters trunk: 15198016
   | number of paramaters mask branch: 1608768
   | number of paramaters score branch: 526337
   | number of paramaters total: 17333121
   | start training
   THCudaCheck FAIL file=/tmp/luarocks_cutorch-scm-1-7759/cutorch/lib/THC/generic/THCStorage.cu line=40 error=2 : out of memory
   /Users/ryan/source/torch/install/bin/luajit: ...ryan/source/torch/install/share/lua/5.1/nn/Container.lua:67:
   In 1 module of nn.Sequential:
   In 7 module of nn.Sequential:
   In 3 module of nn.Sequential:
   In 1 module of nn.Sequential:
   ...an/source/torch/install/share/lua/5.1/nn/ConcatTable.lua:68: cuda runtime error (2) : out of memory at    /tmp/luarocks_cutorch-scm-1-7759/cutorch/lib/THC/generic/THCStorage.cu:40
   stack traceback:
        [C]: in function 'resizeAs'
        ...an/source/torch/install/share/lua/5.1/nn/ConcatTable.lua:68: in function <...an/source/torch/install/share/lua/5.1/nn/ConcatTable.lua:30>
        [C]: in function 'xpcall'
        ...ryan/source/torch/install/share/lua/5.1/nn/Container.lua:63: in function 'rethrowErrors'
        ...yan/source/torch/install/share/lua/5.1/nn/Sequential.lua:88: in function <...yan/source/torch/install/share/lua/5.1/nn/Sequential.lua:78>
        [C]: in function 'xpcall'
        ...ryan/source/torch/install/share/lua/5.1/nn/Container.lua:63: in function 'rethrowErrors'
        ...yan/source/torch/install/share/lua/5.1/nn/Sequential.lua:84: in function <...yan/source/torch/install/share/lua/5.1/nn/Sequential.lua:78>
        [C]: in function 'xpcall'
        ...ryan/source/torch/install/share/lua/5.1/nn/Container.lua:63: in function 'rethrowErrors'
        ...yan/source/torch/install/share/lua/5.1/nn/Sequential.lua:84: in function <...yan/source/torch/install/share/lua/5.1/nn/Sequential.lua:78>
        [C]: in function 'xpcall'
        ...ryan/source/torch/install/share/lua/5.1/nn/Container.lua:63: in function 'rethrowErrors'
        ...yan/source/torch/install/share/lua/5.1/nn/Sequential.lua:88: in function 'backward'
        /Users/ryan/mess/2016/34/deepmask/TrainerDeepMask.lua:88: in function 'train'
        train.lua:117: in main chunk
        [C]: in function 'dofile'
        ...urce/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:145: in main chunk
        [C]: at 0x010f24dcf0

   WARNING: If you see a stack trace below, it doesn't point to the place where this error occurred. Please use only the one above.
stack traceback:
        [C]: in function 'error'
        ...ryan/source/torch/install/share/lua/5.1/nn/Container.lua:67: in function 'rethrowErrors'
        ...yan/source/torch/install/share/lua/5.1/nn/Sequential.lua:88: in function 'backward'
        /Users/ryan/mess/2016/34/deepmask/TrainerDeepMask.lua:88: in function 'train'
        train.lua:117: in main chunk
        [C]: in function 'dofile'
        ...urce/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:145: in main chunk
        [C]: at 0x010f24dcf0
```
   - **REASON**：
      - ```cuda runtime error (2) : out of memory```

   - **SOLUTION**：
      - 运行命令：

      ``` th train.lua -batch 8```
      - 查看选项：

      ``` th train.lua --help ```
***

##基于自建数据集训练
to be continued