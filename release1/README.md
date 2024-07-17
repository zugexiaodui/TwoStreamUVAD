# Context Recovery and Knowledge Retrieval: A Novel Two-Stream Framework for Video Anomaly Detection [[arXiv](https://arxiv.org/abs/2209.02899)]

## 声明
本代码为*Context Recovery and Knowledge Retrieval: A Novel Two-Stream Framework for Video Anomaly Detection (Congqo Cao, Yue Lu, Yanning Zhang)* 论文代码，<del>目前暂未开源，本代码不代表最终开源版本，仅供感兴趣的读者研究使用。</del>

## 环境配置
本代码在pytorch1.8.1下运行，其他接近版本亦可，作者尝试过其他版本暂未发现不兼容的库。其他需要的库请参考各`.py`文件的`import`。

硬件环境建议使用3090Ti显卡或V100显卡，训练模型通常使用一块显存>16GB的显卡就可以，测试时可同时使用多块显卡并行测试加快速度。


## 运行前说明
在训练或测试时，会自动创建几个`save.{ckpts, feats, logs, results, tbxs}`文件夹，其中保存了日志、检查点或测试结果等文件，为了避免在本目录下占用太大空间，可提前手动在外置大硬盘下创建，然后软连接到本目录下。

建议的目录设置如下：
```
.
├── data.Ave
├── data.ST
├── data.Corridor
├── data.model
│   └── I3D_8x8_R50_pre.pth
├── release1
│   ├── 1-train_stream_CR_Ave.sh
│   ├── 1-train_stream_CR.py
│   ├── 1-train_stream_CR_ST.sh
│   ├── 2-test_stream_CR_Ave.sh
│   ├── 2-test_stream_CR.py
│   ├── 2-test_stream_CR_ST.sh
│   ├── 3-feat_stream_KR_Ave.sh
│   ├── 3-feat_stream_KR.py
│   ├── 3-feat_stream_KR_ST.sh
│   ├── 4-train_stream_KR_Ave.sh
│   ├── 4-train_stream_KR.py
│   ├── 4-train_stream_KR_ST.sh
│   ├── 5-test_stream_KR_Ave.sh
│   ├── 5-test_stream_KR.py
│   ├── 5-test_stream_KR_ST.sh
│   ├── 6-two_stream_fuse_score.py
│   ├── 6-two_stream_fuse_score.sh
│   ├── argmanager.py
│   ├── dsets.py
│   ├── hashnet.py
│   ├── il2sh.py
│   ├── metrics.py
│   ├── misc.py
│   ├── mle_pseudo_data.py
│   ├── plan_task.py
│   ├── README.md
│   └── stunet.py
├── save.ckpts
├── save.feats
├── save.logs
├── save.results
├── save.tbxs
```
[I3D_8x8_R50_pre.pth - GoogleDrive](https://drive.google.com/file/d/13u9d1lzvUa85G4OcG6zAsZsK6vcKFw33/view?usp=drive_link)

运行bash脚本时进入release1目录下，通过`bash **.sh`运行。

本仓库主要以ShanghaiTech(ST)和Avenue为主说明运行方法，其中的超参数可能不是最优的设置。Corridor数据集由于数量较大，作者在实验时单独针对该数据集做了数据加载方面的修改优化，读者如要实验根据需要自己修改代码，注意Corridor上默认使用的3个crop是手动选取的3个位置，可参考论文中的说明，本仓库未提供该处理步骤，在不使用手动3 crop的情况下，选择的是15(5\*3)个480\*480的crop区域。

## Context Recovery流模型
### 训练和测试代码
本部分代码主要包含`1-train_stream_CR.py`，`2-test_stream_CR.py`和`mle_pseudo_data.py`文件。

前两个训练和测试程序通过对应的bash脚本调用，比如在ShanghaiTech数据集上训练，就运行`bash 1-train_stream_CR_ST.sh`程序，其中的数据设置可查看该bash脚本的说明。

第三个是为最大局部误差MLE生成伪异常数据的文件，文件内根据需要修改`__main__`函数的路径设置。生成伪异常数据之后，需要对伪异常数据做测试才能得到确定MLE的最佳窗口参数(对应`2-test~.py`中的`patch_size`参数)，测试脚本可以自己写一下，或省略本步骤，直接查看不同`patch_size`的测试结果。

## Knowledge Retrieval流模型
### 提取特征
使用`3-feat_stream_KR.py`提取特征，保存的特征用于后续`4-`和`5-`训练测试iL2SH。

### 训练和测试代码
运行`4-train_stream_KR_ST/Ave.sh`训练，`5-test_stream_KR_Ave.sh`测试，具体参数请查看脚本。

## 融合结果
运行`6-two_stream_fuse_score.sh`，修改其中的参数即可。