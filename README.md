## 文件架构

每个子目录下的文件结构与basic_architecture相同，仅模型名字不同。

```
├── readme.md
├── report.pdf
├── video.mkv                          # the video of GP_model_7000 against pure MCTS
├── basic_architecture                 # all needed files for basic architecture
│   ├── game.py
│   ├── human_play.py
│   ├── mcts_alphaZero.py
│   ├── mcts_pure.py
│   ├── train.py
│   ├── policy_value_net_pytorch.py
│   └── basic_model.model
├── residual
│   ├── Res1                           # all needed files for 1 Residual block
│   │    ├── ...
│   │    ├── policy_value_net_pytorch_res1_1.py
│   │    └── Res1_model.model
│   ├── Res2                           # all needed files for 2 Residual blocks
│   │    ├── ...
│   │    ├── policy_value_net_pytorch_res1_2.py
│   │    └── Res2_model.model
│   ├── Res3                           # all needed files for 3 Residual blocks
│   │    ├── ...
│   │    ├── policy_value_net_pytorch_res1_3.py
│   │    └── Res3_model.model
│   └── Res5                           # all needed files for 5 Residual blocks
│        ├── ...
│        ├── policy_value_net_pytorch_res1_5.py
│        └── Res5_model.model
├── PCR                                # all needed files for PCR
│   ├── ...
│   ├── policy_value_net_pytorch.py
│   └── PCR.model
├── GP
│   ├── 1500_batch                     # all needed files for GP, trained for 1500 batches
│   │    ├── ...
│   │    ├── policy_value_net_withGAP_single.py
│   │    └── GP_model.model
│   └── 7000_batch                     # all needed files for GP, trained for 7000 batches
│        ├── ...
│        ├── train_gpoolsingle.py            # train from scratch
│        ├── train_gpoolsingle_stronger.py   # train on the basis of 1500-batch GP model
│        ├── policy_value_net_withGAP_single.py
│        ├── GP_model.model            # GP model trained for 1500 batches
│        └── GP_model_7000.model       # the best model for testing
├── GARB                               # all needed files for GABR
│   ├── ...
│   ├── policy_value_net_gabr.py
│   ├── 2RBs_1GARB.model
│   └── 3RBs_2GARBs.model
├── GP_padding                         # all needed files for GP with 3 paddings
│   ├── ...
│   ├── policy_value_net_withGAP_single.py
│   └── GP_padding_model.model
├── residual_GP
│   ├── GP_Res1                        # all needed files for GP with 1 residual block
│   │    ├── ...
│   │    ├── policy_value_net_gap_res1.py
│   │    └── GP_Res1_model.model
│   ├── GP_Res2                        # all needed files for GP with 2 residual blocks
│   │    ├── ...
│   │    ├── policy_value_net_gap_res2.py
│   │    └── GP_Res2_model.model
│   └── GP_Res4                        # all needed files for GP with 4 residual blocks
│        ├── ...
│        ├── policy_value_net_gap_res4.py
│        └── GP_Res4_model.model
├── PCR_GP                             # all needed files for PCR+GP
│   ├── ...
│   ├── policy_value_net_withGAP_single.py
│   └── PCR_GP.model
└── visualization                      # an example of visualization
    ├── game.py
    ├── mcts_alphaZero.py
    ├── mcts_pure.py
    ├── GUI.py
    ├── visualization_human.py
    ├── policy_value_net_withGAP_single_pad.py
    └── gp_padding3.model

```


### 注意
+ **GARB**
  GARB当前代码设置为3层residual blocks + 2层GARB，如果需要训练或测试2层residual blocks + 1层GARB，需要修改残差块层数。
+ **visualization**
  可视化文件独立于其余的human_play，由于需要在game.py中额外加入函数，和额外的可视化封装GUI的py文件,因此visualization代码仅示例展示使用GP_padding模型的效果。


## 创建环境
### basic_architecture, GARB, GP_padding

    conda create -n torch python=3.8
    source activate torch
    pip install torch==1.9.0
    pip install numpy==1.23.0
    
### PCR, PCR_GP

<font size=2>
    
    conda create -n torch python=3.8
    conda activate torch
    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch
    
</font>

### residual，GP，residual_GP

    conda create -n torch python=3.9.13
    conda activate torch
    pip install torch==2.0.1
    pip install torchvision==0.15.2
    pip install numpy==1.26.2

### visualization

    pip install pygame

## 运行训练脚本

在对应模型的目录下运行，如：对基础架构，切换到./basic_architecture目录中运行以下代码。

    python train.py

在./GP/7000_batch目录下，若要从头开始训练，则运行以下代码。

    python train_gpoolsingle.py

在./GP/7000_batch目录下，若要在训练了1500 batch的GP_model.model的基础上继续训练，则运行以下代码。

    python train_gpoolsingle_stronger.py

## 运行 pure MCTS 测试脚本

在对应模型的目录下运行，如：对基础架构，切换到./basic_architecture目录中运行以下代码。

    python game.py

注意，此时需要先运行训练模型，在对应模型的文件夹内存有训练好的模型。
其中，./GP/7000_batch/GP_model_7000.model的性能最优。

## 运行 Human 测试脚本

在对应模型的目录下运行，如：对基础架构，切换到./basic_architecture目录中运行以下代码。

    python human_play.py

注意，此时需要先运行训练模型，在对应模型的文件夹内存有训练好的模型。

## 运行 visualization 测试脚本

在visualization的目录下运行以下代码

    python visualization_human.py
    
这里已经提供了GP_padding3模型的环境，直接运行即可。

如果需要运行其他的模型，则需要模型相应的网络py文件，在game和visualization_human中将对应的模型和网络进行修改。注意，因为这个只能本地跑，所以在cpu环境下需要把网络policy_value_net的网络文件中的所有use_gpu改为"False"。