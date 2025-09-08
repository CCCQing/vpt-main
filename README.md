# Visual Prompt Tuning 

https://arxiv.org/abs/2203.12119 

------

This repository contains the official PyTorch implementation for Visual Prompt Tuning.

![vpt_teaser](https://github.com/KMnP/vpt/blob/main/imgs/teaser.png)

## Environment settings

See `env_setup.sh`

## Structure of the this repo (key files are marked with 👉):

- `src/configs`: handles config parameters for the experiments.
                               实验的主要配置设置及其每个配置的解释。
  * 👉 `src/config/config.py`: <u>main config setups for experiments and explanation for each of them. </u> 
              加载和设置输入数据集。这些内容src/data/vtab_datasets借鉴自  
- `src/data`: loading and setup input datasets. The `src/data/vtab_datasets` are borrowed from 

  [VTAB github repo](https://github.com/google-research/task_adaptation/tree/master/task_adaptation/data).

                主要的训练和评估活动在这里。
- `src/engine`: main training and eval actions here.
                处理不同微调协议的主干拱门和头部
- `src/models`: handles backbone archs and heads for different fine-tuning protocols 
                                一个文件夹包含vit_backbones与 VPT 指定的文件夹中相同的骨干文件。此文件夹应包含与 vit_backbones
    * 👉`src/models/vit_prompt`: <u>a folder contains the same backbones in `vit_backbones` folder,</u> specified for VPT. This folder should contain the same file names as those in  `vit_backbones`
                                    基于 Transformer 的模型的主模型 ❗️注意❗️：当前版本仅支持 ViT、Swin 以及带有 mae、moco-v3 的 ViT
    * 👉 `src/models/vit_models.py`: <u>main model for transformer-based models</u> ❗️Note❗️: Current version only support ViT, Swin and ViT with mae, moco-v3
                                    这里的主要操作是利用配置并构建模型来训练/评估。
    * `src/models/build_model.py`: main action here to utilize the config and build the model to train / eval.

- `src/solver`: optimization, losses and learning rate schedules.  优化、损失和学习率计划。
- `src/utils`: helper functions for io, loggings, training, visualizations. 用于 io、日志、训练、可视化的辅助函数。
- 👉`train.py`: call this one for training and eval a model with a specified transfer type. 调用这个来训练并评估具有指定传输类型的模型。
  -                 调用此脚本来调整具有指定迁移类型的模型的学习率和权重衰减。我们将此脚本用于 FGVC 任务。
- 👉`tune_fgvc.py`: call this one for tuning learning rate and weight decay for a model with a specified transfer type. We used this script for FGVC tasks.
  -                  调用此方法调整 vtab 任务：使用 800/200 分割来找到最佳 lr 和 wd，并使用最佳 lr/wd 进行最终运行
- 👉`tune_vtab.py`: call this one for tuning vtab tasks: use 800/200 split to find the best lr and wd, and use the best lr/wd for the final runs
- `launch.py`: contains functions used to launch the job. 包含用于启动作业的功能。

## Experiments

### Key configs:

- 🔥VPT related:
  - MODEL.PROMPT.NUM_TOKENS: prompt length 提示长度
  - MODEL.PROMPT.DEEP: deep or shallow prompt 深提示或浅提示
- Fine-tuning method specification: 微调方法说明
  - MODEL.TRANSFER_TYPE 模型.传输类型
- Vision backbones:
  - DATA.FEATURE: specify which representation to use 指定要使用的表示形式
  - MODEL.TYPE: the general backbone type, e.g., "vit" or "swin" 通用骨干类型，例如“vit”或“swin”
  - MODEL.MODEL_ROOT: folder with pre-trained model checkpoints 包含预训练模型检查点的文件夹
- Optimization related: 优化相关
  - SOLVER.BASE_LR: learning rate for the experiment 实验的学习率
  - SOLVER.WEIGHT_DECAY: weight decay value for the experiment 实验的权重衰减值
  - DATA.BATCH_SIZE 数据.批量大小
- Datasets related:
  - DATA.NAME 数据名称
  - DATA.DATAPATH: where you put the datasets 放置数据集的位置
  - DATA.NUMBER_CLASSES
- Others: 
  - RUN_N_TIMES: ensure only run once in case for duplicated submision, not used during vtab runs 确保只运行一次，以防重复提交，在 vtab 运行期间不使用
  - OUTPUT_DIR: output dir of the final model and logs 最终模型和日志的输出目录
  - MODEL.SAVE_CKPT: if set to `True`, will save model ckpts and final output of both val and test set 如果设置为True，将保存模型 ckpts 以及 val 和测试集的最终输出

### Datasets preperation:

See Table 8 in the Appendix for dataset details. 

- Fine-Grained Visual Classification tasks (FGVC): The datasets can be downloaded following the official links. We split the training data if the public validation set is not available. The splitted dataset can be found here: [Dropbox](https://cornell.box.com/v/vptfgvcsplits), [Google Drive](https://drive.google.com/drive/folders/1mnvxTkYxmOr2W9QjcgS64UBpoJ4UmKaM?usp=sharing). 

  - [CUB200 2011](https://data.caltech.edu/records/65de6-vp158)

  - [NABirds](http://info.allaboutbirds.org/nabirds/)

  - [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/)

  - [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)

  - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
视觉任务适应基准（VTAB）：请参阅VTAB_SETUP.md详细说明和提示。
- [Visual Task Adaptation Benchmark](https://google-research.github.io/task_adaptation/) (VTAB): see [`VTAB_SETUP.md`](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md) for detailed instructions and tips.

### Pre-trained model preperation 预训练模型准备

Download and place the pre-trained Transformer-based backbones to `MODEL.MODEL_ROOT` (ConvNeXt-Base and ResNet50 would be automatically downloaded via the links in the code). Note that you also need to rename the downloaded ViT-B/16 ckpt from `ViT-B_16.npz` to `imagenet21k_ViT-B_16.npz`.

See Table 9 in the Appendix for more details about pre-trained backbones.
下载并放置预先训练好的基于 Transformer 的主干网络到MODEL.MODEL_ROOT（ConvNeXt-Base 和 ResNet50 将通过代码中的链接自动下载）。请注意，您还需要将下载的 ViT-B/16 ckpt 从 重命名ViT-B_16.npz为imagenet21k_ViT-B_16.npz。

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Pre-trained Backbone</th>
<th valign="bottom">Pre-trained Objective</th>
<th valign="bottom">Link</th>
<th valign="bottom">md5sum</th>
<!-- TABLE BODY -->
<tr><td align="left">ViT-B/16</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz">link</a></td>
<td align="center"><tt>d9715d</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MoCo v3</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar">link</a></td>
<td align="center"><tt>8f39ce</tt></td>
</tr>
<tr><td align="left">ViT-B/16</td>
<td align="center">MAE</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth">link</a></td>
<td align="center"><tt>8cad7c</tt></td>
</tr>
<tr><td align="left">Swin-B</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth">link</a></td>
<td align="center"><tt>bf9cc1</tt></td>
</tr>
<tr><td align="left">ConvNeXt-Base</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth">link</a></td>
<td align="center"><tt>-</tt></td>
</tr>
<tr><td align="left">ResNet-50</td>
<td align="center">Supervised</td>
<td align="center"><a href="https://pytorch.org/vision/stable/models.html">link</a></td>
<td align="center"><tt>-</tt></td>
</tr>
</tbody></table>

### Examples for training and aggregating results 训练和汇总结果的示例
请参阅demo.ipynb如何使用这个 repo。
See [`demo.ipynb`](https://github.com/KMnP/vpt/blob/main/demo.ipynb) for how to use this repo.

### Hyperparameters for experiments in paper 论文中实验的超参数
表 1-2、图 3-4、表 4-5 中使用的超参数值（VPT 的提示长度/适配器的减少率、基础学习率、权重衰减值）可以在这里找到
The hyperparameter values used (prompt length for VPT / reduction rate for Adapters, base learning rate, weight decay values) in Table 1-2, Fig. 3-4, Table 4-5 can be found here: [Dropbox](https://cornell.box.com/s/lv10kptgyrm8uxb6v6ctugrhao24rs2z) / [Google Drive](https://drive.google.com/drive/folders/1ldhqkXelHDXq4bG7qpKn5YEfU6sRehJH?usp=sharing). 

## Citation

If you find our work helpful in your research, please cite it as:

```
@inproceedings{jia2022vpt,
  title={Visual Prompt Tuning},
  author={Jia, Menglin and Tang, Luming and Chen, Bor-Chun and Cardie, Claire and Belongie, Serge and Hariharan, Bharath and Lim, Ser-Nam},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## License

The majority of VPT is licensed under the CC-BY-NC 4.0 license (see [LICENSE](https://github.com/KMnP/vpt/blob/main/LICENSE) for details). Portions of the project are available under separate license terms: GitHub - [google-research/task_adaptation](https://github.com/google-research/task_adaptation) and [huggingface/transformers](https://github.com/huggingface/transformers) are licensed under the Apache 2.0 license; [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) and [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) are licensed under the MIT license; and [MoCo-v3](https://github.com/facebookresearch/moco-v3) and [MAE](https://github.com/facebookresearch/mae) are licensed under the Attribution-NonCommercial 4.0 International license.
