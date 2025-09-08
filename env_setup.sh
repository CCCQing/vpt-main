conda create -n prompt python=3.7
conda activate prompt

pip install -q tensorflow
# specifying tfds versions is important to reproduce our results
pip install tfds-nightly==4.4.0.dev202201080107
pip install opencv-python
pip install tensorflow-addons   暂时不考虑:
     官方长期没有提供 Windows 预编译轮子pip 往往装不到而且该项目已停止开发仅做了有限维护到 2024-05建议：跳过 这个包；如果项目里没用到 tfa 的 API，直接不装即可。如果代码里确实 import tensorflow_addons as tfa，要么把相关用法替换为等价的 Keras/TF 实现（常见替代我在下面给了提示），要么在 WSL2/Ubuntu 或 Linux 机器上装
pip install mock


conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
pip install opencv-python

conda install tqdm pandas matplotlib seaborn scikit-learn scipy simplejson termcolor
conda install -c iopath iopath


# for transformers
pip install timm==0.4.12
pip install ml-collections

# Optional: for slurm jobs
pip install submitit -U
pip install slurm_gpustat




安装指令总结

conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 `
  -c pytorch -c defaults --override-channels --solver classic

conda install -y tqdm==4.64.1
conda install -y pandas==1.3.5
conda install -y scikit-learn==0.24.2
conda install -y scipy==1.7.3
conda install -y simplejson==3.17.6
conda install -y termcolor==1.1.0


conda install -c iopath iopath -y

conda config --append channels conda-forge
conda install timm==0.4.12
conda install ml-collections==0.1.1
conda install seaborn==0.11.2
conda install opencv-python==4.5.5.64     1
conda install mock==4.0.3

conda install -y pandas==1.3.5
conda install -c conda-forge opencv       1
conda install dill==0.3.4
conda install attrs==21.4.0
conda install etils==0.8.0
conda install promise==2.3
conda install psutil==5.9.5
pip install --no-deps D:\postgraduate1\project\vpt-main\tfds_nightly-4.4.0.dev202201080107-py3-none-any.whl
conda install -n prompt fvcore=0.1.5.post20221221 -c conda-forge --override-channels --solver classic
pip install D:\postgraduate1\project\vpt-main\pillow-9.3.0-cp37-cp37m-win_amd64.whl
conda install -n prompt libjpeg-turbo zlib vs2015_runtime -c conda-forge --override-channels --solver classic
还是不要用离线下载用conda下吧
conda install -n prompt pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0  -c pytorch -c defaults --override-channels --solver classic
https://pypi.org/project/tfds-nightly/4.4.0.dev202201080107/#files

cd D:\postgraduate1\project\vpt-main\vpt-main