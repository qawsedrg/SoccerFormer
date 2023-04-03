# Environment

```shell
conda create -y -n ActionFormer python=3.7
conda activate ActionFormer
conda install -y cudnn cudatoolkit=10.1
pip install scikit-video tensorflow==2.3.0 imutils opencv-python==3.4.11.41 SoccerNet moviepy scikit-learn
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

# Evaluate on SoccorNet

## Download extracted feature and ground truth labels

```shell
python download.py
```

or from [baidunetdisk](https://pan.baidu.com/s/1Vdgd2veDxyGxENJYLedxug?pwd=cmka)

## Evaluation

```shell
python eval_soccernet.py
```

loose Average mAP on visible action ~ 44%

# Extract Feature

Place [videos](https://pan.baidu.com/s/17X3DzbD1vQ7DIhPL1Tm_mQ?pwd=2455) under ./data folder, under name of xxx.1.mp4 and xxx.2.mp4 for the first and second half of a game.

```shell
cd extraction
python VideoFeatureExtractor.py
```

which will transform xxx.1.mp4 and xxx.2.mp4 to xxx.1.npy and xxx.2.npy.

At least 20G memory is needed, ~3min for a video. 

# Inference

Extracts [models](https://pan.baidu.com/s/1sIwYYO5coxVMQ_V_L0P_xg?pwd=rtza) under ./models

```shell
python inference.py
```

Actions in xxx.1.mp4 and xxx.2.mp4 will be united in xxx.results_spotting.json

```
"gameTime": {which half} - {which secod in this half},
"label": action name,
"position": position,
"confidence": confidence
```

# Acknowledgement

This model is an adaptation of [ActionFormer](https://github.com/happyharrycn/actionformer_release) trained on [SoccerNet](https://github.com/SoccerNet/sn-spotting).
