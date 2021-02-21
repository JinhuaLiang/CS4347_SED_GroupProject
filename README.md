# CS4347_SED_GroupProject
This repo is designed for CS4347 Group Project. It is dedicated to provide a quick start for beginners who are interested in Sound Event Detection (SED). The goal of SED is to classify the events of a recording into a set of provided classes and to identify their corresponding time boundaries.

### Data preparation
The dataset can be directly downloaded from https://goo.gl/PJUVAd (for training set) and https://goo.gl/ip8JXW (for testing set). After downloading, users should prepare the data looks like:
```
dataset_root
├── training (51172 audios)
│    └── ...
├── testing (488 audios)
│    └── ...
└── metadata
     ├── testing_set.csv
     ├── groundtruth_strong_label_testing_set.csv
     ├── groundtruth_weak_label_testing_set.csv
     ├── groundtruth_weak_label_training_set.csv
     └── training_set.csv
```

### Get start with the provided code

#### 1. Install the required package
```
pip install -r requirements.txt
```

#### 2. Run runme.sh
```
sh ./runme.sh
```
runme.sh include:
* Modify the paths to your dataset and workplace.
* Select the model you want to develop.
* Pack the waveforms and targets to hdf5 files.
* Train a specific model.
* Calculate metrics on testset.
* (Optional) Inference on the evaluation dataset.


### Contact
Jinhua Liang (liangjh0903@gmail.com)

### Reference
[1] Mesaros, A. , Heittola, T. , Diment, A. , Elizalde, B. , & Virtanen, T. . (2017). DCASE 2017 Challenge setup: Tasks, datasets and baseline system. Detection & Classification of Acoustic Scenes & Events.<br>
    Code: https://github.com/ankitshah009/Task-4-Large-scale-weakly-supervised-sound-event-detection-for-smart-cars#1-direct-download-for-the-audio-of-the-development-and-evaluation-sets<br>
[2] Kong, Qiuqiang, Yong Xu, Wenwu Wang, and Mark D. Plumbley. "Sound Event Detection of Weakly Labelled Data with CNN-Transformer and Automatic Threshold Optimization." arXiv preprint arXiv:1912.04761 (2019).<br>
    Code: https://github.com/qiuqiangkong/sound_event_detection_dcase2017_task4<br>
[3] Xu, Y. , Kong, Q. , Wang, W. , & Plumbley, M. D. . (2018). Large-scale weakly supervised audio classification using gated convolutional neural network.<br>
    Code: https://github.com/yongxuUSTC/dcase2017_task4_cvssp
