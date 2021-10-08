# Sound Event Detection with FilterAugment

Official implementation of <br>
 - **Heavily Augmented Sound Event Detection utilizing Weak Predictions** (DCASE2021 Challenge Task 4 technical report) <br>
by Hyeonuk Nam, Byeong-Yun Ko, Gyeong-Tae Lee, Seong-Hu Kim, Won-Ho Jung, Sang-Min Choi, Yong-Hwa Park <br>
[![DCASE](https://img.shields.io/badge/DCASE-technical%20report-orange)](http://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Nam_41_t4.pdf) 
[![arXiv](https://img.shields.io/badge/arXiv-2107.03649-brightgreen)](https://arxiv.org/abs/2107.03649)<br>
 - **FilterAugment: An Acoustic Environmental Data Augmentation Method** (Submitted to ICASSP 2022) <br>
by Hyeonuk Nam, Seong-Hu Kim, Yong-Hwa Park <br>
[![arXiv](https://img.shields.io/badge/arXiv-2110.03282-brightgreen)](https://arxiv.org/abs/2110.03282) <br>
implementation of this paper will be updated soon!

Ranked on **[3rd place]** in [IEEE DCASE 2021 Task 4](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments-results).

## FilterAugment
Filter Augment is an audio data augmentation method newly proposed on the above papers for training acoustic models in audio/speech tasks. It applies random weights on randomly selected frequency bands. For more details, refer to the papers mentioned above.<br>
![](./utils/FilterAugment_example.png)<br>
- This example shows FilterAugment applied on log-mel spectrogram of a 10-second audio clip. There are 3 frequency bands, and the low-frequency band(0 ~ 1kHz) is amplified, mid-frequency band(1 ~ 2.5kHz)is diminished while high-frequency band(2.5 ~ 8kHz) is just slightly diminished.

To check the FilterAugment code, refer to function defined "filt_aug" at [utils/data_aug.py](./utils/data_aug.py) @ line 106

## Requirements
Python version of 3.7.10 is used with following libraries
- pytorch==1.8.0
- pytorch-lightning==1.2.4
- pytorchaudio==0.8.0
- scipy==1.4.1
- pandas==1.1.3
- numpy==1.19.2


other requrements in [requirements.txt](./requirements.txt)


## Datasets
You can download datasets by reffering to [DCASE 2021 Task 4 description page](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments) or [DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task). Then, set the dataset directories in [config yaml files](./configs/) accordingly. You need DESED real datasets (weak/unlabeled in domain/validation/public eval) and DESED synthetic datasets (train/validation).

## Training
You can train and save model in `exps` folder by running:
```shell
python main.py
```

#### model settings:
There are 4 model settings in the paper mentioned above. Default hyperparameter setting is set as [model1](./configs/config_model1.yaml), so running the code above will train model 1. To train model [2](./configs/config_model2.yaml), [3](./configs/config_model3.yaml) or [4](./configs/config_model4.yaml) from the paper, you can run the following code instead.
```shell
# for example, to train model 3:
python main.py --model 3
```

#### Results on DESED Real Validation dataset:

Model | PSDS-scenario1 | PSDS-scenario2 | Collar-based F1
------|----------------|----------------|-----------------
1     | 0.408          | 0.628          | 49.0%
2     | **0.414**      | 0.608          | 49.2%
3     | 0.381          | 0.660          | 31.8%
4     | 0.052          | **0.783**      | 19.8%


## Reference
[DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task)

## Citation & Contact
If this repository helped your works, please cite papers below!(will be updated soon)
```bib
@techreport{Nam2021,
    Author = "Nam, Hyeonuk and Ko, Byeong-Yun and Lee, Gyeong-Tae and Kim, Seong-Hu and Jung, Won-Ho and Choi, Sang-Min and Park, Yong-Hwa",
    title = "Heavily Augmented Sound Event Detection utilizing Weak Predictions",
    institution = "DCASE2021 Challenge",
    year = "2021",
    month = "June",
}

@article{nam2021filteraugment,
  title={FilterAugment: An Acoustic Environmental Data Augmentation Method},
  author={Hyeonuk Nam and Seoung-Hu Kim and Yong-Hwa Park},
  journal={arXiv preprint arXiv:2107.13260},
  year={2021}
}

```
Please contact Hyeonuk Nam at frednam@kaist.ac.kr for any query.

