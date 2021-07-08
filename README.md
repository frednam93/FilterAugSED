# Sound Event Detection system with FilterAugment

Official implementation of **Heavily Augmented Sound Event Detection utilizing Weak Predictions**<br>
by Hyeonuk Nam, Byeong-Yun Ko, Gyeong-Tae Lee, Seong-Hu Kim, Won-Ho Jung, Sang-Min Choi, Yong-Hwa Park @ Human Lab, Mechanical Engineering Department, KAIST

Ranked on **3rd place** in [IEEE DCASE 2021 Task 4](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments-results).

Paper is available in [DCASE Technical report](http://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Nam_41_t4.pdf) and arxiv(will be submitted soon).

## FilterAugment
Filter Augment is an original data augmentation method for audio/speech representation learning. It randomly devides frequency domain into several bands, and then apply different amplitudes on each band. For more detail, refer to the paper mentioned above.<br>
![](./utils/FilterAugment_example.png)<br>
- This example shows FilterAugment applied on log-mel spectrogram of a 10-second audio clip. There are 3 frequency bands, and the low-frequency band(0 ~ 1kHz) is amplified, mid-frequency band(1 ~ 2.5kHz)is diminished while high-frequency band(2.5 ~ 8kHz) is just slightly diminished.

## Requirements
- pytorch
- pytorchaudio
- pytorch lightning
- scipy
- numpy
- pathlib
- soundfile
- asteroid

## Dataset
You can download dataset by reffering to [DCASE 2021 Task 4 description page](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments) or [DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task). Then, set the dataset directories in config.yaml accordingly. You will need DESED real datasets (weak/unlabeled in domain/validation/public eval) and DESED synthetic datasets (train/validation).

## Training
You can train and save model in `exps` folder by running:
```shell
python main.py
```

#### Results:

Dataset              | PSDS-scenario1 | PSDS-scenario2 | Collar-based F1
---------------------|----------------|----------------|-----------------
DESED Real Validation| 0.405          | 0.616          | 48.0%

## Reference
[DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task)

## Citation & Contact
If this repository helped your research, please cite the paper below!(will be updated soon)

Please contact Hyeonuk Nam at frednam@kaist.ac.kr for any query.

