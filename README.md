# Sound Event Detection system with FilterAugment

Official implementation of **Heavily Augmented Sound Event Detection utilizing Weak Predictions**<br>
by Hyeonuk Nam, Byeong-Yun Ko, Gyeong-Tae Lee, Seong-Hu Kim, Won-Ho Jung, Sang-Min Choi, Yong-Hwa Park @ Human Lab, Mechanical Engineering Department, KAIST

Ranked on **[3rd place]** in [IEEE DCASE 2021 Task 4](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments-results).

Paper is available in [arxiv](https://arxiv.org/abs/2107.03649) and [DCASE Technical report](http://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Nam_41_t4.pdf).

## FilterAugment
Filter Augment is an original data augmentation method for audio/speech representation learning. It randomly devides frequency domain into several bands, and then apply different amplitudes on each band. For more detail, refer to the paper mentioned above.<br>
![](./utils/FilterAugment_example.png)<br>
- This example shows FilterAugment applied on log-mel spectrogram of a 10-second audio clip. There are 3 frequency bands, and the low-frequency band(0 ~ 1kHz) is amplified, mid-frequency band(1 ~ 2.5kHz)is diminished while high-frequency band(2.5 ~ 8kHz) is just slightly diminished.

## Requirements and versions used
- pytorch: 1.8.0
- pytorchaudio: 0.8.0
- pytorch lightning: 1.2.4
- soundfile: 0.10.3
- sed_eval: 0.2.1
- psds_eval: 0.3.0
- scipy: 1.4.1
- pandas: 1.1.3
- numpy: 1.19.2
- asteroid: 0.4.4

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
```bib
@misc{nam2021heavily,
      title={Heavily Augmented Sound Event Detection utilizing Weak Predictions}, 
      author={Hyeonuk Nam and Byeong-Yun Ko and Gyeong-Tae Lee and Seong-Hu Kim and Won-Ho Jung and Sang-Min Choi and Yong-Hwa Park},
      year={2021},
      eprint={2107.03649},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
Please contact Hyeonuk Nam at frednam@kaist.ac.kr for any query.

