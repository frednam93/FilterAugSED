# Sound Event Detection system with FilterAugment

**Heavily Augmented Sound Event Detection utilizing Weak Predictions**<br>
by Hyeonuk Nam, Byeong-Yun Ko, Gyeong-Tae Lee, Seong-Hu Kim, Won-Ho Jung, Sang-Min Choi, Yong-Hwa Park @ Human Lab, Mechanical Engineering Department, KAIST

Won 3rd place in [IEEE DCASE 2021 Task 4](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments-results).

Paper is available in [DCASE Technical report](http://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Nam_41_t4.pdf) and arxiv(will be submitted soon).

## Requirements
pytorch
pytorchaudio
pytorch lightning
asteroid
scipy
numpy
pathlib
soundfile

## Dataset
You can download dataset by reffering to [DCASE 2021 Task 4 description page](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments) or [DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task). Then, set the dataset directories in config.yaml accordingly. You will need DESED real datasets(weak/unlabeled in domain/validation/public eval) and DESED synthetic dataset(train/validation).

## Training
If you run main.py, it will train and save model in exps folder.
- `python main.py`

#### Results:

Dataset | **PSDS-scenario1** | **PSDS-scenario2** | *Intersection-based F1* | *Collar-based F1*
--------|--------------------|--------------------|-------------------------|-----------------
Dev-test| **0.353**          | **0.553**          | 79.5%                   | 42.1%

## Reference
[DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task)

## Citation & Contact
If this repository helped your research, please cite the paper below!(will be updated soon)

If you have any query, please contact Hyeonuk Nam at frednam@kaist.ac.kr

