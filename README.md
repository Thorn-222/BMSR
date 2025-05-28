# BMSR：A Bidirectional Multi-hop Predictor with Structure-aware Ranking for NAS
The implementation for BMSR：A Bidirectional Multi-hop Predictor with Structure-aware Ranking for NAS
![The overview of BMSR](./images/BMSR.png)
## Requirements
```
python == 3.8.20 
pytorch == 2.4.1
torchvision == 0.19.1
scipy == 1.10.1
nasbench == 1.0
nas_201_api == 2.1
```
## Dataset Preparation

This project uses three datasets: NAS-Bench-101, NAS-Bench-201, and DARTS.

**NAS-Bench-101:**

Project: https://github.com/google-research/nasbench

Dataset: https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord

After downloading, place the dataset under: `NAS-Bench-101/path`.
Then simply run `NAS-Bench-101/main.py`, which will automatically generate the preprocessed data file:`tiny_nas_bench_101_test.pkl` before training.

**NAS-Bench-201:**

Project: https://github.com/D-X-Y/NAS-Bench-201

Dataset: https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_/view

After downloading, place the dataset under: `NAS-Bench-201/data`.
Then run the preprocessing script `NAS-Bench-201/pth_2_npy.py` to generate:`nasbench201_dict_search.npy`.

**DARTS:**

Project: https://github.com/quark0/darts

We randomly selected 100 architectures from the DARTS search space and fully trained them to obtain accuracy labels.
The preprocessed dataset file `darts_data.pkl` is already included in `DARTS/path` and can be used directly.

## How to use

After confirming that the data sets of each space are stored in the designated location, training and testing are performed according to the following process.

## NAS-Bench-101

- To train BMSR using NAS-Bench-101, you can run:
```
cd NAS-Bench-101
python main.py --seed 777 --test False
```
You can change the size of the training set by modifying the partition range in `NAS-Bench-101/loader.py`.

- To evaluate BMSR’s overall Kendall’s Tau and to search for high-performing architectures on NAS-Bench-101, you can run:
```
cd NAS-Bench-101
python main.py --seed 777 --test True
```
