# dcase2026_task6_baseline
[QD-DETR](https://github.com/wjun0830/QD-DETR)-based baseline for DCASE 2026 challenge task 6.

## Model architecture
The model is based on QD-DETR, a Transformer-based encoder-decoder architecture. An overview architecture is described in Figure 2 in the [paper](https://arxiv.org/pdf/2303.13874).
Given an audio and text pair, [CLAP](https://github.com/microsoft/CLAP) encodes them into audio and text features, respectively.
These features are then forwarded into the cross-attention transformers, followed by the Transformer decoder.
Finally, the model outputs multiple candidate moments with start/end timestamps and confidence scores.

## Getting started
0. Clone this repository
```
git clone https://github.com/awkrail/dcase2026_task6_baseline.git
```
1. Install Pytorch & dependency libraries
Install pytorch, torchvision, and torchaudio based on your GPU environments. Note that the inference API is available for CPU environments. We tested the codes on Python 3.9 and CUDA 11.8:
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
2. Prepare feature files
Download [CASTELLA dataset](https://zenodo.org/records/18358706) and [Clotho-Moment dataset](https://zenodo.org/records/17129257).
```
wget https://zenodo.org/records/18358706/files/clap.tar.gz
wget https://zenodo.org/records/18358706/files/clap_text.tar.gz
tar -zxvf clap.tar.gz
tar -zxvf clap_text.tar.gz
```

```
wget https://zenodo.org/api/records/17129257/files-archive
cat clotho-moment_features.tar.part-* > clotho-moment_features.tar
tar -xvf clotho-moment_features.tar
```

These feature files are also available in HuggingFace.
- [CASTELLA dataset](https://huggingface.co/datasets/lighthouse-emnlp2024/CASTELLA_CLAP_features)
- [Clotho-Moment dataset](https://huggingface.co/datasets/lighthouse-emnlp2024/Clotho-Moment_CLAP_features)


## Training and evaluation
0. Train a model
```
python src/train.py --config config.yml  
```
- `config.yml` is for CASTELLA. If you train models on Clotho-Moment, use `config-pretraining.yml`
- If you use pre-trained model weights, use `--resume ./**/{checkpoint}.pth`


1. Evaluation
Reproduce the evaluation on the `val` set.
```
python src/evaluate.py --config config.yml --model_path results/best_checkpoint.pth
```
The result is:
```
2026-03-30 01:14:08.441:INFO:__main__ - Setup config, data and model...
2026-03-30 01:14:08.442:INFO:__main__ - setup model/optimizer/scheduler
2026-03-30 01:14:08.885:INFO:__main__ - CUDA enabled.
2026-03-30 01:14:09.264:INFO:__main__ - Model checkpoint: results/best_checkpoint.pth
2026-03-30 01:14:09.264:INFO:__main__ - Starting inference...
2026-03-30 01:14:09.264:INFO:__main__ - Generate submissions
compute st ed scores: 100%|███████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.93it/s]
convert to multiples of clip_length=1: 100%|███████████████████████████| 352/352 [00:00<00:00, 28908.68it/s]
2026-03-30 01:14:10.652:INFO:__main__ - Saving/Evaluating before nms results
full: [0, 1500], 352/352=100.00 examples.
[eval_moment_retrieval] [full] 0.12 seconds
2026-03-30 01:14:10.795:INFO:__main__ - metrics_no_nms OrderedDict([   ('MR-full-R1@0.5', 27.56),
                ('MR-full-R1@0.7', 16.19),
                ('MR-full-mAP', 11.44),
                ('MR-full-mAP@0.5', 24.02),
                ('MR-full-mAP@0.75', 10.26)])
```

Reproduce the evaluation on the `test` set:
```
python src/evaluate.py --config config.yml --split test --model_path results/best_checkpoint.pth
```
The result is:
```
2026-03-30 01:14:48.156:INFO:__main__ - Setup config, data and model...
2026-03-30 01:14:48.160:INFO:__main__ - setup model/optimizer/scheduler
2026-03-30 01:14:48.599:INFO:__main__ - CUDA enabled.
2026-03-30 01:14:48.986:INFO:__main__ - Model checkpoint: results/best_checkpoint.pth
2026-03-30 01:14:48.986:INFO:__main__ - Starting inference...
2026-03-30 01:14:48.986:INFO:__main__ - Generate submissions
compute st ed scores: 100%|█████████████████████████████████████████████████| 14/14 [00:02<00:00,  5.44it/s]
convert to multiples of clip_length=1: 100%|█████████████████████████| 1347/1347 [00:00<00:00, 28259.52it/s]
2026-03-30 01:14:51.617:INFO:__main__ - Saving/Evaluating before nms results
full: [0, 1500], 1347/1347=100.00 examples.
[eval_moment_retrieval] [full] 0.24 seconds
2026-03-30 01:14:51.886:INFO:__main__ - metrics_no_nms OrderedDict([   ('MR-full-R1@0.5', 23.16),
                ('MR-full-R1@0.7', 10.32),
                ('MR-full-mAP', 9.11),
                ('MR-full-mAP@0.5', 20.34),
                ('MR-full-mAP@0.75', 6.96)])
```

## Preparation for submission.jsonl
Run the following command to create submission file. (Evaluation data for the submission will be publicly available on June 1, and the script will work after that.)
```
python src/create_submission.py --config config.yml --model_path results/best_checkpoint.pth
```
You can get `private_submission.jsonl` file under `results` directory. For details, please read [this README.md](src/standalone_eval/README.md)

## Statistics of scores
Scores may vary slightly due to different random seeds or minor differences in library versions. We conducted five training runs, and the resulting scores on CASTELLA `test` set (mean ± standard deviation) are as follows:
- Only CASTELLA
  - R1@0.5    : 22.74±0.77
  - R1@0.7    : 10.17±0.86
  - mAP (avg)  : 10.49±0.53
  - mAP@0.5   : 21.93±0.57
  - mAP@0.75  : 8.85±0.58
- Clotho-Moment pre-training & CASTELLA fine-tuning
  - R1@0.5    : 25.86±0.74
  - R1@0.7    : 13.85±1.47
  - mAP (avg)  : 11.74±0.39
  - mAP@0.5   : 23.14±0.33
  - mAP@0.75  : 10.54±0.54

## Note
- This recipe includes minor changes from the original paper to improve performance:
  - Training extended from 100 to 200 epochs
  - window sampling controlled by `max_windows` to stabilize the training

## Citation
If you find this code useful for your research, please cite the original paper:
```
@inproceedings{munakata2025audiomoment,
  author = {Munakata, Hokuto and Nishimura, Taichi and Nakada, Shota and Komatsu, Tatsuya},
  title = {Language-based Audio Moment Retrieval},
  booktitle = {Proc. ICASSP},
  year = {2025},
  pages = {1-5},
  _pdf = {https://arxiv.org/pdf/2409.15672}
}
```
QD-DETR citation:
```
@inproceedings{qddetr
    author = {WonJun Moon and Sangeek Hyun and SangUk Park and Dongchan Park and Jae-Pil Heo},
    title = {Query-Dependent Video Representation for Moment Retrieval and Highlight Detection},
    booktitle = {Proc. CVPR},
    year = {2023},
}
```

## Others
This code is based on [lighthouse](https://github.com/line/lighthouse).


## Contact
taichitary@gmail.com

hokuto.munakata@lycorp.co.jp
-------------------------------------------------------------------------------------------------
## M2D-CLAPへの置き換え

### 追加パッケージのインストール
```bash
pip install yt-dlp
pip install --upgrade timm
pip install sentence_transformers nnAudio
```

### M2D-CLAPリポジトリのcloneとチェックポイントのダウンロード
```bash
git clone https://github.com/nttcslab/m2d.git /data/miyamoto/m2d
cd /data/miyamoto/m2d
wget https://github.com/nttcslab/m2d/releases/download/v0.5.0/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip
unzip m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip
```

### 音声ダウンロード
CASTELLAの音声ファイルをYouTubeからダウンロードします。
```bash
python download_audio.py
```

### 特徴量抽出
ダウンロード完了後に実行してください。
```bash
python src/extract_m2d_features.py --split train
python src/extract_m2d_features.py --split val
python src/extract_m2d_features.py --split test
```

### 学習
```bash
python src/train.py --config config.yml
```
