"""
M2D-CLAPを使って音声・テキスト特徴量を抽出するスクリプト
元のCLAP特徴量(features/castella/clap/)をM2D-CLAPで置き換える

使い方:
  python src/extract_m2d_features.py --split train
  python src/extract_m2d_features.py --split val
  python src/extract_m2d_features.py --split test
"""

import sys
import os
import json
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from os.path import join, exists

sys.path.append('/data/miyamoto/m2d')
from examples.portable_m2d import PortableM2D

M2D_CHECKPOINT = '/data/miyamoto/m2d/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025/checkpoint-30.pth'
AUDIO_DIR = '/data/miyamoto/dcase2026_task6_baseline/audio'
DATA_DIR = '/data/miyamoto/dcase2026_task6_baseline/data'
AUDIO_FEAT_DIR = '/data/miyamoto/dcase2026_task6_baseline/features/castella/m2d_clap'
TEXT_FEAT_DIR = '/data/miyamoto/dcase2026_task6_baseline/features/castella/m2d_clap_text'
TARGET_SR = 16000
CLIP_LENGTH = 1  # 1秒ごとのフレーム


def load_audio(vid):
    """音声ファイルを読み込んで16kHzモノラルに変換"""
    wav_path = join(AUDIO_DIR, f'{vid}.wav')
    if not exists(wav_path):
        return None
    wav, sr = torchaudio.load(wav_path)
    if sr != TARGET_SR:
        wav = T.Resample(sr, TARGET_SR)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # ステレオ→モノラル
    return wav.squeeze(0)  # (T,)


def extract_audio_features(model, vid, device):
    """1秒ごとにフレーム特徴量を抽出"""
    wav = load_audio(vid)
    if wav is None:
        return None

    frame_features = []
    total_frames = int(wav.shape[0] / TARGET_SR)  # 秒数=フレーム数

    for i in range(total_frames):
        start = i * TARGET_SR
        end = start + TARGET_SR
        chunk = wav[start:end]

        # 1秒に満たない場合はパディング
        if chunk.shape[0] < TARGET_SR:
            chunk = torch.nn.functional.pad(chunk, (0, TARGET_SR - chunk.shape[0]))

        chunk = chunk.unsqueeze(0).to(device)  # (1, T)

        with torch.no_grad():
            x = model.to_normalized_feature(chunk)
            emb = model.backbone.forward_encoder(x[..., :model.cfg.input_size[1]])
            emb = emb[..., 1:, :]  # CLSトークンを除く
            feat = model.backbone.audio_proj(emb)  # (1, 768)

        frame_features.append(feat.squeeze(0).cpu().numpy())

    if len(frame_features) == 0:
        return None

    return np.array(frame_features)  # (T, 768)


def extract_text_features(model, texts, device):
    """テキスト特徴量を抽出"""
    with torch.no_grad():
        text_emb = model.encode_clap_text(texts)  # (N, 768)
    return text_emb.cpu().numpy()


def main(split):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # モデルのロード
    print('Loading M2D-CLAP model...')
    model = PortableM2D(M2D_CHECKPOINT)
    model.eval()
    model.to(device)

    # 出力ディレクトリ作成
    os.makedirs(AUDIO_FEAT_DIR, exist_ok=True)
    os.makedirs(TEXT_FEAT_DIR, exist_ok=True)

    # データ読み込み
    data_path = join(DATA_DIR, f'castella_{split}_release.jsonl')
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))

    print(f'Processing {split} split: {len(data)} samples')

    # 音声特徴量抽出（vid単位、重複スキップ）
    processed_vids = set()
    print('Extracting audio features...')
    for item in tqdm(data):
        vid = item['vid']
        if vid in processed_vids:
            continue
        out_path = join(AUDIO_FEAT_DIR, f'{vid}.npz')
        if exists(out_path):
            processed_vids.add(vid)
            continue
        features = extract_audio_features(model, vid, device)
        if features is None:
            print(f'Skipped (no audio): {vid}')
            continue
        np.savez(out_path, features=features)
        processed_vids.add(vid)

    # テキスト特徴量抽出（qid単位）
    print('Extracting text features...')
    for item in tqdm(data):
        qid = item['qid']
        out_path = join(TEXT_FEAT_DIR, f'qid-{qid}.npz')
        if exists(out_path):
            continue
        text_emb = extract_text_features(model, [item['query']], device)
        # last_hidden_stateのキー名は元のCLAPに合わせる
        np.savez(out_path, last_hidden_state=text_emb)

    print(f'Done! {split} split completed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'])
    args = parser.parse_args()
    main(args.split)
