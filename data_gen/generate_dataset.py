import os
import re
import cv2
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from mmengine.registry import init_default_scope
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model, inference_topdown

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from data_gen.preprocess import pre_normalization

DET_CONFIG = './checkpoints/rtmdet_tiny_8xb32-300e_coco.py'
DET_CHECKPOINT = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

POSE_CONFIG = './checkpoints/rtmpose-m_8xb256-420e_coco-256x192.py'
POSE_CHECKPOINT = './checkpoints/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth'

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]


COCO_TO_NTU = {
    0: 3,
    5: 4,
    6: 8,
    7: 5,
    8: 9,
    9: 6,
    10: 10,
    11: 12,
    12: 16,
    13: 13,
    14: 17,
    15: 14,
    16: 18,
}

def parse_name(name):
    m = re.match(r'.*S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)', name)
    if not m:
        return None
    s, c, p, r, a = map(int, m.groups())
    return dict(setup=s, camera=c, subject=p, repeat=r, action=a)

def coco17_to_ntu25(seq17):
    T = seq17.shape[0]
    ntu = np.zeros((T, 25, 3), dtype=np.float32)
    for coco_id, ntu_id in COCO_TO_NTU.items():
        ntu[:, ntu_id, :] = seq17[:, coco_id, :]
    ntu[:, 1, :] = (ntu[:, 12, :] + ntu[:, 16, :]) / 2
    ntu[:, 2, :] = (ntu[:, 4, :] + ntu[:, 8, :]) / 2
    ntu[:, 0, :] = ntu[:, 1, :]
    return ntu

def extract_pose_sequence(video_path, det_model, pose_model):
    cap = cv2.VideoCapture(video_path)
    seq = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        init_default_scope('mmdet')
        det_result = inference_detector(det_model, frame)
        bboxes = det_result.pred_instances.bboxes
        scores = det_result.pred_instances.scores
        if len(bboxes) == 0:
            seq.append(np.zeros((17, 3), dtype=np.float32))
            continue
        idx = scores.argmax().item()
        bbox = bboxes[idx].cpu().numpy()
        init_default_scope('mmpose')
        pose_results = inference_topdown(pose_model, frame, [bbox])
        if len(pose_results) == 0 or len(pose_results[0].pred_instances.keypoints) == 0:
            seq.append(np.zeros((17, 3), dtype=np.float32))
            continue
        kpts = pose_results[0].pred_instances.keypoints[0]
        kpts_score = pose_results[0].pred_instances.keypoint_scores[0]
        frame_kps = np.concatenate([kpts, kpts_score[:, None]], axis=1).astype(np.float32)
        seq.append(frame_kps)
    cap.release()
    if len(seq) == 0:
        return np.zeros((1, 17, 3), dtype=np.float32)
    return np.stack(seq)

def seq25_to_data(seq25, max_frame, num_person):
    data = np.zeros((3, max_frame, 25, num_person), dtype=np.float32)
    t = min(seq25.shape[0], max_frame)
    if t > 0:
        data[:, :t, :, 0] = seq25[:t].transpose(2, 0, 1)
    return data

def stack_samples(samples, max_frame, num_person):
    if len(samples) == 0:
        return np.zeros((0, 3, max_frame, 25, num_person), dtype=np.float32)
    return np.stack(samples)

def build_dataset(video_dir, out_folder, device, benchmark, max_frame, num_joint, max_body_true, det_cfg, det_ckpt, pose_cfg, pose_ckpt):
    os.makedirs(out_folder, exist_ok=True)
    det_model = init_detector(det_cfg, det_ckpt, device=device)
    pose_model = init_model(pose_cfg, pose_ckpt, device=device)
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
    videos.sort()
    train_samples, val_samples = [], []
    train_labels, val_labels = [], []
    train_names, val_names = [], []
    for vid in tqdm(videos):
        meta = parse_name(os.path.splitext(vid)[0])
        if meta is None:
            continue
        video_path = os.path.join(video_dir, vid)
        seq17 = extract_pose_sequence(video_path, det_model, pose_model)
        seq25 = coco17_to_ntu25(seq17)
        if benchmark == 'xview':
            is_train = meta['camera'] in training_cameras
        else:
            is_train = meta['subject'] in training_subjects
        data = seq25_to_data(seq25, max_frame, max_body_true)
        label = meta['subject'] - 1
        if is_train:
            train_samples.append(data)
            train_labels.append(label)
            train_names.append(vid.replace('.mp4', '.skeleton'))
        else:
            val_samples.append(data)
            val_labels.append(label)
            val_names.append(vid.replace('.mp4', '.skeleton'))
    train_data = stack_samples(train_samples, max_frame, max_body_true)
    val_data = stack_samples(val_samples, max_frame, max_body_true)
    if train_data.shape[0] == 0 and val_data.shape[0] == 0:
        return
    if train_data.shape[0] > 0:
        train_data = pre_normalization(train_data)
    if val_data.shape[0] > 0:
        val_data = pre_normalization(val_data)
    out_path = os.path.join(out_folder, benchmark)
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, 'train_data_joint.npy'), train_data)
    np.save(os.path.join(out_path, 'val_data_joint.npy'), val_data)
    with open(os.path.join(out_path, 'train_label.pkl'), 'wb') as f:
        pickle.dump((train_names, list(train_labels)), f)
    with open(os.path.join(out_path, 'val_label.pkl'), 'wb') as f:
        pickle.dump((val_names, list(val_labels)), f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='./data/Recordings')
    parser.add_argument('--out_folder', type=str, default='./data/ntu')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--benchmark', type=str, choices=['xview', 'xsub'], default='xview')
    parser.add_argument('--max_frame', type=int, default=300)
    parser.add_argument('--num_joint', type=int, default=25)
    parser.add_argument('--max_body_true', type=int, default=2)
    parser.add_argument('--det_config', type=str, default=DET_CONFIG)
    parser.add_argument('--det_checkpoint', type=str, default=DET_CHECKPOINT)
    parser.add_argument('--pose_config', type=str, default=POSE_CONFIG)
    parser.add_argument('--pose_checkpoint', type=str, default=POSE_CHECKPOINT)
    args = parser.parse_args()
    build_dataset(
        args.video_dir,
        args.out_folder,
        args.device,
        args.benchmark,
        args.max_frame,
        args.num_joint,
        args.max_body_true,
        args.det_config,
        args.det_checkpoint,
        args.pose_config,
        args.pose_checkpoint,
    )

if __name__ == '__main__':
    main()
