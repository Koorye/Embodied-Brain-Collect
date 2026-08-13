#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
session 数据质量检查脚本（修正版，基于真实数据结构）
重点：markers.npz 是否存在、数据是否为空、时长、坏钟、视频缺失
"""

import os
import json
import numpy as np
import csv
from pathlib import Path
import sys

SESSIONS_ROOT = Path(__file__).parent / 'sessions'
OUTPUT_CSV = Path.home() / 'Desktop' / 'session_issues_report_corrected.csv'

def get_field_length(npz_path, field):
    if not npz_path.exists():
        return 0
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            return len(data[field]) if field in data else 0
    except:
        return 0

def get_duration(npz_path, field='gaze_timestamps'):
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if field in data:
                ts = data[field]
                if len(ts) > 1:
                    return float(ts[-1] - ts[0])
                elif len(ts) == 1:
                    return 0.0
    except:
        pass
    return None

def read_marker_codes(npz_path):
    if not npz_path.exists():
        return []
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            return data['codes'].tolist() if 'codes' in data else []
    except:
        return []

def parse_session_json(path):
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def check_monotonic(npz_path, field='gaze_timestamps'):
    if not npz_path.exists():
        return False, None
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if field in data:
                ts = data[field]
                if len(ts) > 1:
                    diffs = np.diff(ts)
                    if np.any(diffs <= 0):
                        return True, '非单调'
                    if np.max(diffs) > 1.0:
                        return True, f'最大间隔{np.max(diffs):.2f}s'
    except:
        pass
    return False, None

def check_session(session_dir):
    p = Path(session_dir)
    issues = []
    details = {}

    # ---------- markers ----------
    marker_path = p / 'markers.npz'
    if marker_path.exists():
        codes = read_marker_codes(marker_path)
        details['marker_count'] = len(codes)
        if len(codes) == 0:
            issues.append('markers.npz存在但无任何码')
        else:
            missing = []
            for code in [241, 242] + list(range(31, 36)):
                if code not in codes:
                    missing.append(str(code))
            if missing:
                issues.append(f'缺少标记码: {", ".join(missing)}')
            details['has_marker'] = True
    else:
        issues.append('markers.npz不存在（marker系统未工作）')
        details['has_marker'] = False

    # ---------- 各模态 ----------
    eye_npz = p / 'eye' / 'eye.npz'
    gaze_len = get_field_length(eye_npz, 'gaze_xy')
    imu_len = get_field_length(eye_npz, 'imu_gyro')
    if gaze_len == 0: issues.append('眼动注视为空')
    if imu_len == 0: issues.append('眼动IMU为空')
    details['gaze_count'] = gaze_len
    details['eye_imu_count'] = imu_len

    emg_npz = p / 'emg' / 'emg.npz'
    emg_data_len = get_field_length(emg_npz, 'emg_data')
    emg_imu_len = get_field_length(emg_npz, 'imu_gyro')
    if emg_data_len == 0: issues.append('EMG数据为空')
    if emg_imu_len == 0: issues.append('EMG-IMU为空')
    details['emg_count'] = emg_data_len
    details['emg_imu_count'] = emg_imu_len

    tactile_npz = p / 'tactile' / 'tactile.npz'
    glove_len = get_field_length(tactile_npz, 'glove_data')
    if glove_len == 0: issues.append('触觉数据为空')
    details['glove_count'] = glove_len

    vive_npz = p / 'vive' / 'vive.npz'
    vive_len = get_field_length(vive_npz, 'positions_m')
    if vive_len == 0: issues.append('Vive数据为空')
    details['vive_count'] = vive_len

    wrist_npz = p / 'wrist_cam' / 'wrist_cam.npz'
    cam0_len = get_field_length(wrist_npz, 'cam0_timestamps')
    cam1_len = get_field_length(wrist_npz, 'cam1_timestamps')
    if cam0_len == 0: issues.append('腕部相机0为空')
    if cam1_len == 0: issues.append('腕部相机1为空')
    details['cam0_count'] = cam0_len
    details['cam1_count'] = cam1_len

    # ---------- 视频 ----------
    video_files = {
        'eye.mp4': p / 'eye' / 'eye.mp4',
        'tactile_cam.mp4': p / 'tactile' / 'tactile_cam.mp4',
        'cam0.mp4': p / 'wrist_cam' / 'cam0.mp4',
        'cam1.mp4': p / 'wrist_cam' / 'cam1.mp4',
    }
    missing_videos = [name for name, path in video_files.items() if not path.exists()]
    if missing_videos:
        issues.append(f'缺失视频: {", ".join(missing_videos)}')
    if not any(path.exists() for path in video_files.values()):
        issues.append('完全无视频')
    details['videos'] = {k: v.exists() for k, v in video_files.items()}

    # ---------- 时长 ----------
    duration = get_duration(eye_npz, 'gaze_timestamps')
    if duration is None:
        duration = get_duration(emg_npz, 'emg_timestamps')
    if duration is None:
        sj = parse_session_json(p / 'session.json')
        if sj and 'duration' in sj:
            duration = sj['duration']
    if duration is not None:
        details['duration_s'] = duration
        if duration < 5:
            issues.append(f'时长过短: {duration:.1f}s')
        elif duration > 60:
            issues.append(f'时长过长: {duration:.1f}s')
    else:
        issues.append('无法确定时长')

    # ---------- 坏钟 ----------
    bad, msg = check_monotonic(eye_npz, 'gaze_timestamps')
    if bad:
        issues.append(f'坏钟(眼动): {msg}')
    else:
        bad, msg = check_monotonic(emg_npz, 'emg_timestamps')
        if bad:
            issues.append(f'坏钟(EMG): {msg}')

    # ---------- 掉线 ----------
    sj = parse_session_json(p / 'session.json')
    if sj and 'degraded_modalities' in sj and sj['degraded_modalities']:
        issues.append(f"模态掉线: {', '.join(sj['degraded_modalities'])}")

    return {
        'session_dir': str(p),
        'issues': issues,
        'issue_count': len(issues),
        'details': details
    }

def main():
    if not SESSIONS_ROOT.exists():
        print(f"错误: 找不到 sessions 目录 {SESSIONS_ROOT}")
        sys.exit(1)
    
    dirs = [d for d in SESSIONS_ROOT.iterdir() if d.is_dir()]
    print(f"共发现 {len(dirs)} 个 session 目录，开始检查...")
    results = []
    for i, d in enumerate(dirs, 1):
        print(f"[{i}/{len(dirs)}] 检查 {d.name}...")
        try:
            results.append(check_session(d))
        except Exception as e:
            results.append({
                'session_dir': str(d),
                'issues': [f'检查异常: {e}'],
                'issue_count': 1,
                'details': {}
            })

    # 写入 CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Session目录', '问题数量', '问题列表', '详细信息'])
        for r in results:
            w.writerow([
                r['session_dir'],
                r['issue_count'],
                '；'.join(r['issues']) if r['issues'] else '无问题',
                json.dumps(r['details'], ensure_ascii=False)
            ])

    print(f"\n✅ 完成！报告已生成: {OUTPUT_CSV}")
    print(f"总 session: {len(results)}")
    print(f"有问题的 session: {sum(1 for r in results if r['issue_count'] > 0)}")

if __name__ == '__main__':
    main()