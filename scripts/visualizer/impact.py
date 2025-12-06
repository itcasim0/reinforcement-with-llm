"""
🔥 임팩트 있는 학습 결과 시각화 🔥
- 핵심 메시지 강조
- 자극적인 색상과 디자인
- 수치 강조
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import glob
import sys


def get_latest_checkpoint(base_dir: str = "logs/checkpoints") -> Path:
    """가장 최근 체크포인트 찾기"""
    checkpoint_path = Path(base_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트 디렉토리가 없습니다: {checkpoint_path}")
    
    checkpoint_dirs = [
        d for d in checkpoint_path.iterdir() 
        if d.is_dir() and len(d.name) == 15 and 'T' in d.name
    ]
    
    if not checkpoint_dirs:
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}")
    
    return sorted(checkpoint_dirs, key=lambda x: x.name, reverse=True)[0]


def load_trajectories(traj_dir: Path) -> List[Dict]:
    """trajectories 디렉토리에서 모든 에피소드 파일 로드"""
    traj_files = sorted(glob.glob(str(traj_dir / "episode_*.json")))
    trajectories = []
    
    for traj_file in traj_files:
        with open(traj_file, 'r', encoding='utf-8') as f:
            traj_data = json.load(f)
            trajectories.append(traj_data)
    
    return trajectories


def calculate_score_improvements(episodes: List[Dict]) -> Dict:
    """에피소드들의 점수 개선도 계산"""
    score_keys = ['structure', 'length', 'academic_style', 
                  'information_density', 'clarity', 'overall']
    
    initial = {key: [] for key in score_keys}
    final = {key: [] for key in score_keys}
    deltas = {key: [] for key in score_keys}
    returns = []
    
    for traj in episodes:
        if not traj['steps']:
            continue
        
        initial_state = traj['steps'][0]['state']
        last_step = traj['steps'][-1]
        if 'info' in last_step and 'new_scores' in last_step['info']:
            final_state = last_step['info']['new_scores']
        else:
            final_state = last_step['state']
        
        for key in score_keys:
            init_val = initial_state[key]
            final_val = final_state[key]
            
            initial[key].append(init_val)
            final[key].append(final_val)
            deltas[key].append(final_val - init_val)
        
        returns.append(traj['total_return'])
    
    return {
        'initial': initial,
        'final': final,
        'deltas': deltas,
        'returns': returns
    }


def plot_key_improvements_bar(early_data: Dict, late_data: Dict, save_dir: Path):
    """
    핵심 개선 지표 막대 그래프 (top 3만)
    """
    score_keys = ['structure', 'academic_style', 'overall']
    labels = ['Structure', 'Academic\nStyle', 'Overall']
    
    early_values = [np.mean(early_data['deltas'][key]) for key in score_keys]
    late_values = [np.mean(late_data['deltas'][key]) for key in score_keys]
    
    improvements = [(late_values[i] - early_values[i]) / abs(early_values[i] + 1e-8) * 100 
                   for i in range(len(score_keys))]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    fig.patch.set_facecolor('#0f0f23')
    ax.set_facecolor('#1a1a2e')
    
    x = np.arange(len(labels))
    width = 0.35
    
    # Early bars
    bars1 = ax.bar(x - width/2, early_values, width, 
                   label='Early (ep 1-100)',
                   color='#e63946', alpha=0.8, edgecolor='white', linewidth=2)
    
    # Late bars
    bars2 = ax.bar(x + width/2, late_values, width,
                   label='Late (ep 901-1000)', 
                   color='#06ffa5', alpha=0.8, edgecolor='white', linewidth=2)
    
    # 값 표시 (굵게)
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        
        ax.text(bar1.get_x() + bar1.get_width()/2., height1,
                f'{height1:.2f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold',
                color='white')
        
        ax.text(bar2.get_x() + bar2.get_width()/2., height2,
                f'{height2:.2f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold',
                color='white')
        
        # 개선률 화살표
        ax.annotate(f'↑{improvements[i]:+.0f}%', 
                   xy=(x[i], max(height1, height2) + 0.3),
                   ha='center', fontsize=16, fontweight='bold',
                   color='#ffd700')
    
    ax.set_ylabel('Score Improvement', fontsize=14, fontweight='bold', color='white')
    ax.set_title('TOP 3 IMPROVEMENTS: Early vs Late Training', 
                 fontsize=18, fontweight='bold', color='white', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold', color='white')
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.2, color='white', linestyle='--')
    ax.tick_params(colors='white')
    
    # 스파인 색상
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'key_improvements_bar.png', dpi=200, bbox_inches='tight',
                facecolor='#0f0f23')
    print(f"핵심 개선 막대그래프 저장: {save_dir / 'key_improvements_bar.png'}")
    plt.close()


def plot_before_after_showcase(early_data: Dict, late_data: Dict, save_dir: Path):
    """
    Before & After 쇼케이스 (점수 분포)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0a0e27')
    
    score_keys = ['structure', 'academic_style', 'overall']
    labels = ['Structure', 'Academic\nStyle', 'Overall']
    
    # BEFORE (Early)
    ax = axes[0]
    ax.set_facecolor('#1a1a2e')
    early_final = [np.mean(early_data['final'][key]) for key in score_keys]
    
    bars = ax.barh(labels, early_final, color='#e63946', alpha=0.9, 
                   edgecolor='white', linewidth=3)
    
    for i, (bar, val) in enumerate(zip(bars, early_final)):
        ax.text(val + 0.2, i, f'{val:.1f}', 
               va='center', fontsize=18, fontweight='bold', color='white')
    
    ax.set_xlim(0, 10)
    ax.set_xlabel('Score', fontsize=14, fontweight='bold', color='white')
    ax.set_title('BEFORE\n(Episodes 1-100)', 
                 fontsize=20, fontweight='bold', color='#e63946', pad=15)
    ax.tick_params(colors='white', labelsize=12)
    ax.grid(True, alpha=0.2, color='white', axis='x')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    
    # AFTER (Late)
    ax = axes[1]
    ax.set_facecolor('#1a1a2e')
    late_final = [np.mean(late_data['final'][key]) for key in score_keys]
    
    bars = ax.barh(labels, late_final, color='#06ffa5', alpha=0.9,
                   edgecolor='white', linewidth=3)
    
    for i, (bar, val) in enumerate(zip(bars, late_final)):
        improvement = ((val - early_final[i]) / early_final[i]) * 100
        ax.text(val + 0.2, i, f'{val:.1f} (+{improvement:.0f}%)', 
               va='center', fontsize=18, fontweight='bold', color='white')
    
    ax.set_xlim(0, 10)
    ax.set_xlabel('Score', fontsize=14, fontweight='bold', color='white')
    ax.set_title('AFTER\n(Episodes 901-1000)', 
                 fontsize=20, fontweight='bold', color='#06ffa5', pad=15)
    ax.tick_params(colors='white', labelsize=12)
    ax.grid(True, alpha=0.2, color='white', axis='x')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'before_after_showcase.png', dpi=200, bbox_inches='tight',
                facecolor='#0a0e27')
    print(f"Before/After 쇼케이스 저장: {save_dir / 'before_after_showcase.png'}")
    plt.close()


def main(checkpoint_dir: str = None, split_n: int = 100):
    """메인 실행 함수"""
    if checkpoint_dir is None:
        print("가장 최근 체크포인트를 찾는 중...")
        checkpoint_path = get_latest_checkpoint()
        print(f"찾은 체크포인트: {checkpoint_path}")
    else:
        checkpoint_path = Path(checkpoint_dir)
    
    traj_dir = checkpoint_path / "trajectories"
    
    if not traj_dir.exists():
        print(f"Trajectory 디렉토리를 찾을 수 없습니다: {traj_dir}")
        return
    
    print(f"\n Trajectory 로드 중: {traj_dir}")
    trajectories = load_trajectories(traj_dir)
    print(f"총 {len(trajectories)}개 에피소드 로드 완료")
    
    if not trajectories:
        print("로드된 trajectory가 없습니다.")
        return
    
    # 초기/후기 분할
    early = trajectories[:min(split_n, len(trajectories))]
    late = trajectories[-min(split_n, len(trajectories)):]
    
    early_data = calculate_score_improvements(early)
    late_data = calculate_score_improvements(late)
    
    plot_key_improvements_bar(early_data, late_data, checkpoint_path)
    plot_before_after_showcase(early_data, late_data, checkpoint_path)
    


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()