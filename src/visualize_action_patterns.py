import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import glob
from collections import Counter


def load_trajectories(traj_dir: Path) -> List[Dict]:
    """trajectories 디렉토리에서 모든 에피소드 파일 로드"""
    traj_files = sorted(glob.glob(str(traj_dir / "episode_*.json")))
    trajectories = []
    
    for traj_file in traj_files:
        with open(traj_file, 'r', encoding='utf-8') as f:
            traj_data = json.load(f)
            trajectories.append(traj_data)
    
    return trajectories


def extract_action_data(trajectories: List[Dict]) -> Dict:
    """
    에피소드별 액션 선택 데이터 추출
    
    Returns:
        {
            'episodes': [1, 2, 3, ...],
            'action_counts': {episode: {action_name: count, ...}, ...},
            'action_sequences': {episode: [action1, action2, ...], ...},
            'action_names': ['fix_grammar', 'improve_clarity', ...]
        }
    """
    episodes = []
    action_counts = {}
    action_sequences = {}
    action_names_set = set()
    
    for traj in trajectories:
        ep = traj['episode']
        episodes.append(ep)
        
        # 액션 시퀀스 추출
        actions = [step['action_name'] for step in traj['steps']]
        action_sequences[ep] = actions
        
        # 액션 카운트
        counter = Counter(actions)
        action_counts[ep] = dict(counter)
        
        # 모든 액션 이름 수집
        action_names_set.update(actions)
    
    action_names = sorted(list(action_names_set))
    
    return {
        'episodes': episodes,
        'action_counts': action_counts,
        'action_sequences': action_sequences,
        'action_names': action_names
    }


def plot_action_distribution_over_time(action_data: Dict, save_dir: Path, window_size=100):
    """
    학습 구간별 액션 분포 변화 (누적 막대 그래프)
    """
    episodes = action_data['episodes']
    action_counts = action_data['action_counts']
    action_names = action_data['action_names']
    
    # 구간별로 나누기
    num_windows = len(episodes) // window_size
    if num_windows == 0:
        num_windows = 1
        window_size = len(episodes)
    
    window_labels = []
    window_data = {action: [] for action in action_names}
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(episodes))
        
        window_eps = episodes[start_idx:end_idx]
        window_labels.append(f'{window_eps[0]}-{window_eps[-1]}')
        
        # 구간 내 액션 합산
        total_counts = {action: 0 for action in action_names}
        for ep in window_eps:
            counts = action_counts.get(ep, {})
            for action in action_names:
                total_counts[action] += counts.get(action, 0)
        
        # 비율로 변환
        total = sum(total_counts.values())
        if total > 0:
            for action in action_names:
                window_data[action].append(total_counts[action] / total * 100)
        else:
            for action in action_names:
                window_data[action].append(0)
    
    # 누적 막대 그래프
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    x = np.arange(len(window_labels))
    width = 0.6
    
    bottom = np.zeros(len(window_labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(action_names)))
    
    for idx, action in enumerate(action_names):
        values = window_data[action]
        ax.bar(x, values, width, label=action, bottom=bottom, color=colors[idx])
        bottom += values
    
    ax.set_xlabel('Episode Range')
    ax.set_ylabel('Action Distribution (%)')
    ax.set_title(f'Action Distribution Over Time (Window Size: {window_size})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(window_labels, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'action_distribution_over_time.png', dpi=150, bbox_inches='tight')
    print(f"그래프 저장: {save_dir / 'action_distribution_over_time.png'}")
    plt.close()


def plot_action_sequence_patterns(action_data: Dict, save_dir: Path, top_n=10):
    """
    가장 많이 사용된 액션 시퀀스 패턴 분석
    """
    action_sequences = action_data['action_sequences']
    
    # 시퀀스를 문자열로 변환하여 카운트
    sequence_counter = Counter()
    for ep, actions in action_sequences.items():
        seq_str = ' -> '.join(actions)
        sequence_counter[seq_str] += 1
    
    # 상위 N개 시퀀스
    top_sequences = sequence_counter.most_common(top_n)
    
    if not top_sequences:
        print("액션 시퀀스 데이터가 없습니다.")
        return
    
    # 막대 그래프
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    sequences = [seq for seq, _ in top_sequences]
    counts = [count for _, count in top_sequences]
    
    y_pos = np.arange(len(sequences))
    ax.barh(y_pos, counts, color='steelblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sequences, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title(f'Top {top_n} Most Common Action Sequences', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'action_sequence_patterns.png', dpi=150, bbox_inches='tight')
    print(f"그래프 저장: {save_dir / 'action_sequence_patterns.png'}")
    plt.close()


def print_action_statistics(action_data: Dict):
    """
    액션 선택 통계 출력
    """
    action_counts = action_data['action_counts']
    action_names = action_data['action_names']
    episodes = action_data['episodes']
    
    # 전체 액션 통계
    total_action_counts = {action: 0 for action in action_names}
    
    for ep in episodes:
        counts = action_counts.get(ep, {})
        for action in action_names:
            total_action_counts[action] += counts.get(action, 0)
    
    total = sum(total_action_counts.values())
    
    print("\n" + "="*80)
    print("액션 선택 통계")
    print("="*80)
    print(f"\n총 액션 선택 횟수: {total}")
    print(f"총 에피소드 수: {len(episodes)}")
    print(f"\n액션별 선택 빈도:")
    
    # 빈도순 정렬
    sorted_actions = sorted(total_action_counts.items(), key=lambda x: x[1], reverse=True)
    
    for action, count in sorted_actions:
        ratio = count / total * 100 if total > 0 else 0
        bar = '*' * int(ratio / 2)
        print(f"  {action:25s}: {count:4d} ({ratio:5.1f}%) {bar}")
    
    # 초기 vs 후기 비교
    mid_point = len(episodes) // 2
    early_eps = episodes[:mid_point]
    late_eps = episodes[mid_point:]
    
    early_counts = {action: 0 for action in action_names}
    late_counts = {action: 0 for action in action_names}
    
    for ep in early_eps:
        counts = action_counts.get(ep, {})
        for action in action_names:
            early_counts[action] += counts.get(action, 0)
    
    for ep in late_eps:
        counts = action_counts.get(ep, {})
        for action in action_names:
            late_counts[action] += counts.get(action, 0)
    
    early_total = sum(early_counts.values())
    late_total = sum(late_counts.values())
    
    print(f"\n초기 {len(early_eps)}ep vs 후기 {len(late_eps)}ep 비교:")
    print(f"{'액션':25s} {'초기 비율':>10s} {'후기 비율':>10s} {'변화':>10s}")
    print("-" * 60)
    
    for action in action_names:
        early_ratio = early_counts[action] / early_total * 100 if early_total > 0 else 0
        late_ratio = late_counts[action] / late_total * 100 if late_total > 0 else 0
        change = late_ratio - early_ratio
        
        print(f"{action:25s} {early_ratio:9.1f}% {late_ratio:9.1f}% {change:+9.1f}%")

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

def main(checkpoint_dir: str):
    """
    메인 실행 함수
    
    Args:
        checkpoint_dir: 체크포인트 디렉토리 경로
    """
    checkpoint_path = Path(checkpoint_dir)
    traj_dir = checkpoint_path / "trajectories"
    
    if not traj_dir.exists():
        print(f"[ERROR] Trajectory 디렉토리를 찾을 수 없습니다: {traj_dir}")
        return
    
    print(f"Trajectory 로드 중: {traj_dir}")
    trajectories = load_trajectories(traj_dir)
    print(f"총 {len(trajectories)}개 에피소드 로드 완료")
    
    if not trajectories:
        print("로드된 trajectory가 없습니다.")
        return
    
    # 액션 데이터 추출
    action_data = extract_action_data(trajectories)
    
    # 시각화
    plot_action_distribution_over_time(action_data, checkpoint_path, window_size=100)
    plot_action_sequence_patterns(action_data, checkpoint_path, top_n=10)
    
    # 통계 출력
    print_action_statistics(action_data)
    


if __name__ == "__main__":
    # 사용 예시
    checkpoint_dir = get_latest_checkpoint()
    main(checkpoint_dir)