import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
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


def smooth_curve(values, window=20):
    """이동평균으로 곡선 스무딩"""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode='valid')


def remove_outliers(values, threshold=3):
    """이상치 제거 (z-score 기반)"""
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    z_scores = np.abs((values - mean) / (std + 1e-8))
    return np.where(z_scores < threshold, values, np.nan)


def plot_training_improved(log_path):
    """개선된 DQN 학습 로그 시각화"""
    with open(log_path) as f:
        data = json.load(f)
    
    episodes = np.array(data['episodes'])
    returns = np.array(data['returns'])
    losses = np.array(data['losses'])
    epsilons = np.array(data['epsilons'])
    
    # 스타일 설정
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('DQN Training Metrics', fontsize=18, fontweight='bold', y=0.995)
    
    window = 20
    
    # 1) Episode Returns
    ax = axes[0, 0]
    ax.plot(episodes, returns, alpha=0.25, linewidth=0.5, color='steelblue', label='Raw')
    
    if len(returns) > window:
        smoothed = smooth_curve(returns, window)
        ax.plot(episodes[window-1:], smoothed, color='#e74c3c', linewidth=2.5, 
                label=f'{window}-episode MA', zorder=10)
        
        # 표준편차 범위 (신뢰구간 느낌)
        std_upper = []
        std_lower = []
        for i in range(window-1, len(returns)):
            window_data = returns[max(0, i-window+1):i+1]
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)
            std_upper.append(mean_val + std_val)
            std_lower.append(mean_val - std_val)
        
        ax.fill_between(episodes[window-1:], std_lower, std_upper, 
                        color='#e74c3c', alpha=0.15, label='±1 std')
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Return', fontsize=11, fontweight='bold')
    ax.set_title('Episode Returns', fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # 최종 평균 표시
    final_mean = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
    ax.text(0.02, 0.98, f'Final 100-ep avg: {final_mean:.3f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2) Q-Network Loss (smoothed + outlier removed)
    ax = axes[0, 1]
    
    # 이상치 제거
    losses_clean = remove_outliers(losses, threshold=3)
    
    ax.plot(episodes, losses_clean, alpha=0.2, linewidth=0.5, 
            color='green', label='Raw (outliers removed)')
    
    if len(losses) > window:
        # NaN 무시하고 이동평균
        smoothed = []
        for i in range(len(losses)):
            start = max(0, i - window + 1)
            window_data = losses_clean[start:i+1]
            window_data = window_data[~np.isnan(window_data)]
            if len(window_data) > 0:
                smoothed.append(np.mean(window_data))
            else:
                smoothed.append(np.nan)
        
        ax.plot(episodes, smoothed, color='#27ae60', linewidth=2.5, 
                label=f'{window}-episode MA', zorder=10)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Q-Network Loss (MSE)', fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # y축 범위 제한 (99 percentile 기준)
    valid_losses = losses_clean[~np.isnan(losses_clean)]
    if len(valid_losses) > 0:
        y_max = np.percentile(valid_losses, 99)
        ax.set_ylim(bottom=-0.1, top=y_max * 1.1)
    
    # 3) Epsilon (탐색 정도)
    ax = axes[1, 0]
    ax.plot(episodes, epsilons, color='#8e44ad', linewidth=2.0, label='Epsilon')
    
    ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
    ax.set_ylabel('Epsilon', fontsize=11, fontweight='bold')
    ax.set_title('Exploration Rate (Epsilon)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle='--')
    
    # 초기/최종 epsilon 표시
    initial_epsilon = epsilons[0] if len(epsilons) > 0 else 0
    final_epsilon = epsilons[-1] if len(epsilons) > 0 else 0
    ax.text(0.02, 0.98, f'Initial: {initial_epsilon:.3f}\nFinal: {final_epsilon:.3f}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4) 학습 진행 통계
    ax = axes[1, 1]
    ax.axis('off')
    
    # 통계 정보 계산
    total_episodes = len(episodes)
    initial_100_mean = np.mean(returns[:100]) if len(returns) >= 100 else np.mean(returns[:len(returns)])
    initial_100_std = np.std(returns[:100]) if len(returns) >= 100 else np.std(returns[:len(returns)])
    final_100_mean = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns[-len(returns):])
    final_100_std = np.std(returns[-100:]) if len(returns) >= 100 else np.std(returns[-len(returns):])
    improvement = final_100_mean - initial_100_mean
    
    best_return = np.max(returns)
    worst_return = np.min(returns)
    avg_return = np.mean(returns)
    
    valid_losses = losses_clean[~np.isnan(losses_clean)]
    avg_loss = np.mean(valid_losses) if len(valid_losses) > 0 else 0
    final_100_loss = np.mean(valid_losses[-100:]) if len(valid_losses) >= 100 else np.mean(valid_losses) if len(valid_losses) > 0 else 0
    
    epsilon_decay_rate = (initial_epsilon - final_epsilon) / initial_epsilon * 100 if initial_epsilon > 0 else 0
    
    # 텍스트 정보 구성
    stats_text = f"""DQN 학습 통계 요약

{'='*40}
Episode 정보
{'='*40}
총 에피소드: {total_episodes}

{'='*40}
Returns
{'='*40}
초기 100-ep 평균: {initial_100_mean:.3f} ± {initial_100_std:.3f}
최종 100-ep 평균: {final_100_mean:.3f} ± {final_100_std:.3f}
개선도: {improvement:+.3f}

최고 Return: {best_return:.3f}
최저 Return: {worst_return:.3f}
전체 평균: {avg_return:.3f}

{'='*40}
Q-Network Loss
{'='*40}
전체 평균: {avg_loss:.4f}
최종 100-ep: {final_100_loss:.4f}

{'='*40}
탐색 (Epsilon)
{'='*40}
초기: {initial_epsilon:.3f}
최종: {final_epsilon:.3f}
감소율: {epsilon_decay_rate:.1f}%
"""
    
    ax.text(0.05, 0.95, stats_text, 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    save_path = Path(log_path).parent / 'training_metrics.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"저장 완료: {save_path}")
    plt.close()
    
    # 학습 요약 통계
    print("\n" + "="*60)
    print("DQN 학습 요약 통계")
    print("="*60)
    print(f"총 에피소드: {total_episodes}")
    print(f"\nReturns:")
    print(f"  초기 100-ep 평균: {initial_100_mean:.3f} ± {initial_100_std:.3f}")
    print(f"  최종 100-ep 평균: {final_100_mean:.3f} ± {final_100_std:.3f}")
    print(f"  개선도: {improvement:+.3f}")
    print(f"  최고: {best_return:.3f}")
    print(f"  최저: {worst_return:.3f}")
    
    print(f"\nEpsilon (탐색 정도):")
    print(f"  초기: {initial_epsilon:.3f}")
    print(f"  최종: {final_epsilon:.3f}")
    print(f"  감소율: {epsilon_decay_rate:.1f}%")
    
    if len(valid_losses) > 0:
        print(f"\nQ-Network Loss:")
        print(f"  평균: {avg_loss:.4f}")
        print(f"  최종 100-ep: {final_100_loss:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_dir = sys.argv[1]
    else:
        checkpoint_dir = get_latest_checkpoint()
    
    log_path = Path(checkpoint_dir) / "training_log.json"
    
    if not log_path.exists():
        print(f"학습 로그 파일을 찾을 수 없습니다: {log_path}")
        sys.exit(1)
    
    plot_training_improved(log_path)
