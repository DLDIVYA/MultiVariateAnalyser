
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns
from collections import Counter
import os
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data(csv_path="TEP_Train_Test_with_anomalies.csv"):
    """Load the anomaly detection results."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Get feature columns (exclude Time, Abnormality_score, top_feature_*)
    exclude_cols = ['Time', 'Abnormality_score'] + [f'top_feature_{i}' for i in range(1, 8)]
    feature_columns = [col for col in df.columns if col not in exclude_cols]
    top_feature_columns = [f'top_feature_{i}' for i in range(1, 8)]
    
    print(f"Loaded {len(df)} rows, {len(feature_columns)} features")
    print(f"Score range: {df['Abnormality_score'].min():.2f} - {df['Abnormality_score'].max():.2f}")
    
    return df, feature_columns, top_feature_columns

def plot_timeline(df, output_dir):
    """Generate timeline plot of anomaly scores."""
    print("Generating timeline plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Main timeline
    ax1.plot(df['Time'], df['Abnormality_score'], linewidth=0.8, alpha=0.7, color='darkblue')
    ax1.fill_between(df['Time'], df['Abnormality_score'], alpha=0.3, color='lightblue')
    
    # Threshold lines
    ax1.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Medium (30)')
    ax1.axhline(y=60, color='red', linestyle='--', alpha=0.7, label='High (60)')
    ax1.axhline(y=90, color='darkred', linestyle='--', alpha=0.7, label='Severe (90)')
    
    ax1.set_title('Anomaly Scores Over Time', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Abnormality Score')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # High anomaly events
    high_anomalies = df[df['Abnormality_score'] > 60]
    if not high_anomalies.empty:
        ax2.scatter(high_anomalies['Time'], high_anomalies['Abnormality_score'], 
                   c='red', alpha=0.6, s=20)
        ax2.set_title('High Anomaly Events (Score > 60)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Abnormality Score')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No high anomaly events found', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=14)
        ax2.set_title('High Anomaly Events (Score > 60)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/timeline.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Timeline plot saved")

def plot_feature_bars(df, top_feature_columns, output_dir):
    """Generate feature contribution bar charts."""
    print("Generating feature contribution bars...")
    
    # Collect all features
    all_features = []
    for col in top_feature_columns:
        features = df[col].dropna()
        features = features[features != '']
        all_features.extend(features.tolist())
    
    feature_counts = Counter(all_features)
    top_features = dict(feature_counts.most_common(20))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Overall contributions
    features = list(top_features.keys())
    counts = list(top_features.values())
    
    bars1 = ax1.bar(range(len(features)), counts, color='steelblue', alpha=0.7)
    ax1.set_title('Top 20 Most Contributing Features (Overall)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Contribution Frequency')
    ax1.set_xticks(range(len(features)))
    ax1.set_xticklabels(features, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add counts on bars
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom', fontsize=10)
    
    # High anomaly contributions
    high_anomaly_df = df[df['Abnormality_score'] > 60]
    if not high_anomaly_df.empty:
        high_features = []
        for col in top_feature_columns:
            features = high_anomaly_df[col].dropna()
            features = features[features != '']
            high_features.extend(features.tolist())
        
        high_feature_counts = Counter(high_features)
        top_high_features = dict(high_feature_counts.most_common(15))
        
        h_features = list(top_high_features.keys())
        h_counts = list(top_high_features.values())
        
        bars2 = ax2.bar(range(len(h_features)), h_counts, color='crimson', alpha=0.7)
        ax2.set_title('Top Contributing Features (High Anomalies > 60)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Contribution Frequency')
        ax2.set_xticks(range(len(h_features)))
        ax2.set_xticklabels(h_features, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars2, h_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(h_counts)*0.01,
                    str(count), ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No high anomaly events found', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=14)
        ax2.set_title('Top Contributing Features (High Anomalies > 60)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_bars.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature bar charts saved")


def generate_summary(df, top_feature_columns, output_dir):
    """Generate summary report."""
    print("Generating summary report...")
    
    scores = df['Abnormality_score']
    
    # Collect features
    all_features = []
    for col in top_feature_columns:
        features = df[col].dropna()
        features = features[features != '']
        all_features.extend(features.tolist())
    
    feature_counts = Counter(all_features)
    
    report = []
    report.append("ANOMALY DETECTION SUMMARY REPORT")
    report.append("=" * 50)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total points: {len(df):,}")
    report.append(f"Time range: {df['Time'].min()} to {df['Time'].max()}")
    report.append("")
    
    report.append("SCORE STATISTICS:")
    report.append(f"Mean: {scores.mean():.3f}")
    report.append(f"Median: {scores.median():.3f}")
    report.append(f"Std Dev: {scores.std():.3f}")
    report.append(f"Min: {scores.min():.3f}")
    report.append(f"Max: {scores.max():.3f}")
    report.append(f"95th percentile: {scores.quantile(0.95):.3f}")
    report.append("")
    
    # Score bands
    bands = [(0, 10, 'Normal'), (10, 30, 'Low'), (30, 60, 'Medium'), 
            (60, 90, 'High'), (90, 100, 'Severe')]
    
    report.append("SCORE BANDS:")
    total = len(df)
    for low, high, label in bands:
        count = len(df[(scores >= low) & (scores <= high)])
        percentage = (count / total) * 100
        report.append(f"{label:8} ({low:2d}-{high:3d}): {count:6,} ({percentage:5.1f}%)")
    report.append("")
    
    # Top features
    report.append("TOP 10 FEATURES:")
    for i, (feature, count) in enumerate(feature_counts.most_common(10), 1):
        percentage = (count / len(all_features)) * 100
        report.append(f"{i:2d}. {feature[:30]:30} : {count:5,} ({percentage:4.1f}%)")
    
    # Save report
    with open(f"{output_dir}/summary.txt", 'w') as f:
        f.write('\n'.join(report))
    
    print("Summary report saved")
    print("\nSUMMARY:")
    for line in report:
        print(line)

def main():
    """Main function to generate all plots."""
    print("Starting anomaly visualization generation...")
    
    # Check if CSV exists
    csv_path = "TEP_Train_Test_with_anomalies.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        print("Make sure the file is in the current directory.")
        return
    
    # Create output directory
    output_dir = "anomaly_plots"
    Path(output_dir).mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}/")
    
    # Load data
    df, feature_columns, top_feature_columns = load_data(csv_path)
    
    # Generate all plots
    plot_timeline(df, output_dir)
    plot_feature_bars(df, top_feature_columns, output_dir)
    generate_summary(df, top_feature_columns, output_dir)
    
    print("\n All visualizations generated successfully!")
    print(f" Check the '{output_dir}/' directory for:")
    print("   - timeline.png (anomaly scores over time)")
    print("   - feature_bars.png (feature contributions)")
    print("   - summary.txt (detailed report)")

if __name__ == "__main__":
    main()
