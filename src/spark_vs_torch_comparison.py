#!/usr/bin/env python3
"""
This script compares the benchmarking results between Spark cores and PyTorch threads,
generating comparative visualizations to help understand the scaling characteristics
of each approach.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse

def load_benchmark_results(spark_file, torch_file):
    """Load benchmark results from CSV files"""
    # Check if files exist
    if not os.path.exists(spark_file):
        raise FileNotFoundError(f"Spark benchmark results file not found: {spark_file}")
    
    if not os.path.exists(torch_file):
        raise FileNotFoundError(f"PyTorch benchmark results file not found: {torch_file}")
    
    # Load the data
    spark_results = pd.read_csv(spark_file)
    torch_results = pd.read_csv(torch_file)
    
    # Rename columns for clarity if needed
    if "cores" in spark_results.columns:
        spark_results = spark_results.rename(columns={"cores": "parallelism"})
    
    if "threads" in torch_results.columns:
        torch_results = torch_results.rename(columns={"threads": "parallelism"})
    
    # Add framework column
    spark_results["framework"] = "Spark"
    torch_results["framework"] = "PyTorch"
    
    return spark_results, torch_results

def calculate_comparative_metrics(spark_results, torch_results):
    """Calculate comparative metrics for analysis"""
    # Calculate speedup relative to single core/thread
    spark_base_time = spark_results[spark_results["parallelism"] == 1]["total_time"].values[0]
    torch_base_time = torch_results[torch_results["parallelism"] == 1]["total_time"].values[0]
    
    spark_results["speedup"] = spark_base_time / spark_results["total_time"]
    torch_results["speedup"] = torch_base_time / torch_results["total_time"]
    
    # Calculate efficiency (speedup / parallelism)
    spark_results["efficiency"] = spark_results["speedup"] / spark_results["parallelism"]
    torch_results["efficiency"] = torch_results["speedup"] / torch_results["parallelism"]
    
    # Calculate throughput ratio to single core/thread
    spark_base_throughput = spark_results[spark_results["parallelism"] == 1]["throughput_tokens_per_second"].values[0]
    spark_results["throughput_ratio"] = spark_results["throughput_tokens_per_second"] / spark_base_throughput
    
    torch_base_throughput = torch_results[torch_results["parallelism"] == 1]["throughput_tokens_per_second"].values[0]
    torch_results["throughput_ratio"] = torch_results["throughput_tokens_per_second"] / torch_base_throughput
    
    return spark_results, torch_results

def create_comparative_plots(spark_results, torch_results):
    """Create comparative plots for Spark vs PyTorch scaling"""
    # Set up figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get the range of parallelism values
    parallelism_values = sorted(set(spark_results["parallelism"].tolist() + torch_results["parallelism"].tolist()))
    
    # 1. Processing Time Comparison
    axs[0, 0].plot(spark_results["parallelism"], spark_results["total_time"], 'o-', 
                   color='blue', linewidth=2, label="Spark Cores")
    axs[0, 0].plot(torch_results["parallelism"], torch_results["total_time"], 'o-', 
                   color='red', linewidth=2, label="PyTorch Threads")
    axs[0, 0].set_title('Processing Time Comparison')
    axs[0, 0].set_xlabel('Parallelism Level (Cores/Threads)')
    axs[0, 0].set_ylabel('Time (seconds)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # 2. Speedup Comparison
    axs[0, 1].plot(spark_results["parallelism"], spark_results["speedup"], 'o-', 
                  color='blue', linewidth=2, label="Spark Cores")
    axs[0, 1].plot(torch_results["parallelism"], torch_results["speedup"], 'o-', 
                  color='red', linewidth=2, label="PyTorch Threads")
    axs[0, 1].plot(parallelism_values, parallelism_values, '--', 
                  color='gray', linewidth=1, label="Perfect Scaling")
    axs[0, 1].set_title('Speedup Comparison')
    axs[0, 1].set_xlabel('Parallelism Level (Cores/Threads)')
    axs[0, 1].set_ylabel('Speedup Factor')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # 3. Scaling Efficiency Comparison
    axs[1, 0].plot(spark_results["parallelism"], spark_results["efficiency"], 'o-', 
                  color='blue', linewidth=2, label="Spark Cores")
    axs[1, 0].plot(torch_results["parallelism"], torch_results["efficiency"], 'o-', 
                  color='red', linewidth=2, label="PyTorch Threads")
    axs[1, 0].axhline(y=1.0, color='gray', linestyle='--', label="Perfect Efficiency")
    axs[1, 0].set_title('Scaling Efficiency Comparison')
    axs[1, 0].set_xlabel('Parallelism Level (Cores/Threads)')
    axs[1, 0].set_ylabel('Efficiency (Speedup/Parallelism)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # 4. Throughput Ratio Comparison
    axs[1, 1].plot(spark_results["parallelism"], spark_results["throughput_ratio"], 'o-', 
                  color='blue', linewidth=2, label="Spark Cores")
    axs[1, 1].plot(torch_results["parallelism"], torch_results["throughput_ratio"], 'o-', 
                  color='red', linewidth=2, label="PyTorch Threads")
    axs[1, 1].plot(parallelism_values, parallelism_values, '--', 
                  color='gray', linewidth=1, label="Perfect Scaling")
    axs[1, 1].set_title('Throughput Scaling Comparison')
    axs[1, 1].set_xlabel('Parallelism Level (Cores/Threads)')
    axs[1, 1].set_ylabel('Throughput Ratio')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('spark_vs_pytorch_comparison.png')
    print("Comparison plot saved to 'spark_vs_pytorch_comparison.png'")
    
    # Save individual comparison plots
    # 1. Processing Time
    plt.figure(figsize=(10, 6))
    plt.plot(spark_results["parallelism"], spark_results["total_time"], 'o-', 
             color='blue', linewidth=2, label="Spark Cores")
    plt.plot(torch_results["parallelism"], torch_results["total_time"], 'o-', 
             color='red', linewidth=2, label="PyTorch Threads")
    plt.title('Processing Time Comparison')
    plt.xlabel('Parallelism Level (Cores/Threads)')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    plt.savefig('comparison_processing_time.png')
    
    # 2. Speedup
    plt.figure(figsize=(10, 6))
    plt.plot(spark_results["parallelism"], spark_results["speedup"], 'o-', 
             color='blue', linewidth=2, label="Spark Cores")
    plt.plot(torch_results["parallelism"], torch_results["speedup"], 'o-', 
             color='red', linewidth=2, label="PyTorch Threads")
    plt.plot(parallelism_values, parallelism_values, '--', 
             color='gray', linewidth=1, label="Perfect Scaling")
    plt.title('Speedup Comparison')
    plt.xlabel('Parallelism Level (Cores/Threads)')
    plt.ylabel('Speedup Factor')
    plt.grid(True)
    plt.legend()
    plt.savefig('comparison_speedup.png')
    
    # 3. Efficiency
    plt.figure(figsize=(10, 6))
    plt.plot(spark_results["parallelism"], spark_results["efficiency"], 'o-', 
             color='blue', linewidth=2, label="Spark Cores")
    plt.plot(torch_results["parallelism"], torch_results["efficiency"], 'o-', 
             color='red', linewidth=2, label="PyTorch Threads")
    plt.axhline(y=1.0, color='gray', linestyle='--', label="Perfect Efficiency")
    plt.title('Scaling Efficiency Comparison')
    plt.xlabel('Parallelism Level (Cores/Threads)')
    plt.ylabel('Efficiency (Speedup/Parallelism)')
    plt.grid(True)
    plt.legend()
    plt.savefig('comparison_efficiency.png')
    
    # 4. Throughput
    plt.figure(figsize=(10, 6))
    plt.plot(spark_results["parallelism"], spark_results["throughput_ratio"], 'o-', 
             color='blue', linewidth=2, label="Spark Cores")
    plt.plot(torch_results["parallelism"], torch_results["throughput_ratio"], 'o-', 
             color='red', linewidth=2, label="PyTorch Threads")
    plt.plot(parallelism_values, parallelism_values, '--', 
             color='gray', linewidth=1, label="Perfect Scaling")
    plt.title('Throughput Scaling Comparison')
    plt.xlabel('Parallelism Level (Cores/Threads)')
    plt.ylabel('Throughput Ratio')
    plt.grid(True)
    plt.legend()
    plt.savefig('comparison_throughput.png')

def create_summary_table(spark_results, torch_results):
    """Create a summary table comparing Spark and PyTorch performance"""
    # Combine results
    comparison_rows = []
    
    for level in sorted(set(spark_results["parallelism"].tolist())):
        # Get Spark metrics for this parallelism level
        if level in spark_results["parallelism"].values:
            spark_row = spark_results[spark_results["parallelism"] == level].iloc[0]
            spark_time = spark_row["total_time"]
            spark_speedup = spark_row["speedup"]
            spark_efficiency = spark_row["efficiency"]
        else:
            spark_time = None
            spark_speedup = None
            spark_efficiency = None
        
        # Get PyTorch metrics for this parallelism level
        if level in torch_results["parallelism"].values:
            torch_row = torch_results[torch_results["parallelism"] == level].iloc[0]
            torch_time = torch_row["total_time"]
            torch_speedup = torch_row["speedup"]
            torch_efficiency = torch_row["efficiency"]
        else:
            torch_time = None
            torch_speedup = None
            torch_efficiency = None
        
        # Calculate ratio of PyTorch to Spark performance (if both exist)
        if spark_time is not None and torch_time is not None:
            time_ratio = torch_time / spark_time
            speedup_ratio = torch_speedup / spark_speedup if spark_speedup > 0 else 0
            efficiency_ratio = torch_efficiency / spark_efficiency if spark_efficiency > 0 else 0
        else:
            time_ratio = None
            speedup_ratio = None
            efficiency_ratio = None
        
        comparison_rows.append({
            "parallelism": level,
            "spark_time": spark_time,
            "torch_time": torch_time,
            "time_ratio": time_ratio,
            "spark_speedup": spark_speedup,
            "torch_speedup": torch_speedup,
            "speedup_ratio": speedup_ratio,
            "spark_efficiency": spark_efficiency,
            "torch_efficiency": torch_efficiency,
            "efficiency_ratio": efficiency_ratio
        })
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Save to CSV
    comparison_df.to_csv("spark_vs_pytorch_comparison.csv", index=False)
    print("Comparison table saved to 'spark_vs_pytorch_comparison.csv'")
    
    # Print formatted table
    print("\nSpark vs PyTorch Performance Comparison:")
    print("="*100)
    print(f"{'Parallelism':<12} {'Spark Time(s)':<15} {'PyTorch Time(s)':<15} {'Time Ratio':<12} "
          f"{'Spark Speedup':<15} {'PyTorch Speedup':<15} {'Speedup Ratio':<15}")
    print("-"*100)
    
    for row in comparison_rows:
        parallelism = row["parallelism"]
        spark_time = f"{row['spark_time']:.2f}" if row['spark_time'] is not None else "N/A"
        torch_time = f"{row['torch_time']:.2f}" if row['torch_time'] is not None else "N/A"
        time_ratio = f"{row['time_ratio']:.2f}" if row['time_ratio'] is not None else "N/A"
        spark_speedup = f"{row['spark_speedup']:.2f}x" if row['spark_speedup'] is not None else "N/A"
        torch_speedup = f"{row['torch_speedup']:.2f}x" if row['torch_speedup'] is not None else "N/A"
        speedup_ratio = f"{row['speedup_ratio']:.2f}" if row['speedup_ratio'] is not None else "N/A"
        
        print(f"{parallelism:<12} {spark_time:<15} {torch_time:<15} {time_ratio:<12} "
              f"{spark_speedup:<15} {torch_speedup:<15} {speedup_ratio:<15}")
    
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description="Compare Spark and PyTorch benchmark results")
    parser.add_argument("--spark", default="spark_benchmark_results.csv", 
                        help="Path to Spark benchmark results CSV")
    parser.add_argument("--torch", default="thread_benchmark_results.csv", 
                        help="Path to PyTorch benchmark results CSV")
    
    args = parser.parse_args()
    
    try:
        # Load benchmark results
        spark_results, torch_results = load_benchmark_results(args.spark, args.torch)
        
        # Calculate comparative metrics
        spark_results, torch_results = calculate_comparative_metrics(spark_results, torch_results)
        
        # Create comparison plots
        create_comparative_plots(spark_results, torch_results)
        
        # Create summary table
        create_summary_table(spark_results, torch_results)
        
        print("Comparison completed successfully.")
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
