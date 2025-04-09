#!/usr/bin/env python3
"""
PyTorch thread scaling benchmark with simplified memory tracking.
Only records the most important memory metrics at key points.
"""

import json
import os
import sys
import time
import psutil
import threading
import gc
import numpy as np
import pandas as pd
import torch
import subprocess
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

def load_data(file_path, limit=1000):
    """Load Amazon reviews from JSONL file"""
    reviews = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            reviews.append(json.loads(line))
    
    print(f"Loaded {len(reviews)} reviews from {file_path}")
    return [review["text"] for review in reviews]

def measure_memory():
    """Measure memory usage after forcing garbage collection"""
    # Force garbage collection to get accurate measurements
    gc.collect()
    
    # Get process memory info
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    return memory_mb

def monitor_cpu_during_processing(reviews_to_process, tokenizer, model):
    """Continuously monitor CPU while processing reviews"""
    # Storage for CPU measurements and results
    cpu_samples = []
    stop_monitoring = threading.Event()
    processing_times = []
    token_counts = []
    
    # Memory before processing
    memory_before = measure_memory()
    
    # CPU monitoring thread function
    def cpu_monitor_thread():
        # Wait briefly for processing to begin
        time.sleep(1.0)
        
        # Take samples until stopped
        while not stop_monitoring.is_set():
            # Get per-CPU utilization (0.1s interval)
            per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Store the sample
            cpu_samples.append(per_cpu)
            
            # Don't sample too frequently
            time.sleep(0.1)
    
    # Start the monitoring thread
    monitor = threading.Thread(target=cpu_monitor_thread)
    monitor.daemon = True
    monitor.start()
    
    # Process the reviews
    start_total = time.time()
    
    # Track peak memory during processing
    peak_memory = memory_before
    
    # Process each review
    for review_text in tqdm(reviews_to_process, desc="Processing reviews"):
        start_time = time.time()
        
        # Tokenize the review text
        inputs = tokenizer(
            review_text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True
        )
        
        # Count tokens
        token_count = inputs["attention_mask"].sum().item()
        token_counts.append(token_count)
        
        # Perform sentiment analysis
        with torch.no_grad():
            sentiment_batch = model(**inputs).logits
        
        # Record time
        batch_time = time.time() - start_time
        processing_times.append(batch_time)
        
        # Check if memory has increased
        if len(processing_times) % 50 == 0:
            current_memory = measure_memory()
            peak_memory = max(peak_memory, current_memory)
    
    total_time = time.time() - start_total
    
    # Memory after processing
    memory_after = measure_memory()
    
    # Signal monitoring thread to stop
    stop_monitoring.set()
    monitor.join(timeout=1.0)
    
    # Process CPU data
    num_cpus = psutil.cpu_count(logical=True)
    if not cpu_samples:
        print("Warning: No CPU samples collected!")
        cpu_metrics = {
            "num_cpus": num_cpus,
            "per_core_usage": [0] * num_cpus,
            "active_cores": 0,
            "avg_usage": 0,
            "max_core_usage": 0
        }
    else:
        # Calculate average usage for each core
        per_core_usage = []
        for core in range(num_cpus):
            core_samples = [sample[core] for sample in cpu_samples]
            per_core_usage.append(np.mean(core_samples))
        
        # Count active cores (those with >50% usage)
        active_cores = sum(1 for usage in per_core_usage if usage > 50.0)
        
        cpu_metrics = {
            "num_cpus": num_cpus,
            "per_core_usage": per_core_usage,
            "active_cores": active_cores,
            "avg_usage": np.mean(per_core_usage),
            "max_core_usage": np.max(per_core_usage),
            "samples_collected": len(cpu_samples)
        }
    
    # Memory metrics
    memory_metrics = {
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "peak_memory_mb": peak_memory,
        "memory_increase_mb": peak_memory - memory_before
    }
    
    # Return all results
    return {
        "processing_times": processing_times,
        "token_counts": token_counts,
        "total_time": total_time,
        "cpu_metrics": cpu_metrics,
        "memory_metrics": memory_metrics
    }

def run_single_benchmark(threads, reviews, num_samples=500):
    """Run a single benchmark with equal intra/inter thread configuration"""
    # Clear memory before starting
    gc.collect()
    
    # Set thread counts (equal for both intra and inter)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    
    print(f"\n=== Testing with {threads} threads (both intra-op and inter-op) ===")
    print(f"Actual torch.get_num_threads(): {torch.get_num_threads()}")
    print(f"Actual torch.get_num_interop_threads(): {torch.get_num_interop_threads()}")
    
    # Initial memory measurement (before model load)
    initial_memory = measure_memory()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Memory after model load
    model_memory = measure_memory()
    model_increase = model_memory - initial_memory
    print(f"Memory after model loading: {model_memory:.2f} MB (increase: {model_increase:.2f} MB)")
    
    # Limit reviews to requested sample size
    reviews_to_process = reviews[:num_samples]
    print(f"Processing {len(reviews_to_process)} reviews")
    
    # Process reviews with CPU and memory monitoring
    results = monitor_cpu_during_processing(reviews_to_process, tokenizer, model)
    
    # Extract results
    processing_times = results["processing_times"]
    token_counts = results["token_counts"]
    total_time = results["total_time"]
    cpu_metrics = results["cpu_metrics"]
    memory_metrics = results["memory_metrics"]
    
    # Calculate metrics
    metrics = {
        "threads": threads,
        "total_threads": threads * 2,  # Total of intra + inter
        "total_samples": len(reviews_to_process),
        "total_time": total_time,
        "avg_time_per_sample": np.mean(processing_times),
        "max_time_per_sample": np.max(processing_times),
        "min_time_per_sample": np.min(processing_times),
        "total_tokens": sum(token_counts),
        "throughput_tokens_per_second": sum(token_counts) / total_time,
        "throughput_reviews_per_second": len(reviews_to_process) / total_time,
        "initial_memory_mb": initial_memory,
        "model_memory_mb": model_memory,
        "peak_memory_mb": memory_metrics["peak_memory_mb"],
        "memory_increase_mb": memory_metrics["memory_increase_mb"],
        "num_cpu_cores": cpu_metrics["num_cpus"],
        "active_cpu_cores": cpu_metrics["active_cores"],
        "avg_core_usage": cpu_metrics["avg_usage"],
        "max_core_usage": cpu_metrics["max_core_usage"],
        "per_core_usage": cpu_metrics["per_core_usage"],
        "cpu_samples_collected": cpu_metrics.get("samples_collected", 0)
    }
    
    # Print metrics
    print("\nResults:")
    print(f"  Total time: {metrics['total_time']:.2f} seconds")
    print(f"  Average time per sample: {metrics['avg_time_per_sample']:.4f} seconds")
    print(f"  Throughput: {metrics['throughput_tokens_per_second']:.2f} tokens/second")
    
    # Print memory metrics
    print("\nMemory Usage:")
    print(f"  Initial memory: {metrics['initial_memory_mb']:.2f} MB")
    print(f"  After model load: {metrics['model_memory_mb']:.2f} MB")
    print(f"  Peak memory during processing: {metrics['peak_memory_mb']:.2f} MB")
    print(f"  Memory increase: {metrics['memory_increase_mb']:.2f} MB")
    
    # Print CPU metrics
    print("\nCPU Utilization:")
    print(f"  CPU cores: {metrics['active_cpu_cores']} active out of {metrics['num_cpu_cores']} total")
    print(f"  Average core usage: {metrics['avg_core_usage']:.2f}%")
    print(f"  Maximum core usage: {metrics['max_core_usage']:.2f}%")
    
    # Save metrics to a file
    result_file = f"benchmark_threads_{threads}.csv"
    pd.DataFrame([metrics]).to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")
    
    return metrics

def run_subprocess_benchmark(threads, data_file, num_samples):
    """Run a benchmark as a separate process to avoid thread setting limitations"""
    cmd = [
        sys.executable,  # Current Python interpreter
        __file__,  # This script
        "--mode", "single",
        "--threads", str(threads),
        "--data", data_file,
        "--samples", str(num_samples)
    ]
    
    print(f"Starting subprocess for {threads} threads...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error in subprocess: {stderr}")
        return None
    
    # Load results from the CSV file
    result_file = f"benchmark_threads_{threads}.csv"
    if os.path.exists(result_file):
        metrics = pd.read_csv(result_file).to_dict('records')[0]
        return metrics
    else:
        print(f"Result file not found: {result_file}")
        return None

def plot_results(all_metrics):
    """Create plots for benchmark results"""
    # Extract configurations and metrics
    threads = [m["threads"] for m in all_metrics]
    times = [m["total_time"] for m in all_metrics]
    throughputs = [m["throughput_tokens_per_second"] for m in all_metrics]
    active_cores = [m["active_cpu_cores"] for m in all_metrics]
    core_usage = [m["avg_core_usage"] for m in all_metrics]
    memory_increase = [m["memory_increase_mb"] for m in all_metrics]
    
    # Find baseline (1 thread)
    baseline_idx = next((i for i, m in enumerate(all_metrics) if m["threads"] == 1), 0)
    base_time = times[baseline_idx]
    
    # Calculate speedup
    speedups = [base_time / t for t in times]
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 15))
    
    # Plot 1: Processing Time
    axs[0, 0].plot(threads, times, 'o-', linewidth=2, color='blue')
    axs[0, 0].set_title('Processing Time vs. Thread Count')
    axs[0, 0].set_xlabel('Threads Count')
    axs[0, 0].set_ylabel('Time (seconds)')
    axs[0, 0].grid(True)
    
    # Plot 2: Throughput
    axs[0, 1].plot(threads, throughputs, 'o-', color='green', linewidth=2)
    axs[0, 1].set_title('Throughput vs. Thread Count')
    axs[0, 1].set_xlabel('Threads Count')
    axs[0, 1].set_ylabel('Tokens per Second')
    axs[0, 1].grid(True)
    
    # Plot 3: Speedup vs. Perfect Scaling
    axs[1, 0].plot(threads, speedups, 'o-', color='red', linewidth=2, label="Actual Speedup")
    axs[1, 0].plot(threads, threads, '--', color='gray', linewidth=1, label="Perfect Scaling")
    axs[1, 0].set_title('Speedup vs. Thread Count')
    axs[1, 0].set_xlabel('Threads Count')
    axs[1, 0].set_ylabel('Speedup Factor')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Plot 4: CPU Core Usage
    axs[1, 1].plot(threads, core_usage, 'o-', color='purple', linewidth=2)
    axs[1, 1].set_title('Average CPU Core Usage vs. Thread Count')
    axs[1, 1].set_xlabel('Threads Count')
    axs[1, 1].set_ylabel('Average Core Usage (%)')
    axs[1, 1].grid(True)
    
    # Plot 5: Active CPU Cores
    axs[2, 0].plot(threads, active_cores, 'o-', color='orange', linewidth=2)
    axs[2, 0].set_title('Active CPU Cores vs. Thread Count')
    axs[2, 0].set_xlabel('Threads Count')
    axs[2, 0].set_ylabel('Number of Active Cores')
    axs[2, 0].grid(True)
    
    # Plot 6: Memory Increase
    axs[2, 1].plot(threads, memory_increase, 'o-', color='brown', linewidth=2)
    axs[2, 1].set_title('Memory Increase vs. Thread Count')
    axs[2, 1].set_xlabel('Threads Count')
    axs[2, 1].set_ylabel('Memory Increase (MB)')
    axs[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('thread_scaling_results.png')
    print("Results plots saved to 'thread_scaling_results.png'")
    
    # Create detailed summary
    print("\nThread Configuration Performance Summary:")
    print("="*100)
    print(f"{'Threads':<8} {'Time(s)':<10} {'Throughput':<12} {'Speedup':<10} {'Efficiency':<10} {'Active Cores':<13} {'Mem Incr(MB)':<12}")
    print("-"*100)
    
    sorted_metrics = sorted(all_metrics, key=lambda m: m["threads"])
    
    for m in sorted_metrics:
        speedup = base_time / m["total_time"]
        efficiency = speedup / m["threads"] if m["threads"] > 0 else 0
        
        print(f"{m['threads']:<8} {m['total_time']:<10.2f} "
              f"{m['throughput_tokens_per_second']:<12.2f} {speedup:<10.2f}x {efficiency:<10.2f} "
              f"{m['active_cpu_cores']:<13d} {m['memory_increase_mb']:<12.2f}")
    
    print("="*100)
    
    # Save all metrics to CSV
    pd.DataFrame(all_metrics).to_csv('thread_benchmark_results.csv', index=False)
    print("Full results saved to 'thread_benchmark_results.csv'")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Equal Thread Configuration Benchmark")
    parser.add_argument("--mode", choices=["single", "multi"], default="multi",
                        help="Run a single benchmark or multiple configurations")
    parser.add_argument("--threads", type=int, default=1,
                        help="Number of threads (for single mode)")
    parser.add_argument("--data", default="Subscription_Boxes.jsonl",
                        help="Path to data file")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to process")
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        return 1
    
    # Single benchmark mode (called as subprocess)
    if args.mode == "single":
        reviews = load_data(args.data, limit=args.samples * 2)  # Load a bit more than needed
        run_single_benchmark(args.threads, reviews, args.samples)
        return 0
    
    # Multi-configuration benchmark mode
    print("Starting PyTorch Thread Scaling Benchmark")
    print(f"Data file: {args.data}")
    print(f"Samples per test: {args.samples}")
    
    # Define thread counts to test (equal intra and inter)
    thread_counts = [1, 2, 3, 4]
    
    # Run benchmarks for each configuration
    all_metrics = []
    
    for threads in thread_counts:
        metrics = run_subprocess_benchmark(threads, args.data, args.samples)
        if metrics:
            all_metrics.append(metrics)
        
        # Short pause between runs
        time.sleep(2)
    
    # Generate plots and summary
    plot_results(all_metrics)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
