#!/usr/bin/env python3
"""
Simple and reliable benchmark for PyTorch threading with transformers models.
Fixes threading configuration issues by initializing all threading settings upfront.
"""

import json
import os
import time
import psutil
import numpy as np
import pandas as pd
import torch
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
    """Simple function to measure current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    return memory_mb

def benchmark_thread_count(num_threads, reviews, model, tokenizer, num_samples=500):
    """Run benchmark with specified thread count"""
    # Set thread count (interop threads are set once at startup)
    torch.set_num_threads(num_threads)
    
    print(f"\n=== Testing with {num_threads} thread(s) ===")
    print(f"Actual torch.get_num_threads(): {torch.get_num_threads()}")
    
    # Limit reviews to requested sample size
    reviews_to_process = reviews[:num_samples]
    print(f"Processing {len(reviews_to_process)} reviews")
    
    # Memory before processing
    memory_before = measure_memory()
    print(f"Memory before processing: {memory_before:.2f} MB")
    
    # Process reviews
    processing_times = []
    token_counts = []
    
    start_total = time.time()
    
    # Process each review and measure time
    for review_text in tqdm(reviews_to_process, desc=f"Processing with {num_threads} threads"):
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
    
    # Calculate total time
    total_time = time.time() - start_total
    
    # Memory after processing
    memory_after = measure_memory()
    memory_increase = memory_after - memory_before
    
    # CPU usage 
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Calculate metrics
    metrics = {
        "num_threads": num_threads,
        "total_samples": len(reviews_to_process),
        "total_time": total_time,
        "avg_time_per_sample": np.mean(processing_times),
        "max_time_per_sample": np.max(processing_times),
        "min_time_per_sample": np.min(processing_times),
        "total_tokens": sum(token_counts),
        "throughput_tokens_per_second": sum(token_counts) / total_time,
        "throughput_reviews_per_second": len(reviews_to_process) / total_time,
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "memory_increase_mb": memory_increase,
        "cpu_percent": cpu_percent
    }
    
    # Print metrics
    print("\nResults:")
    print(f"  Total time: {metrics['total_time']:.2f} seconds")
    print(f"  Average time per sample: {metrics['avg_time_per_sample']:.4f} seconds")
    print(f"  Throughput: {metrics['throughput_tokens_per_second']:.2f} tokens/second")
    print(f"  Memory usage: {memory_before:.1f} MB → {memory_after:.1f} MB (increase: {memory_increase:.1f} MB)")
    print(f"  CPU usage: {cpu_percent:.1f}%")
    
    return metrics

def plot_results(all_metrics):
    """Create simple plots of benchmark results"""
    threads = [m["num_threads"] for m in all_metrics]
    times = [m["total_time"] for m in all_metrics]
    throughputs = [m["throughput_tokens_per_second"] for m in all_metrics]
    
    # Calculate speedup relative to single thread
    base_time = times[0]
    speedups = [base_time / time for time in times]
    
    # Create figure with simple plots
    plt.figure(figsize=(10, 6))
    
    # Plot speedup
    plt.plot(threads, speedups, 'o-', color='blue', linewidth=2, label="Actual Speedup")
    plt.plot(threads, threads, '--', color='gray', linewidth=1, label="Perfect Scaling")
    
    plt.title('Thread Scaling Performance')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup Factor')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('thread_speedup.png')
    print("Speedup plot saved to 'thread_speedup.png'")
    
    # Create a simple table for the report
    print("\nThread Scaling Summary:")
    print("="*70)
    print(f"{'Threads':<8} {'Time (s)':<10} {'Throughput':<15} {'Speedup':<10} {'Efficiency':<10}")
    print("-"*70)
    
    for i, m in enumerate(all_metrics):
        speedup = speedups[i]
        efficiency = speedup / m["num_threads"]
        print(f"{m['num_threads']:<8} {m['total_time']:<10.2f} {m['throughput_tokens_per_second']:<15.2f} {speedup:<10.2f}x {efficiency:<10.2f}")
    
    print("="*70)
    
    # Save metrics to CSV
    pd.DataFrame(all_metrics).to_csv('thread_benchmark_results.csv', index=False)
    print("Full results saved to 'thread_benchmark_results.csv'")

def main():
    # First, configure PyTorch threading at startup
    # Set interop threads once before any other operations
    torch.set_num_interop_threads(1)
    print(f"Set interop threads to 1")
    
    # Load the data
    data_file = "Subscription_Boxes.jsonl"
    if not os.path.exists(data_file):
        print(f"Error: File not found: {data_file}")
        return
    
    reviews = load_data(data_file, limit=2000)
    
    # Load model and tokenizer once
    print("\nLoading model and tokenizer...")
    model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Define thread counts to test
    thread_counts = [1, 2, 3, 4]
    num_samples = 500  # Number of reviews to process in each test
    
    # Run benchmarks for each thread count
    all_metrics = []
    for num_threads in thread_counts:
        metrics = benchmark_thread_count(
            num_threads=num_threads,
            reviews=reviews,
            model=model,
            tokenizer=tokenizer,
            num_samples=num_samples
        )
        all_metrics.append(metrics)
        
        # Short pause between runs
        time.sleep(2)
    
    # Generate report and visualizations
    plot_results(all_metrics)

if __name__ == "__main__":
    main()
    