#!/usr/bin/env python3
"""
PySpark core scaling benchmark for transformer models.
Tests performance with different numbers of Spark executors/cores
while keeping PyTorch threads fixed at 1.
"""

import json
import os
import sys
import time
import psutil
import gc
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import subprocess
from tqdm import tqdm

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import (
    StringType,
    FloatType,
    StructType,
    StructField,
)

from sentiment_analysis.config import logger
from sentiment_analysis.utils import delete_local_file, run_command
from sentiment_analysis.load import load_amazon_reviews, load_model
import sentiment_analysis.process as process

# PyTorch and transformers imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def measure_memory():
    """Measure current process memory usage"""
    # Force garbage collection to get accurate measurements
    gc.collect()

    # Get process memory info
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    return memory_mb


def load_data(spark, file_path, limit=None):
    """Load Amazon reviews from JSONL file into a Spark DataFrame"""
    # Read the JSON Lines file
    reviews_df = spark.read.json(file_path)

    # Apply limit if specified
    if limit:
        reviews_df = reviews_df.limit(limit)

    # Select necessary columns
    reviews_df = reviews_df.select(
        col("asin"),
        col("user_id"),
        col("rating"),
        col("text"),
    )

    # Cache the DataFrame to improve performance
    reviews_df.cache()

    # Count and log the number of reviews
    count = reviews_df.count()
    logger.info(f"Loaded {count} reviews from {file_path}")

    return reviews_df


def run_spark_benchmark(data_file, num_cores, num_samples):
    """Run a benchmark with the specified number of Spark cores"""
    logger.info(f"\n=== Testing with {num_cores} Spark cores (PyTorch threads=1) ===")
    
    # Create a new SparkSession with the correct core count
    spark_session = (
        SparkSession.builder.appName(f"SparkCoreScalingBenchmark{num_cores}")
        .config("spark.cores.max", str(num_cores))
        .getOrCreate()
    )
    
    # Log the actual configuration
    actual_cores = spark_session.conf.get("spark.cores.max")
    logger.info(f"Actual spark.cores.max: {actual_cores}")
    
    # Load data
    data_df = load_data(spark_session, data_file, num_samples)
    logger.info(f"Loaded {data_df.count()} reviews for processing")
    
    # Repartition the DataFrame to match the number of cores
    reviews_df = data_df.repartition(num_cores)
    logger.info(f"Repartitioned DataFrame to {num_cores} partitions")
    
    # Load model and tokenizer
    tokenizer, model = load_model()
    # Broadcast model and tokenizer to all workers
    process.bc_tokenizer = spark_session.sparkContext.broadcast(tokenizer)
    process.bc_model = spark_session.sparkContext.broadcast(model)
    
    # Memory before processing
    memory_before = measure_memory()
    logger.info(f"Memory before processing: {memory_before:.2f} MB")
    
    # Apply sentiment analysis
    sentiment_results_df = reviews_df.withColumn(
        "result", process.batch_sentiment_analysis(reviews_df["text"])
    )
    
    # Flatten the result column and force execution
    results_df = sentiment_results_df.select(
        col("asin"),
        col("user_id"),
        col("result.review_text"),
        col("result.sentiment"),
        col("result.score"),
    )
    
    # Count to force execution of the entire pipeline
    total_results = results_df.count()
    
    # Start timing
    start_time = time.time()
    
    # Write results to csv
    output_path = f"/analysis_outputs/cores_{num_cores}_results.csv"
    logger.info(f"Writing results to {output_path}")
    results_df.write.option("header", "true").mode("overwrite").csv(
        output_path
    )
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Memory after processing
    memory_after = measure_memory()
    memory_increase = memory_after - memory_before
    
    # Calculate metrics
    metrics = {
        "cores": num_cores,
        "total_samples": total_results,
        "total_time": total_time,
        "throughput_samples_per_second": total_results / total_time,
        "memory_before_mb": memory_before,
        "memory_after_mb": memory_after,
        "memory_increase_mb": memory_increase,
    }
    
    # Print results
    logger.info(f"Results for {num_cores} cores:")
    logger.info(f"  Total time: {metrics['total_time']:.2f} seconds")
    logger.info(
        f"  Throughput: {metrics['throughput_samples_per_second']:.2f} samples/second"
    )
    logger.info(
        f"  Memory: {memory_before:.1f} MB → {memory_after:.1f} MB (increase: {memory_increase:.1f} MB)"
    )
    
    # Save metrics to file
    metrics_file = f"spark_benchmark_cores_{num_cores}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    logger.info(f"Metrics saved to {metrics_file}")
    # Stop this Spark session
    spark_session.stop()
    # Allow JVM to clean up resources
    time.sleep(2)

    return metrics
    


def plot_results(all_metrics):
    """Create plots for benchmark results"""
    # Extract metrics
    cores = [m["cores"] for m in all_metrics]
    times = [m["total_time"] for m in all_metrics]
    throughputs = [m["throughput_samples_per_second"] for m in all_metrics]

    # Find baseline (1 core)
    baseline_idx = next((i for i, m in enumerate(all_metrics) if m["cores"] == 1), 0)
    base_time = times[baseline_idx]

    # Calculate speedup
    speedups = [base_time / t for t in times]

    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Processing Time
    axs[0, 0].plot(cores, times, "o-", linewidth=2, color="blue")
    axs[0, 0].set_title("Processing Time vs. Core Count")
    axs[0, 0].set_xlabel("Spark Cores")
    axs[0, 0].set_ylabel("Time (seconds)")
    axs[0, 0].grid(True)

    # Plot 2: Throughput
    axs[0, 1].plot(cores, throughputs, "o-", color="green", linewidth=2)
    axs[0, 1].set_title("Throughput vs. Core Count")
    axs[0, 1].set_xlabel("Spark Cores")
    axs[0, 1].set_ylabel("Samples per Second")
    axs[0, 1].grid(True)

    # Plot 3: Speedup vs. Perfect Scaling
    axs[1, 0].plot(
        cores, speedups, "o-", color="red", linewidth=2, label="Actual Speedup"
    )
    axs[1, 0].plot(
        cores, cores, "--", color="gray", linewidth=1, label="Perfect Scaling"
    )
    axs[1, 0].set_title("Speedup vs. Core Count")
    axs[1, 0].set_xlabel("Spark Cores")
    axs[1, 0].set_ylabel("Speedup Factor")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    # Plot 4: Efficiency (Speedup/Cores)
    efficiency = [s / c for s, c in zip(speedups, cores)]
    axs[1, 1].plot(cores, efficiency, "o-", color="purple", linewidth=2)
    axs[1, 1].set_title("Scaling Efficiency vs. Core Count")
    axs[1, 1].set_xlabel("Spark Cores")
    axs[1, 1].set_ylabel("Efficiency (Speedup/Cores)")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("spark_scaling_results.png")
    logger.info("Results plots saved to 'spark_scaling_results.png'")

    # Save individual plots
    # Plot 1: Processing Time
    plt.figure(figsize=(8, 6))
    plt.plot(cores, times, "o-", linewidth=2, color="blue")
    plt.title("Processing Time vs. Core Count")
    plt.xlabel("Spark Cores")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.savefig("plot1_spark_processing_time.png")

    # Plot 2: Throughput
    plt.figure(figsize=(8, 6))
    plt.plot(cores, throughputs, "o-", color="green", linewidth=2)
    plt.title("Throughput vs. Core Count")
    plt.xlabel("Spark Cores")
    plt.ylabel("Samples per Second")
    plt.grid(True)
    plt.savefig("plot2_spark_throughput.png")

    # Plot 3: Speedup
    plt.figure(figsize=(8, 6))
    plt.plot(cores, speedups, "o-", color="red", linewidth=2, label="Actual Speedup")
    plt.plot(cores, cores, "--", color="gray", linewidth=1, label="Perfect Scaling")
    plt.title("Speedup vs. Core Count")
    plt.xlabel("Spark Cores")
    plt.ylabel("Speedup Factor")
    plt.grid(True)
    plt.legend()
    plt.savefig("plot3_spark_speedup.png")

    # Plot 4: Efficiency
    plt.figure(figsize=(8, 6))
    plt.plot(cores, efficiency, "o-", color="purple", linewidth=2)
    plt.title("Scaling Efficiency vs. Core Count")
    plt.xlabel("Spark Cores")
    plt.ylabel("Efficiency (Speedup/Cores)")
    plt.grid(True)
    plt.savefig("plot4_spark_efficiency.png")

    # Create summary table
    print("\nSpark Core Scaling Performance Summary:")
    print("=" * 100)
    print(
        f"{'Cores':<8} {'Time(s)':<10} {'Throughput':<15} {'Speedup':<10} {'Efficiency':<10}"
    )
    print("-" * 100)

    sorted_metrics = sorted(all_metrics, key=lambda m: m["cores"])

    for m in sorted_metrics:
        speedup = base_time / m["total_time"]
        efficiency = speedup / m["cores"] if m["cores"] > 0 else 0

        print(
            f"{m['cores']:<8} {m['total_time']:<10.2f} "
            f"{m['throughput_samples_per_second']:<15.2f} {speedup:<10.2f}x {efficiency:<10.2f}"
        )

    print("=" * 100)

    # Save all metrics to CSV
    pd.DataFrame(all_metrics).to_csv("spark_benchmark_results.csv", index=False)
    logger.info("Full results saved to 'spark_benchmark_results.csv'")


def main():
    parser = argparse.ArgumentParser(description="PySpark Core Scaling Benchmark")
    parser.add_argument(
        "--data",
        default="Subscription_Boxes.jsonl",
        help="Path to JSONL file with reviews",
    )
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of samples to process"
    )
    parser.add_argument(
        "--cores",
        type=str,
        default="1,2,3,4",
        help="Comma-separated list of core counts to test",
    )

    args = parser.parse_args()

    # Parse core counts
    core_counts = [int(c) for c in args.cores.split(",")]

    # Always set PyTorch threads to 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    logger.info(
        f"Fixed PyTorch threads to 1 (get_num_threads={torch.get_num_threads()}, get_num_interop_threads={torch.get_num_interop_threads()})"
    )


    # Run benchmarks for each core count
    all_metrics = []

    for cores in core_counts:
        metrics = run_spark_benchmark(args.data, cores, args.samples)
        if metrics:
            all_metrics.append(metrics)

        # Short pause between runs
        time.sleep(2)

    # Generate plots and summary
    plot_results(all_metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
