#!/usr/bin/env python3
"""
PySpark core scaling benchmark for transformer models.
Tests performance with different numbers of Spark executors/cores
while keeping PyTorch threads fixed at 1.
Includes token statistics for input dataset.
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
from pyspark.sql.functions import col, pandas_udf, expr, min as spark_min, max as spark_max, avg as spark_avg, sum as spark_sum
from pyspark.sql.types import (
    StringType,
    FloatType,
    IntegerType,
    StructType,
    StructField,
)

from sentiment_analysis.config import logger
from sentiment_analysis.utils import delete_local_file, run_command
from sentiment_analysis.load import load_amazon_reviews, load_model
import sentiment_analysis.process as process
import sentiment_analysis.postprocess as postprocess


# PyTorch and transformers imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def compute_token_counts(df, tokenizer):
    """
    Compute token count statistics for the reviews dataframe
    Returns a new dataframe with token counts added
    """
    # Create a UDF to count tokens
    @pandas_udf(IntegerType())
    def count_tokens(texts):
        token_counts = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=128)
            token_counts.append(len(tokens))
        return pd.Series(token_counts)
    
    # Apply the UDF to compute token counts
    df_with_tokens = df.withColumn("token_count", count_tokens(df["text"]))
    
    # Compute statistics
    token_stats = df_with_tokens.agg(
        spark_sum("token_count").alias("total_tokens"),
        spark_min("token_count").alias("min_tokens"),
        spark_max("token_count").alias("max_tokens"),
        spark_avg("token_count").alias("avg_tokens"),
    ).collect()[0]
    
    # Log token statistics
    logger.info(f"Token statistics:")
    logger.info(f"  Total tokens: {token_stats['total_tokens']}")
    logger.info(f"  Min tokens per review: {token_stats['min_tokens']}")
    logger.info(f"  Max tokens per review: {token_stats['max_tokens']}")
    logger.info(f"  Avg tokens per review: {token_stats['avg_tokens']:.2f}")
    
    return df_with_tokens, token_stats


def load_data(spark, file_path, tokenizer, limit=None):
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

    # Compute token counts and get statistics
    # reviews_df_with_tokens, token_stats = compute_token_counts(reviews_df, tokenizer)

    # Cache the DataFrame to improve performance
    reviews_df.cache()

    # Count and log the number of reviews
    count = reviews_df.count()
    logger.info(f"Loaded {count} reviews from {file_path}")

    return reviews_df


def run_spark_benchmark(data_file, num_cores, sample_ratio, num_samples):
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

    # Load model and tokenizer first (needed for token counting)
    tokenizer, model = load_model()
    
    # Load data
    data_df = load_amazon_reviews(spark=spark_session, file_path=data_file, sample_ratio=sample_ratio, num_samples=num_samples)
    logger.info(f"Loaded {data_df.count()} reviews for processing")

    # Repartition the DataFrame to match the number of cores
    reviews_df = data_df.repartition(num_cores)
    logger.info(f"Repartitioned DataFrame to {num_cores} partitions")

    # Broadcast model and tokenizer to all workers
    logger.info("Broadcasting model and tokenizer to all workers")
    process.bc_tokenizer = spark_session.sparkContext.broadcast(tokenizer)
    process.bc_model = spark_session.sparkContext.broadcast(model)

    # Apply sentiment analysis
    analysis_output_path = f"/analysis_outputs/cores_{num_cores}_results.csv"
    summary_output_path = f"/analysis_outputs/summary_{num_cores}_results.csv"
    start_time = time.time()
    sentiment_analysis_results_df = process.process_reviews(
        reviews_df=reviews_df,
        output_path=analysis_output_path,
        review_text_column="text",
    )
    total_time = time.time() - start_time
    total_results = sentiment_analysis_results_df.count()
    logger.info(
        f"Sentiment analysis completed in {total_time:.1f} seconds with average speed of {total_results / total_time:.1f} reviews/second"
    )

    # Generate token statistics
    token_stats = postprocess.generate_token_statistics(sentiment_analysis_results_df, summary_output_path)

    # Calculate metrics
    metrics = {
        "cores": num_cores,
        "total_samples": total_results,
        "total_tokens": token_stats["total_tokens"],
        "min_tokens": token_stats["min_token_count"],
        "max_tokens": token_stats["max_token_count"],
        "avg_tokens": float(token_stats["mean_token_count"]),
        "total_time": total_time,
        "throughput_samples_per_second": total_results / total_time,
        "throughput_tokens_per_second": token_stats["total_tokens"] / total_time
    }

    # Print results
    logger.info(f"Results for {num_cores} cores:")
    logger.info(f"  Total time: {metrics['total_time']:.2f} seconds")
    logger.info(
        f"  Throughput: {metrics['throughput_samples_per_second']:.2f} samples/second"
    )
    logger.info(
        f"  Token throughput: {metrics['throughput_tokens_per_second']:.2f} tokens/second"
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
    token_throughputs = [m["throughput_tokens_per_second"] for m in all_metrics]

    # Find baseline (1 core)
    baseline_idx = next((i for i, m in enumerate(all_metrics) if m["cores"] == 1), 0)
    base_time = times[baseline_idx]

    # Calculate speedup
    speedups = [base_time / t for t in times]

    # Create figure with subplots - now 3x2 to include token metrics
    fig, axs = plt.subplots(3, 2, figsize=(14, 18))

    # Plot 1: Processing Time
    axs[0, 0].plot(cores, times, "o-", linewidth=2, color="blue")
    axs[0, 0].set_title("Processing Time vs. Core Count")
    axs[0, 0].set_xlabel("Spark Cores")
    axs[0, 0].set_ylabel("Time (seconds)")
    axs[0, 0].grid(True)

    # Plot 2: Sample Throughput
    axs[0, 1].plot(cores, throughputs, "o-", color="green", linewidth=2)
    axs[0, 1].set_title("Sample Throughput vs. Core Count")
    axs[0, 1].set_xlabel("Spark Cores")
    axs[0, 1].set_ylabel("Samples per Second")
    axs[0, 1].grid(True)

    # Plot 3: Token Throughput
    axs[1, 0].plot(cores, token_throughputs, "o-", color="orange", linewidth=2)
    axs[1, 0].set_title("Token Throughput vs. Core Count")
    axs[1, 0].set_xlabel("Spark Cores")
    axs[1, 0].set_ylabel("Tokens per Second")
    axs[1, 0].grid(True)

    # Plot 4: Speedup vs. Perfect Scaling
    axs[1, 1].plot(
        cores, speedups, "o-", color="red", linewidth=2, label="Actual Speedup"
    )
    axs[1, 1].plot(
        cores, cores, "--", color="gray", linewidth=1, label="Perfect Scaling"
    )
    axs[1, 1].set_title("Speedup vs. Core Count")
    axs[1, 1].set_xlabel("Spark Cores")
    axs[1, 1].set_ylabel("Speedup Factor")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # Plot 5: Efficiency (Speedup/Cores)
    efficiency = [s / c for s, c in zip(speedups, cores)]
    axs[2, 0].plot(cores, efficiency, "o-", color="purple", linewidth=2)
    axs[2, 0].set_title("Scaling Efficiency vs. Core Count")
    axs[2, 0].set_xlabel("Spark Cores")
    axs[2, 0].set_ylabel("Efficiency (Speedup/Cores)")
    axs[2, 0].grid(True)

    # Plot 6: Tokens per Review Distribution (use the last metrics for this)
    axs[2, 1].bar(["Min", "Avg", "Max"], 
                 [all_metrics[-1]["min_tokens"], 
                  all_metrics[-1]["avg_tokens"], 
                  all_metrics[-1]["max_tokens"]], 
                 color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axs[2, 1].set_title("Token Count Distribution")
    axs[2, 1].set_ylabel("Number of Tokens")
    axs[2, 1].grid(axis="y")

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

    # Plot 2: Sample Throughput
    plt.figure(figsize=(8, 6))
    plt.plot(cores, throughputs, "o-", color="green", linewidth=2)
    plt.title("Sample Throughput vs. Core Count")
    plt.xlabel("Spark Cores")
    plt.ylabel("Samples per Second")
    plt.grid(True)
    plt.savefig("plot2_spark_throughput.png")

    # Plot 3: Token Throughput
    plt.figure(figsize=(8, 6))
    plt.plot(cores, token_throughputs, "o-", color="orange", linewidth=2)
    plt.title("Token Throughput vs. Core Count")
    plt.xlabel("Spark Cores")
    plt.ylabel("Tokens per Second")
    plt.grid(True)
    plt.savefig("plot3_spark_token_throughput.png")

    # Plot 4: Speedup
    plt.figure(figsize=(8, 6))
    plt.plot(cores, speedups, "o-", color="red", linewidth=2, label="Actual Speedup")
    plt.plot(cores, cores, "--", color="gray", linewidth=1, label="Perfect Scaling")
    plt.title("Speedup vs. Core Count")
    plt.xlabel("Spark Cores")
    plt.ylabel("Speedup Factor")
    plt.grid(True)
    plt.legend()
    plt.savefig("plot4_spark_speedup.png")

    # Plot 5: Efficiency
    plt.figure(figsize=(8, 6))
    plt.plot(cores, efficiency, "o-", color="purple", linewidth=2)
    plt.title("Scaling Efficiency vs. Core Count")
    plt.xlabel("Spark Cores")
    plt.ylabel("Efficiency (Speedup/Cores)")
    plt.grid(True)
    plt.savefig("plot5_spark_efficiency.png")

    # Plot 6: Token Distribution
    plt.figure(figsize=(8, 6))
    plt.bar(["Min", "Avg", "Max"], 
            [all_metrics[-1]["min_tokens"], 
             all_metrics[-1]["avg_tokens"], 
             all_metrics[-1]["max_tokens"]], 
            color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    plt.title("Token Count Distribution")
    plt.ylabel("Number of Tokens")
    plt.grid(axis="y")
    plt.savefig("plot6_token_distribution.png")

    # Create summary table
    print("\nSpark Core Scaling Performance Summary:")
    print("=" * 140)
    print(
        f"{'Cores':<8} {'Time(s)':<10} {'Samples/s':<12} {'Tokens/s':<12} {'Speedup':<10} {'Efficiency':<10} {'Avg Tokens':<12} {'Min Tokens':<12} {'Max Tokens':<12}"
    )
    print("-" * 140)

    sorted_metrics = sorted(all_metrics, key=lambda m: m["cores"])

    for m in sorted_metrics:
        speedup = base_time / m["total_time"]
        efficiency = speedup / m["cores"] if m["cores"] > 0 else 0

        print(
            f"{m['cores']:<8} {m['total_time']:<10.2f} "
            f"{m['throughput_samples_per_second']:<12.2f} {m['throughput_tokens_per_second']:<12.2f} "
            f"{speedup:<10.2f}x {efficiency:<10.2f} {m['avg_tokens']:<12.2f} "
            f"{m['min_tokens']:<12d} {m['max_tokens']:<12d}"
        )

    print("=" * 140)

    # Save all metrics to CSV
    pd.DataFrame(all_metrics).to_csv("spark_benchmark_results.csv", index=False)
    logger.info("Full results saved to 'spark_benchmark_results.csv'")


def main():
    parser = argparse.ArgumentParser(description="PySpark Core Scaling Benchmark")
    parser.add_argument(
        "--data",
        default="/Subscription_Boxes.jsonl",
        help="Path to JSONL file with reviews",
    )
    parser.add_argument(
        "--sample-ratio", type=float, default=1.0, help="Sample ratio (0.0-1.0) of the dataset to process"
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
        metrics = run_spark_benchmark(args.data, cores, args.sample_ratio, args.samples)
        if metrics:
            all_metrics.append(metrics)

        # Short pause between runs
        time.sleep(2)

    # Generate plots and summary
    plot_results(all_metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
