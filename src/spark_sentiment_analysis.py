import argparse
import sys
import time
import json
import os
import threading

import sentiment_analysis.process as process
import sentiment_analysis.preprocess as preprocess
import sentiment_analysis.postprocess as postprocess

from pyspark.sql import SparkSession
from sentiment_analysis.config import logger
from sentiment_analysis.load import load_amazon_reviews, load_model

# Define progress file path as a constant
PROGRESS_FILE = "/tmp/sentiment_analysis_progress.txt"

# Define available datasets
PREDEFINED_DATASETS = ["subscription_boxes", "magazine_subscriptions", "all_beauty", "appliances", "baby_products"]


def update_progress(stage, current=0, total=0, start_time=None):
    """Update the progress file with current status"""
    if start_time is None:
        elapsed = 0
    else:
        elapsed = time.time() - start_time

    percentage = (current / total * 100) if total > 0 else 0

    with open(PROGRESS_FILE, "w") as f:
        f.write(f"Stage: {stage}\n")
        if total > 0:
            f.write(f"Progress: {percentage:.1f}% ({current}/{total})\n")
        f.write(f"Elapsed time: {elapsed:.1f} seconds\n")

    # Also log the progress
    if total > 0:
        logger.info(f"Progress: {stage} - {percentage:.1f}% ({current}/{total})")
    else:
        logger.info(f"Progress: {stage}")


def track_batch(processed_count, batch_size=1000):
    """Returns a function to track progress in Spark partitions"""

    def _track_batch_fn(partition_index, iterator):
        counter = 0
        for row in iterator:
            counter += 1
            if counter % batch_size == 0:
                processed_count.add(batch_size)
            yield row

    return _track_batch_fn


def monitor_progress_thread(processed_count, total_count, start_time, stop_event):
    """Thread function to monitor progress and update the progress file"""
    last_count = 0
    while not stop_event.is_set():
        current = processed_count.value
        if current > last_count:
            # Only update if there's been progress
            update_progress("Processing sentiment", current, total_count, start_time)
            last_count = current
        time.sleep(2)  # Check every 2 seconds


def save_final_summary(total_reviews, total_duration, process_duration, partitions_count):
    """Save the final summary and metrics to files."""
    # Save final summary to progress file
    with open(PROGRESS_FILE, "w") as f:
        f.write("COMPLETE!\n")
        f.write(f"Total reviews processed: {total_reviews}\n")
        f.write(f"Number of partitions: {partitions_count}\n")
        f.write(f"Reviews per partition: {total_reviews // partitions_count}\n")
        f.write(f"Total processing time: {process_duration:.1f} seconds\n")
        f.write(f"Average processing speed: {total_reviews/process_duration:.1f} reviews/second\n")
        f.write(f"Total time: {total_duration:.1f} seconds\n")

    logger.info(f"Done! Processed {total_reviews} reviews in {total_duration:.1f} seconds")

    # Save basic metrics to a simple JSON file
    metrics = {
        "total_reviews": total_reviews,
        "partitions_count": partitions_count,
        "reviews_per_partition": total_reviews // partitions_count,
        "processing_time_seconds": process_duration,
        "reviews_per_second": (total_reviews / process_duration if process_duration > 0 else 0),
        "total_time_seconds": total_duration,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open("/tmp/sentiment_analysis_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run sentiment analysis on Amazon reviews")

    # Dataset selection
    parser.add_argument(
        "--dataset",
        choices=[dataset.lower() for dataset in PREDEFINED_DATASETS],
        required=True,
        help="Predefined dataset to analyze",
    )

    # Additional options
    parser.add_argument(
        "--sample-ratio", type=float, default=1.0, help="Sample ratio (0.0-1.0) of the dataset to process"
    )
    parser.add_argument(
        "--output-dir", type=str, default="/analysis_outputs", help="HDFS output directory for analysis results"
    )
    parser.add_argument(
        "--summary-dir", type=str, default="/summary_outputs", help="HDFS output directory for summary results"
    )

    return parser.parse_args()


def main():
    """Main function to run the sentiment analysis"""
    # Parse command line arguments
    args = parse_arguments()
    overall_start_time = time.time()

    dataset_name = args.dataset.lower()
    # Transform dataset name to path format
    path_name = "_".join(word.capitalize() for word in dataset_name.split("_"))
    input_path = f"/{path_name}.jsonl"
    # Set output paths
    analysis_output_path, summary_output_path = args.output_dir, args.summary_dir

    # Create initial progress file
    update_progress("Starting up")
    logger.info(f"Progress can be monitored at: {PROGRESS_FILE}")

    # Initialize Spark session
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

    # Load the dataset
    update_progress("Loading data", start_time=overall_start_time)
    reviews_df = load_amazon_reviews(spark, input_path, args.sample_ratio)

    # Count total reviews
    total_reviews = reviews_df.count()
    logger.info(f"Starting to handle all {total_reviews:,} reviews")
    update_progress("Data loaded", 0, total_reviews, overall_start_time)

    # Load model and tokenizer
    update_progress("Loading model", 0, total_reviews, overall_start_time)
    tokenizer, model = load_model()

    # Broadcast model and tokenizer to all workers
    if process.bc_model is None:
        logger.info("Broadcasting model and tokenizer to all workers")
        process.bc_tokenizer = spark.sparkContext.broadcast(tokenizer)
        process.bc_model = spark.sparkContext.broadcast(model)

    # Repartition the DataFrame for optimal processing
    update_progress("Repartitioning data", 0, total_reviews, overall_start_time)
    target_partition_size_mb = 128
    reviews_df = preprocess.repartition_dataset(spark, input_path, reviews_df, target_partition_size_mb)

    # Preprocess the reviews
    update_progress("Preprocessing reviews", 0, total_reviews, overall_start_time)
    reviews_df = preprocess.preprocess_reviews(reviews_df, text_column="text", output_column="preprocessed_text")

    # Get current count after preprocessing
    total_reviews = reviews_df.count()

    # Process reviews for sentiment analysis
    logger.info("Starting to process reviews for sentiment analysis...")
    update_progress("Starting sentiment analysis", 0, total_reviews, overall_start_time)

    # Create a counter to track progress
    processed_count = spark.sparkContext.accumulator(0)

    # Apply tracking to the DataFrame using the tracker function
    batch_tracker = track_batch(processed_count)
    tracked_df = reviews_df.rdd.mapPartitionsWithIndex(batch_tracker).toDF(reviews_df.schema)

    # Start a background thread to monitor progress accumulator
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_progress_thread,
        args=(processed_count, total_reviews, overall_start_time, stop_monitor),
    )
    monitor_thread.daemon = True
    monitor_thread.start()

    # Process with tracking
    process_start_time = time.time()
    sentiment_analysis_results_df = process.process_reviews(
        reviews_df=tracked_df,
        output_path=analysis_output_path,
        review_text_column="preprocessed_text",
    )
    process_duration = time.time() - process_start_time
    logger.info(
        f"Sentiment analysis completed in {process_duration:.1f} seconds with average speed of {total_reviews / process_duration:.1f} reviews/second"
    )
    partitions_count = sentiment_analysis_results_df.rdd.getNumPartitions()
    # Stop the monitoring thread
    stop_monitor.set()
    monitor_thread.join(timeout=1.0)

    # Update final progress for sentiment analysis
    update_progress("Sentiment analysis complete", total_reviews, total_reviews, overall_start_time)

    # Combine results into a single csv file
    update_progress("Merging results", total_reviews, total_reviews, overall_start_time)
    postprocess.merge_results_csv_in_hdfs(analysis_output_path, summary_output_path, "sentiment_analysis_full_results")

    # Generate sentiment statistics
    update_progress("Generating statistics", total_reviews, total_reviews, overall_start_time)
    postprocess.generate_sentiment_statistics(sentiment_analysis_results_df, summary_output_path)

    # Clean up temporary files
    update_progress("Cleaning up", total_reviews, total_reviews, overall_start_time)

    # Add explicit unpersist for cached DataFrames
    reviews_df.unpersist()
    sentiment_analysis_results_df.unpersist()
    # Clear broadcasted variables when done
    process.bc_tokenizer.unpersist()
    process.bc_model.unpersist()

    # Calculate overall metrics
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time

    # Save final summary and metrics
    save_final_summary(total_reviews, total_duration, process_duration, partitions_count)

    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    sys.exit(main())
