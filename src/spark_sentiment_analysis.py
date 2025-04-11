import sys
import time


import sentiment_analysis.process as process
import sentiment_analysis.preprocess as preprocess
import sentiment_analysis.postprocess as postprocess

from pyspark.sql import SparkSession
from sentiment_analysis.config import logger
from sentiment_analysis.load import load_amazon_reviews, load_model


def main():
    input_path, analysis_output_path, summary_output_path = (
        "/Subscription_Boxes.jsonl",
        "/analysis_outputs",
        "/summary_outputs",
    )
    # Initialize Spark session
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    # Load the dataset
    reviews_df = load_amazon_reviews(spark, input_path)
    # Count total reviews
    total_reviews = reviews_df.count()
    logger.info(f"Starting to handle all {total_reviews:,} reviews")
    # Load model and tokenizer
    tokenizer, model = load_model()
    # Broadcast model and tokenizer to all workers
    if process.bc_model is None:
        logger.info("Broadcasting model and tokenizer to all workers")
        process.bc_tokenizer = spark.sparkContext.broadcast(tokenizer)
        process.bc_model = spark.sparkContext.broadcast(model)

    # Repartition the DataFrame for optimal processing
    target_partition_size_mb = 128
    reviews_df = preprocess.repartition_dataset(
        spark, input_path, reviews_df, target_partition_size_mb
    )
    # Preprocess the reviews
    logger.info("Starting to preprocess reviews...")
    start_time = time.time()
    reviews_df = preprocess.preprocess_reviews(
        reviews_df, text_column="text", output_column="preprocessed_text"
    )
    end_time = time.time()
    logger.info(
        f"Done preprocessing all reviews in {end_time - start_time:.2f} seconds"
    )
    # Process reviews for sentiment analysis
    logger.info("Starting to process reviews for sentiment analysis...")
    start_time = time.time()
    sentiment_analysis_results_df = process.process_reviews(
        reviews_df=reviews_df,
        output_path=analysis_output_path,
        review_text_column="preprocessed_text",
    )
    end_time = time.time()
    logger.info(f"Done processing all reviews in {end_time - start_time:.2f} seconds")
    # Combine results into a single csv file
    logger.info("Merging results into a single CSV file...")
    merge_start_time = time.time()
    postprocess.merge_results_csv_in_hdfs(
        analysis_output_path, summary_output_path, "sentiment_analysis_full_results"
    )
    merge_end_time = time.time()
    logger.info(
        f"Done merging results in {merge_end_time - merge_start_time:.2f} seconds"
    )
    # Generate sentiment statistics
    logger.info("Generating sentiment statistics...")
    generate_stats_start_time = time.time()
    postprocess.generate_sentiment_statistics(
        sentiment_analysis_results_df, summary_output_path
    )
    generate_stats_end_time = time.time()
    logger.info(
        f"Done generating sentiment statistics in {generate_stats_end_time - generate_stats_start_time:.2f} seconds"
    )
    # Clean up temporary files
    logger.info("Cleaning up temporary files...")
    # Add explicit unpersist for cached DataFrames
    reviews_df.unpersist()
    sentiment_analysis_results_df.unpersist()
    # Clear broadcasted variables when done
    process.bc_tokenizer.unpersist()
    process.bc_model.unpersist()
    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    sys.exit(main())
