#!/bin/bash

# Sentiment Analysis Pipeline Wrapper
# This script allows users to run sentiment analysis on Amazon reviews datasets

# Display help information
function show_help {
    echo "Sentiment Analysis Pipeline"
    echo "============================"
    echo "Run sentiment analysis on Amazon reviews or similar datasets."
    echo
    echo "Usage:"
    echo "  ./run_sentiment_analysis.sh -d DATASET_NAME [options]"
    echo
    echo "Options:"
    echo "  -h, --help                 Show this help message"
    echo "  -d, --dataset NAME         Dataset name/identifier (REQUIRED)"
    echo "                             Predefined datasets: (Subscription_Boxes, Magazine_Subscriptions,"
    echo "                             All_beauty, Appliances, Arts_Crafts_and_Sewing)"
    echo "  -i, --input-file PATH      Path to input file (local path, optional)"
    echo "                             Only needed if dataset isn't predefined"
    echo "  -s, --sample RATIO         Sample ratio of data to process (0.0-1.0)"
    echo "                             Default: 1.0 (use all data)"
    echo "  -c, --sample-count COUNT   Sample count of data to process (integer)"
    echo "                             Default: use all data"
    echo
    echo "Examples:"
    echo "  ./run_sentiment_analysis.sh --dataset Subscription_Boxes"
    echo "  ./run_sentiment_analysis.sh --dataset my_custom_reviews --input-file ./my_reviews.jsonl"
    echo
}

# Default values
SAMPLE_RATIO=1.0
SAMPLE_COUNT=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -i|--input-file)
            LOCAL_INPUT_FILE="$2"
            shift 2
            ;;
        -s|--sample-ratio)
            SAMPLE_RATIO="$2"
            shift 2
            ;;
        -c|--sample-count)
            SAMPLE_COUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if dataset name is provided (required)
if [[ -z "$DATASET" ]]; then
    echo "Error: Dataset name is required. Use --dataset to specify it."
    echo "Use --help for usage information"
    exit 1
fi

# Check if the dataset already exists on HDFS
echo "Checking if dataset file is available..."
hdfs dfs -test -e "/$DATASET.jsonl"
HDFS_EXISTS=$?

# If dataset exists on HDFS, use it directly
if [[ $HDFS_EXISTS -ne 0 ]]; then
    
    echo "Dataset $DATASET is not predefined."
    
    # If input file is not provided, show error
    if [[ -z "$LOCAL_INPUT_FILE" ]]; then
        echo "Error: Dataset is not predefined and no input file provided."
        echo "Please provide an input file using --input-file or use a predefined dataset."
        echo "Predefined datasets are:"
        echo "Subscription_Boxes, Magazine_Subscriptions, All_beauty, Appliances, Arts_Crafts_and_Sewing"
        exit 1
    fi
    
    # Check if local input file exists
    if [[ ! -f "$LOCAL_INPUT_FILE" ]]; then
        echo "Error: Local input file not found: $LOCAL_INPUT_FILE"
        echo "Please provide a valid path to the input file."
        exit 1
    fi
    
    # Upload the local file to HDFS with the dataset name
    HDFS_PATH="/$DATASET.jsonl"
    echo "Uploading input file to HDFS as $HDFS_PATH..."
    hdfs dfs -put -f "$LOCAL_INPUT_FILE" "$HDFS_PATH"
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to upload file to HDFS"
        exit 1
    fi
    
    echo "File uploaded successfully to HDFS: $HDFS_PATH"
    

fi

echo "Running sentiment analysis job on dataset $DATASET..."
echo "You can monitor the job status using YARN dashboard: https://ucabhhk-yarn.comp0235.condenser.arc.ucl.ac.uk/cluster"

nohup spark-submit --deploy-mode cluster --master yarn spark_sentiment_analysis.py \
        --dataset $DATASET --sample-ratio $SAMPLE_RATIO --sample-count $SAMPLE_COUNT > job_output.log 2>&1 

echo "Job is done. The job output files are saved in HDFS under /summary_outputs/$DATASET"
echo "Downloading the output files from HDFS..."
hdfs dfs -get /summary_outputs/$DATASET summary_outputs_$DATASET
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to download output files from HDFS"
    exit 1
fi
echo "Output files downloaded successfully to summary_outputs_$DATASET in the current directory."
echo "You can also download the output files using curl as .tar.gz file:"
echo "curl 'https://ucabhhk-nginx.comp0235.condenser.arc.ucl.ac.uk/webhdfs/v1/summary_outputs_$DATASET.tar.gz?op=OPEN&user.name=almalinux' -o summary_outputs_$DATASET.tar.gz"
