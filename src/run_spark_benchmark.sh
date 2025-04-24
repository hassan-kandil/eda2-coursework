#!/bin/bash

# Simplified Spark configuration testing script
# Tests different executor configurations for sentiment analysis jobs

# Display help information
function show_help {
    echo "Spark Configuration Testing Script"
    echo "============================"
    echo "Run sentiment analysis jobs with different Spark configurations."
    echo
    echo "Usage:"
    echo "  ./run_spark_config_tests.sh -d DATASET_NAME -s SAMPLES [options]"
    echo
    echo "Required Arguments:"
    echo "  -d, --dataset NAME         Dataset name/identifier to analyze"
    echo "  -s, --samples NUMBER       Number of samples to process"
    echo
    echo "Optional Arguments:"
    echo "  -h, --help                 Show this help message"
    echo "  -o, --output-dir PATH      HDFS path for analysis outputs (default: /analysis_outputs_test)"
    echo "  -u, --summary-dir PATH     HDFS path for summary outputs (default: /summary_outputs_test)"
    echo
    echo "Examples:"
    echo "  ./run_spark_config_tests.sh --dataset Arts_Crafts_and_Sewing --samples 10000"
    echo "  ./run_spark_config_tests.sh -d Subscription_Boxes -s 5000 -o /my_analysis -u /my_summary"
    echo
}

# Default values
DATASET=""
SAMPLES=""
OUTPUT_DIR="/analysis_outputs_test"
SUMMARY_DIR="/summary_outputs_test"

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
        -s|--samples)
            SAMPLES="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -u|--summary-dir)
            SUMMARY_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [[ -z "$DATASET" || -z "$SAMPLES" ]]; then
    echo "Error: Dataset name and number of samples are required."
    echo "Use --help for usage information"
    exit 1
fi

# Function to run Spark job with specific configuration
run_spark_test() {
    local config_name=$1
    local num_executors=$2
    local executor_cores=$3
    local executor_memory=$4
    
    echo "=========================================================="
    echo "Starting test with configuration: $config_name"
    echo "  Executors: $num_executors"
    echo "  Cores per executor: $executor_cores"
    echo "  Memory per executor: $executor_memory"
    echo "=========================================================="
    
    # Calculate total cores and set default parallelism
    local total_cores=$((num_executors * executor_cores))
    
    # Build the command
    local cmd="spark-submit \
        --deploy-mode cluster \
        --master yarn \
        --num-executors $num_executors \
        --executor-cores $executor_cores \
        --executor-memory $executor_memory \
        --driver-memory 6G \
        --driver-cores 1 \
        --conf spark.default.parallelism=$total_cores \
        spark_sentiment_analysis.py \
        --dataset $DATASET \
        --samples $SAMPLES \
        --output-dir $OUTPUT_DIR/${config_name} \
        --summary-dir $SUMMARY_DIR/${config_name}"
    
    echo "Running command:"
    echo "$cmd"
    
    # Run the spark job
    nohup $cmd > ${config_name}_yarn.log 2>&1 &
    
    echo "Job submitted. Output is being logged to ${config_name}_yarn.log"
    echo "Waiting 10 seconds before submitting next job..."
    sleep 10
}

# Log script start
echo "Starting Spark configuration tests at $(date)"
echo "Dataset: $DATASET"
echo "Number of samples: $SAMPLES"

# Run tests for each configuration
echo "Will run 4 configuration tests:"
echo "1. 15 executors with 1 core 6G RAM per executor"
echo "2. 12 executors with 1 core 8G RAM per executor"
echo "3. 7 executors with 2 cores 12G RAM per executor"
echo "4. 4 executors with 3 cores 24G RAM per executor"

# Configuration 1: 15 executors with 1 core 6G RAM per executor
run_spark_test "config1_15exec_1core_6G" 15 1 "6G"

# Configuration 2: 12 executors with 1 core 8G RAM per executor
run_spark_test "config2_12exec_1core_8G" 12 1 "8G"

# Configuration 3: 7 executors with 2 cores 12G RAM per executor
run_spark_test "config3_7exec_2core_12G" 7 2 "12G"

# Configuration 4: 4 executors with 3 cores 24G RAM per executor
run_spark_test "config4_4exec_3core_24G" 4 3 "24G"

echo "All jobs have been submitted!"
echo "You can check the status using 'yarn application -list'"
echo "Logs are being saved to config*_yarn.log files"
