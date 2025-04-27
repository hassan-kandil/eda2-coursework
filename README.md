# EDA2 Coursework: Sentiment Analysis Distributed System

This repository contains the infrastructure and code to deploy a distributed sentiment analysis system using Hadoop, Spark, and various supporting services. The system is designed to classify text sentiment at scale. Follow the steps below for provisioning a new cluster, running the analysis jobs, downloading output files, and monitoring the cluster.

## Existing Cluster Specifications

The sentiment analysis system runs currently on a distributed cluster with the following hardware specifications:

| Machine ID | IP Address | CPU Cores | RAM | Storage |
|------------|------------|-----------|-----|---------|
| ucabhhk-host-01-6a3b7c6d69 | 10.134.12.236 | 2 | 4GB | 10GB |
| ucabhhk-worker-01-6a3b7c6d69 | 10.134.12.226 | 4 | 32GB | 50GB |
| ucabhhk-worker-02-6a3b7c6d69 | 10.134.12.78 | 4 | 32GB | 50GB |
| ucabhhk-worker-03-6a3b7c6d69 | 10.134.12.227 | 4 | 32GB | 50GB |
| ucabhhk-worker-04-6a3b7c6d69 | 10.134.12.200 | 4 | 32GB | 50GB |
| **Total Resources** | | **18 cores** | **132GB** | **210GB** |


## New Cluster Setup Instructions


### Prerequisites

- Terraform installed locally
- Access to Harvester cloud provider with appropriate credentials
- SSH access to the created VMs using lecturer private_key
- Ansible 2.9+ installed locally
- Update your SSH config file to include:
   ```plaintext
   StrictHostKeyChecking accept-new
   ```

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/eda2-coursework.git
cd eda2-coursework
```

### 2. Deploy the Infrastructure with Terraform

```bash
# Initialize Terraform
terraform init

# Apply the configuration to create VMs
terraform apply -auto-approve
```

This will create 1 host node and 4 worker nodes on the Harvester cloud.

### 3. Run the Ansible Playbook to Set Up the Cluster

```bash
# Run the full setup playbook
ansible-playbook -i generate_inventory.py full_setup.yaml --private-key=/path/to/lecturer_private_key
```

This playbook will:
1. Install basic tools on all nodes
2. Generate and share SSH keys for passwordless access
3. Install and configure Hadoop on all nodes
4. Set up Nginx on the host node
5. Install and configure Spark on all nodes
6. Set up monitoring with Prometheus and Grafana
7. Install the sentiment analysis pipeline


## Running Sentiment Analysis Jobs

### Using Predefined Datasets

The system comes pre-loaded with several datasets extracted from the [Amazon Reviews 2023 dataset](https://amazon-reviews-2023.github.io/). Each dataset contains customer reviews from a specific product category, ranging from small to large-scale datasets for testing different workloads.

```bash
# SSH to the host node
ssh -i /path/to/lecturer_private_key -J condenser-proxy almalinux@10.134.12.236

# Run analysis on a predefined dataset
cd ~/sentiment_analysis
./run_sentiment_analysis.sh --dataset [DATASET_NAME]
```

Available predefined datasets:

| Dataset Name | Size | Review Count | Complexity |
|--------------|------|--------------|------------|
| Subscription_Boxes | 8.5 MB | 16,200 | Small (Quick Test) |
| Magazine_Subscriptions | 31.8 MB | 71,500 | Medium-Small |
| All_Beauty | 311.5 MB | 701,500 | Medium |
| Appliances | 886.4 MB | 2.1 million | Large |
| Arts_Crafts_and_Sewing | 3.7 GB | 9.0 million | Very Large |

For quick testing and validation, start with smaller datasets like Subscription_Boxes. For performance benchmarking or large-scale analysis, use the larger datasets.

### Using Custom Datasets

The system allows you to upload and analyze your own custom datasets. Custom datasets should be formatted as `.jsonl` files (JSON Lines), where each line contains a valid JSON object representing a single review.

#### Required Schema

Each review in your dataset should follow this schema:

| Field | Description | Required |
|-------|-------------|----------|
| `text` | The review text to analyze for sentiment | Required |
| `asin` | Amazon Standard Identification Number (product ID) | Optional |
| `user_id` | Identifier for the reviewer | Optional |
| `rating` | Numerical rating (typically 1-5 stars) | Optional |
| `title` | Title or headline of the review | Optional |

#### Example JSON Records
```json
{"text": "This product exceeded all my expectations!", "asin": "B07X123YZ", "user_id": "A2SUAM1J3GNN3B", "rating": 5, "title": "Fantastic purchase"}
{"text": "Disappointed with the quality. Not worth the price.", "asin": "B08Y456AB", "user_id": "B1CDE2F3GHIJ4K", "rating": 2, "title": "Not as advertised"}
{"text": "Average product. Does the job but nothing special.", "asin": "C09Z789XY", "user_id": "C5LMNO6P7QRS8T", "rating": 3, "title": "It's okay"}
```

#### Upload and Analysis Process

```bash
# Step 1: Upload your custom dataset to the host node
scp -i /path/to/lecturer_private_key -J condenser-proxy /path/to/your_local_dataset.jsonl almalinux@10.134.12.236:~/

# Step 2: SSH to the host node
ssh -i /path/to/lecturer_private_key -J condenser-proxy almalinux@10.134.12.236

# Step 3: Run analysis using your custom dataset
cd ~/sentiment_analysis
./run_sentiment_analysis.sh --dataset [YOUR_CUSTOM_DATASET_NAME] --input-file ~/your_local_dataset.jsonl
```

### Additional Options

You can control the amount of data processed:

```bash
# Process only 10% of the dataset
./run_sentiment_analysis.sh --dataset [DATASET_NAME] --sample-ratio 0.1  # Replace [DATASET_NAME] with your dataset

# Process only 1000 records (picked randomly)
./run_sentiment_analysis.sh --dataset [DATASET_NAME] --sample-count 1000  # Replace [DATASET_NAME] with your dataset
```

## Monitoring the Analysis

Access the following dashboards to monitor the analysis:

1. **HDFS Dashboard:** [https://ucabhhk-hadoop.comp0235.condenser.arc.ucl.ac.uk/](https://ucabhhk-hadoop.comp0235.condenser.arc.ucl.ac.uk/)
2. **YARN Dashboard:** [https://ucabhhk-yarn.comp0235.condenser.arc.ucl.ac.uk/](https://ucabhhk-yarn.comp0235.condenser.arc.ucl.ac.uk/)
3. **Prometheus Dashboard:** [https://ucabhhk-prometheus.comp0235.condenser.arc.ucl.ac.uk/](https://ucabhhk-prometheus.comp0235.condenser.arc.ucl.ac.uk/)
4. **Grafana Dashboard:** [https://ucabhhk-grafana.comp0235.condenser.arc.ucl.ac.uk/](https://ucabhhk-grafana.comp0235.condenser.arc.ucl.ac.uk/)


## Accessing Results

### Directly from HDFS

```bash
# SSH to the host node
ssh -i /path/to/lecturer_private_key -J condenser-proxy almalinux@10.134.12.236

# List available analysis outputs
hdfs dfs -ls /summary_outputs/

# Download results for a specific dataset
hdfs dfs -get /summary_outputs/[DATASET_NAME]  # Replace [DATASET_NAME] with your dataset
```

### Via HTTP

Results can be downloaded as a compressed archive:

```bash
# From your local machine
curl "https://ucabhhk-nginx.comp0235.condenser.arc.ucl.ac.uk/webhdfs/v1/summary_outputs_[DATASET_NAME].tar.gz?op=OPEN&user.name=almalinux" -o summary_outputs_[DATASET_NAME].tar.gz # Replace [DATASET_NAME] with your dataset
```
## Output Files

The sentiment analysis system generates several output files that provide detailed insights into the analyzed reviews:

### CSV Files

1. **sentiment_analysis_full_results.csv**
   - Contains the complete analysis results for each review
2. **sentiment_distribution.csv**
   - Summarizes the distribution of sentiments across all reviews
3. **sentiment_overall_stats.csv**
   - Provides overall statistics about the sentiment analysis
4. **sentiment_token_stats.csv**
   - Statistics about tokenization of reviews
5. **summary_metrics.csv**
   - Performance metrics of the sentiment analysis job

## Cleaning Up

To destroy the infrastructure when you're done:

```bash
terraform destroy -auto-approve
```

## Directory Structure

- `src/` - Source code for the sentiment analysis pipeline
- `roles/` - Ansible roles for setting up different components
- `templates/` - Templates for configuration files
- `*.tf` - Terraform files for infrastructure setup

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


