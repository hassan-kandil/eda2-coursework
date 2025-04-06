import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sentiment_analysis_pipeline.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("sentiment_analysis_pipeline_logger")
