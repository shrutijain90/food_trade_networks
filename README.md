# Global Cereal Flow Mapping at Subnational Scales

This repository contains the code for the manuscript: **"Mapping Global Cereal Flow at Subnational Scales Unveils Key Insights for Food Systems Resilience"**

The code implements a modeling pipeline to downscale national-level cereal production and trade data to estimate cereal flows between 3,536 subnational administrative regions across 195 countries. 

## System Requirements

* **Operating System:** Developed on macOS; adaptable to Linux.
* **RAM:** Minimum 16 GB recommended.
* **Processor:** Single core used for development.
* **Disk Space:** Minimum 50 GB free space recommended for data and outputs.
* **Python:** Python 3.7 or above 

## Installation

1.  **Clone the Repository:**

2.  **Set up Conda Environment:**
    * Create and activate the environment:
        ```
        conda env create -f environment.yml
        conda activate food_trade # Or the name specified in your environment.yml
        ```

3.  **Google Earth Engine (GEE) Python API Setup:**
    Useful for handling geospatial datasets.
    * Install: `pip install earthengine-api --upgrade`
    * Authenticate: `earthengine authenticate`
    * Initialize in scripts: `import ee; ee.Initialize()`

## Usage Workflow

The notebook `demo.ipynb` walks through the usage workflow. On a normal desktop computer, all but the third part, i.e. harmonization, should execute in a couple of minutes. Harmonization can take up to an hour.

1.  **Data Preparation:** Pre-process raw data and prepare features.
2.  **Machine Learning Model Training & Prediction:** Train classification and regression models, then predict raw subnational flows.
3.  **Harmonization:** Estimate subnational consumption and run the iterative harmonization algorithm.

To recreate the figures and tables presented in the manuscript, use the following notebook: `generate_results.ipynb`