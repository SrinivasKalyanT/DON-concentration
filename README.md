## 1)Setup Instructions

# Cloning the repository
git clone https://github.com/SrinivasKalyanT/DON-concentration.git
cd DON-concentration.git

# Create a Virtual Environment

# Install Dependencies
pip install -r requirements.txt

# DON-concentration.git/
│── data/                 # csv file
│── models/               # Trained models (saved PCA, RF, XGB, NN)
│── pipeline.ipynb/       # Jupyter notebooks for analysis
│── app.py                # FastAPI service for predictions
│── requirements.txt      # Python dependencies
│── Dockerfile            # Deployment setup
│── README.md             # Project documentation


# Run the pipeline.ipynb file

## Run API Service (FastAPI)
uvicorn src.api:app --host 0.0.0.0 --port 8000

## Then, open http://0.0.0.0:8000/docs in your browser to test the API.

## Deployment with Docker
# Build the Docker Image
  docker build -t don-concentration-predictor .
# Run the Container
  docker run -p 8000:8000 don-concentration-predictor
