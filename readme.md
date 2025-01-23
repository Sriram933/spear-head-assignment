# FastAPI CSV Model Training and Prediction

This project provides a FastAPI application for uploading CSV files, training a model, and making predictions.

## Project Structure

## Setup

1. **Clone the repository:**

   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment:**

   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the FastAPI server:**

   ```sh
   uvicorn main:app --reload
   ```

2. **Access the API documentation:**
   Open your browser and navigate to `http://127.0.0.1:8000/docs` to view the interactive API documentation.

## API Endpoints

- **POST /upload**

  - Upload a CSV file for training.
  - Request: `multipart/form-data` with a file field named .
  - Response: HTTP 200 on success, HTTP 400 if the file is not a CSV.

- **POST /train**

  - Train the model using the uploaded CSV file.
  - Request: No parameters.
  - Response: Training status.

- **POST /predict**
  - Make predictions using the trained model.
  - Request: JSON object with input data.
  - Response: Prediction results.


