# Question Answering API with Fine-Tuned Model

This project provides a FastAPI-based question answering service using a fine-tuned DistilBERT model. The model is trained on the SQuAD v2.0 dataset and can answer questions based on provided context.

## Features

- **Automatic Model Training**: The model is automatically trained on startup if no pre-trained checkpoint exists.
- **REST API Endpoint**: `/qa` endpoint accepts a question and context, returning the best matching answer span.
- **Docker Support**: Fully containerized with `docker-compose.yml` for easy deployment.

## Project Structure

```
.
├── app.py              # Main FastAPI application
├── trainer.py          # Model training logic
├── docker-compose.yml  # Docker orchestration
├── Dockerfile          # Container build instructions
├── requirements.txt    # Python dependencies
└── models/             # Directory for storing trained model checkpoints (persisted via volume)
```

## Setup & Running

1. **Prerequisites**:
   - Docker and Docker Compose installed

2. **Build and Run**:
   ```bash
   docker-compose up --build
   ```

3. **API Usage**:
   Send a POST request to `http://localhost:8000/qa` with JSON body:
   ```json
   {
     "question": "What is the capital of France?",
     "context": "Paris is the capital and most populous city of France."
   }
   ```

## Notes

- The model will be automatically trained on first run using a small subset of SQuAD v2.0.
- Trained models are saved in the `models/` directory and persisted across container restarts.
- Replace the dummy dataset loader in `trainer.py` with your own PDF-to-dataset pipeline for real-world usage.

## Requirements

- Python 3.10
- Docker & Docker Compose
- GPU support (optional, for faster training)