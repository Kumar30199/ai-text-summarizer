# AI Text Summarizer Backend

A local, privacy-focused AI text summarizer backend using Python, FastAPI, and Hugging Face Transformers.

## Setup

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the App

1.  **Start the server:**
    ```bash
    uvicorn app.main:app --reload
    ```
    The app will download the `facebook/bart-large-cnn` model on the first run.

2.  **Health Check:**
    Open `http://127.0.0.1:8000/health` in your browser.

## Usage

**Endpoint:** `POST /api/v1/summarize`

**Example Request:**

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/summarize" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Your long text goes here...",
           "length": "medium"
         }'
```

**Length Options:**
*   `short`: 40-80 tokens
*   `medium`: 80-150 tokens
*   `long`: 150-250 tokens
