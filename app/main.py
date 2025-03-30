from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from transformers import pipeline

# Initialize the FastAPI app
app = FastAPI(title="Sentiment Prediction API")

# Enable CORS middleware (adjust origins as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint: returns a welcome message
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Prediction API"}

# Pydantic model for the sentiment prediction request
class SentimentRequest(BaseModel):
    text: str

# Load the pre-trained sentiment analysis pipeline from Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis")

# POST endpoint for text sentiment prediction
@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text input is required")
    
    # Analyze the sentiment of the provided text
    results = sentiment_pipeline(request.text)
    prediction = results[0]
    label = prediction['label'].lower()
    score = prediction['score']
    
    # Basic threshold logic for neutrality
    sentiment = "neutral" if score < 0.6 else label

    return {
        "input_text": request.text,
        "predicted_sentiment": sentiment,
        "confidence": score
    }

# Function to fetch YouTube comments using the API key from environment variables
def fetch_youtube_comments(video_id: str):
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    if not YOUTUBE_API_KEY:
        print("Error: YouTube API key not set in environment variables.")
        return []
    
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "key": YOUTUBE_API_KEY,
        "textFormat": "plainText",
        "maxResults": 20  # Adjust this number as needed
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching comments: {response.status_code} {response.text}")
        return []
    
    data = response.json()
    comments = []
    for item in data.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
    
    return comments

# GET endpoint for analyzing YouTube comments sentiment
@app.get("/youtube/{video_id}")
async def analyze_youtube_comments(video_id: str):
    # Fetch comments for the given YouTube video ID
    comments = fetch_youtube_comments(video_id)
    if not comments:
        raise HTTPException(status_code=404, detail="No comments found or error fetching comments")
    
    # Analyze sentiment for each comment
    sentiment_results = [sentiment_pipeline(comment)[0] for comment in comments]
    return {
        "video_id": video_id,
        "analysis": sentiment_results
    }

# Run the FastAPI app with Uvicorn if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
