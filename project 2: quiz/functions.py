import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import ollama
import time
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec


class YoutubeProcessor:
    def __init__(self, video_url: str):
        """
        Initialize the processor with the YouTube video URL.
        """
        self.video_url = video_url
        self.video_id = self.extract_video_id()

    def extract_video_id(self) -> str:
        """
        Extract the video ID from the YouTube URL.
        """
        if "v=" in self.video_url:
            return self.video_url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in self.video_url:
            return self.video_url.split("youtu.be/")[-1].split("?")[0]
        else:
            raise ValueError("Invalid YouTube URL format.")

    def extract_title(self) -> str:
        """
        Extract the title from the YouTube video.
        """
        ydl_opts = {"quiet": True}  # Suppress unnecessary output

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(self.video_url, download=False)
                return info_dict.get("title", "Title not found")
        except yt_dlp.utils.DownloadError as e:
            print(f"Error extracting title: {e}")
            return "Error extracting title"

    def extract_transcript(self) -> str:
        """
        Extract the transcript from the YouTube video.
        """
        try:
            transcript = YouTubeTranscriptApi.get_transcript(self.video_id)
            return " ".join([entry["text"] for entry in transcript])
        except Exception as e:
            print(f"Error extracting transcript: {e}")
            return "Transcript not available"

    def generate_summary(self):
        """
        Generate a summary using Llama from the video title and transcript.
        """
        start_time = time.time()
        try:
            title = self.extract_title()
            transcript = self.extract_transcript()

            prompt = (
                f"You are an expert learner. You have just watched a YouTube video titled '{title}'. "
                f"The transcript of the video is as follows: {transcript}. "
                f"Please generate summary paragraphs that retains enough detail about the key concepts covered."
            )

            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
            )
            summary_runtime = time.time() - start_time
            return (
                response.get("message", {}).get("content", "Error generating summary"),
                summary_runtime,
                title,
            )
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating summary"


class PineconeDB:
    def __init__(self, index_name: str, dimension: int):
        """
        Initialize the Pinecone vector database
        """
        self.index_name = index_name
        self.dimension = dimension

    def create_index(self):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )  # dimension matches embedding model

        try:
            # Get index details to confirm it's available and accessible
            index_description = pc.describe_index(self.index_name)
            print(
                f"Index '{self.index_name}' is available. Details: {index_description}"
            )
        except Exception as e:
            print(f"Error accessing the index '{self.index_name}': {str(e)}")
        return pc.Index(self.index_name)
