import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import ollama


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

    def generate_summary(self, title: str, transcript: str) -> str:
        """
        Generate a summary using Llama from the video title and transcript.
        """
        if not transcript:
            return "No transcript available to summarize."

        prompt = (
            f"You are an expert learner. You have just watched a YouTube video titled '{title}'. "
            f"The transcript of the video is as follows: {transcript}. "
            f"Please generate a concise summary in paragraph form that highlights the key concepts covered."
        )

        try:
            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.get("message", {}).get(
                "content", "Error generating summary"
            )
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating summary"
