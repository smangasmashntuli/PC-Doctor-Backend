#youtube_service.py
import os
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from backend.app.models import VideoTutorial, LaptopSetup
from backend.app.database import SessionLocal

logger = logging.getLogger(__name__)


class YouTubeService:
    """Service for fetching and caching YouTube tutorial videos"""
    
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            logger.warning("YOUTUBE_API_KEY not found in environment variables. YouTube search will be unavailable.")
            self.enabled = False
        else:
            self.enabled = True
            try:
                self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize YouTube API client: {e}")
                self.enabled = False
    
    async def search_youtube_tutorials(
        self,
        brand: str,
        model: str,
        issue_keyword: str,
        user_id: int,
        db: SessionLocal
    ) -> List[Dict[str, Any]]:
        """
        Search for YouTube tutorials with caching (24-hour TTL)
        
        Args:
            brand: Laptop brand (e.g., "Dell", "Lenovo")
            model: Laptop model (e.g., "XPS 15 9520")
            issue_keyword: Issue description (e.g., "overheating", "won't boot")
            user_id: User ID for linking to laptop setup
            db: Database session
        
        Returns:
            List of formatted video tutorial cards
        """
        # Check if YouTube API is enabled
        if not self.enabled:
            logger.warning("YouTube API not enabled, returning empty results")
            return []
        
        # Check if user has a laptop setup
        laptop_setup = db.query(LaptopSetup).filter(
            LaptopSetup.user_id == user_id
        ).first()
        
        laptop_setup_id = laptop_setup.id if laptop_setup else None
        
        # Check cache first (24-hour TTL)
        cache_cutoff = datetime.utcnow() - timedelta(hours=24)
        cached_videos = db.query(VideoTutorial).filter(
            VideoTutorial.laptop_setup_id == laptop_setup_id,
            VideoTutorial.issue_keyword == issue_keyword,
            VideoTutorial.cached_at >= cache_cutoff
        ).limit(5).all()
        
        if cached_videos:
            logger.info(f"Returning {len(cached_videos)} cached YouTube videos for {brand} {model} - {issue_keyword}")
            return self._format_video_cards(cached_videos)
        
        # Cache miss - fetch from YouTube API
        if not self.enabled:
            logger.warning("YouTube API not enabled, returning empty results")
            return []
        
        try:
            videos = await self._fetch_from_youtube(brand, model, issue_keyword)
            
            # Cache the results
            if videos and laptop_setup_id:
                self._cache_videos(db, laptop_setup_id, videos, issue_keyword)
            
            return videos
            
        except Exception as e:
            logger.error(f"Error fetching YouTube videos: {e}")
            return []
    
    async def _fetch_from_youtube(
        self,
        brand: str,
        model: str,
        issue_keyword: str
    ) -> List[Dict[str, Any]]:
        """Fetch videos from YouTube Data API v3"""
        try:
            # Formulate search query
            query = f"{brand} {model} how to fix {issue_keyword}"
            
            # Call YouTube API
            search_response = self.youtube.search().list(
                q=query,
                part='id,snippet',
                maxResults=5,
                type='video',
                videoDefinition='high',
                relevanceLanguage='en',
                order='relevance'
            ).execute()
            
            videos = []
            for item in search_response.get('items', []):
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                description = item['snippet']['description']
                thumbnail_url = item['snippet']['thumbnails']['medium']['url']
                channel_title = item['snippet']['channelTitle']
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                videos.append({
                    'video_id': video_id,
                    'title': title,
                    'description': description[:200] + '...' if len(description) > 200 else description,
                    'thumbnail_url': thumbnail_url,
                    'channel_title': channel_title,
                    'url': video_url
                })
            
            logger.info(f"Fetched {len(videos)} videos from YouTube for query: {query}")
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching YouTube videos: {e}")
            return []
    
    def _cache_videos(
        self,
        db: SessionLocal,
        laptop_setup_id: int,
        videos: List[Dict[str, Any]],
        issue_keyword: str
    ):
        """Cache videos in database"""
        try:
            for video in videos:
                video_tutorial = VideoTutorial(
                    laptop_setup_id=laptop_setup_id,
                    video_id=video['video_id'],
                    title=video['title'],
                    description=video.get('description', ''),
                    thumbnail_url=video['thumbnail_url'],
                    video_url=video['url'],
                    issue_keyword=issue_keyword,
                    expires_at=datetime.utcnow() + timedelta(hours=24)
                )
                db.add(video_tutorial)
            
            db.commit()
            logger.info(f"Cached {len(videos)} videos for laptop_setup_id={laptop_setup_id}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error caching videos: {e}")
    
    def _format_video_cards(self, videos: List[VideoTutorial]) -> List[Dict[str, Any]]:
        """Format database records into mobile-friendly video cards"""
        return [
            {
                'video_id': video.video_id,
                'title': video.title,
                'description': video.description,
                'thumbnail_url': video.thumbnail_url,
                'channel_title': video.channel_title if hasattr(video, 'channel_title') else 'YouTube',
                'url': video.video_url,
                'cached_at': video.cached_at.isoformat()
            }
            for video in videos
        ]
    
    async def invalidate_cache(
        self,
        laptop_setup_id: int,
        issue_keyword: Optional[str] = None,
        db: Optional[SessionLocal] = None
    ):
        """
        Invalidate cached videos (e.g., when new videos are needed)
        
        Args:
            laptop_setup_id: Laptop setup ID
            issue_keyword: Optional specific issue to invalidate (None = all)
            db: Database session (creates new one if not provided)
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        
        try:
            query = db.query(VideoTutorial).filter(
                VideoTutorial.laptop_setup_id == laptop_setup_id
            )
            
            if issue_keyword:
                query = query.filter(VideoTutorial.issue_keyword == issue_keyword)
            
            deleted_count = query.delete(synchronize_session=False)
            db.commit()
            
            logger.info(f"Invalidated {deleted_count} cached videos for laptop_setup_id={laptop_setup_id}")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error invalidating cache: {e}")
        finally:
            if close_db:
                db.close()


# Global instance
_youtube_service_instance = None

def get_youtube_service() -> YouTubeService:
    """Get or create the global YouTube service instance"""
    global _youtube_service_instance
    if _youtube_service_instance is None:
        _youtube_service_instance = YouTubeService()
    return _youtube_service_instance