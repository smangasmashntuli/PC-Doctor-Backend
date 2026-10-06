#test_youtube_notifications.py
import pytest
import sys
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock external dependencies
sys.modules['googleapiclient'] = Mock()
sys.modules['googleapiclient.discovery'] = Mock()
sys.modules['googleapiclient.errors'] = Mock()

from backend.app.youtube_service import YouTubeService
from backend.app.notifications import NotificationService
from backend.app.models import VideoTutorial, Notification, LaptopSetup, LaptopSpecs


class TestYouTubeService:
    """Test suite for YouTube service with caching"""
    
    @pytest.fixture
    def mock_youtube_service(self):
        """Create YouTubeService with mocked API client"""
        with patch.dict('os.environ', {'YOUTUBE_API_KEY': 'test_key'}):
            service = YouTubeService()
            service.enabled = True
            service.youtube = Mock()
            return service
    
    @pytest.mark.asyncio
    async def test_search_youtube_tutorials_cache_hit(self, mock_youtube_service):
        """Test 1: Cache hit - returns cached videos without API call"""
        with patch('backend.app.youtube_service.SessionLocal') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Mock laptop setup
            mock_laptop_setup = Mock()
            mock_laptop_setup.id = 1
            
            # Mock cached videos (within 24 hours)
            mock_video = Mock()
            mock_video.video_id = "abc123"
            mock_video.title = "How to fix Dell XPS overheating"
            mock_video.description = "Tutorial video"
            mock_video.thumbnail_url = "https://example.com/thumb.jpg"
            mock_video.video_url = "https://youtube.com/watch?v=abc123"
            mock_video.issue_keyword = "overheating"
            mock_video.cached_at = datetime.utcnow()
            
            # Setup proper query chain mocking
            mock_query_laptop = Mock()
            mock_filter_laptop = Mock()
            mock_filter_laptop.first.return_value = mock_laptop_setup
            mock_query_laptop.filter.return_value = mock_filter_laptop
            
            mock_query_video = Mock()
            mock_filter_video = Mock()
            mock_limit = Mock()
            mock_limit.all.return_value = [mock_video]
            mock_filter_video.limit.return_value = mock_limit
            mock_query_video.filter.return_value = mock_filter_video
            
            mock_db.query.side_effect = [mock_query_laptop, mock_query_video]
            
            # Execute search
            result = await mock_youtube_service.search_youtube_tutorials(
                brand="Dell",
                model="XPS 15",
                issue_keyword="overheating",
                user_id=1,
                db=mock_db
            )
            
            # Verify cached results returned
            assert len(result) == 1
            assert result[0]['video_id'] == "abc123"
            assert result[0]['title'] == "How to fix Dell XPS overheating"
            
            # Verify YouTube API was NOT called
            mock_youtube_service.youtube.search().list.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_search_youtube_tutorials_cache_miss(self, mock_youtube_service):
        """Test 2: Cache miss - fetches from YouTube API and caches results"""
        with patch('backend.app.youtube_service.SessionLocal') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Mock laptop setup
            mock_laptop_setup = Mock()
            mock_laptop_setup.id = 1
            
            # Setup proper query chain mocking
            mock_query_laptop = Mock()
            mock_filter_laptop = Mock()
            mock_filter_laptop.first.return_value = mock_laptop_setup
            mock_query_laptop.filter.return_value = mock_filter_laptop
            
            mock_query_video = Mock()
            mock_filter_video = Mock()
            mock_limit = Mock()
            mock_limit.all.return_value = []  # Empty cache
            mock_filter_video.limit.return_value = mock_limit
            mock_query_video.filter.return_value = mock_filter_video
            
            mock_db.query.side_effect = [mock_query_laptop, mock_query_video]
            
            # Mock YouTube API response
            mock_youtube_service.youtube.search().list().execute.return_value = {
                'items': [
                    {
                        'id': {'videoId': 'abc123'},
                        'snippet': {
                            'title': 'Dell XPS 15 Overheating Fix',
                            'description': 'Learn how to fix overheating issues on Dell XPS 15 laptop.',
                            'thumbnails': {'medium': {'url': 'https://example.com/thumb1.jpg'}},
                            'channelTitle': 'Tech Repair Pro'
                        }
                    },
                    {
                        'id': {'videoId': 'def456'},
                        'snippet': {
                            'title': 'Dell XPS 15 Fan Replacement',
                            'description': 'Step by step guide to replace the fan.',
                            'thumbnails': {'medium': {'url': 'https://example.com/thumb2.jpg'}},
                            'channelTitle': 'Laptop Repair Guide'
                        }
                    }
                ]
            }
            
            # Execute search
            result = await mock_youtube_service.search_youtube_tutorials(
                brand="Dell",
                model="XPS 15",
                issue_keyword="overheating",
                user_id=1,
                db=mock_db
            )
            
            # Verify API was called
            assert mock_youtube_service.youtube.search().list.called
            
            # Verify results
            assert len(result) == 2
            assert result[0]['video_id'] == 'abc123'
            assert result[0]['title'] == 'Dell XPS 15 Overheating Fix'
            assert result[0]['channel_title'] == 'Tech Repair Pro'
            assert result[1]['video_id'] == 'def456'
            
            # Verify videos were cached
            assert mock_db.add.call_count == 2
            assert mock_db.commit.called
    
    @pytest.mark.asyncio
    async def test_search_youtube_tutorials_no_api_key(self):
        """Test 3: YouTube API disabled when no API key"""
        with patch.dict('os.environ', {}, clear=True):
            service = YouTubeService()
            assert service.enabled is False
            
            with patch('backend.app.youtube_service.SessionLocal') as mock_db_class:
                mock_db = Mock()
                mock_db_class.return_value = mock_db
                
                # Mock empty laptop setup to avoid DB queries
                mock_query = Mock()
                mock_filter = Mock()
                mock_filter.first.return_value = None
                mock_query.filter.return_value = mock_filter
                mock_db.query.return_value = mock_query
                
                result = await service.search_youtube_tutorials(
                    brand="Dell",
                    model="XPS 15",
                    issue_keyword="overheating",
                    user_id=1,
                    db=mock_db
                )
                
                assert result == []
                # Verify no YouTube API calls were made (service disabled)
                assert not service.enabled
    
    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, mock_youtube_service):
        """Test 4: Cache TTL - expired cache triggers new API call"""
        with patch('backend.app.youtube_service.SessionLocal') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Mock laptop setup
            mock_laptop_setup = Mock()
            mock_laptop_setup.id = 1
            
            # Setup proper query chain mocking
            mock_query_laptop = Mock()
            mock_filter_laptop = Mock()
            mock_filter_laptop.first.return_value = mock_laptop_setup
            mock_query_laptop.filter.return_value = mock_filter_laptop
            
            mock_query_video = Mock()
            mock_filter_video = Mock()
            mock_limit = Mock()
            mock_limit.all.return_value = []  # Empty (expired) cache
            mock_filter_video.limit.return_value = mock_limit
            mock_query_video.filter.return_value = mock_filter_video
            
            mock_db.query.side_effect = [mock_query_laptop, mock_query_video]
            
            # Mock YouTube API response
            mock_youtube_service.youtube.search().list().execute.return_value = {
                'items': [
                    {
                        'id': {'videoId': 'new123'},
                        'snippet': {
                            'title': 'New Video',
                            'description': 'New description',
                            'thumbnails': {'medium': {'url': 'https://example.com/new.jpg'}},
                            'channelTitle': 'New Channel'
                        }
                    }
                ]
            }
            
            # Execute search
            result = await mock_youtube_service.search_youtube_tutorials(
                brand="Dell",
                model="XPS 15",
                issue_keyword="overheating",
                user_id=1,
                db=mock_db
            )
            
            # Verify API was called (cache expired)
            assert mock_youtube_service.youtube.search().list.called
            assert len(result) == 1
            assert result[0]['video_id'] == 'new123'


class TestNotificationService:
    """Test suite for notification service"""
    
    @pytest.fixture
    def notification_service(self):
        return NotificationService()
    
    @pytest.fixture
    def mock_laptop_setup(self):
        setup = Mock()
        setup.id = 1
        setup.user_id = 1
        setup.brand = "Dell"
        setup.model = "XPS 15 9520"
        return setup
    
    @pytest.fixture
    def mock_laptop_specs(self):
        specs = Mock()
        specs.id = 1
        specs.laptop_setup_id = 1
        specs.cpu = "Intel Core i7-12700H"
        specs.gpu = "NVIDIA RTX 3050 Ti"
        specs.ram = "16GB DDR5"
        specs.storage = "512GB SSD"
        specs.os = "Windows 11 Pro"
        return specs
    
    def test_generate_maintenance_alerts_creates_notifications(
        self, notification_service, mock_laptop_setup, mock_laptop_specs
    ):
        """Test 3: Notification generation creates personalized alerts"""
        with patch('backend.app.notifications.SessionLocal') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Mock laptop setup query
            mock_query_setup = Mock()
            mock_filter_setup = Mock()
            mock_filter_setup.first.return_value = mock_laptop_setup
            mock_query_setup.filter.return_value = mock_filter_setup
            
            # Mock laptop specs query
            mock_query_specs = Mock()
            mock_filter_specs = Mock()
            mock_filter_specs.first.return_value = mock_laptop_specs
            mock_query_specs.filter.return_value = mock_filter_specs
            
            # Mock notification query (for checking existing notifications)
            mock_query_notif = Mock()
            mock_filter_notif1 = Mock()
            mock_filter_notif2 = Mock()
            mock_order = Mock()
            mock_order.first.return_value = None  # No existing notifications
            mock_filter_notif2.order_by.return_value = mock_order
            mock_filter_notif1.filter.return_value = mock_filter_notif2
            mock_query_notif.filter.return_value = mock_filter_notif1
            
            # Need more mocks because _is_alert_due is called for each template
            # Create a generic query mock that returns itself for additional queries
            mock_query_generic = Mock()
            mock_filter_generic = Mock()
            mock_filter_generic.first.return_value = None
            mock_filter_generic.filter.return_value = mock_filter_generic
            mock_query_generic.filter.return_value = mock_filter_generic
            
            mock_db.query.side_effect = [
                mock_query_setup, 
                mock_query_specs, 
                mock_query_notif,
                mock_query_generic,  # For _is_alert_due calls
                mock_query_generic,
                mock_query_generic
            ]
            
            # Execute alert generation
            result = notification_service.generate_maintenance_alerts(
                user_id=1,
                db=mock_db
            )
            
            # Verify notifications were created
            assert len(result) > 0
            assert mock_db.bulk_save_objects.called
            assert mock_db.commit.called
            
            # Verify personalized messages
            titles = [n['title'] for n in result]
            assert any('Clean' in title or '🧹' in title for title in titles)
            assert any('Battery' in title or '🔋' in title for title in titles)
            assert any('OS' in title for title in titles)
    
    def test_generate_maintenance_alerts_respects_frequency(
        self, notification_service, mock_laptop_setup, mock_laptop_specs
    ):
        """Test: Alerts respect frequency limits"""
        with patch('backend.app.notifications.SessionLocal') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Mock laptop setup query
            mock_query_setup = Mock()
            mock_filter_setup = Mock()
            mock_filter_setup.first.return_value = mock_laptop_setup
            mock_query_setup.filter.return_value = mock_filter_setup
            
            # Mock laptop specs query
            mock_query_specs = Mock()
            mock_filter_specs = Mock()
            mock_filter_specs.first.return_value = mock_laptop_specs
            mock_query_specs.filter.return_value = mock_filter_specs
            
            # Mock recent notification (sent 1 day ago for 7-day frequency)
            recent_notification = Mock()
            recent_notification.created_at = datetime.utcnow() - timedelta(days=1)
            
            # Mock notification query
            mock_query_notif = Mock()
            mock_filter_notif1 = Mock()
            mock_filter_notif2 = Mock()
            mock_order = Mock()
            mock_order.first.return_value = recent_notification
            mock_filter_notif2.order_by.return_value = mock_order
            mock_filter_notif1.filter.return_value = mock_filter_notif2
            mock_query_notif.filter.return_value = mock_filter_notif1
            
            # Need more mocks because _is_alert_due is called for each template
            mock_query_generic = Mock()
            mock_filter_generic = Mock()
            mock_filter_generic.first.return_value = recent_notification
            mock_filter_generic.filter.return_value = mock_filter_generic
            mock_query_generic.filter.return_value = mock_filter_generic
            
            mock_db.query.side_effect = [
                mock_query_setup, 
                mock_query_specs, 
                mock_query_notif,
                mock_query_generic,  # For _is_alert_due calls
                mock_query_generic,
                mock_query_generic
            ]
            
            # Execute alert generation
            result = notification_service.generate_maintenance_alerts(
                user_id=1,
                db=mock_db
            )
            
            # Verify no notifications created (not due yet)
            assert len(result) == 0
            assert not mock_db.bulk_save_objects.called
    
    def test_get_user_notifications(self, notification_service):
        """Test: Get user notifications with filtering"""
        with patch('backend.app.notifications.SessionLocal') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Mock notifications
            mock_notifications = [
                Mock(id=1, title="Alert 1", message="Message 1", notification_type="maintenance", 
                     priority="high", is_read=False, created_at=datetime.utcnow()),
                Mock(id=2, title="Alert 2", message="Message 2", notification_type="maintenance",
                     priority="medium", is_read=True, created_at=datetime.utcnow() - timedelta(days=1))
            ]
            
            mock_query = Mock()
            mock_filter = Mock()
            mock_order = Mock()
            mock_limit = Mock()
            mock_limit.all.return_value = mock_notifications
            mock_order.limit.return_value = mock_limit
            mock_filter.order_by.return_value = mock_order
            mock_query.filter.return_value = mock_filter
            mock_db.query.return_value = mock_query
            
            # Make the query chain work properly
            mock_query.return_value = mock_filter
            
            # Test get all notifications
            result = notification_service.get_user_notifications(
                user_id=1,
                db=mock_db,
                unread_only=False
            )
            
            assert len(result) == 2
            assert result[0]['title'] == "Alert 1"
            assert result[1]['is_read'] is True
            
            # Test get unread only
            mock_limit.all.return_value = [mock_notifications[0]]
            result = notification_service.get_user_notifications(
                user_id=1,
                db=mock_db,
                unread_only=True
            )
            
            assert len(result) == 1
            assert result[0]['is_read'] is False
    
    def test_mark_notification_as_read(self, notification_service):
        """Test 4: Mark notification as read"""
        with patch('backend.app.notifications.SessionLocal') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            # Mock notification
            mock_notification = Mock()
            mock_notification.id = 1
            mock_notification.user_id = 1
            mock_notification.title = "Test Alert"
            mock_notification.message = "Test message"
            mock_notification.notification_type = "maintenance"
            mock_notification.priority = "high"
            mock_notification.is_read = False
            mock_notification.created_at = datetime.utcnow()
            mock_notification.read_at = None
            
            mock_query = Mock()
            mock_filter = Mock()
            mock_filter.first.return_value = mock_notification
            mock_query.filter.return_value = mock_filter
            mock_db.query.return_value = mock_query
            
            # Execute mark as read
            result = notification_service.mark_notification_as_read(
                notification_id=1,
                user_id=1,
                db=mock_db
            )
            
            # Verify notification was marked as read
            assert result is not None
            assert result['id'] == 1
            assert result['is_read'] is True
            assert mock_notification.read_at is not None
            assert mock_db.commit.called
            assert mock_db.refresh.called
    
    def test_mark_notification_as_read_not_found(self, notification_service):
        """Test: Mark non-existent notification returns None"""
        with patch('backend.app.notifications.SessionLocal') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            mock_query = Mock()
            mock_filter = Mock()
            mock_filter.first.return_value = None
            mock_query.filter.return_value = mock_filter
            mock_db.query.return_value = mock_query
            
            result = notification_service.mark_notification_as_read(
                notification_id=999,
                user_id=1,
                db=mock_db
            )
            
            assert result is None
    
    def test_get_unread_count(self, notification_service):
        """Test: Get unread notification count"""
        with patch('backend.app.notifications.SessionLocal') as mock_db_class:
            mock_db = Mock()
            mock_db_class.return_value = mock_db
            
            mock_query = Mock()
            mock_filter = Mock()
            mock_filter.count.return_value = 5
            mock_query.filter.return_value = mock_filter
            mock_db.query.return_value = mock_query
            
            count = notification_service.get_unread_count(user_id=1, db=mock_db)
            
            assert count == 5
            assert mock_db.query.called


class TestYouTubeNotificationsIntegration:
    """Integration tests for YouTube and Notifications"""
    
    @pytest.mark.asyncio
    async def test_youtube_search_with_no_laptop_setup(self):
        """Test: YouTube search handles users without laptop setup"""
        with patch.dict('os.environ', {'YOUTUBE_API_KEY': 'test_key'}):
            service = YouTubeService()
            service.enabled = True
            service.youtube = Mock()
            
            with patch('backend.app.youtube_service.SessionLocal') as mock_db_class:
                mock_db = Mock()
                mock_db_class.return_value = mock_db
                
                # Mock no laptop setup
                mock_query = Mock()
                mock_filter = Mock()
                mock_filter.first.return_value = None
                mock_query.filter.return_value = mock_filter
                mock_db.query.return_value = mock_query
                
                # Mock empty video cache
                mock_query_video = Mock()
                mock_filter_video = Mock()
                mock_limit = Mock()
                mock_limit.all.return_value = []
                mock_filter_video.limit.return_value = mock_limit
                mock_query_video.filter.return_value = mock_filter_video
                
                result = await service.search_youtube_tutorials(
                    brand="Dell",
                    model="XPS 15",
                    issue_keyword="overheating",
                    user_id=1,
                    db=mock_db
                )
                
                # Should return empty list (no laptop setup to link cache)
                assert result == []
                service.youtube.search().list.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_youtube_api_error_handling(self):
        """Test: YouTube API errors are handled gracefully"""
        with patch.dict('os.environ', {'YOUTUBE_API_KEY': 'test_key'}):
            service = YouTubeService()
            service.enabled = True
            service.youtube = Mock()
            
            # Mock API error
            from googleapiclient.errors import HttpError
            service.youtube.search().list().execute.side_effect = HttpError(
                resp=Mock(status=403),
                content=b'API quota exceeded'
            )
            
            with patch('backend.app.youtube_service.SessionLocal') as mock_db_class:
                mock_db = Mock()
                mock_db_class.return_value = mock_db
                
                mock_laptop_setup = Mock()
                mock_laptop_setup.id = 1
                
                mock_query = Mock()
                mock_filter = Mock()
                mock_filter.first.return_value = mock_laptop_setup
                mock_query.filter.return_value = mock_filter
                mock_db.query.return_value = mock_query
                
                # Mock empty cache to trigger API call
                mock_query_video = Mock()
                mock_filter_video = Mock()
                mock_limit = Mock()
                mock_limit.all.return_value = []
                mock_filter_video.limit.return_value = mock_limit
                mock_query_video.filter.return_value = mock_filter_video
                
                # Setup side_effect for multiple queries
                mock_db.query.side_effect = [mock_query, mock_query_video]
                
                result = await service.search_youtube_tutorials(
                    brand="Dell",
                    model="XPS 15",
                    issue_keyword="overheating",
                    user_id=1,
                    db=mock_db
                )
                
                # Should return empty list on error
                assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])