#notifications.py
import os
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from backend.app.models import Notification, LaptopSetup, LaptopSpecs
from backend.app.database import SessionLocal

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for managing notifications and maintenance alerts"""
    
    # Maintenance alert templates based on laptop specs
    MAINTENANCE_TEMPLATES = {
        'dust_cleaning': {
            'title': '🧹 Time to Clean Your Laptop Fans',
            'message': 'Dust buildup can cause overheating. Consider cleaning the fan vents on your {brand} {model}.',
            'frequency_days': 90,  # Every 3 months
            'priority': 'medium'
        },
        'battery_calibration': {
            'title': '🔋 Battery Health Check',
            'message': 'Calibrate your {brand} {model} battery by fully charging and discharging it once every 3 months.',
            'frequency_days': 90,
            'priority': 'low'
        },
        'os_updates': {
            'title': '💻 Operating System Updates',
            'message': 'Check for {os} updates on your {brand} {model} to ensure security and performance.',
            'frequency_days': 14,  # Every 2 weeks
            'priority': 'high'
        },
        'driver_updates': {
            'title': '🖥️ Graphics Driver Update',
            'message': 'Update your {gpu} drivers for optimal performance on {brand} {model}.',
            'frequency_days': 30,  # Monthly
            'priority': 'medium'
        },
        'disk_cleanup': {
            'title': '💾 Disk Cleanup Reminder',
            'message': 'Free up space on your {storage} by removing temporary files and old downloads.',
            'frequency_days': 60,  # Every 2 months
            'priority': 'low'
        },
        'security_scan': {
            'title': '🔒 Security Scan',
            'message': 'Run a full security scan on your {brand} {model} to detect any malware or threats.',
            'frequency_days': 7,  # Weekly
            'priority': 'high'
        }
    }
    
    def generate_maintenance_alerts(
        self,
        user_id: int,
        db: Session
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized maintenance alerts based on laptop specs
        
        Args:
            user_id: User ID
            db: Database session
        
        Returns:
            List of created notifications
        """
        # Get user's laptop setup and specs
        laptop_setup = db.query(LaptopSetup).filter(
            LaptopSetup.user_id == user_id
        ).first()
        
        if not laptop_setup:
            logger.warning(f"No laptop setup found for user {user_id}")
            return []
        
        laptop_specs = db.query(LaptopSpecs).filter(
            LaptopSpecs.laptop_setup_id == laptop_setup.id
        ).first()
        
        if not laptop_specs:
            logger.warning(f"No laptop specs found for user {user_id}")
            return []
        
        # Check which alerts are due
        notifications_to_create = []
        
        for template_key, template in self.MAINTENANCE_TEMPLATES.items():
            if self._is_alert_due(user_id, template_key, db):
                # Personalize message with laptop specs
                message = template['message'].format(
                    brand=laptop_setup.brand,
                    model=laptop_setup.model,
                    os=laptop_specs.os or 'your operating system',
                    gpu=laptop_specs.gpu or 'graphics',
                    storage=laptop_specs.storage or 'storage'
                )
                
                notification = Notification(
                    user_id=user_id,
                    title=template['title'],
                    message=message,
                    notification_type='maintenance',
                    priority=template['priority'],
                    metadata_json={
                        'template_key': template_key,
                        'laptop_setup_id': laptop_setup.id
                    }
                )
                notifications_to_create.append(notification)
        
        # Bulk insert notifications
        created_notifications = []
        if notifications_to_create:
            try:
                db.bulk_save_objects(notifications_to_create)
                db.commit()
                created_notifications = notifications_to_create
                logger.info(f"Created {len(created_notifications)} maintenance alerts for user {user_id}")
            except Exception as e:
                db.rollback()
                logger.error(f"Error creating notifications: {e}")
        
        return [
            {
                'id': n.id,
                'title': n.title,
                'message': n.message,
                'notification_type': n.notification_type,
                'priority': n.priority,
                'is_read': n.is_read,
                'created_at': n.created_at.isoformat()
            }
            for n in created_notifications
        ]
    
    def _is_alert_due(
        self,
        user_id: int,
        template_key: str,
        db: Session
    ) -> bool:
        """
        Check if an alert is due based on last sent time
        
        Args:
            user_id: User ID
            template_key: Template key from MAINTENANCE_TEMPLATES
            db: Database session
        
        Returns:
            True if alert is due, False otherwise
        """
        template = self.MAINTENANCE_TEMPLATES.get(template_key)
        if not template:
            return False
        
        # Find the most recent notification of this type
        last_notification = db.query(Notification).filter(
            Notification.user_id == user_id,
            Notification.notification_type == 'maintenance'
        ).filter(
            Notification.metadata_json.contains(template_key)
        ).order_by(Notification.created_at.desc()).first()
        
        if not last_notification:
            return True  # Never sent before
        
        # Check if enough time has passed
        time_since_last = datetime.utcnow() - last_notification.created_at
        return time_since_last >= timedelta(days=template['frequency_days'])
    
    def get_user_notifications(
        self,
        user_id: int,
        db: Session,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get notifications for a user
        
        Args:
            user_id: User ID
            db: Database session
            unread_only: If True, only return unread notifications
            limit: Maximum number of notifications to return
        
        Returns:
            List of notifications
        """
        query = db.query(Notification).filter(Notification.user_id == user_id)
        
        if unread_only:
            query = query.filter(Notification.is_read == False)
        
        notifications = query.order_by(Notification.created_at.desc()).limit(limit).all()
        
        return [
            {
                'id': n.id,
                'title': n.title,
                'message': n.message,
                'notification_type': n.notification_type,
                'priority': n.priority,
                'is_read': n.is_read,
                'created_at': n.created_at.isoformat()
            }
            for n in notifications
        ]
    
    def mark_notification_as_read(
        self,
        notification_id: int,
        user_id: int,
        db: Session
    ) -> Optional[Dict[str, Any]]:
        """
        Mark a notification as read
        
        Args:
            notification_id: Notification ID
            user_id: User ID (for authorization)
            db: Database session
        
        Returns:
            Updated notification dict or None if not found
        """
        notification = db.query(Notification).filter(
            Notification.id == notification_id,
            Notification.user_id == user_id
        ).first()
        
        if not notification:
            return None
        
        notification.is_read = True
        notification.read_at = datetime.utcnow()
        db.commit()
        db.refresh(notification)
        
        return {
            'id': notification.id,
            'title': notification.title,
            'message': notification.message,
            'notification_type': notification.notification_type,
            'priority': notification.priority,
            'is_read': notification.is_read,
            'created_at': notification.created_at.isoformat()
        }
    
    def get_unread_count(self, user_id: int, db: Session) -> int:
        """Get count of unread notifications for a user"""
        return db.query(Notification).filter(
            Notification.user_id == user_id,
            Notification.is_read == False
        ).count()


# Global instance
_notification_service_instance = None

def get_notification_service() -> NotificationService:
    """Get or create the global notification service instance"""
    global _notification_service_instance
    if _notification_service_instance is None:
        _notification_service_instance = NotificationService()
    return _notification_service_instance