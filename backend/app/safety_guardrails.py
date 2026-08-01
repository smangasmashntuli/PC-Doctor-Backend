#safety_guardrails.py
import re
import logging
from typing import Dict, List, Tuple, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SafetyGuardrails:
    """Safety guardrail system to detect high-risk hardware issues"""
    
    # High-risk patterns that require professional technician
    HIGH_RISK_PATTERNS = {
        'swollen_battery': [
            r'battery.*(?:bulg|swell|expand|puff)',
            r'(?:bulg|swell|expand|puff).*(?:battery|bottom.*case)',
            r'bottom.*(?:case|cover).*(?:bulg|swell|expand|puff)',
            r'battery.*(?:leak|burst|explode)',
            r'(?:leak|burst|explode).*(?:battery)',
        ],
        'smoke_spark': [
            r'(?:smoke|smoking|fume)',
            r'(?:spark|sparking|electrical.*arc)',
            r'(?:burn|burning|burnt).*(?:smell|odor)',
            r'(?:smell|odor).*(?:burn|burning|burnt|plastic)',
        ],
        'liquid_damage': [
            r'(?:spill|spilled|poured).*(?:water|liquid|coffee|tea|juice|soda)',
            r'(?:water|liquid|coffee|tea|juice|soda).*(?:spill|spilled|poured)',
            r'(?:wet|damp|moist).*(?:keyboard|screen|inside)',
            r'(?:keyboard|screen|inside).*(?:wet|damp|moist)',
        ],
        'display_replacement': [
            r'(?:display|screen).*(?:replace|repair|fix).*(?:panel|glass)',
            r'(?:replace|repair|fix).*(?:display|screen).*(?:panel|glass)',
            r'crack.*(?:screen|display)',
            r'(?:screen|display).*(?:crack|shatter|broken)',
        ],
        'motherboard_soldering': [
            r'(?:motherboard|mainboard).*(?:solder|reflow|heat)',
            r'(?:solder|reflow|heat).*(?:motherboard|mainboard)',
            r'(?:capacitor|component).*(?:burst|leak|swell)',
            r'(?:short|shorting).*(?:circuit|motherboard)',
        ],
    }
    
    # Warning messages for each risk category
    WARNING_MESSAGES = {
        'swollen_battery': {
            'title': '🚨 Battery Swelling Detected - Immediate Safety Risk',
            'message': 'A swollen or bulging battery is a serious safety hazard. Lithium-ion batteries can leak, overheat, or even catch fire when damaged. Do NOT attempt to open the laptop or remove the battery yourself.',
            'action': 'Please visit a certified repair technician or authorized service center immediately. If you notice heat, smoke, or smell burning, disconnect from power and evacuate the area.'
        },
        'smoke_spark': {
            'title': '🚨 Electrical Fire Hazard - Immediate Danger',
            'message': 'Smoke, sparks, or burning smells indicate serious electrical issues that could lead to fire or electric shock. This is extremely dangerous and requires immediate professional attention.',
            'action': 'Please power off the laptop immediately, disconnect from any power source, and visit a certified repair technician or authorized service center. Do NOT use the device until inspected.'
        },
        'liquid_damage': {
            'title': '⚠️ Liquid Damage - Risk of Short Circuit',
            'message': 'Liquid inside the laptop can cause short circuits, corrosion, and permanent damage. Using the device while wet can cause further damage or electrical hazards.',
            'action': 'Please power off the laptop immediately, disconnect from power, and do NOT turn it back on. Visit a certified repair technician who can properly clean and inspect the internal components.'
        },
        'display_replacement': {
            'title': '⚠️ Display Repair - Specialized Tools Required',
            'message': 'Replacing display panels requires specialized tools, calibration equipment, and careful handling of fragile components. Improper repair can cause permanent damage or injury from broken glass.',
            'action': 'Please visit a certified display repair specialist or authorized service center. They have the proper tools and expertise to safely replace your display.'
        },
        'motherboard_soldering': {
            'title': '🚨 Motherboard Repair - Professional Service Required',
            'message': 'Motherboard repairs involving soldering, component replacement, or heat work require professional equipment and expertise. Incorrect handling can permanently destroy the motherboard or cause electrical hazards.',
            'action': 'Please visit a certified motherboard repair specialist or authorized service center. This type of repair should only be performed by trained professionals.'
        }
    }
    
    def assess_risk(self, user_query: str) -> Tuple[RiskLevel, Optional[Dict]]:
        """
        Assess the risk level of a user's query
        
        Args:
            user_query: User's message/question
        
        Returns:
            Tuple of (RiskLevel, warning_data_dict or None)
        """
        query_lower = user_query.lower()
        
        # Check each high-risk category
        for category, patterns in self.HIGH_RISK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    warning_data = self.WARNING_MESSAGES.get(category)
                    if warning_data:
                        logger.warning(f"High-risk issue detected: {category} in query: {user_query}")
                        return RiskLevel.HIGH, {
                            'is_high_risk': True,
                            'warning_title': warning_data['title'],
                            'warning_message': warning_data['message'],
                            'action_recommendation': warning_data['action'],
                            'category': category
                        }
        
        return RiskLevel.LOW, None
    
    def is_safe_query(self, user_query: str) -> bool:
        """Quick check if query is safe (no high-risk indicators)"""
        risk_level, _ = self.assess_risk(user_query)
        return risk_level != RiskLevel.HIGH
    
    def get_safety_warning(self, user_query: str) -> Optional[Dict]:
        """Get safety warning data if high-risk detected"""
        _, warning_data = self.assess_risk(user_query)
        return warning_data


# Global instance
_safety_guardrails_instance = None

def get_safety_guardrails() -> SafetyGuardrails:
    """Get or create the global safety guardrails instance"""
    global _safety_guardrails_instance
    if _safety_guardrails_instance is None:
        _safety_guardrails_instance = SafetyGuardrails()
    return _safety_guardrails_instance