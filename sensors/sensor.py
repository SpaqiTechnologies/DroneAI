"""
Base sensor class for Drone_AI_App
"""

class Sensor:
    def update(self):
        """Fetch latest measurement."""
        raise NotImplementedError

    def is_valid(self):
        """True if last reading is valid."""
        return False

    def to_dict(self):
        """Convenience serialization."""
        return {
            'type': self.__class__.__name__,
            'value': vars(self)
        }
