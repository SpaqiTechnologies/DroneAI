"""
Mission Storage Module

Handles saving and loading missions to/from files (JSON format).
"""

import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from .mission import Mission, MissionState
from .waypoint import Waypoint


class MissionStorage:
    """
    Handles mission persistence to JSON files.

    Supports saving individual missions and managing a mission library.
    """

    DEFAULT_EXTENSION = ".mission.json"
    LIBRARY_FILE = "mission_library.json"

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize mission storage.

        Args:
            storage_dir: Directory for mission files (default: ./missions)
        """
        if storage_dir is None:
            storage_dir = os.path.join(os.getcwd(), "missions")

        self.storage_dir = Path(storage_dir)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_mission_path(self, mission_id: str) -> Path:
        """Get path for a mission file."""
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in mission_id)
        return self.storage_dir / f"{safe_id}{self.DEFAULT_EXTENSION}"

    def save(self, mission: Mission) -> str:
        """
        Save a mission to file.

        Args:
            mission: Mission to save

        Returns:
            Path to saved file
        """
        mission.modified_at = datetime.now()
        file_path = self._get_mission_path(mission.id)

        data = mission.to_dict()

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return str(file_path)

    def load(self, mission_id: str) -> Optional[Mission]:
        """
        Load a mission from file.

        Args:
            mission_id: ID of mission to load

        Returns:
            Mission or None if not found
        """
        file_path = self._get_mission_path(mission_id)

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Mission.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading mission {mission_id}: {e}")
            return None

    def load_from_path(self, file_path: str) -> Optional[Mission]:
        """
        Load a mission from a specific file path.

        Args:
            file_path: Path to mission file

        Returns:
            Mission or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Mission.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError) as e:
            print(f"Error loading mission from {file_path}: {e}")
            return None

    def delete(self, mission_id: str) -> bool:
        """
        Delete a mission file.

        Args:
            mission_id: ID of mission to delete

        Returns:
            True if deleted, False if not found
        """
        file_path = self._get_mission_path(mission_id)

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def exists(self, mission_id: str) -> bool:
        """Check if a mission exists."""
        return self._get_mission_path(mission_id).exists()

    def list_missions(self) -> List[Dict[str, Any]]:
        """
        List all saved missions.

        Returns:
            List of mission summaries (id, name, modified_at, waypoint_count)
        """
        missions = []

        for file_path in self.storage_dir.glob(f"*{self.DEFAULT_EXTENSION}"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                missions.append({
                    'id': data.get('id', file_path.stem),
                    'name': data.get('name', 'Unknown'),
                    'description': data.get('description', ''),
                    'mission_type': data.get('mission_type', 'waypoint'),
                    'waypoint_count': len(data.get('waypoints', [])),
                    'created_at': data.get('created_at'),
                    'modified_at': data.get('modified_at'),
                    'tags': data.get('tags', []),
                })
            except (json.JSONDecodeError, KeyError):
                continue

        # Sort by modified date (newest first)
        missions.sort(key=lambda m: m.get('modified_at', ''), reverse=True)
        return missions

    def search_missions(
        self,
        name_contains: Optional[str] = None,
        tags: Optional[List[str]] = None,
        mission_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search missions by criteria.

        Args:
            name_contains: Filter by name substring
            tags: Filter by tags (any match)
            mission_type: Filter by mission type

        Returns:
            List of matching mission summaries
        """
        all_missions = self.list_missions()
        results = []

        for mission in all_missions:
            # Check name filter
            if name_contains:
                if name_contains.lower() not in mission['name'].lower():
                    continue

            # Check tags filter
            if tags:
                mission_tags = mission.get('tags', [])
                if not any(tag in mission_tags for tag in tags):
                    continue

            # Check type filter
            if mission_type:
                if mission.get('mission_type') != mission_type:
                    continue

            results.append(mission)

        return results

    def export_to_json(self, mission: Mission, file_path: str) -> bool:
        """
        Export mission to a specific JSON file.

        Args:
            mission: Mission to export
            file_path: Target file path

        Returns:
            True if successful
        """
        try:
            data = mission.to_dict()
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except (IOError, OSError) as e:
            print(f"Error exporting mission: {e}")
            return False

    def import_from_json(self, file_path: str) -> Optional[Mission]:
        """
        Import mission from a JSON file.

        Args:
            file_path: Source file path

        Returns:
            Mission or None if import fails
        """
        return self.load_from_path(file_path)

    def export_to_kml(self, mission: Mission, file_path: str) -> bool:
        """
        Export mission waypoints to KML format.

        Args:
            mission: Mission to export
            file_path: Target KML file path

        Returns:
            True if successful
        """
        try:
            kml_content = self._generate_kml(mission)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(kml_content)
            return True
        except (IOError, OSError) as e:
            print(f"Error exporting KML: {e}")
            return False

    def _generate_kml(self, mission: Mission) -> str:
        """Generate KML content for a mission."""
        coordinates = []
        placemarks = []

        for i, wp in enumerate(mission.waypoints):
            coordinates.append(f"{wp.longitude},{wp.latitude},{wp.altitude}")

            placemarks.append(f"""
    <Placemark>
      <name>WP{i + 1}</name>
      <description>{wp.description or f'Waypoint {i + 1}'}</description>
      <Point>
        <coordinates>{wp.longitude},{wp.latitude},{wp.altitude}</coordinates>
      </Point>
    </Placemark>""")

        coords_string = " ".join(coordinates)

        kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>{mission.name}</name>
  <description>{mission.description}</description>

  <Style id="missionPath">
    <LineStyle>
      <color>ff0000ff</color>
      <width>3</width>
    </LineStyle>
  </Style>

  <Placemark>
    <name>Mission Path</name>
    <styleUrl>#missionPath</styleUrl>
    <LineString>
      <altitudeMode>relativeToGround</altitudeMode>
      <coordinates>{coords_string}</coordinates>
    </LineString>
  </Placemark>
{''.join(placemarks)}
</Document>
</kml>"""

        return kml

    def export_to_csv(self, mission: Mission, file_path: str) -> bool:
        """
        Export mission waypoints to CSV format.

        Args:
            mission: Mission to export
            file_path: Target CSV file path

        Returns:
            True if successful
        """
        try:
            lines = ["sequence,latitude,longitude,altitude,speed,hold_time,pass_through"]

            for wp in mission.waypoints:
                lines.append(
                    f"{wp.sequence},{wp.latitude},{wp.longitude},{wp.altitude},"
                    f"{wp.speed},{wp.hold_time},{wp.pass_through}"
                )

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            return True
        except (IOError, OSError) as e:
            print(f"Error exporting CSV: {e}")
            return False

    def import_from_csv(self, file_path: str, mission_name: str = "Imported Mission") -> Optional[Mission]:
        """
        Import waypoints from CSV file.

        Args:
            file_path: Source CSV file path
            mission_name: Name for the imported mission

        Returns:
            Mission or None if import fails
        """
        try:
            mission = Mission(name=mission_name)

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Skip header
            for i, line in enumerate(lines[1:]):
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    wp = Waypoint(
                        latitude=float(parts[1]),
                        longitude=float(parts[2]),
                        altitude=float(parts[3]),
                        sequence=i,
                        speed=float(parts[4]) if len(parts) > 4 else 5.0,
                        hold_time=float(parts[5]) if len(parts) > 5 else 0.0,
                        pass_through=parts[6].lower() == 'true' if len(parts) > 6 else False,
                    )
                    mission.add_waypoint(wp)

            return mission if mission.waypoints else None

        except (IOError, OSError, ValueError) as e:
            print(f"Error importing CSV: {e}")
            return None

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage info
        """
        missions = self.list_missions()
        total_waypoints = sum(m['waypoint_count'] for m in missions)

        return {
            'storage_dir': str(self.storage_dir),
            'mission_count': len(missions),
            'total_waypoints': total_waypoints,
            'storage_exists': self.storage_dir.exists(),
        }
