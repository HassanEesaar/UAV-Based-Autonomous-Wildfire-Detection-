"""
GPS-based route planner for UAV dispatch to wildfire incident sites.
"""

import math
from dataclasses import dataclass


@dataclass
class Waypoint:
    lat: float
    lon: float
    altitude: float  # meters above ground level
    action: str = "fly"  # 'fly', 'hover', 'capture', 'return'


class RoutePlanner:
    """
    Plans a UAV flight route from origin to an incident site.

    Args:
        default_altitude (float): Default flight altitude in meters.
        capture_radius (float): Radius (meters) around incident for capture passes.
    """

    EARTH_RADIUS_M = 6_371_000

    def __init__(self, default_altitude: float = 100.0, capture_radius: float = 50.0):
        self.default_altitude = default_altitude
        self.capture_radius = capture_radius

    def plan(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
        altitude: float = None,
        n_capture_passes: int = 4,
    ) -> list[Waypoint]:
        """
        Generate a list of waypoints for a wildfire survey mission.

        Args:
            origin: (lat, lon) of UAV launch point.
            target: (lat, lon) of incident site.
            altitude: Flight altitude in meters. Uses default if None.
            n_capture_passes: Number of orbit passes around incident.

        Returns:
            list[Waypoint]: Ordered waypoints for the mission.
        """
        alt = altitude or self.default_altitude
        route = []

        # 1. Takeoff
        route.append(Waypoint(*origin, alt, action="fly"))

        # 2. Transit to target
        route.append(Waypoint(*target, alt, action="hover"))

        # 3. Capture orbit around incident
        for i in range(n_capture_passes):
            angle_deg = i * (360.0 / n_capture_passes)
            wp_lat, wp_lon = self._offset_point(
                target[0], target[1], self.capture_radius, angle_deg
            )
            route.append(Waypoint(wp_lat, wp_lon, alt, action="capture"))

        # 4. Return to origin
        route.append(Waypoint(*origin, alt, action="return"))

        return route

    def _offset_point(
        self, lat: float, lon: float, distance_m: float, bearing_deg: float
    ) -> tuple[float, float]:
        """Compute a lat/lon offset from a point given distance and bearing."""
        bearing = math.radians(bearing_deg)
        lat_r = math.radians(lat)
        lon_r = math.radians(lon)
        d_r = distance_m / self.EARTH_RADIUS_M

        new_lat = math.asin(
            math.sin(lat_r) * math.cos(d_r)
            + math.cos(lat_r) * math.sin(d_r) * math.cos(bearing)
        )
        new_lon = lon_r + math.atan2(
            math.sin(bearing) * math.sin(d_r) * math.cos(lat_r),
            math.cos(d_r) - math.sin(lat_r) * math.sin(new_lat),
        )
        return math.degrees(new_lat), math.degrees(new_lon)

    def total_distance_m(self, route: list[Waypoint]) -> float:
        """Compute total flight distance in meters for a route."""
        total = 0.0
        for i in range(1, len(route)):
            a, b = route[i - 1], route[i]
            total += self._haversine(a.lat, a.lon, b.lat, b.lon)
        return total

    def _haversine(self, lat1, lon1, lat2, lon2) -> float:
        """Haversine distance between two lat/lon points in meters."""
        r = self.EARTH_RADIUS_M
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
        return 2 * r * math.asin(math.sqrt(a))
