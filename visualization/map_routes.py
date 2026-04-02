"""
Взято за основу SPB_calculation.py из репо Stephic-Hardy/Fair_VRP_ala_Yandex, 
адаптировано под существующий формат солверов.
"""

import folium

_COLORS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "cadetblue", "darkgreen", "darkpurple", "black",
    "pink", "lightblue", "lightgreen", "gray", "lightgray",
]


def plot_routes_on_map(
    routes: list[list[int]],
    coordinates: list[tuple[float, float]],
    output_path: str,
    center: tuple[float, float] | None = None,
    zoom: int = 13,
) -> None:
    if center is None:
        center = coordinates[0]

    m = folium.Map(location=center, zoom_start=zoom)

    folium.Marker(
        location=coordinates[0],
        popup="Depot",
        icon=folium.Icon(color="black", icon="home"),
    ).add_to(m)

    for idx, route in enumerate(routes):
        if not route:
            continue
        color = _COLORS[idx % len(_COLORS)]

        route_coords = (
            [coordinates[0]]
            + [coordinates[c] for c in route]
            + [coordinates[0]]
        )

        folium.PolyLine(
            route_coords,
            color=color,
            weight=5,
            opacity=0.8,
            tooltip=f"Route {idx + 1} ({len(route)} stops)",
        ).add_to(m)

        for stop_num, client_idx in enumerate(route):
            folium.CircleMarker(
                location=coordinates[client_idx],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=f"Route {idx + 1}, stop {stop_num + 1} (node {client_idx})",
            ).add_to(m)

    m.save(output_path)
    print(f"Map saved → {output_path}")
