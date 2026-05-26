from __future__ import annotations

from pathlib import Path

import yaml

from autolocalize.map.grid import CellState, OccupancyGrid


def load_map(yaml_path: str | Path) -> OccupancyGrid:
    """Load a ROS map_server YAML + PGM pair into an OccupancyGrid."""
    yaml_path = Path(yaml_path)
    with yaml_path.open() as f:
        meta = yaml.safe_load(f)

    pgm_path = yaml_path.parent / meta["image"]
    pixels, width, height = _read_pgm(pgm_path)

    negate = int(meta.get("negate", 0))
    occupied_thresh = float(meta["occupied_thresh"])
    free_thresh = float(meta["free_thresh"])
    resolution = float(meta["resolution"])
    origin = meta["origin"]
    origin_x, origin_y, origin_yaw = float(origin[0]), float(origin[1]), float(origin[2])

    cells: list[CellState] = [CellState.UNKNOWN] * (width * height)
    for gy in range(height):
        for gx in range(width):
            pixel = pixels[(height - 1 - gy) * width + gx]
            if negate:
                occ = pixel / 255.0
            else:
                occ = (255 - pixel) / 255.0

            if occ > occupied_thresh:
                state = CellState.OCCUPIED
            elif occ < free_thresh:
                state = CellState.FREE
            else:
                state = CellState.UNKNOWN

            cells[gy * width + gx] = state

    return OccupancyGrid(
        width=width,
        height=height,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_yaw=origin_yaw,
        cells=tuple(cells),
    )


def _read_pgm(path: Path) -> tuple[bytes, int, int]:
    with path.open("rb") as f:
        magic = _read_token(f)
        if magic != b"P5":
            raise ValueError(f"Expected P5 PGM, got {magic!r}")

        _skip_comments(f)
        width = int(_read_token(f))
        _skip_comments(f)
        height = int(_read_token(f))
        _skip_comments(f)
        max_val = int(_read_token(f))
        if max_val > 255:
            raise ValueError(f"Unsupported PGM maxval {max_val}")

        data = f.read(width * height)
        if len(data) != width * height:
            raise ValueError(
                f"PGM {path}: expected {width * height} bytes, got {len(data)}"
            )
        return data, width, height


def _read_token(f) -> bytes:
    while True:
        ch = f.read(1)
        if not ch:
            raise EOFError("Unexpected end of PGM file")
        if ch.isspace():
            continue
        if ch == b"#":
            f.readline()
            continue
        break

    token = bytearray(ch)
    while True:
        ch = f.read(1)
        if not ch or ch.isspace():
            break
        token.extend(ch)
    return bytes(token)


def _skip_comments(f) -> None:
    pos = f.tell()
    ch = f.read(1)
    if ch == b"#":
        f.readline()
    else:
        f.seek(pos)
