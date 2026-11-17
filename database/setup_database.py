"""
Database Initialization & Image Dataset Builder
Generates a structured, multi-class visual dataset, stores images on disk,
and populates the SQLite metadata database with indexing for pipeline optimization.
"""

import os
import math
import sqlite3
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    DB_PATH,
    SCHEMA_PATH,
    IMAGES_DIR,
    CLASS_NAMES,
    IMG_HEIGHT,
    IMG_WIDTH,
    CHANNELS,
    RANDOM_SEED
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def initialize_schema(conn: sqlite3.Connection, schema_file: Path = SCHEMA_PATH) -> None:
    """Executes the DDL statements from schema.sql to create tables and indexes."""
    with open(schema_file, "r") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    conn.commit()
    print("Database schema and performance indexes initialized successfully.")


def render_pattern_image(class_id: int, size: int = 32) -> np.ndarray:
    """
    Renders distinct geometric and visual texture classes with independent background colors,
    random foreground colors, translations, scale variations, and sensor noise.
    Forces models to learn spatial convolutional features rather than trivial color shortcuts.
    """
    # Independent random background and foreground colors to decouple color from class
    bg_color = (random.randint(25, 75), random.randint(25, 75), random.randint(25, 75))
    fg_color = (random.randint(140, 245), random.randint(140, 245), random.randint(140, 245))

    img = Image.new("RGB", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Random center translation (+/- 4 pixels)
    cx = size // 2 + random.randint(-4, 4)
    cy = size // 2 + random.randint(-4, 4)
    r = random.randint(6, 11)

    if class_id == 0:  # Circle
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fg_color, outline=(255, 255, 255))
    elif class_id == 1:  # Plus / Cross
        w = random.randint(2, 4)
        draw.rectangle([cx - w, cy - r, cx + w, cy + r], fill=fg_color)
        draw.rectangle([cx - r, cy - w, cx + r, cy + w], fill=fg_color)
    elif class_id == 2:  # Triangle
        points = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
        draw.polygon(points, fill=fg_color, outline=(255, 255, 255))
    elif class_id == 3:  # Horizontal Wave / Stripes
        step = random.choice([5, 6])
        for y in range(2, size, step):
            draw.line([(0, y), (size, y)], fill=fg_color, width=2)
    elif class_id == 4:  # Diagonal Stripes
        step = random.choice([7, 8])
        for offset in range(-size, size * 2, step):
            draw.line([(offset, 0), (offset + size, size)], fill=fg_color, width=2)
    elif class_id == 5:  # Checkerboard / Grid
        step = 8
        for gx in range(0, size, step):
            for gy in range(0, size, step):
                if (gx // step + gy // step) % 2 == 0:
                    draw.rectangle([gx, gy, gx + step - 1, gy + step - 1], fill=fg_color)
    elif class_id == 6:  # Star / Radial Spokes
        for angle in [0, 45, 90, 135]:
            rad = math.radians(angle)
            ex = int(cx + r * math.cos(rad))
            ey = int(cy + r * math.sin(rad))
            sx = int(cx - r * math.cos(rad))
            sy = int(cy - r * math.sin(rad))
            draw.line([(sx, sy), (ex, ey)], fill=fg_color, width=2)
    elif class_id == 7:  # Concentric Rings
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fg_color, width=2)
        r2 = max(2, r - 5)
        draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], outline=(255, 255, 255), width=1)
    elif class_id == 8:  # Diamond / Rotated Square
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=fg_color)
    else:  # Multi-dots Grid
        for dx in [-7, 0, 7]:
            for dy in [-7, 0, 7]:
                draw.rectangle([cx + dx - 1, cy + dy - 1, cx + dx + 1, cy + dy + 1], fill=fg_color)

    # Convert to array and add sensor noise
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 14.0, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return arr


def populate_dataset_and_database(total_samples: int = 6000) -> None:
    """Generates images, saves to filesystem, and indexes into SQLite database."""
    print(f"Initializing SQLite database at: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))
    initialize_schema(conn)

    samples_per_class = total_samples // len(CLASS_NAMES)
    train_per_class = int(samples_per_class * 0.70)
    val_per_class = int(samples_per_class * 0.15)
    test_per_class = samples_per_class - train_per_class - val_per_class

    cursor = conn.cursor()
    records = []
    total_generated = 0

    print(f"Generating {total_samples} images across {len(CLASS_NAMES)} classes...")
    for class_id, class_name in enumerate(CLASS_NAMES):
        for i in range(samples_per_class):
            if i < train_per_class:
                split = "train"
            elif i < train_per_class + val_per_class:
                split = "val"
            else:
                split = "test"

            class_dir = IMAGES_DIR / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            img_arr = render_pattern_image(class_id, size=IMG_HEIGHT)
            filename = f"{split}_{class_name}_{i:05d}.png"
            filepath = class_dir / filename

            # Save PNG to disk
            Image.fromarray(img_arr).save(filepath, format="PNG")

            file_size = filepath.stat().st_size
            mean_pixel = float(np.mean(img_arr))
            quality_score = round(random.uniform(0.85, 1.0), 3)

            records.append((
                filename,
                str(filepath),
                int(class_id),
                class_name,
                split,
                IMG_WIDTH,
                IMG_HEIGHT,
                CHANNELS,
                file_size,
                quality_score,
                mean_pixel
            ))
            total_generated += 1

    print("Inserting metadata into SQLite database...")
    cursor.executemany("""
        INSERT INTO image_metadata (
            filename, filepath, class_id, class_name, split,
            width, height, channels, file_size_bytes, quality_score, mean_pixel_val
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, records)
    conn.commit()

    # Verify counts
    cursor.execute("SELECT split, count(*) FROM image_metadata GROUP BY split;")
    splits = cursor.fetchall()
    print("\nDataset Partition Distribution:")
    for s, count in splits:
        print(f"  [{s.upper()}]: {count} records")

    cursor.execute("SELECT count(*) FROM image_metadata;")
    total_in_db = cursor.fetchone()[0]
    print(f"\nTotal image records in SQLite: {total_in_db}")

    conn.close()
    print("Database and visual dataset generated successfully.")


if __name__ == "__main__":
    populate_dataset_and_database(total_samples=6000)
