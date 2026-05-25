"""Generate synthetic example images for testing and demo purposes."""

import cv2
import numpy as np
import os


def draw_resistor(img, x, y, scale=1.0, color=0):
    """Draw a resistor symbol."""
    s = scale
    pts = np.array([
        [x, y],
        [x + int(10*s), y],
        [x + int(15*s), y - int(8*s)],
        [x + int(25*s), y + int(8*s)],
        [x + int(35*s), y - int(8*s)],
        [x + int(45*s), y + int(8*s)],
        [x + int(55*s), y - int(8*s)],
        [x + int(60*s), y],
        [x + int(70*s), y],
    ], dtype=np.int32)
    cv2.polylines(img, [pts], False, color, max(1, int(2*s)))


def draw_capacitor(img, x, y, scale=1.0, color=0):
    """Draw a capacitor symbol."""
    s = scale
    cv2.line(img, (x, y), (x + int(20*s), y), color, max(1, int(2*s)))
    cv2.line(img, (x + int(20*s), y - int(15*s)), (x + int(20*s), y + int(15*s)), color, max(1, int(2*s)))
    cv2.line(img, (x + int(30*s), y - int(15*s)), (x + int(30*s), y + int(15*s)), color, max(1, int(2*s)))
    cv2.line(img, (x + int(30*s), y), (x + int(50*s), y), color, max(1, int(2*s)))


def draw_circle_symbol(img, x, y, radius=20, color=0, thickness=2):
    """Draw a circle with cross inside (generic component)."""
    cv2.circle(img, (x, y), radius, color, thickness)
    cv2.line(img, (x - radius, y), (x + radius, y), color, thickness)
    cv2.line(img, (x, y - radius), (x, y + radius), color, thickness)


def draw_triangle(img, x, y, size=30, color=0, thickness=2):
    """Draw a triangle symbol."""
    pts = np.array([
        [x, y - size],
        [x - size, y + size],
        [x + size, y + size],
    ], dtype=np.int32)
    cv2.polylines(img, [pts], True, color, thickness)


def draw_diamond(img, x, y, size=20, color=0, thickness=2):
    """Draw a diamond/rhombus symbol."""
    pts = np.array([
        [x, y - size],
        [x + size, y],
        [x, y + size],
        [x - size, y],
    ], dtype=np.int32)
    cv2.polylines(img, [pts], True, color, thickness)


def generate_example_1():
    """Generate: circle-cross pattern on a drawing with multiple instances."""
    drawing = np.ones((800, 1200), dtype=np.uint8) * 255

    cv2.rectangle(drawing, (20, 20), (1180, 780), 0, 2)
    cv2.rectangle(drawing, (30, 30), (1170, 770), 0, 1)
    cv2.line(drawing, (30, 700), (1170, 700), 0, 1)
    cv2.line(drawing, (800, 30), (800, 700), 0, 1)

    for i in range(5):
        cv2.line(drawing, (50 + i * 150, 650), (50 + i * 150, 700), 0, 1)

    positions = [(150, 200), (400, 300), (600, 150), (200, 500), (500, 500), (900, 400)]
    for px, py in positions:
        draw_circle_symbol(drawing, px, py, radius=22, thickness=2)
        cv2.line(drawing, (px - 50, py), (px - 22, py), 0, 1)
        cv2.line(drawing, (px + 22, py), (px + 50, py), 0, 1)

    for i in range(3):
        draw_resistor(drawing, 850, 150 + i * 120, scale=1.0)

    draw_triangle(drawing, 700, 600, size=25)
    draw_diamond(drawing, 950, 600, size=20)

    pattern = np.ones((60, 60), dtype=np.uint8) * 255
    draw_circle_symbol(pattern, 30, 30, radius=22, thickness=2)

    return drawing, pattern


def generate_example_2():
    """Generate: resistor pattern on a drawing."""
    drawing = np.ones((600, 1000), dtype=np.uint8) * 255

    cv2.rectangle(drawing, (10, 10), (990, 590), 0, 2)
    cv2.line(drawing, (10, 500), (990, 500), 0, 1)

    positions = [
        (100, 100), (300, 100), (500, 100),
        (100, 250), (300, 250),
        (500, 350), (700, 200),
    ]
    for px, py in positions:
        draw_resistor(drawing, px, py, scale=1.0)
        cv2.line(drawing, (px - 30, py), (px, py), 0, 1)
        cv2.line(drawing, (px + 70, py), (px + 100, py), 0, 1)

    for px, py in [(700, 400), (200, 400)]:
        draw_capacitor(drawing, px, py, scale=1.0)

    draw_circle_symbol(drawing, 850, 100, radius=20)
    draw_triangle(drawing, 850, 300, size=20)

    pattern = np.ones((40, 90), dtype=np.uint8) * 255
    draw_resistor(pattern, 10, 20, scale=1.0)

    return drawing, pattern


def generate_example_3():
    """Generate: diamond pattern on a drawing with mixed symbols."""
    drawing = np.ones((700, 1100), dtype=np.uint8) * 255

    cv2.rectangle(drawing, (15, 15), (1085, 685), 0, 2)

    for y in range(100, 600, 100):
        cv2.line(drawing, (50, y), (1050, y), 0, 1)

    diamond_positions = [
        (200, 150), (500, 150), (800, 150),
        (350, 350), (650, 350),
        (200, 550), (800, 550),
    ]
    for px, py in diamond_positions:
        draw_diamond(drawing, px, py, size=22, thickness=2)

    for px, py in [(100, 350), (900, 250)]:
        draw_circle_symbol(drawing, px, py, radius=18)

    for px, py in [(400, 550), (600, 550)]:
        draw_triangle(drawing, px, py, size=18)

    pattern = np.ones((60, 60), dtype=np.uint8) * 255
    draw_diamond(pattern, 30, 30, size=22, thickness=2)

    return drawing, pattern


def main():
    os.makedirs("examples/patterns", exist_ok=True)
    os.makedirs("examples/drawings", exist_ok=True)

    generators = [
        ("example1", generate_example_1),
        ("example2", generate_example_2),
        ("example3", generate_example_3),
    ]

    for name, gen_fn in generators:
        drawing, pattern = gen_fn()
        cv2.imwrite(f"examples/drawings/{name}_drawing.png", drawing)
        cv2.imwrite(f"examples/patterns/{name}_pattern.png", pattern)
        print(f"Generated {name}: drawing={drawing.shape}, pattern={pattern.shape}")

    print("All examples generated successfully.")


if __name__ == "__main__":
    main()
