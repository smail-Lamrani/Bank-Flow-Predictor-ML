import os
import sys
import cairosvg
from pathlib import Path

def convert_svg_to_png(svg_path, png_path, width=32, height=32):
    """Convert SVG file to PNG with specified dimensions."""
    try:
        cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=width, output_height=height)
        print(f"Converted {svg_path} to {png_path}")
        return True
    except Exception as e:
        print(f"Error converting {svg_path}: {e}")
        return False

def convert_all_svg_in_directory(directory):
    """Convert all SVG files in a directory to PNG."""
    svg_files = list(Path(directory).glob("*.svg"))
    
    if not svg_files:
        print(f"No SVG files found in {directory}")
        return
    
    for svg_file in svg_files:
        png_file = svg_file.with_suffix(".png")
        convert_svg_to_png(str(svg_file), str(png_file))

if __name__ == "__main__":
    # Get the directory from command line or use default
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        # Use the icons directory relative to this script
        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "icons")
    
    print(f"Converting SVG files in {directory}")
    convert_all_svg_in_directory(directory)