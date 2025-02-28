import os
import subprocess
import sys

def upscale_video(input_file, resolution):
    """
    Upscale a video to either 1080p (Full HD) or 4K resolution.
    
    Args:
        input_file (str): Path to the input video
        resolution (str): Either '1080p' or '4k'
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist.")
        return
    
    # Get the file name without extension
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Set resolution parameters
    if resolution == '1080p':
        width, height = 1920, 1080
        output_file = f"{file_name}_fullhd.mp4"
    elif resolution == '4k':
        width, height = 3840, 2160
        output_file = f"{file_name}_4k.mp4"
    else:
        print("Invalid resolution. Choose either '1080p' or '4k'")
        return
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'medium',
        '-c:a', 'copy',
        output_file
    ]
    
    # Run the command
    try:
        print(f"Converting {input_file} to {resolution}...")
        subprocess.run(cmd, check=True)
        print(f"Successfully created {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg and make sure it's in your PATH.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python upscale_video.py input_file resolution")
        print("Example: python upscale_video.py my_video.mp4 1080p")
        print("Example: python upscale_video.py my_video.mp4 4k")
        sys.exit(1)
    
    input_file = sys.argv[1]
    resolution = sys.argv[2]
    
    upscale_video(input_file, resolution)
