import subprocess

# RTMP URL of the Twitch stream (example format)
rtmp_url = "rtmp://jfk50.contribute.live-video.net/app/live_1058042320_CECjGvxU1Kn5aZaYX5y5R2bOr6DWv6"

# FFmpeg command to capture the RTMP stream and save it to a file
# Replace 'output.mp4' with your desired file name and location
ffmpeg_command = [
    'ffmpeg',
    '-i', rtmp_url,  # Input from RTMP URL
    '-acodec', 'copy',  # Use the same audio codec without re-encoding
    '-vcodec', 'copy',  # Use the same video codec without re-encoding
    'output.mp4'  # Output file
]

# Run the FFmpeg command
process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Wait for FFmpeg to finish
stdout, stderr = process.communicate()

# Check for errors
if process.returncode != 0:
    print(f"FFmpeg error:\n{stderr.decode('utf-8')}")
else:
    print("Stream capture complete.")
