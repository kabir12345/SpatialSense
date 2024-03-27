import streamlit as st
from streamlit.components.v1 import html
import cv2
import numpy as np
import time
from PIL import Image
import io
from selenium import webdriver

def main():
    

    # User inputs the Twitch channel name
    channel_name = 'kabirjaiswal900'

    if channel_name:
        # Embed the Twitch live stream using an iframe
        embed_twitch_stream(channel_name)
    else:
        st.error("Please enter a valid Twitch channel name.")

def embed_twitch_stream(channel_name):
    # Constructing the Twitch embed URL using the channel name
    embed_url = f"https://player.twitch.tv/?channel={channel_name}&parent=localhost"
    
    # Using HTML component to embed Twitch live stream
    html_code = f'<iframe src="{embed_url}" height="394" width="700" frameborder="0" allowfullscreen="true" scrolling="no" allow="autoplay; fullscreen"></iframe>'
    html(html_code, height=400)

if __name__ == "__main__":
    main()
