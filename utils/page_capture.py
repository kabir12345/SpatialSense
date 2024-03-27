from selenium import webdriver
import cv2
import numpy as np
import time

# Set up Selenium with a headless browser
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
driver = webdriver.Chrome(options=options)

# Navigate to the page you want to stream
driver.get('https://www.twitch.tv/kabirjaiswal900')

frame_rate = 1  # It's not feasible to achieve 600 FPS due to hardware and network limitations, so setting it to a realistic value
wait_time = 1 / frame_rate

try:
    while True:
        # Take screenshot and convert it to a format that OpenCV can display
        screenshot = driver.get_screenshot_as_png()  # Take screenshot as PNG
        nparr = np.frombuffer(screenshot, np.uint8)  # Convert PNG to numpy array
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode numpy array into OpenCV format
        
        # Crop the center of the image
        h, w, _ = img_np.shape
        center_x, center_y = w // 2, h // 2
        size = min(center_x, center_y)  # Size of the cropped area (make it a square for simplicity)

        # Define the top left corner of the cropping rectangle
        start_x = center_x - size // 2
        start_y = center_y - size // 2

        # Crop the image using array slicing
        cropped_img_np = img_np[start_y:start_y + size, start_x:start_x + size]

        # Display the cropped image using OpenCV
        cv2.imshow('Cropped Stream', cropped_img_np)
        
        # Delay to control the frame rate
        time.sleep(wait_time)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
            break
except KeyboardInterrupt:
    print("Stopping stream...")

finally:
    driver.quit()
    cv2.destroyAllWindows()  # Make sure all OpenCV windows are closed
