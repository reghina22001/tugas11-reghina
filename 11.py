import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
from matplotlib import colors

def get_dominant_colors(image, k=5):
    # Function to get the dominant colors from an image
    # Convert image to RGB from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    
    # KMeans to find dominant colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    
    # Count and get the colors
    counts = Counter(kmeans.labels_)
    centers = kmeans.cluster_centers_
    
    # Order colors by frequency
    ordered_colors = [centers[i] for i in counts.keys()]
    hex_colors = [colors.rgb2hex(ordered_colors[i] / 255.0).upper() for i in range(k)]  # Upper case hex colors
    rgb_colors = [ordered_colors[i] for i in range(k)]
    
    return hex_colors, rgb_colors

def display_color_palette(hex_colors, rgb_colors):
    # Function to display the color palette
    st.markdown('<div style="display: flex;">', unsafe_allow_html=True)
    for hex_color, rgb_color in zip(hex_colors, rgb_colors):
        rgb_text = f"RGB({int(rgb_color[0])}, {int(rgb_color[1])}, {int(rgb_color[2])})"
        st.markdown(f'''
            <div style="text-align: center; margin: 10px;">
                <div style="display: inline-block; width: 150px; height: 150px; background-color: {hex_color}; border-radius: 10px;"></div>
                <div style="margin-top: 5px; font-weight: bold;">{hex_color}</div>
                <div style="font-size: 12px; color: #555;">{rgb_text}</div>
            </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Streamlit UI
st.title("Color Picker Generator")

st.write("""
Upload an image and get the 5 most dominant colors as a color palette.
""")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "heic"])

if uploaded_file is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display uploaded image
    st.image(image_rgb, caption='Uploaded Image.', use_column_width=True)
    
    st.write("Generating palette...")
    try:
        hex_colors, rgb_colors = get_dominant_colors(image)
        st.write("Here are the 5 most dominant colors in the image:")
        display_color_palette(hex_colors, rgb_colors)
    except Exception as e:
        st.error("Error processing the image. Please make sure it's a valid image file.")
