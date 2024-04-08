import cv2
import numpy as np
import streamlit as st

# Function to perform dark channel prior dehazing
def dark_channel(image, window_size=15):
    min_channel = np.min(image, axis=2)
    return cv2.erode(min_channel, np.ones((window_size, window_size)))

# Function to estimate atmosphere light
def estimate_atmosphere(image, dark_channel, percentile=0.001):
    flat_dark_channel = dark_channel.flatten()
    flat_image = image.reshape(-1, 3)
    num_pixels = flat_image.shape[0]
    num_pixels_to_keep = int(num_pixels * percentile)
    indices = np.argpartition(flat_dark_channel, -num_pixels_to_keep)[-num_pixels_to_keep:]
    atmosphere = np.max(flat_image[indices], axis=0)
    return atmosphere

# Function to perform dehazing
def dehaze(image, tmin=0.1, omega=0.95, window_size=15):
    if image is None:
        return None

    image = image.astype(np.float64) / 255.0
    dark_ch = dark_channel(image, window_size)
    atmosphere = estimate_atmosphere(image, dark_ch)
    transmission = 1 - omega * dark_ch
    transmission = np.maximum(transmission, tmin)
    dehazed = np.zeros_like(image)

    for channel in range(3):
        dehazed[:, :, channel] = (image[:, :, channel] - atmosphere[channel]) / transmission + atmosphere[channel]

    dehazed = np.clip(dehazed, 0, 1)
    dehazed = (dehazed * 255).astype(np.uint8)

    return dehazed

# Function to perform object detection
def detect_objects(frame):
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    return frame

# Load YOLO for object detection
net = cv2.dnn.readNet(r'C:\Users\VINAY\Desktop\final project\yolov4.weights', r'C:\Users\VINAY\Desktop\final project\yolov4.cfg')
classes = []
with open('C:/Users/VINAY/Desktop/final project/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def main():
    st.title("Dehazing and Object Detection Web App")

    st.sidebar.header("Choose Option")
    option = st.sidebar.radio("", ("Image", "Video"))

    if option == "Image":
        st.subheader("Upload Image")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            st.subheader("Dehazed Image with Object Detection")
            dehazed_image = dehaze(image)
            detected_image = detect_objects(dehazed_image)
            st.image(detected_image, caption="Dehazed Image with Object Detection", use_column_width=True)

    elif option == "Video":
        st.subheader("Upload Video")
        uploaded_video = st.file_uploader("Choose a video file...", type=["mp4"])

        if uploaded_video is not None:
            file_bytes = np.asarray(bytearray(uploaded_video.read()), dtype=np.uint8)
            video = cv2.imdecode(file_bytes, 1)
            st.video(uploaded_video, format='video/mp4')

            st.subheader("Dehazed Video with Object Detection")
            cap = cv2.VideoCapture(uploaded_video)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                dehazed_frame = dehaze(frame)
                detected_frame = detect_objects(dehazed_frame)
                st.image(detected_frame, channels="BGR")

    else:
        st.error("Please choose an option from the sidebar.")

if __name__ == "__main__":
    main()
