# Object-Detection
# Dog and Cat Object Detection using Faster R-CNN

This project utilizes a pre-trained Faster R-CNN model for detecting and classifying dogs and cats in images. The model is based on the Faster R-CNN architecture with a ResNet-50 backbone, pre-trained on the COCO dataset.

## Requirements

- Python 3
- PyTorch
- torchvision
- OpenCV
- NumPy
- Matplotlib

## Usage

1. First, make sure to mount Google Drive if you're running this code in Google Colab:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. Run the provided code in a Python environment with the required dependencies installed.

3. Provide the path to the image you want to perform object detection on. You can specify the image path directly in the code or use user input for dynamic image selection.

4. Execute the code, and it will display the input image with bounding boxes around the detected dogs and cats, along with their respective confidence scores.

## Pre-trained Model

The Faster R-CNN model with a ResNet-50 backbone used in this project is pre-trained on the COCO dataset. You can load the pre-trained model directly using `torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)`.

## Customization

- You can adjust the confidence threshold for object detection by changing the `detection_threshold` parameter.
- Modify the input image path according to your dataset or input requirements.
- The code currently saves the output image with detected objects in the `outputs/` directory. You can customize the save location or disable saving as needed.

## Example

Here's a simple example of how to use the provided code:

```python
image_path = '/content/drive/MyDrive/Colab Notebooks/Dog&Cat.jpeg'

# Run object detection
boxes, classes, labels = predict(image_path, model, device, detection_threshold=0.8)

# Display and save the output image
image = draw_boxes(boxes, classes, labels, image)
cv2_imshow(image)
save_name = f"{image_path.split('/')[-1].split('.')[0]}_output"
cv2.imwrite(f"outputs/{save_name}.jpg", image)
