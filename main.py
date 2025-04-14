import argparse
import torch
import cv2
import numpy as np
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description='Detect objects in images using YOLOv5 (CPU-only)')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='yolov5s', help='YOLOv5 model to use (default: yolov5s)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output', type=str, help='Path to save the output image (optional)')
    return parser.parse_args()

def detect_objects(image_path, model_name, conf_threshold):

    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  

    print(f"Loading YOLOv5 model: {model_name}")
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, force_reload=True)

    model.cpu()

    model.conf = conf_threshold

    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    results = model(img_rgb)
    end_time = time.time()

    df = results.pandas().xyxy[0]  
    detection_count = len(df)

    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        class_name = row['name']
        label = f"{class_name} {conf:.2f}"

        if class_name == 'person':
            color = (0, 255, 0) 
        else:
            color = (147, 20, 255) 

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


    print(f"Detection completed in {end_time - start_time:.2f} seconds")
    print(f"Found {detection_count} objects")

    if detection_count > 0:
        class_counts = df['name'].value_counts().to_dict()

        print("\nDetected objects:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")

    return img, detection_count

def main():
    args = parse_arguments()

    try:

        print("Using CPU-only mode for PyTorch")

        annotated_img, _ = detect_objects(args.image, args.model, args.conf)

        cv2.imshow("YOLOv5 Detection", annotated_img)
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if args.output:
            cv2.imwrite(args.output, annotated_img)
            print(f"Output image saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()