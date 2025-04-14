
# YOLOv5 Object Detection (CPU-Only)

A simple script to perform object detection on images using YOLOv5 in **CPU-only** mode with PyTorch.  
**Persons** are highlighted in **green**, and all **other objects** in **red/pink** for easy visual distinction.

---

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/defaltastra/imgdetect.git
   cd imgdetect
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install opencv-python numpy
   ```

---

## 🚀 Usage

```bash
python detect.py --image path/to/your/image.jpg
```

### Optional Arguments:

- `--model` : YOLOv5 model variant to use (default: `yolov5s`)  
- `--conf` : Confidence threshold (default: `0.25`)  
- `--output` : Path to save the output image with annotations

### Example:

```bash
python detect.py --image input.jpg --model --conf 0.3 --output result.jpg
```

---

## 📦 Features

- CPU-only mode (no CUDA/GPU needed)
- Highlights detected:
  - **Persons** in ✅ **green**
  - **Objects** in 💖 **red/pink**
- Displays and optionally saves the annotated image
- Simple CLI interface

---



## 📄 License

This project is under the [MIT License](LICENSE).

---

## 🤝 Credits

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- PyTorch, OpenCV, NumPy

