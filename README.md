# Adaptive License Plate Recognition and Speed Estimation Framework

## Overview

This project presents an adaptive framework for **vehicle speed estimation** and **license plate recognition (LPR)** tailored to Indian road conditions. It is designed to assist intelligent traffic monitoring systems in automatically identifying over-speeding vehicles and extracting license plate characters from surveillance footage.

The framework combines state-of-the-art deep learning models and computer vision techniques to process video input and deliver high-accuracy results in real-time environments.

Published as a research paper in **AIP Conference Proceedings**:  
**[An adaptive license plate recognition framework for Indian roads](https://pubs.aip.org/aip/acp/article-abstract/2794/1/020020/2914513/An-adaptive-license-plate-recognition-framework?redirectedFrom=fulltext)**

---

## Key Features

- **Real-Time Object Detection** using **YOLOv5**
- **Vehicle Speed Estimation** via inter-frame displacement analysis
- **License Plate Detection** using **WPOD-NET**
- **Character Recognition** with deep learning models: **MobileNet**, **Xception**, **ResNet50**
- Evaluation on an Indian License Plate Dataset

---

## System Architecture

1. **Object Detection**: YOLOv5 is applied to processed video streams to detect fast-approaching vehicles.
2. **Speed Estimation**: Vehicle speed is estimated based on the distance covered over sequential frames.
3. **License Plate Recognition**:
   - **Localization**: WPOD-NET is used to detect and crop license plates from vehicle images.
   - **Character Recognition**: Pre-trained models (MobileNet, Xception, ResNet50) are evaluated for alphanumeric character extraction.

---

## Dataset

- 100 real-world vehicle images captured from Indian roads.
- 34,575 multi-styled license plate alphanumeric characters for training and evaluation.

---

## Performance

- **Speed Estimation Accuracy**: Absolute error of **0.98 km/h**
- **License Plate Recognition Accuracy**: Achieved **90%** accuracy using ResNet50

---

## Conclusion

This research contributes a scalable and adaptive framework for intelligent traffic systems, particularly for urban environments. The system not only estimates vehicle speed with high precision but also achieves high accuracy in license plate recognition across multiple styles and fonts.

---

## Future Work

- Incorporating **low-light and low-resolution surveillance footage**
- Extending the model to handle **partial and obscured license plates**
- Enhancing support for various regional license plate formats

---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{singh2023adaptive,
  title={An adaptive license plate recognition framework for moving vehicles},
  author={Singh, Guruanjan and Mehta, Krutik and Mishra, Apoorva and Chawla, Harnish and Shekokar, Narendra},
  booktitle={AIP Conference Proceedings},
  volume={2794},
  number={1},
  year={2023},
  organization={AIP Publishing}
}
```

---

## License

This repository is released under the [MIT License](LICENSE).
