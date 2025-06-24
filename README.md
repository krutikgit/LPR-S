# An adaptive License Plate Recognition Framework for Moving Vehicles

## Overview

This project presents an adaptive framework for **vehicle speed estimation** and **license plate recognition (LPR)** tailored to Indian road conditions. It is designed to assist intelligent traffic monitoring systems in automatically identifying over-speeding vehicles and extracting license plate characters from surveillance footage.

The framework combines state-of-the-art deep learning models and computer vision techniques to process video input and deliver high-accuracy results in real-time environments.

**[Reserach Paper Publication](https://pubs.aip.org/aip/acp/article-abstract/2794/1/020020/2914513/An-adaptive-license-plate-recognition-framework?redirectedFrom=fulltext)**

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

- 100+ real-world vehicle images and videos captured manually from Indian roads.
- 34,575 multi-styled license plate alphanumeric characters for training and evaluation.

---

## Performance

- **Speed Estimation Accuracy**: Absolute error of **0.98 km/h**
- **License Plate Recognition Accuracy**: Achieved **90%** accuracy using MobileNet

---

## Conclusion

This research contributes a scalable and adaptive framework for intelligent traffic systems, particularly for urban environments. The system not only estimates vehicle speed with high precision but also achieves high accuracy in license plate recognition across multiple styles and fonts.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{10.1063/5.0165667,
    author = {Singh, Guruanjan and Mehta, Krutik and Mishra, Apoorva and Chawla, Harnish and Shekokar, Narendra},
    title = {An adaptive license plate recognition framework for moving vehicles},
    journal = {AIP Conference Proceedings},
    volume = {2794},
    number = {1},
    pages = {020020},
    year = {2023},
    month = {10},
    issn = {0094-243X},
    doi = {10.1063/5.0165667},
    url = {https://doi.org/10.1063/5.0165667},
    eprint = {https://pubs.aip.org/aip/acp/article-pdf/doi/10.1063/5.0165667/18153233/020020\_1\_5.0165667.pdf},
}
```

---

## License

This repository is released under the [MIT License](LICENSE).
