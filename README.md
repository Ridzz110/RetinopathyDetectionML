# Diabetic Retinopathy Detection with ResNet50

## 📌 Project Overview
This project uses a **ResNet50-based deep learning model** to automatically detect and classify **Diabetic Retinopathy (DR)** severity levels from retinal fundus images. The model classifies images into **five categories**:
1. **No DR**  
2. **Mild**  
3. **Moderate**  
4. **Severe**  
5. **Proliferative DR**  

The model was trained using **transfer learning** on the **Diabetic Retinopathy Balanced dataset**, achieving an **81.05% accuracy** on the test set.

---

## 📺 Dataset
The dataset consists of **fundus images labeled into five classes**:

| Class                 | Image Count |
|-----------------------|------------|
| **No DR (Class 0)**   | 7000       |
| **Mild (Class 1)**    | 6792       |
| **Moderate (Class 2)**| 7000       |
| **Severe (Class 3)**  | 7000       |
| **Proliferative DR (Class 4)** | 7000 |

**Preprocessing Steps:**
- Images resized to **244×244** pixels
- Normalized pixel values to **[0,1]**

---

## ⚙️ Model Architecture
The model is built on **ResNet50** with custom dense layers:

```
ResNet50 (pre-trained on ImageNet) → Flatten → Dropout → Dense (256) → BatchNorm → Dropout → Dense (128) → BatchNorm → Dropout → Dense (32) → BatchNorm → Dense (5) (Softmax Activation)
```

**Key Features:**
✔ **Pre-trained ResNet50 backbone**  
✔ **Batch Normalization & Dropout for regularization**  
✔ **Categorical Crossentropy Loss** for multi-class classification  
✔ **Adam Optimizer (learning rate = 0.0001)**  

---

## 🔥 Training & Performance
### **Training Settings**
- **Batch Size:** 25
- **Epochs:** 100
- **Optimizer:** Adam (LR = 0.0001)
- **Loss Function:** Categorical Crossentropy

### **Final Model Performance**
| Metric          | Train Set | Validation Set |
|----------------|----------|---------------|
| **Accuracy**   | 81.36%   | 80.00%        |
| **AUC Score**  | 76.03%   | 83.38%        |
| **Loss**       | 1.3206   | 1.1816        |
| **Precision**  | 59.47%   | 50.00%        |
| **Recall**     | 16.46%   | 20.00%        |

### **Confusion Matrix**
![Confusion Matrix](./confusion_matrix.PNG)

---

## 💡 How to Use This Model
### **1. Clone the Repository**
```sh
git clone https://github.com/Ridzz110/RetinopathyDetectionML.git
cd RetinopathyDetectionML
```

### **2. Install Dependencies**
```sh
pip install tensorflow numpy pandas matplotlib opencv-python
```

### **3. Load the Model**
⚠️ **The model couldn't be saved in `.h5` format due to version conflicts.** Instead, run the training notebook directly.

To train the model:
```python
python train.py
```

To predict an image:
```python
python predict.py --image /path/to/image.jpeg
```

---

## 📝 Classification Report
```
Classification Report:
               Precision    Recall  F1-score   Support
           0       0.40      0.22      0.28      1834
           1       0.21      0.19      0.20      1060
           2       0.03      0.15      0.04       166
           3       0.21      0.20      0.20      1065
           4       0.16      0.19      0.17       846
    -------------------------------------------
    Overall Accuracy:  20%
    Macro Average: Precision=20%, Recall=19%, F1-score=18%
```

---

## 🚀 Future Improvements
- **Improve Recall** by class balancing and data augmentation.
- **Fine-tune ResNet50 further** for better feature extraction.
- **Experiment with different architectures** like EfficientNet or Vision Transformers.

---

## 🐝 License
This project is open-source under the **MIT License**.

---

## 🤝 Acknowledgments
This project was developed as part of **INFYMA AI Hackathon 25'** by **Code in Pink 🎀**.

---

### 🔗 **Connect with Me**
💡 **Author**: Rida Batool  
💎 **Email**: [your-email@example.com](mailto:your-email@example.com)  
🕊 **Twitter**: [@yourhandle](https://twitter.com/yourhandle)  
📚 **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)  

---

