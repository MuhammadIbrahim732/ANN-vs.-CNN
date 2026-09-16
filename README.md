# 🔢 Perceptron vs ANN vs CNN — MNIST Digit Classification

A deep learning project comparing three neural network architectures of increasing complexity — a single-layer **Perceptron**, a fully connected **Artificial Neural Network (ANN)**, and a **Convolutional Neural Network (CNN)** — on the classic MNIST handwritten digit dataset, to see how architecture choice affects performance on image data.

## 📊 Dataset

- **MNIST digit dataset** in CSV form (`train.csv` / `test.csv`), where each row is a flattened 28×28 grayscale image (784 pixel columns) with a `label` column (digit 0–9)
- Training set: **42,000 samples**, 785 columns (`label` + 784 pixels)
- Pixel values normalized to `[0, 1]` and reshaped back to `28×28` images for the models

## 🛠️ Workflow

1. **Import Libraries** — NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras
2. **Load Datasets** — `train.csv` and `test.csv`
3. **Data Analysis** — inspect shape, columns, and structure of the pixel data
4. **Preprocessing**
   - Split features (`X`) and labels (`y`) from the training data
   - Normalize pixel values (divide by 255)
   - Reshape flat pixel vectors into `28x28` image arrays
   - One-hot encode labels into 10 classes (`to_categorical`)
5. **Model Implementation** — three models trained for 5 epochs each (Adam/SGD optimizer, categorical cross-entropy loss):
   - **Perceptron** — single dense layer with softmax (no hidden layers)
   - **ANN** — fully connected network: `Flatten → Dense(128, relu) → Dense(64, relu) → Dense(10, softmax)`
   - **CNN** — convolutional network: `Conv2D(32) → MaxPooling2D → Conv2D(64) → MaxPooling2D → Flatten → Dense(128, relu) → Dense(10, softmax)`

## 📈 Results

Final training accuracy after 5 epochs:

| Model | Training Accuracy | Training Time (5 epochs) |
|---|---|---|
| Perceptron | 89.6% | ~13s |
| ANN | 98.5% | ~32s |
| **CNN** | **99.4%** | ~198s |

> The CNN achieves the highest accuracy by learning spatial features (edges, curves) directly from the 2D image structure through convolution and pooling — something the Perceptron and ANN can't exploit since they treat pixels as a flat vector. The trade-off is significantly higher training time per epoch.

## 🧰 Tech Stack

- **Python 3**
- **Pandas** & **NumPy** — data handling
- **Matplotlib** & **Seaborn** — visualization
- **Scikit-learn** — preprocessing utilities and metrics
- **TensorFlow / Keras** — building and training the Perceptron, ANN, and CNN models

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### Data

Place `train.csv` and `test.csv` (MNIST digit data in CSV format, e.g. from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/c/digit-recognizer/data)) in the working directory, or update the file paths in the notebook.

### Run

```bash
jupyter notebook ANN_vs__CNN.ipynb
```

Run all cells sequentially to reproduce training and results.

## 📁 Project Structure

```
.
├── ANN_vs__CNN.ipynb   # Main notebook: preprocessing, Perceptron/ANN/CNN training & comparison
└── README.md
```

## 🔮 Future Improvements

- Evaluate on a held-out validation/test split rather than only training accuracy
- Train for more epochs with early stopping to compare converged performance
- Add dropout/batch normalization to reduce overfitting risk
- Visualize learned CNN filters and misclassified digits
- Compare inference time and model size across architectures

## 👤 Author

**Muhammad Ibrahim**

- 📧 Email: [mibrahim.seng@gmail.com](mailto:mibrahim.seng@gmail.com)
- 💼 LinkedIn: [muhammad-ibrahim-python](https://www.linkedin.com/in/muhammad-ibrahim-python)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
