# ViViT: Video Vision Transformer for Video Classification

## Project Overview
This project implements **ViViT (Video Vision Transformer)**, a **pure Transformer-based model** for **video classification**. The model learns spatial and temporal representations using a **Tubelet Embedding** scheme and a **Transformer Encoder**. The implementation is based on Keras and TensorFlow, using the **OrganMNIST3D** dataset from MedMNIST.

## Dataset: OrganMNIST3D (MedMNIST)
The dataset used in this project is **OrganMNIST3D**, a **3D medical imaging dataset** from the **MedMNIST** collection. It consists of **11 organ classes** in MRI scans, making it an excellent benchmark for **3D medical video classification**.

- **Dataset Name:** OrganMNIST3D
- **Input Shape:** (28, 28, 28, 1) (3D images as video sequences)
- **Number of Classes:** 11
- **Training Samples:** Provided in **train_images.npy**
- **Validation Samples:** Provided in **val_images.npy**
- **Test Samples:** Provided in **test_images.npy**
- **Source:** MedMNIST 2D/3D datasets
- **Dataset Link:** [MedMNIST GitHub](https://github.com/MedMNIST/MedMNIST)

## Model Architecture
### 🔹 **Tubelet Embedding**
- Splits video frames into **small 3D patches**.
- Uses **Conv3D layers** to extract features from patches.

### 🔹 **Positional Encoding**
- Encodes the **order of patches** in the video sequence.
- Uses **learnable embeddings**.

### 🔹 **Transformer Encoder**
- **Multi-Head Self-Attention (MHSA)** to model long-range dependencies.
- **Feed-Forward MLP** with **GELU activation**.
- **Layer Normalization and Skip Connections**.

### 🔹 **Classification Head**
- **Global Average Pooling (GAP)** layer.
- **Dense layer with Softmax activation** for class prediction.

## Installation
To run this project, install the required dependencies:
```bash
pip install -q tensorflow keras numpy medmnist imageio ipywidgets
```

## Dataset Preparation
The dataset is automatically downloaded and processed using the following function:
```python
import medmnist
from keras.utils import get_file

def download_and_prepare_dataset(data_info: dict):
    data_path = get_file(origin=data_info["url"], md5_hash=data_info["MD5"])
    with np.load(data_path) as data:
        train_videos = data["train_images"]
        valid_videos = data["val_images"]
        test_videos = data["test_images"]
        train_labels = data["train_labels"].flatten()
        valid_labels = data["val_labels"].flatten()
        test_labels = data["test_labels"].flatten()
    return (train_videos, train_labels), (valid_videos, valid_labels), (test_videos, test_labels)
```

## Training the ViViT Model
Run the **experiment** to train the model using the following command:
```python
model = run_experiment()
```

### Hyperparameters:
- **Batch Size:** 32
- **Learning Rate:** 1e-4
- **Weight Decay:** 1e-5
- **Number of Layers:** 8
- **Number of Attention Heads:** 8
- **Epochs:** 60

## Model Evaluation
After training, evaluate the model on the **test set**:
```python
_, accuracy, top_5_accuracy = model.evaluate(testloader)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Top-5 Accuracy: {top_5_accuracy * 100:.2f}%")
```

## Visualization of Predictions
- The project generates **GIFs** of MRI sequences.
- Predictions are displayed using **ipywidgets**.

```python
import imageio
from IPython.display import display

def visualize_predictions(model, testloader, num_samples=25):
    testsamples, labels = next(iter(testloader))
    for i in range(num_samples):
        sample = np.reshape(testsamples[i].numpy(), (-1, 28, 28))
        gif = imageio.mimsave("temp.gif", (sample * 255).astype("uint8"), "GIF", fps=5)
        pred = np.argmax(model.predict(ops.expand_dims(testsample, axis=0))[0])
        print(f"Ground Truth: {labels[i]}, Prediction: {pred}")
        display(gif)
```

## Conclusion
- **ViViT successfully classifies 3D medical video data.**
- **Uses Transformer-based attention instead of CNNs.**
- **Can be extended to other video classification tasks.**

