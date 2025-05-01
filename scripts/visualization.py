import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import pandas as pd
import matplotlib.pyplot as plt


with open('results/training_data2.pkl', 'rb') as f:
    data = pickle.load(f)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1,)

    plt.plot(data.get('accuracy', []), label='Training Accuracy')
    plt.plot(data.get('val_accuracy', []), label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)


    plt.subplot(1, 2, 2)
    plt.plot(data.get('loss', []), label='Training Loss')
    plt.plot(data.get('val_loss', []), label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()