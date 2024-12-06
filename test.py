import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
from Program.Utils.getData import Data  # Import class Data untuk pengelolaan dataset

# Fungsi untuk mengevaluasi performa model
def evaluate_model(model, data_loader):
    """
    Mengevaluasi model menggunakan data test dan menghitung berbagai metrik evaluasi.
    Args:
        model: Model yang akan dievaluasi
        data_loader: DataLoader untuk dataset test
    Returns:
        accuracy: Akurasi model
        precision: Precision rata-rata model
        recall: Recall rata-rata model
        f1: F1-Score rata-rata model
        auc: ROC-AUC model
        cm: Confusion Matrix
    """
    model.eval()  # Mengatur model ke mode evaluasi
    all_preds = []  # Untuk menyimpan semua prediksi
    all_labels = []  # Untuk menyimpan semua label asli
    all_probs = []  # Untuk menyimpan semua probabilitas prediksi

    with torch.no_grad():  # Mematikan autograd untuk efisiensi
        for src, trg in data_loader:
            src = src.permute(0, 3, 1, 2).float()  # Mengubah format data dari NHWC ke NCHW
            trg = torch.argmax(trg, dim=1)  # Mengubah label one-hot menjadi indeks label
            
            outputs = model(src)  # Hasil prediksi model
            probs = torch.softmax(outputs, dim=1)  # Menghitung probabilitas dari output model
            _, preds = torch.max(outputs, 1)  # Mengambil label prediksi dengan nilai probabilitas maksimum

            # Menyimpan hasil batch ke daftar
            all_preds.append(preds.cpu().numpy())  
            all_labels.append(trg.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    # Menggabungkan hasil dari semua batch
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Konversi label asli ke one-hot encoding
    unique_labels = np.unique(all_labels)  # Label unik dari dataset
    num_classes = len(unique_labels)
    all_labels_onehot = np.eye(num_classes)[all_labels]

    # Menyesuaikan probabilitas dengan label unik
    all_probs = all_probs[:, unique_labels]

    # Menghitung metrik evaluasi
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', labels=unique_labels, zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', labels=unique_labels, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', labels=unique_labels, zero_division=0)
    auc = roc_auc_score(all_labels_onehot, all_probs, multi_class='ovr', average='weighted')

    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)  # Membuat confusion matrix

    return accuracy, precision, recall, f1, auc, cm  # Mengembalikan hasil evaluasi


# Fungsi untuk visualisasi confusion matrix
def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """
    Membuat heatmap dari confusion matrix.
    Args:
        cm: Confusion matrix (numpy array)
        class_names: Nama-nama kelas
        save_path: Path untuk menyimpan gambar confusion matrix
    """
    plt.figure(figsize=(8, 6))  # Ukuran plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")  # Label sumbu X
    plt.ylabel("True Labels")  # Label sumbu Y
    plt.title("Confusion Matrix")  # Judul plot
    plt.savefig(save_path)  # Simpan gambar
    plt.close()  # Tutup plot untuk menghemat memori


# Fungsi utama program
def main():
    """
    Proses utama evaluasi model: memuat dataset, model, menghitung metrik, dan menampilkan hasil evaluasi.
    """
    BATCH_SIZE = 4  # Ukuran batch
    NUM_CLASSES = 6  # Jumlah kelas

    # Path dataset
    aug_path = "./Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "./Dataset/Original Images/Original Images/FOLDS/"

    # Memuat dataset
    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
    test_data = dataset.dataset_test  # Dataset test
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)  # DataLoader untuk test

    # Memuat model pre-trained Swin Transformer
    model = models.swin_t(weights="IMAGENET1K_V1")
    model.head = torch.nn.Linear(model.head.in_features, NUM_CLASSES)  # Menyesuaikan jumlah kelas
    model.load_state_dict(torch.load("trained_modelswin_t.pth"))  # Memuat parameter model terlatih
    model.eval()  # Mengatur model ke mode evaluasi

    # Evaluasi model pada data test
    accuracy, precision, recall, f1, auc, cm = evaluate_model(model, test_loader)

    # Menampilkan hasil evaluasi
    print("Evaluasi pada data test:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Membuat heatmap confusion matrix
    class_names = ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles", "Monkeypox"]  # Nama kelas
    plot_confusion_matrix(cm, class_names, save_path="./confusion_matrix.png")  # Visualisasi


# Menjalankan program utama
if __name__ == "__main__":
    main()