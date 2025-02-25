import os
from openimages.download import download_dataset
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from PIL import Image
from numpy import np

data_dir = "data"
number_for_samples = 334
classes = ["Orange", "Broccoli", "Lemon"]

if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print("Directory was created")

print("Downloading is starting...")
download_dataset(data_dir, classes, limit=number_for_samples)
print("Download is finished")

model = models.resnet50(pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.class_to_idx = {
            "orange": 950,
            "broccoli": 937,
            "lemon": 951
        }
        # Get image paths for each class
        self.classes = list(self.class_to_idx.keys())

        for class_name in self.classes:
            class_folder = os.path.join(root_dir, class_name, "images")
            label = self.class_to_idx[class_name]

            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# Create dataset and dataloader
dataset = CustomDataset(data_dir, transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    
    TN = np.sum((y_true == 0) & (y_pred == 0))
    
    FP = np.sum((y_true == 0) & (y_pred == 1))
    
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1

def evaluate_model_for_both_classes(model, dataloader, device, thresholds):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    class_indices = {
        "orange": 950,
        "broccoli": 937,
        "lemon": 951
    }

    results = {}

    for class_name, class_idx in class_indices.items():
        print(f"\nEvaluating for class: {class_name} (Index: {class_idx})")
        y_true = (all_labels == class_idx).astype(int)
        y_scores = all_probs[:, class_idx]

        class_results = []

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
            class_results.append((threshold, accuracy, precision, recall, f1))

        results[class_name] = class_results

    return results

thresholds = np.linspace(0, 1, 11)

results = evaluate_model_for_both_classes(model, dataloader, device, thresholds)

for class_name, class_results in results.items():
    print(f"\nResults for {class_name}:")
    print("Threshold\tAccuracy\tPrecision\tRecall\t\tF1 Score")
    for threshold, accuracy, precision, recall, f1 in class_results:
        print(f"{threshold:.2f}\t\t{accuracy:.4f}\t\t{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}")