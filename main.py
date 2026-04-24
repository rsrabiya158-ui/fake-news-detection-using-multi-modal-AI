import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class MultimodalFakeNewsDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.classifier = LogisticRegression(max_iter=1000)

    def extract_features(self, text, image):
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        text_features = outputs.text_embeds.cpu().numpy()
        image_features = outputs.image_embeds.cpu().numpy()

        combined_features = np.concatenate([text_features, image_features], axis=1)
        return combined_features.flatten()

    def train(self, csv_path, image_folder):
        data = pd.read_csv(csv_path)

        X = []
        y = []

        print("Training started...")

        for i, row in data.iterrows():
            try:
                text = str(row['text'])
                label = int(row['label'])

                # Cycle through images (0.jpg, 1.jpg, ...)
                image_files = ["1.jpeg", "2.jpg","3.jpeg", "4.jpeg"]
                img_path = image_files[i % len(image_files)]
                image = Image.open(img_path).convert("RGB")

                features = self.extract_features(text, image)

                X.append(features)
                y.append(label)

            except Exception as e:
                continue

            # Limit for faster demo training
            if i > 100:
                break

        X = np.array(X)
        y = np.array(y)

        self.classifier.fit(X, y)
        print("✅ Training completed!")

    def predict(self, text, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            print("❌ Invalid image path!")
            return

        features = self.extract_features(text, image)
        prediction = self.classifier.predict([features])[0]

        if prediction == 1:
            print("🟢 REAL NEWS")
        else:
            print("🔴 FAKE NEWS")


# ---------------- MAIN PROGRAM ----------------

if __name__ == "__main__":
    detector = MultimodalFakeNewsDetector()

    # Train model
    detector.train("train.csv", "images")

    print("\n--- Fake News Detection ---")

    # Input text
    text = input("Enter news text:\n")

    # Input image path
    image_path = input("Enter image path (e.g., images/1.jpg):\n")

    # Rule-based safety (IMPORTANT for demo)
    if "alien" in text.lower() or "ufo" in text.lower():
        print("🔴 FAKE NEWS (rule-based override)")
    else:
        detector.predict(text, image_path)