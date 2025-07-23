import os
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import numpy as np
from PIL import Image
import easyocr
import json
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_paragraphs_with_ocr(pdf_path):
    structured_blocks = []
    page_sizes = {}
    order = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, 0):
            page_sizes[str(page_idx)] = {"width": page.width, "height": page.height}
            # Try to extract normal lines first
            lines = list(page.extract_text_lines())
            if lines and any(line.get("text", "").strip() for line in lines) and len(lines)>=5:
                # (Use your previous merging logic)
                for line in lines:
                    block = {
                        "box": [line["x0"], line["top"], line["x1"], line["bottom"]],
                        "text": line["text"],
                        "page": page_idx,
                        "id": order,    
                        "order": order
                    }
                    structured_blocks.append(block)
                    order += 1
            else:
                # If no text, do OCR
                pil_img = page.to_image(resolution=300).original
                ocr_text = pytesseract.image_to_string(pil_img)
                block = {
                    "box": [0,0,page.width,page.height],
                    "text": ocr_text,
                    "page": page_idx,
                    "id": order,
                    "order": order
                }
                structured_blocks.append(block)
                order += 1

    result = {
        "contents": structured_blocks
    }
    return result


import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import time
import pickle

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class DocIENetDataset(Dataset):
    VALID_LABELS = {'title', 'section-title'}

    def __init__(self, input_files, output_files, tokenizer, max_length=64, label_encoder=None):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.order_counts_dict = {}

        if label_encoder is None:
            all_labels = set(['title', 'section-title'])
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(sorted(list(all_labels)))
        else:
            self.label_encoder = label_encoder

        self.link_counts = 0
        self.order_counts = 0
        for input_file, output_file in tqdm(zip(input_files, output_files), desc="Loading dataset"):
            self._process_file_pair(input_file, output_file)

        print(f"Dataset loaded with {len(self.data)} samples")
        print(f"Valid labels: {list(self.label_encoder.classes_)}")
        print(f"Total valid links: {self.link_counts}")
        print(f"Total items with reassigned orders: {self.order_counts}")
        print(f"Order distribution in dataset: {self.order_counts_dict}")

    def _process_file_pair(self, input_file, output_file):
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            with open(output_file, 'r', encoding='utf-8') as f:
                output_data = json.load(f)

            output_map = {item['id']: item for item in output_data.get('contents', [])}
            valid_ids = {item['id'] for item in output_data.get('contents', []) if item.get('label') in self.VALID_LABELS}

            for item in input_data.get('contents', []):
                item_id = item.get('id')
                if item_id not in output_map:
                    continue
                output_item = output_map[item_id]
                label = output_item.get('label', 'other')
                if label not in self.VALID_LABELS:
                    continue
                if label not in self.label_encoder.classes_:
                    continue

                page_num = item.get('page', 1)
                page_info = input_data.get('pages', {}).get(f'page{page_num}', 
                                                           input_data.get('pages', {}).get(str(page_num), 
                                                           {'width': 1000, 'height': 1000}))
                box = item.get('box', [0, 0, 0, 0])
                normalized_box = [
                    box[0] / page_info['width'],
                    box[1] / page_info['height'],
                    box[2] / page_info['width'],
                    box[3] / page_info['height']
                ]

                parent_id = 0
                for link in output_item.get('linking', []):
                    if len(link) == 2 and link[0] in valid_ids and link[1] in valid_ids and link[0] != 0 and link[1] != 0:
                        parent_id = link[1]
                        self.link_counts += 1
                        break

                if parent_id == 0:
                    for other_item in input_data.get('contents', []):
                        other_id = other_item.get('id')
                        if other_id != item_id and other_id in output_map:
                            other_label = output_map[other_id].get('label', 'other')
                            if other_label in self.VALID_LABELS:
                                other_box = other_item.get('box', [0, 0, 0, 0])
                                other_box_norm = [
                                    other_box[0] / page_info['width'],
                                    other_box[1] / page_info['height'],
                                    other_box[2] / page_info['width'],
                                    other_box[3] / page_info['height']
                                ]
                                dist = np.sqrt((normalized_box[0] - other_box_norm[0])**2 + (normalized_box[1] - other_box_norm[1])**2)
                                if dist < 0.2 and parent_id == 0:
                                    parent_id = other_id
                                    self.link_counts += 1
                                    break

                text = item.get('text', '') or '[EMPTY]'
                order = min(output_item.get('order', 0), 6)  # Cap at 6 based on GT distribution
                self.order_counts_dict[order] = self.order_counts_dict.get(order, 0) + 1

                self.data.append({
                    'text': text,
                    'box': normalized_box,
                    'page': page_num,
                    'id': item_id,
                    'label': label,
                    'parent_id': parent_id,
                    'order': order,
                    'file_idx': len(self.data)
                })
                self.order_counts += 1

        except Exception as e:
            print(f"Error processing {input_file}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'box': torch.tensor(item['box'], dtype=torch.float32),
            'page': torch.tensor(item['page'], dtype=torch.long),
            'label': item['label'],
            'id': item['id'],
            'parent_id': item['parent_id'],
            'order': torch.tensor(item['order'], dtype=torch.long),
            'file_idx': item['file_idx']
        }

def custom_collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    box = torch.stack([item['box'] for item in batch])
    page = torch.stack([item['page'] for item in batch])
    order = torch.stack([item['order'] for item in batch])
    labels = [item['label'] for item in batch]
    ids = [item['id'] for item in batch]
    file_idx = [item['file_idx'] for item in batch]
    parent_ids = [item['parent_id'] for item in batch]

    batch_size = len(batch)
    id_to_idx = {id_val: i for i, id_val in enumerate(ids)}
    linking_targets = torch.zeros(batch_size, dtype=torch.long)
    for i, parent_id in enumerate(parent_ids):
        if parent_id in id_to_idx:
            linking_targets[i] = id_to_idx[parent_id]
        else:
            linking_targets[i] = 0  # No parent (H1)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'box': box,
        'page': page,
        'order': order,
        'label': labels,
        'id': ids,
        'file_idx': file_idx,
        'linking_targets': linking_targets
    }

class CompactDocumentModel(nn.Module):
    def __init__(self, num_labels, model_name='prajjwal1/bert-tiny'):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        text_dim = self.text_encoder.config.hidden_size

        self.spatial_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        combined_dim = text_dim + 128
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )

        self.order_head = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)  # 7 classes for orders 0-6
        )

        self.feature_projector = nn.Linear(combined_dim, 256)

    def forward(self, input_ids, attention_mask, box, page):
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0]

        spatial_input = torch.cat([box, page.unsqueeze(-1).float()], dim=-1)
        spatial_features = self.spatial_encoder(spatial_input)

        combined_features = torch.cat([text_features, spatial_features], dim=-1)
        label_logits = self.classifier(combined_features)
        order_pred = self.order_head(combined_features)
        linking_features = self.feature_projector(combined_features)

        return {
            'label_logits': label_logits,
            'order_pred': order_pred,
            'features': combined_features,
            'linking_features': linking_features
        }

class ImprovedLinkingPredictor(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        self.similarity_net = nn.Sequential(
            nn.Linear(feature_dim * 3 + 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, features, boxes):
        batch_size, feature_dim = features.shape
        attended_features, _ = self.attention(
            features.unsqueeze(0), features.unsqueeze(0), features.unsqueeze(0)
        )
        attended_features = attended_features.squeeze(0)

        features_i = attended_features.unsqueeze(1).expand(-1, batch_size, -1)
        features_j = attended_features.unsqueeze(0).expand(batch_size, -1, -1)
        concat_features = torch.cat([features_i, features_j], dim=-1)
        product_features = features_i * features_j

        boxes_i = boxes.unsqueeze(1).expand(-1, batch_size, -1)
        boxes_j = boxes.unsqueeze(0).expand(batch_size, -1, -1)
        x_dist = (boxes_i[:, :, 0] - boxes_j[:, :, 0]).abs()
        y_dist = (boxes_i[:, :, 1] - boxes_j[:, :, 1]).abs()
        y_diff = boxes_i[:, :, 1] - boxes_j[:, :, 1]  # Vertical alignment
        spatial_dists = torch.stack([x_dist, y_dist, y_diff], dim=-1)

        pairs = torch.cat([concat_features, product_features, spatial_dists], dim=-1)
        linking_logits = self.similarity_net(pairs.reshape(-1, feature_dim * 3 + 3))
        linking_logits = linking_logits.reshape(batch_size, batch_size)

        mask = torch.eye(batch_size, device=linking_logits.device)
        linking_logits = linking_logits * (1 - mask) - 1e9 * mask  # Prevent self-linking
        return linking_logits

class DocumentTrainer:
    def __init__(self, model, linking_predictor, tokenizer, label_encoder, device):
        self.model = model
        self.linking_predictor = linking_predictor
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.device = device
        self.model.to(device)
        self.linking_predictor.to(device)

    def train(self, train_loader, val_loader, num_epochs=20, save_path="./models"):
        os.makedirs(save_path, exist_ok=True)
        optimizer = torch.optim.AdamW([
            {'params': self.model.parameters(), 'lr': 5e-5},
            {'params': self.linking_predictor.parameters(), 'lr': 1e-5}
        ], weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        label_criterion = FocalLoss(alpha=0.75, gamma=2.0)
        order_criterion = nn.CrossEntropyLoss()  # Will compute weights dynamically
        linking_criterion = nn.CrossEntropyLoss()

        best_val_f1 = 0.0
        best_linking_threshold = 0.5
        patience = 30
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            self.model.train()
            self.linking_predictor.train()
            train_loss, train_label_acc, train_linking_acc = 0, 0, 0
            num_batches = 0

            # Compute dynamic order weights
            order_counts = train_loader.dataset.order_counts_dict
            total_orders = sum(order_counts.values())
            order_weights = torch.tensor([total_orders / (order_counts.get(i, 1) * 7) for i in range(7)], dtype=torch.float).to(self.device)
            order_criterion.weight = order_weights

            for batch in tqdm(train_loader, desc="Training"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                box = batch['box'].to(self.device)
                page = batch['page'].to(self.device)
                order = batch['order'].to(self.device)
                labels = torch.tensor(self.label_encoder.transform(batch['label']), dtype=torch.long).to(self.device)
                linking_targets = batch['linking_targets'].to(self.device)

                outputs = self.model(input_ids, attention_mask, box, page)
                label_loss = label_criterion(outputs['label_logits'], labels)
                order_loss = order_criterion(outputs['order_pred'], order.long())
                linking_logits = self.linking_predictor(outputs['linking_features'], box)
                linking_loss = linking_criterion(linking_logits, linking_targets)

                total_loss = label_loss + 0.5 * order_loss + 5.0 * linking_loss

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.linking_predictor.parameters()), 0.5)
                optimizer.step()

                train_loss += total_loss.item()
                pred_labels = torch.argmax(outputs['label_logits'], dim=1)
                train_label_acc += (pred_labels == labels).float().mean().item()
                linking_preds = torch.argmax(linking_logits, dim=1)
                train_linking_acc += (linking_preds == linking_targets).float().mean().item()
                num_batches += 1

            scheduler.step()
            val_loss, val_metrics = self.validate(val_loader, label_criterion, order_criterion, linking_criterion, best_linking_threshold)
            best_linking_threshold = self.optimize_threshold(val_loader)

            print(f"Train Loss: {train_loss/num_batches:.4f}, Train Acc: {train_label_acc/num_batches:.4f}, Train Linking Acc: {train_linking_acc/num_batches:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}, Val F1: {val_metrics['f1']:.4f}, Val Linking Acc: {val_metrics['linking_acc']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Linking Threshold: {best_linking_threshold:.4f}")

            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                self.save_model(save_path)
                print("Saved best model!")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    def optimize_threshold(self, val_loader):
        self.model.eval()
        self.linking_predictor.eval()
        all_linking_probs, all_linking_targets = [], []
        thresholds = np.arange(0.3, 0.7, 0.05)
        best_acc, best_threshold = 0.0, 0.5

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                box = batch['box'].to(self.device)
                page = batch['page'].to(self.device)
                linking_targets = batch['linking_targets'].to(self.device)
                outputs = self.model(input_ids, attention_mask, box, page)
                linking_logits = self.linking_predictor(outputs['linking_features'], box)
                linking_probs = torch.sigmoid(linking_logits)
                all_linking_probs.extend(linking_probs.cpu().numpy())
                all_linking_targets.extend(linking_targets.cpu().numpy())

        all_linking_probs = np.array(all_linking_probs)
        all_linking_targets = np.array(all_linking_targets)
        for thresh in thresholds:
            linking_preds = np.argmax(all_linking_probs, axis=1)
            linking_preds[all_linking_probs.max(axis=1) < thresh] = 0  # No parent if below threshold
            acc = sum(p == t for p, t in zip(linking_preds, all_linking_targets)) / len(all_linking_targets)
            if acc > best_acc:
                best_acc = acc
                best_threshold = thresh

        return best_threshold

    def validate(self, val_loader, label_criterion, order_criterion, linking_criterion, linking_threshold):
        self.model.eval()
        self.linking_predictor.eval()
        val_loss, all_preds, all_labels, all_linking_preds, all_linking_targets = 0, [], [], [], []
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                box = batch['box'].to(self.device)
                page = batch['page'].to(self.device)
                order = batch['order'].to(self.device)
                labels = torch.tensor(self.label_encoder.transform(batch['label']), dtype=torch.long).to(self.device)
                linking_targets = batch['linking_targets'].to(self.device)

                outputs = self.model(input_ids, attention_mask, box, page)
                label_loss = label_criterion(outputs['label_logits'], labels)
                order_loss = order_criterion(outputs['order_pred'], order.long())
                linking_logits = self.linking_predictor(outputs['linking_features'], box)
                linking_loss = linking_criterion(linking_logits, linking_targets)

                total_loss = label_loss + 0.5 * order_loss + 5.0 * linking_loss
                val_loss += total_loss.item()
                pred_labels = torch.argmax(outputs['label_logits'], dim=-1)
                all_preds.extend(pred_labels.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                linking_preds = torch.argmax(linking_logits, dim=1)
                all_linking_preds.extend(linking_preds.cpu().numpy())
                all_linking_targets.extend(linking_targets.cpu().numpy())
                num_batches += 1

        val_loss = val_loss / num_batches
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        linking_acc = sum(p == t for p, t in zip(all_linking_preds, all_linking_targets)) / len(all_linking_targets)
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        return val_loss, {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'linking_acc': linking_acc,
            'accuracy': accuracy
        }

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), os.path.join(save_path, "document_model.pth"))
        torch.save(self.linking_predictor.state_dict(), os.path.join(save_path, "linking_predictor.pth"))
        with open(os.path.join(save_path, "label_encoder.pkl"), 'wb') as f:
            pickle.dump(self.label_encoder, f)

class FastInferenceModel:
    def __init__(self, model_path, linking_threshold=0.5):
        self.device = torch.device('cpu')
        self.linking_threshold = linking_threshold
        self.tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
        with open(os.path.join(model_path, "label_encoder.pkl"), 'rb') as f:
            self.label_encoder = pickle.load(f)

        self.model = CompactDocumentModel(num_labels=len(self.label_encoder.classes_))
        self.linking_predictor = ImprovedLinkingPredictor()
        self.model.load_state_dict(torch.load(os.path.join(model_path, "document_model.pth"), map_location='cpu'))
        self.linking_predictor.load_state_dict(torch.load(os.path.join(model_path, "linking_predictor.pth"), map_location='cpu'))
        self.model.eval()
        self.linking_predictor.eval()

    def predict(self, input_file):
        start_time = time.time()
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        items = data.get('contents', [])
        if not items:
            return []

        texts, boxes, pages, item_ids = [], [], [], []
        for item in items:
            text = item.get('text', '') or '[EMPTY]'
            texts.append(text)
            page_num = item.get('page', 1)
            page_info = data.get('pages', {}).get(f'page{page_num}', 
                                                 data.get('pages', {}).get(str(page_num), 
                                                 {'width': 1000, 'height': 1000}))
            box = item.get('box', [0, 0, 0, 0])
            normalized_box = [
                box[0] / page_info['width'],
                box[1] / page_info['height'],
                box[2] / page_info['width'],
                box[3] / page_info['height']
            ]
            boxes.append(normalized_box)
            pages.append(page_num)
            item_id = item.get('id', 0)
            if item_id == 0:
                continue
            item_ids.append(item_id)

        if not item_ids:
            return []

        encodings = self.tokenizer(texts, truncation=True, padding='max_length', max_length=64, return_tensors='pt')
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        box_tensor = torch.tensor(boxes, dtype=torch.float32).to(self.device)
        page_tensor = torch.tensor(pages, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, box_tensor, page_tensor)
            label_probs = torch.softmax(outputs['label_logits'], dim=-1)
            label_preds = torch.argmax(label_probs, dim=-1)
            order_preds = torch.argmax(outputs['order_pred'], dim=-1)
            linking_logits = self.linking_predictor(outputs['linking_features'], box_tensor)
            linking_probs = torch.sigmoid(linking_logits)

        try:
            with open(input_file.replace('input', 'output'), 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            gt_map = {item['id']: item for item in gt_data.get('contents', [])}
        except Exception as e:
            print(f"Error loading ground truth for {input_file}: {e}")
            gt_map = {}

        results = []
        for i in range(len(item_ids)):
            pred_label = self.label_encoder.inverse_transform([label_preds[i].item()])[0]
            if pred_label not in ['title', 'section-title']:
                continue
            max_prob, max_idx = torch.max(linking_probs[i], dim=0)
            parent_id = item_ids[max_idx] if max_prob.item() > self.linking_threshold and max_idx != i and item_ids[max_idx] != item_ids[i] else 0
            results.append({
                'id': item_ids[i],
                'label': pred_label,
                'parent_id': parent_id,
                'order': int(order_preds[i].item()),
                'confidence': label_probs[i, label_preds[i]].item(),
                'text': texts[i],
                'gt_text': gt_map.get(item_ids[i], {}).get('text', '')
            })

        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.2f} seconds")
        return results

class DocIENetPreprocessor:
    def __init__(self, dataset_path="../datasets/dochienet_dataset/dochienet_dataset/labels", input_dir="./input", output_dir="./output"):
        self.dataset_path = dataset_path
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def split_dataset(self):
        print("Processing DocIENet dataset...")
        if not os.path.exists(self.dataset_path):
            print(f"Dataset path {self.dataset_path} not found!")
            return [], []

        input_files, output_files, label_counts, order_counts = [], [], {}, {}
        print("Processing DocIENet dataset...")
        if not os.path.exists(self.dataset_path):
            print(f"Dataset path {self.dataset_path} not found!")
            return [], []

        input_files, output_files, label_counts, order_counts = [], [], {}, {}
        link_counts, discarded_links = 0, 0
        for filename in tqdm(os.listdir(self.dataset_path)):
            if filename.endswith('.json'):
                file_path = os.path.join(self.dataset_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Validate IDs
                    valid_items = [item for item in data.get('contents', []) if item.get('label') in DocIENetDataset.VALID_LABELS]
                    valid_ids = {item['id'] for item in valid_items}
                    if not valid_ids:
                        continue

                    for item in data.get('contents', []):
                        if item.get('id', 0) == 0:
                            continue
                        label = item.get('label', 'other')
                        label_counts[label] = label_counts.get(label, 0) + 1
                        order = min(item.get('order', 0), 6)  # Cap at 6
                        order_counts[order] = order_counts.get(order, 0) + 1

                    input_data = {"pages": data.get("pages", {}), "contents": []}
                    output_data = {"contents": []}

                    for item in data.get('contents', []):
                        if item.get('id', 0) == 0:
                            continue
                        input_item = {
                            "box": item.get("box", [0, 0, 0, 0]),
                            "text": item.get("text", "") or "[EMPTY]",
                            "page": item.get("page", 1),
                            "id": item.get("id", 0)
                        }
                        input_data["contents"].append(input_item)

                    valid_items = sorted(valid_items, key=lambda x: x.get('order', 0))
                    order_map = {item['id']: idx for idx, item in enumerate(valid_items)}

                    for item in valid_items:
                        parent_id = 0
                        for link in item.get("linking", []):
                            if len(link) != 2:
                                continue
                            source_id, target_id = link
                            if source_id not in valid_ids or target_id not in valid_ids or source_id == 0 or target_id == 0:
                                discarded_links += 1
                                continue
                            if source_id == target_id:
                                print(f"Invalid self-link in {filename}: {link}")
                                discarded_links += 1
                                continue
                            parent_id = target_id
                            link_counts += 1
                            break

                        output_item = {
                            "id": item.get("id", 0),
                            "label": item.get("label", "other"),
                            "text": item.get("text", "") or "[EMPTY]",
                            "parent_id": parent_id,
                            "order": min(order_map.get(item.get("id", 0), 0), 6)
                        }
                        output_data["contents"].append(output_item)

                    if input_data["contents"] and output_data["contents"]:
                        input_file_path = os.path.join(self.input_dir, filename)
                        with open(input_file_path, 'w', encoding='utf-8') as f:
                            json.dump(input_data, f, indent=2, ensure_ascii=False)
                        input_files.append(input_file_path)
                        output_file_path = os.path.join(self.output_dir, filename)
                        with open(output_file_path, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, indent=2, ensure_ascii=False)
                        output_files.append(output_file_path)

                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

        print(f"Processed {len(input_files)} files with valid labels")
        print(f"Input files saved to: {self.input_dir}")
        print(f"Output files saved to: {self.output_dir}")
        print(f"Label distribution: {label_counts}")
        print(f"Order distribution: {order_counts}")
        print(f"Total valid links: {link_counts}")
        print(f"Discarded links: {discarded_links}")
        return input_files, output_files

def main():
    print("=== Improved DocIENet Model Training (Titles & Section-Titles as Separate Labels) ===")
    preprocessor = DocIENetPreprocessor()
    input_files, output_files = preprocessor.split_dataset()

    if not input_files:
        print("No files with valid labels processed. Please check your dataset path.")
        return

    total_files = len(input_files)
    train_size = int(0.7 * total_files)
    val_size = int(0.1 * total_files)
    print(f"\nDataset split:")
    print(f"Total files: {total_files}")
    print(f"Train files: {train_size} (70%)")
    print(f"Validation files: {val_size} (10%)")
    print(f"Unused files: {total_files - train_size - val_size} (20%)")

    train_files = input_files[:train_size]
    train_output_files = output_files[:train_size]
    val_files = input_files[train_size:train_size + val_size]
    val_output_files = output_files[train_size:train_size + val_size]

    tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
    train_dataset = DocIENetDataset(train_files, train_output_files, tokenizer)
    val_dataset = DocIENetDataset(val_files, val_output_files, tokenizer, label_encoder=train_dataset.label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn, drop_last=True)

    print(f"\nFinal dataset statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of labels: {len(train_dataset.label_encoder.classes_)}")
    print(f"Labels: {list(train_dataset.label_encoder.classes_)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = CompactDocumentModel(num_labels=len(train_dataset.label_encoder.classes_))
    linking_predictor = ImprovedLinkingPredictor()
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in linking_predictor.parameters())
    print(f"Total model parameters: {total_params / 1e6:.1f}M")

    trainer = DocumentTrainer(model, linking_predictor, tokenizer, train_dataset.label_encoder, device)
    print("\nStarting training...")
    trainer.train(train_loader, val_loader, num_epochs=20)

    print("\nTraining completed!")
    print("\nTesting inference speed...")
    inference_model = FastInferenceModel("./models")

    if input_files:
        val = int(0.9 * len(input_files))
        test_file = input_files[val]
        print(f"Testing inference on {test_file}")
        results = inference_model.predict(test_file)
        with open(test_file.replace('input', 'output'), 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        gt_map = {item['id']: item for item in gt_data.get('contents', [])}

        print("\nComparison of predictions vs ground truth:")
        for pred in results:
            item_id = pred['id']
            gt = gt_map.get(item_id)
            if gt:
                print(f"ID: {item_id}")
                print(f"  Pred label: {pred['label']}, GT label: {gt['label']}")
                print(f"  Pred order: {pred['order']}, GT order: {gt['order']}")
                print(f"  Pred parent_id: {pred['parent_id']}")
                print(f"  GT parent_id: {gt['parent_id']}")
                print(f"  Text: {pred['text']}")
                print(f"  GT Text: {gt['text']}")
                print("---")
            else:
                print(f"ID {item_id} not found in ground truth.")

if __name__ == "__main__":
    main()

def extract_paragraphs_with_ocr(pdf_path):
    structured_blocks = []
    page_sizes = {}
    order = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, 0):
            page_sizes[str(page_idx)] = {"width": page.width, "height": page.height}
            lines = list(page.extract_text_lines())
            if lines and any(line.get("text", "").strip() for line in lines) and len(lines) >= 5:
                for line in lines:
                    box = [line["x0"], line["top"], line["x1"], line["bottom"]]
                    if any(coord < 0 or coord > max(page.width, page.height) for coord in box):
                        continue
                    block = {
                        "box": box,
                        "text": line["text"] or "[EMPTY]",
                        "page": page_idx,
                        "id": order,
                        "order": order
                    }
                    structured_blocks.append(block)
                    order += 1
            else:
                try:
                    pil_img = page.to_image(resolution=300).original
                    ocr_text = pytesseract.image_to_string(pil_img, lang='eng')
                except Exception as e:
                    print(f"OCR failed for page {page_idx} in {pdf_path}: {e}")
                    ocr_text = "[EMPTY]"
                block = {
                    "box": [0, 0, page.width, page.height],
                    "text": ocr_text or "[EMPTY]",
                    "page": page_idx,
                    "id": order,
                    "order": order
                }
                structured_blocks.append(block)
                order += 1

    return {
        "pages": page_sizes,
        "contents": structured_blocks
    }