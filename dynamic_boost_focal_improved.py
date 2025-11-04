# ============================================================
# IMPROVED VERSION: ADAPTIVE WEIGHTS + VALIDATION + AUGMENTATION
# Improvements:
# 1. Per-class metrics monitoring
# 2. Validation check in self-training
# 3. Adaptive weight calculation (no hard-coded)
# 4. Early stopping
# 5. Simple augmentation
# ============================================================
import os, shutil, random, gc, json, pickle, warnings
from pathlib import Path
from collections import Counter
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
import timm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, balanced_accuracy_score
from PIL import Image, ImageOps
warnings.filterwarnings("ignore")

# --------------------- CONFIG ---------------------
class CFG:
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_ROOT = "/kaggle/input/data-mining-action-2025"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train", "train")
    TEST_DIR = os.path.join(DATA_ROOT, "test", "test")
    WORKDIR = "/kaggle/working"
    CLEAN_DIR = os.path.join(WORKDIR, "train_clean")
    LABELED_CSV = "/kaggle/input/labeled-action/gabungan_label.csv"
    SUBMISSION = os.path.join(WORKDIR, "submission.csv")
    FEATURE_CACHE = os.path.join(WORKDIR, "train_features.npz")
    
    IMG_SIZE = 448
    BATCH = 64
    DINO_MODEL = "vit_giant_patch14_dinov2.lvd142m"
    
    # Self-training
    PSEUDO_EPOCHS = 8
    PSEUDO_LR = 5e-4
    CONF_THRESH = 0.86
    MAX_PER_CLASS = 300
    
    # Final training
    EPOCHS = 45
    LR = 8e-4
    DROPOUT = 0.55
    LABEL_SMOOTH = 0.08
    
    # NEW: Adaptive weights config
    WEIGHT_ALPHA = 1.3  # Smoothing factor for adaptive weights
    
    # NEW: Early stopping config
    EARLY_STOP_PATIENCE = 5
    EARLY_STOP_MIN_DELTA = 0.001

# --------------------- SEED & UTILS ---------------------
def seed_everything():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
seed_everything()

def cleanup():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# --------------------- FOCAL LOSS ---------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none', label_smoothing=label_smoothing)
    
    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# --------------------- IMAGE CLEANING ---------------------
def is_noise(img: Image.Image) -> bool:
    arr = np.array(img)
    return (arr.mean() > 245 and arr.std() < 5) or np.all(arr > 240)

def clean_images():
    shutil.rmtree(CFG.CLEAN_DIR, ignore_errors=True)
    os.makedirs(CFG.CLEAN_DIR, exist_ok=True)
    paths = list(Path(CFG.TRAIN_DIR).rglob("*"))
    paths = [p for p in paths if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp",".jfif"}]
    kept = 0
    print(f"Using IMG_SIZE: {CFG.IMG_SIZE}x{CFG.IMG_SIZE}")
    for p in tqdm(paths, desc="Cleaning"):
        try:
            img = Image.open(p).convert("RGB")
            img = ImageOps.exif_transpose(img).resize((CFG.IMG_SIZE, CFG.IMG_SIZE), Image.LANCZOS)
            if not is_noise(img):
                rel = p.relative_to(CFG.TRAIN_DIR)
                dst = Path(CFG.CLEAN_DIR) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                img.save(dst, quality=95, optimize=True)
                kept += 1
        except Exception:
            pass
    print(f"Kept: {kept}/{len(paths)}")
clean_images()

# --------------------- DINOv2 FEATURES ---------------------
norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

# NEW: Transform WITHOUT augmentation for feature extraction
transform_base = T.Compose([T.ToTensor(), norm])

# NEW: Transform WITH augmentation (will be applied during training if needed)
transform_aug = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.ToTensor(),
    norm
])

class ImgDS(Dataset):
    def __init__(self, root, transform=None):
        exts = {".jpg",".jpeg",".png",".bmp",".webp",".jfif"}
        self.paths = sorted([p for p in Path(root).rglob("*") if p.suffix.lower() in exts])
        self.transform = transform if transform else transform_base
    
    def __len__(self): 
        return len(self.paths)
    
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = ImageOps.exif_transpose(img).resize((CFG.IMG_SIZE, CFG.IMG_SIZE), Image.LANCZOS)
        return self.transform(img), self.paths[i].name

# NEW: Feature extraction with caching
def extract_features():
    # Check cache first
    if os.path.exists(CFG.FEATURE_CACHE):
        print(f"Loading cached features from {CFG.FEATURE_CACHE}")
        data = np.load(CFG.FEATURE_CACHE)
        return data['features'], data['names'].tolist()
    
    print("Extracting features (this will be cached)...")
    ds = ImgDS(CFG.CLEAN_DIR, transform=transform_base)
    dl = DataLoader(ds, batch_size=CFG.BATCH, shuffle=False, num_workers=4, pin_memory=True)
    model = timm.create_model(
        CFG.DINO_MODEL,
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True
    ).eval().to(CFG.DEVICE)
    
    feats, names = [], []
    with torch.no_grad():
        for x, n in tqdm(dl, desc="Extracting"):
            f = model(x.to(CFG.DEVICE)).cpu().numpy()
            feats.append(f)
            names.extend(n)
    
    feats = np.concatenate(feats)
    
    # Save to cache
    np.savez_compressed(CFG.FEATURE_CACHE, features=feats, names=names)
    print(f"Features cached to {CFG.FEATURE_CACHE}")
    
    return feats, names

FEATS, NAMES = extract_features()
name2idx = {n:i for i,n in enumerate(NAMES)}
print(f"Features: {FEATS.shape}")

# --------------------- LOAD SEED ---------------------
df_seed = pd.read_csv(CFG.LABELED_CSV)
df_seed = df_seed[df_seed["file_name"].isin(NAMES)]
classes = sorted(df_seed["class"].unique())
cls2id = {c:i for i,c in enumerate(classes)}
id2cls = {i:c for c,i in cls2id.items()}
print(f"Classes: {len(classes)} | Seed: {len(df_seed)}")

X_seed = FEATS[[name2idx[f] for f in df_seed["file_name"]]]
y_seed = np.array([cls2id[c] for c in df_seed["class"]])

# --------------------- MLP CLASSIFIER ---------------------
class EnhancedClassifier(nn.Module):
    def __init__(self, input_dim=1536, num_classes=15, dropout=0.55):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 768)
        self.bn0 = nn.BatchNorm1d(768)
        self.block1 = nn.Sequential(
            nn.Linear(768,768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(768,768), nn.BatchNorm1d(768)
        )
        self.trans1 = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(768,384))
        self.block2 = nn.Sequential(
            nn.Linear(384,384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(384,384), nn.BatchNorm1d(384)
        )
        self.trans2 = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(384,192))
        self.block3 = nn.Sequential(
            nn.Linear(192,192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(dropout*0.7),
            nn.Linear(192,192), nn.BatchNorm1d(192)
        )
        self.classifier = nn.Sequential(
            nn.ReLU(), nn.Dropout(dropout*0.5), nn.Linear(192, num_classes)
        )
    
    def forward(self, x):
        x = self.bn0(self.input_proj(x))
        identity = x
        x = self.block1(x) + identity
        x = self.trans1(x)
        identity = x
        x = self.block2(x) + identity
        x = self.trans2(x)
        identity = x
        x = self.block3(x) + identity
        return self.classifier(x)

# NEW: Adaptive class weight calculation
def get_adaptive_weights(y, alpha=1.3):
    """
    Calculate adaptive class weights based on frequency
    alpha: smoothing factor (1.0 = linear, >1.0 = more aggressive)
    """
    counts = np.bincount(y)
    total = len(y)
    weights = (total / (len(counts) * counts)) ** alpha
    
    # Normalize to mean = 1.0
    weights = weights / weights.mean()
    
    print(f"\n{'='*60}")
    print("Adaptive Class Weights:")
    print(f"{'='*60}")
    for i, w in enumerate(weights):
        class_name = id2cls.get(i, f"Class_{i}")
        count = counts[i] if i < len(counts) else 0
        print(f"{class_name:20s} | Count: {count:4d} | Weight: {w:.3f}")
    print(f"{'='*60}\n")
    
    return torch.FloatTensor(weights)

# NEW: Per-class metrics evaluation
def evaluate_per_class(model, X_val, y_val, id2cls):
    """Evaluate model and print per-class metrics"""
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_val).to(CFG.DEVICE))
        preds = logits.argmax(1).cpu().numpy()
    
    print(f"\n{'='*60}")
    print("Per-Class Metrics:")
    print(f"{'='*60}")
    print(classification_report(
        y_val, 
        preds, 
        target_names=[id2cls[i] for i in sorted(id2cls.keys())],
        digits=4
    ))
    print(f"{'='*60}\n")
    
    # Calculate balanced accuracy
    bal_acc = balanced_accuracy_score(y_val, preds)
    return bal_acc, preds

# NEW: Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0.0
        self.should_stop = False
    
    def __call__(self, val_acc):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered! No improvement for {self.patience} checks.")
                self.should_stop = True
                return True
            return False

# --------------------- TRAINER CLASS ---------------------
class Trainer:
    def __init__(self, model, lr, weight=None):
        self.model = model.to(CFG.DEVICE)
        self.crit = FocalLoss(alpha=weight, gamma=2.0, label_smoothing=CFG.LABEL_SMOOTH)
        self.opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.best_state = None
        self.best_acc = 0.0
        self.early_stopping = EarlyStopping(
            patience=CFG.EARLY_STOP_PATIENCE, 
            min_delta=CFG.EARLY_STOP_MIN_DELTA
        )
    
    def fit_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(CFG.DEVICE), y.to(CFG.DEVICE)
                if train: self.opt.zero_grad()
                logits = self.model(x)
                loss = self.crit(logits, y)
                if train:
                    loss.backward()
                    self.opt.step()
                loss_sum += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += x.size(0)
        return loss_sum/total, correct/total
    
    def train(self, X, y, epochs, val_split=0.15, verbose=True):
        idx = np.arange(len(X))
        tr_idx, val_idx = train_test_split(idx, test_size=val_split, stratify=y, random_state=CFG.SEED)
        tr_dl = DataLoader(
            TensorDataset(torch.FloatTensor(X[tr_idx]), torch.LongTensor(y[tr_idx])), 
            batch_size=CFG.BATCH, 
            shuffle=True
        )
        val_dl = DataLoader(
            TensorDataset(torch.FloatTensor(X[val_idx]), torch.LongTensor(y[val_idx])), 
            batch_size=CFG.BATCH
        )
        
        for ep in range(1, epochs+1):
            tr_loss, tr_acc = self.fit_epoch(tr_dl, train=True)
            val_loss, val_acc = self.fit_epoch(val_dl, train=False)
            
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_state = {k:v.cpu().clone() for k,v in self.model.state_dict().items()}
            
            if verbose:
                print(f"Ep {ep:02d} | TrAcc: {tr_acc:.4f} | ValAcc: {val_acc:.4f} | Best: {self.best_acc:.4f}")
            
            # NEW: Early stopping check
            if self.early_stopping(val_acc):
                print(f"Stopping at epoch {ep}/{epochs}")
                break
        
        if self.best_state:
            self.model.load_state_dict({k:v.to(CFG.DEVICE) for k,v in self.best_state.items()})
        
        return self.model, X[val_idx], y[val_idx]

# --------------------- SELF-TRAINING WITH VALIDATION ---------------------
X_lab, y_lab = X_seed.copy(), y_seed.copy()
unlabeled_idx = np.array([i for i,n in enumerate(NAMES) if n not in df_seed["file_name"].values])

# NEW: Adaptive weights instead of hard-coded
weights = get_adaptive_weights(y_lab, alpha=CFG.WEIGHT_ALPHA).to(CFG.DEVICE)

# Initial model training
print(f"\n{'='*60}")
print("INITIAL MODEL TRAINING")
print(f"{'='*60}\n")
model = EnhancedClassifier(num_classes=len(classes), dropout=CFG.DROPOUT)
trainer = Trainer(model, lr=CFG.PSEUDO_LR, weight=weights)
model, X_val_init, y_val_init = trainer.train(X_lab, y_lab, epochs=CFG.PSEUDO_EPOCHS)

# NEW: Evaluate initial model with per-class metrics
print(f"\n{'='*60}")
print("INITIAL MODEL EVALUATION")
print(f"{'='*60}")
bal_acc_init, _ = evaluate_per_class(model, X_val_init, y_val_init, id2cls)
print(f"Initial Balanced Accuracy: {bal_acc_init:.4f}")

# Self-training loops with validation
prev_bal_acc = bal_acc_init
for rnd in range(5):  # Increased from 3 to 5
    print(f"\n{'='*60}")
    print(f"SELF-TRAINING ROUND {rnd+1}")
    print(f"{'='*60}\n")
    
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(unlabeled_idx), CFG.BATCH):
            idx = unlabeled_idx[i:i+CFG.BATCH]
            p = torch.softmax(model(torch.FloatTensor(FEATS[idx]).to(CFG.DEVICE)), dim=1).cpu().numpy()
            probs.append(p)
    probs = np.concatenate(probs)
    conf, pred = probs.max(1), probs.argmax(1)
    
    # Select pseudo-labels
    selected = []
    for c in range(len(classes)):
        mask = (pred == c) & (conf >= CFG.CONF_THRESH)
        idx_c = np.where(mask)[0]
        if len(idx_c) == 0: 
            continue
        idx_c = idx_c[np.argsort(-conf[idx_c])]
        take = min(CFG.MAX_PER_CLASS, len(idx_c))
        selected.extend(idx_c[:take])
    
    if len(selected) < 30:
        print(f"⚠️ Only {len(selected)} samples selected, stopping self-training")
        break
    
    # Add pseudo-labels
    sel_abs = unlabeled_idx[selected]
    X_lab = np.vstack([X_lab, FEATS[sel_abs]])
    y_lab = np.concatenate([y_lab, pred[selected]])
    unlabeled_idx = np.delete(unlabeled_idx, selected)
    
    print(f"Added: {len(selected)} pseudo-labels | Total labeled: {len(X_lab)}")
    
    # NEW: Recalculate adaptive weights
    weights = get_adaptive_weights(y_lab, alpha=CFG.WEIGHT_ALPHA).to(CFG.DEVICE)
    
    # Retrain
    trainer = Trainer(model, lr=CFG.PSEUDO_LR*0.8, weight=weights)
    model, X_val_round, y_val_round = trainer.train(
        X_lab, 
        y_lab, 
        epochs=CFG.PSEUDO_EPOCHS//2,
        verbose=False
    )
    
    # NEW: Validate on original validation set
    bal_acc_round, _ = evaluate_per_class(model, X_val_init, y_val_init, id2cls)
    
    print(f"\nRound {rnd+1} Balanced Accuracy: {bal_acc_round:.4f}")
    
    # NEW: Quality check - stop if accuracy drops significantly
    if bal_acc_round < prev_bal_acc - 0.02:
        print(f"⚠️ Model quality dropped from {prev_bal_acc:.4f} to {bal_acc_round:.4f}")
        print("Stopping self-training to prevent error propagation")
        break
    
    prev_bal_acc = bal_acc_round
    cleanup()

# --------------------- FINAL TRAINING ---------------------
print(f"\n{'='*60}")
print(f"FINAL TRAINING: {CFG.EPOCHS} epochs")
print(f"{'='*60}\n")

# NEW: Final adaptive weights
weights = get_adaptive_weights(y_lab, alpha=CFG.WEIGHT_ALPHA).to(CFG.DEVICE)

final_model = EnhancedClassifier(num_classes=len(classes), dropout=CFG.DROPOUT)
trainer = Trainer(final_model, lr=CFG.LR, weight=weights)
final_model, X_val_final, y_val_final = trainer.train(X_lab, y_lab, epochs=CFG.EPOCHS)

# NEW: Final evaluation with per-class metrics
print(f"\n{'='*60}")
print("FINAL MODEL EVALUATION")
print(f"{'='*60}")
bal_acc_final, _ = evaluate_per_class(final_model, X_val_final, y_val_final, id2cls)
print(f"Final Balanced Accuracy: {bal_acc_final:.4f}")

# Save model & label map
torch.save(final_model.state_dict(), "final_mlp.pth")
pickle.dump({"cls2id": cls2id, "id2cls": id2cls}, open("label_map.pkl", "wb"))
print("Model & map saved.")

# --------------------- TEST INFERENCE ---------------------
print(f"\n{'='*60}")
print("TEST SET INFERENCE")
print(f"{'='*60}\n")


test_ds = ImgDS(CFG.TEST_DIR, transform=transform_base)
test_dl = DataLoader(test_ds, batch_size=CFG.BATCH, num_workers=4)
dino = timm.create_model(
    CFG.DINO_MODEL,
    pretrained=True,
    num_classes=0,
    dynamic_img_size=True
).eval().to(CFG.DEVICE)

X_test, test_names = [], []
with torch.no_grad():
    for x, n in tqdm(test_dl, desc="Test feats"):
        f = dino(x.to(CFG.DEVICE)).cpu().numpy()
        X_test.append(f)
        test_names.extend(n)
X_test = np.concatenate(X_test)

final_model.eval()
preds = []
with torch.no_grad():
    for i in range(0, len(X_test), CFG.BATCH):
        batch = torch.FloatTensor(X_test[i:i+CFG.BATCH]).to(CFG.DEVICE)
        out = final_model(batch)
        pred = out.argmax(1).cpu().numpy()
        preds.extend(pred)
preds = [id2cls[i] for i in preds]

# Submission
sub = pd.DataFrame({"ID": range(1, len(preds)+1), "label": preds})
sub.to_csv(CFG.SUBMISSION, index=False)

print(f"\n{'='*60}")
print(f"Submission saved: {CFG.SUBMISSION}")
print(f"{'='*60}")
print("\nTop 10 predicted classes:")
print(f"{'='*60}")
print(sub["label"].value_counts().head(10))
print(f"{'='*60}\n")
