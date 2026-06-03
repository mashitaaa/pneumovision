# 1. IMPORT LIBRARIES & SETTING DIRECTORY
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV2
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
from pathlib import Path

# Path dataset Kaggle
BASE_DIR = Path('/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray')
TRAIN_DIR = BASE_DIR / 'train'
TEST_DIR  = BASE_DIR / 'test'

IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 32
EPOCHS_FREEZE    = 10   # Fase 1: Semua base layer di-freeze
EPOCHS_FINETUNE  = 5    # Fase 2: Fine-tuning layer atas

print("TensorFlow Version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

# 2. FUNGSI HELPER: BUAT GENERATOR
def make_generators(preprocess_fn):
    """
    Buat train/val/test generator dengan preprocessing yang sesuai tiap model.
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', subset='training', seed=42
    )
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', subset='validation', shuffle=False, seed=42
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )
    return train_gen, val_gen, test_gen

# 3. FUNGSI HELPER: HITUNG CLASS WEIGHT
def get_class_weights(train_gen):
    cls = train_gen.classes
    weights = compute_class_weight('balanced', classes=np.unique(cls), y=cls)
    return dict(zip(np.unique(cls), weights))

# 4. FUNGSI HELPER: BUAT MODEL PRETRAINED
def build_pretrained_model(base_model, freeze=True):
    """
    Bangun model dengan base pretrained.
    freeze=True  → Skenario A: Hanya top layer yang dilatih
    freeze=False → Skenario B: Seluruh layer dilatih (fine-tuning)
    """
    base_model.trainable = not freeze  # False = freeze, True = unfreeze

    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    lr = 1e-4 if freeze else 1e-5  # LR lebih kecil saat fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

# 5. FUNGSI HELPER: BUAT MODEL CNN SCRATCH
def build_cnn_scratch():
    """
    Model CNN sederhana dibangun dari nol (tanpa pretrained weights).
    Digunakan sebagai baseline pembanding.
    """
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

# 6. FUNGSI HELPER: TRAIN + FINE-TUNE
def train_with_finetune(model, train_gen, val_gen, class_weight_dict,
                        model_name, epochs_freeze, epochs_finetune,
                        is_scratch=False):
    """
    Training 2 fase:
    - Fase 1 (Skenario A): Base layer di-freeze, hanya top layer yang belajar
    - Fase 2 (Skenario B): Unfreeze seluruh layer, fine-tuning dengan LR kecil
    CNN Scratch hanya punya 1 fase (tidak ada base model untuk di-freeze).
    """
    checkpoint_freeze    = tf.keras.callbacks.ModelCheckpoint(
        f'{model_name}_freeze_best.keras', monitor='val_loss',
        save_best_only=True, verbose=0
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    print(f"\n{'='*60}")
    print(f"  [SKENARIO A - FREEZE] Training: {model_name}")
    print(f"{'='*60}")

    # --- FASE 1: FREEZE ---
    history_freeze = model.fit(
        train_gen,
        epochs=epochs_freeze,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=[checkpoint_freeze, early_stop],
        verbose=1
    )

    if is_scratch:
        # CNN Scratch tidak punya base model → langsung return
        return history_freeze, None, model

    # --- FASE 2: FINE-TUNING ---
    print(f"\n{'='*60}")
    print(f"  [SKENARIO B - FINE-TUNE] Training: {model_name}")
    print(f"{'='*60}")

    # Unfreeze semua layer base model (layer pertama di Sequential adalah base)
    model.layers[0].trainable = True

    # Re-compile dengan LR lebih kecil agar tidak overwrite pretrained weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )

    checkpoint_ft = tf.keras.callbacks.ModelCheckpoint(
        f'{model_name}_finetune_best.keras', monitor='val_loss',
        save_best_only=True, verbose=0
    )
    early_stop_ft = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )

    history_finetune = model.fit(
        train_gen,
        epochs=epochs_finetune,
        validation_data=val_gen,
        class_weight=class_weight_dict,
        callbacks=[checkpoint_ft, early_stop_ft],
        verbose=1
    )

    return history_freeze, history_finetune, model

# 7. FUNGSI EVALUASI MODEL
def evaluate_model(model, test_gen, model_name):
    """
    Evaluasi model pada test set dan kembalikan dict metrik.
    """
    test_gen.reset()
    y_true = test_gen.classes
    y_prob = model.predict(test_gen, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_true, y_pred,
                                   target_names=['NORMAL', 'PNEUMONIA'],
                                   output_dict=True)
    auc = roc_auc_score(y_true, y_prob)

    metrics = {
        'Model'     : model_name,
        'Accuracy'  : report['accuracy'],
        'Precision' : report['PNEUMONIA']['precision'],
        'Recall'    : report['PNEUMONIA']['recall'],
        'F1-Score'  : report['PNEUMONIA']['f1-score'],
        'AUC-ROC'   : auc
    }

    print(f"\n📊 Hasil Evaluasi [{model_name}]")
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))
    print(f"   AUC-ROC: {auc:.4f}")

    return metrics, y_true, y_prob, y_pred

# 8. FUNGSI PLOT CONFUSION MATRIX
def plot_confusion_matrix(y_true, y_pred, model_name, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'], ax=ax)
    ax.set_title(f'Confusion Matrix\n{model_name}', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

# 9. FUNGSI PLOT TRAINING HISTORY
def plot_history(history_freeze, history_finetune, model_name):
    """
    Plot accuracy & loss untuk kedua skenario (freeze & fine-tune).
    Jika fine-tune None (CNN Scratch), hanya tampilkan 1 skenario.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training History: {model_name}', fontsize=14, fontweight='bold')

    # Gabungkan history jika ada fine-tuning
    acc      = history_freeze.history['accuracy']
    val_acc  = history_freeze.history['val_accuracy']
    loss     = history_freeze.history['loss']
    val_loss = history_freeze.history['val_loss']

    if history_finetune:
        acc      += history_finetune.history['accuracy']
        val_acc  += history_finetune.history['val_accuracy']
        loss     += history_finetune.history['loss']
        val_loss += history_finetune.history['val_loss']
        split_epoch = len(history_freeze.history['accuracy'])
    else:
        split_epoch = None

    epochs_range = range(1, len(acc) + 1)

    # Plot Accuracy
    axes[0].plot(epochs_range, acc,     label='Train Accuracy',  color='steelblue')
    axes[0].plot(epochs_range, val_acc, label='Val Accuracy',    color='orange', linestyle='--')
    if split_epoch:
        axes[0].axvline(x=split_epoch, color='red', linestyle=':', label='Fine-tune Start')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot Loss
    axes[1].plot(epochs_range, loss,     label='Train Loss',  color='steelblue')
    axes[1].plot(epochs_range, val_loss, label='Val Loss',    color='orange', linestyle='--')
    if split_epoch:
        axes[1].axvline(x=split_epoch, color='red', linestyle=':', label='Fine-tune Start')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'history_{model_name.replace(" ", "_")}.png', dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✅ Plot history '{model_name}' disimpan.")

# 10. MAIN PIPELINE - JALANKAN SEMUA MODEL
all_results_freeze   = []   # Metrik Skenario A (Freeze saja)
all_results_finetune = []   # Metrik Skenario B (Setelah Fine-tune)
confusion_data       = []   # Untuk grid confusion matrix

# MODEL 1: EfficientNetB0
print("\n" + "="*60)
print("  MODEL 1: EfficientNetB0 (Pretrained ImageNet)")
print("="*60)

train_gen, val_gen, test_gen = make_generators(eff_preprocess)
cw = get_class_weights(train_gen)
print(f"Class weights: {cw}")

base_eff = EfficientNetB0(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))
model_eff = build_pretrained_model(base_eff, freeze=True)

hist_eff_freeze, hist_eff_ft, model_eff = train_with_finetune(
    model_eff, train_gen, val_gen, cw,
    'EfficientNetB0', EPOCHS_FREEZE, EPOCHS_FINETUNE
)

# Evaluasi Skenario A (ambil checkpoint freeze terbaik)
model_eff_freeze = tf.keras.models.load_model('EfficientNetB0_freeze_best.keras')
metrics_eff_A, yt, yp, ypred = evaluate_model(model_eff_freeze, test_gen, 'EfficientNetB0 [Freeze]')
all_results_freeze.append(metrics_eff_A)
confusion_data.append(('EfficientNetB0\n[Freeze]', yt, ypred))

# Evaluasi Skenario B (ambil checkpoint fine-tune terbaik)
model_eff_ft = tf.keras.models.load_model('EfficientNetB0_finetune_best.keras')
metrics_eff_B, yt, yp, ypred = evaluate_model(model_eff_ft, test_gen, 'EfficientNetB0 [Fine-tune]')
all_results_finetune.append(metrics_eff_B)
confusion_data.append(('EfficientNetB0\n[Fine-tune]', yt, ypred))

plot_history(hist_eff_freeze, hist_eff_ft, 'EfficientNetB0')

# Simpan model terbaik EfficientNetB0 Fine-tune sebagai model final
model_eff_ft.save('pneumonia_model.keras')

# MODEL 2: ResNet50
print("\n" + "="*60)
print("  MODEL 2: ResNet50 (Pretrained ImageNet)")
print("="*60)

train_gen, val_gen, test_gen = make_generators(resnet_preprocess)
cw = get_class_weights(train_gen)

base_res = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(224, 224, 3))
model_res = build_pretrained_model(base_res, freeze=True)

hist_res_freeze, hist_res_ft, model_res = train_with_finetune(
    model_res, train_gen, val_gen, cw,
    'ResNet50', EPOCHS_FREEZE, EPOCHS_FINETUNE
)

model_res_freeze = tf.keras.models.load_model('ResNet50_freeze_best.keras')
metrics_res_A, yt, yp, ypred = evaluate_model(model_res_freeze, test_gen, 'ResNet50 [Freeze]')
all_results_freeze.append(metrics_res_A)
confusion_data.append(('ResNet50\n[Freeze]', yt, ypred))

model_res_ft = tf.keras.models.load_model('ResNet50_finetune_best.keras')
metrics_res_B, yt, yp, ypred = evaluate_model(model_res_ft, test_gen, 'ResNet50 [Fine-tune]')
all_results_finetune.append(metrics_res_B)
confusion_data.append(('ResNet50\n[Fine-tune]', yt, ypred))

plot_history(hist_res_freeze, hist_res_ft, 'ResNet50')

# MODEL 3: MobileNetV2
print("\n" + "="*60)
print("  MODEL 3: MobileNetV2 (Pretrained ImageNet)")
print("="*60)

train_gen, val_gen, test_gen = make_generators(mobilenet_preprocess)
cw = get_class_weights(train_gen)

base_mob = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))
model_mob = build_pretrained_model(base_mob, freeze=True)

hist_mob_freeze, hist_mob_ft, model_mob = train_with_finetune(
    model_mob, train_gen, val_gen, cw,
    'MobileNetV2', EPOCHS_FREEZE, EPOCHS_FINETUNE
)

model_mob_freeze = tf.keras.models.load_model('MobileNetV2_freeze_best.keras')
metrics_mob_A, yt, yp, ypred = evaluate_model(model_mob_freeze, test_gen, 'MobileNetV2 [Freeze]')
all_results_freeze.append(metrics_mob_A)
confusion_data.append(('MobileNetV2\n[Freeze]', yt, ypred))

model_mob_ft = tf.keras.models.load_model('MobileNetV2_finetune_best.keras')
metrics_mob_B, yt, yp, ypred = evaluate_model(model_mob_ft, test_gen, 'MobileNetV2 [Fine-tune]')
all_results_finetune.append(metrics_mob_B)
confusion_data.append(('MobileNetV2\n[Fine-tune]', yt, ypred))

plot_history(hist_mob_freeze, hist_mob_ft, 'MobileNetV2')

# MODEL 4: CNN Scratch (Baseline)
print("\n" + "="*60)
print("  MODEL 4: CNN Scratch (Baseline - Tanpa Pretrained)")
print("="*60)

# CNN scratch pakai normalisasi sederhana (rescale)
scratch_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                     rotation_range=15, width_shift_range=0.1,
                                     height_shift_range=0.1, shear_range=0.1,
                                     zoom_range=0.1, horizontal_flip=True)
scratch_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen_sc = scratch_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='training', seed=42
)
val_gen_sc = scratch_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', subset='validation', shuffle=False, seed=42
)
test_gen_sc = scratch_test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary', shuffle=False
)

cw_sc = get_class_weights(train_gen_sc)
model_sc = build_cnn_scratch()

hist_sc_freeze, _, model_sc = train_with_finetune(
    model_sc, train_gen_sc, val_gen_sc, cw_sc,
    'CNN_Scratch', EPOCHS_FREEZE, EPOCHS_FINETUNE,
    is_scratch=True
)

model_sc_best = tf.keras.models.load_model('CNN_Scratch_freeze_best.keras')
metrics_sc, yt, yp, ypred = evaluate_model(model_sc_best, test_gen_sc, 'CNN Scratch')
# CNN Scratch masuk kedua tabel sebagai baseline
all_results_freeze.append(metrics_sc)
all_results_finetune.append({**metrics_sc, 'Model': 'CNN Scratch (Baseline)'})
confusion_data.append(('CNN Scratch\n[Baseline]', yt, ypred))

plot_history(hist_sc_freeze, None, 'CNN Scratch')

# 11. TABEL PERBANDINGAN HASIL
print("\n" + "="*60)
print("  RINGKASAN PERBANDINGAN SEMUA MODEL")
print("="*60)

df_freeze   = pd.DataFrame(all_results_freeze)
df_finetune = pd.DataFrame(all_results_finetune)

print("\n📋 Skenario A — FREEZE (Hanya Top Layer Dilatih):")
print(df_freeze.to_string(index=False))

print("\n📋 Skenario B — FINE-TUNE (Seluruh Layer Dilatih):")
print(df_finetune.to_string(index=False))

# 12. VISUALISASI PERBANDINGAN METRIK
metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
colors_freeze   = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
colors_finetune = ['#1565C0', '#B71C1C', '#1B5E20', '#E65100']

fig, axes = plt.subplots(1, len(metrics_cols), figsize=(20, 5))
fig.suptitle('Perbandingan Metrik: Skenario A (Freeze) vs B (Fine-tune)',
             fontsize=14, fontweight='bold')

for i, metric in enumerate(metrics_cols):
    x    = np.arange(len(df_freeze))
    w    = 0.35
    vals_A = df_freeze[metric].values
    vals_B = df_finetune[metric].values

    bars_A = axes[i].bar(x - w/2, vals_A, w, label='Freeze',    color=colors_freeze,   alpha=0.85)
    bars_B = axes[i].bar(x + w/2, vals_B, w, label='Fine-tune', color=colors_finetune, alpha=0.85)

    axes[i].set_title(metric, fontweight='bold')
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(df_freeze['Model'].str.replace(' [Freeze]','',regex=False),
                             rotation=20, ha='right', fontsize=8)
    axes[i].set_ylim(0.7, 1.02)
    axes[i].set_ylabel(metric if i == 0 else '')
    axes[i].legend(fontsize=7)
    axes[i].grid(axis='y', alpha=0.3)

    # Annotate nilai
    for bar in bars_A:
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003,
                     f'{bar.get_height():.3f}',
                     ha='center', va='bottom', fontsize=6, rotation=90)
    for bar in bars_B:
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003,
                     f'{bar.get_height():.3f}',
                     ha='center', va='bottom', fontsize=6, rotation=90)

plt.tight_layout()
plt.savefig('comparison_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Plot perbandingan metrik disimpan: comparison_metrics.png")

# 13. GRID CONFUSION MATRIX (SEMUA MODEL)
n_models = len(confusion_data)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Confusion Matrix - Semua Model & Skenario',
             fontsize=15, fontweight='bold')
axes_flat = axes.flatten()

for idx, (name, yt, ypred) in enumerate(confusion_data):
    plot_confusion_matrix(yt, ypred, name, axes_flat[idx])

# Sembunyikan subplot kosong jika ada
for idx in range(n_models, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Grid confusion matrix disimpan: confusion_matrices.png")

# 14. SIMPAN HASIL KE CSV & ASET FINAL
df_freeze.to_csv('results_freeze.csv', index=False)
df_finetune.to_csv('results_finetune.csv', index=False)
print("✅ Hasil evaluasi disimpan: results_freeze.csv & results_finetune.csv")

label_map = {"0": "NORMAL", "1": "PNEUMONIA"}
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)
print("✅ label_map.json berhasil dibuat.")

print("\n" + "="*60)
print("  SELESAI! Semua aset berhasil dibuat:")
print("  - pneumonia_model.keras        (model terbaik: EfficientNetB0 FT)")
print("  - label_map.json")
print("  - results_freeze.csv")
print("  - results_finetune.csv")
print("  - comparison_metrics.png")
print("  - confusion_matrices.png")
print("  - history_*.png (per model)")
print("="*60)
