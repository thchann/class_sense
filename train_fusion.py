"""Trainer application code

Author - Ximi
License - MIT
export LD_LIBRARY_PATH=/home/surbhi/anaconda3/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_DIR=/usr/lib/cuda
export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_DIR}
"""
import numpy as np
import tensorflow as tf
import os
import math
import keras
import config
from keras.callbacks import LearningRateScheduler
from keras import backend as K
from keras.saving import register_keras_serializable
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.models import load_model
from keras import Input, Model
from keras.layers import Dense, Dropout, GlobalAveragePooling1D, Concatenate, LayerNormalization, MultiHeadAttention
from keras.losses import SparseCategoricalCrossentropy

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score 
from sklearn.model_selection import StratifiedShuffleSplit

from collections import defaultdict

from data_prep import data_loader_fusion

scaler = RobustScaler()

@register_keras_serializable(package="Custom")      
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.int64)

        tp = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.not_equal(y_true, y_pred), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.not_equal(y_true, y_pred), tf.float32))  # Same as FP for macro f1

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-6)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-6)
        return 2 * ((precision * recall) / (precision + recall + 1e-6))

    def reset_states(self):
        for var in self.variables:
            var.assign(0)

class F1ScoreCallback(Callback):
    def __init__(self, val_data):
        super().__init__()
        self.validation_data = val_data
        self.classwise_acc = []  # ✅ Will store list of classwise accuracy per epoch

    def on_epoch_end(self, epoch, logs=None):
        val_x1, val_x2, val_y = self.validation_data
        y_pred = self.model.predict([val_x1, val_x2], verbose=0)
        y_pred_labels = np.argmax(y_pred, axis=1)

        # Compute and log macro F1
        f1 = f1_score(val_y, y_pred_labels, average="macro")
        print(f"\n🔍 Epoch {epoch + 1}: Val Macro F1 Score = {f1:.4f}")
        logs["val_f1"] = f1

        # Print full classification report
        print(classification_report(val_y, y_pred_labels, digits=3))

        # ✅ Per-class accuracy this epoch
        per_class_acc = []
        for cls in range(4):
            cls_mask = val_y == cls
            cls_acc = np.mean(y_pred_labels[cls_mask] == val_y[cls_mask]) if np.sum(cls_mask) > 0 else 0
            per_class_acc.append(cls_acc)
        self.classwise_acc.append(per_class_acc)

        print(f"Per-class accuracy (Epoch {epoch + 1}): {per_class_acc}")

        (unique, counts) = np.unique(y_pred_labels, return_counts=True)
        print(f"🧮 Class predictions: {dict(zip(unique, counts))}")

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')  # <-- Fix added
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return loss

def cosine_annealing(epoch, lr, T_max=200, eta_min=1e-6):
    return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2

lr_scheduler = LearningRateScheduler(cosine_annealing, verbose=1)

def build_fusion_model(hidden_dim=128, dropout_rate=0.3, num_heads=2, num_transformer_layers=1):
    # MARLIN: (batch, 1, 768)
    marlin_input = Input(shape=(1, 768), name="marlin_input")
    x1 = LayerNormalization()(marlin_input)
    x1 = GlobalAveragePooling1D()(x1)
    x1 = Dense(256, activation='relu')(x1)
    x1 = Dropout(dropout_rate)(x1)
    x1 = Dense(hidden_dim, activation='relu')(x1)

    # OpenFace: (batch, 9, 768)
    openface_input = Input(shape=(9, 768), name="openface_input")
    x2 = LayerNormalization()(openface_input)
    for _ in range(num_transformer_layers):
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=64)(x2, x2)
        x2 = LayerNormalization()(x2 + attn_out)  # residual connection
    x2 = GlobalAveragePooling1D()(x2)
    x2 = Dense(256, activation='relu')(x2)
    x2 = Dropout(dropout_rate)(x2)
    x2 = Dense(hidden_dim, activation='relu')(x2)

    # Fusion
    fused = Concatenate()([x1, x2])
    fused = Dense(hidden_dim, activation='relu')(fused)
    fused = Dropout(dropout_rate)(fused)
    output = Dense(4, activation='softmax')(fused)

    return Model(inputs=[marlin_input, openface_input], outputs=output)


BUFFER_SIZE = 100000
def make_ds(features, labels):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    return ds

def add_noise(X, std=0.05):
    noise = np.random.normal(0, std, X.shape)
    return X + noise

def get_best_weights_by_val_acc():
    def extract_val_acc(x):
        return float(x.split('acc')[-1].replace('.keras', ''))
    return sorted(os.listdir('./checkpoints/'), key=extract_val_acc)[-1]

def oversample_minority_classes(train_data, target_distribution=1000):
    grouped = defaultdict(list)
    for sample in train_data:
        marlin, openface, label = sample
        grouped[label].append(sample)

    oversampled = []
    for cls, samples in grouped.items():
        if len(samples) < target_distribution:
            repeats = target_distribution // len(samples) + 1
            oversampled.extend((samples * repeats)[:target_distribution])
        else:
            oversampled.extend(samples)
    return oversampled

def train(model_name, val=True):
    (x1, x2, y), (test_x1, test_x2, test_y) = data_loader_fusion(model_name, val=False)

    # ⚠️ Stratified train/val split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(splitter.split(x1, y))

    train_x1, val_x1 = x1[train_idx], x1[val_idx]
    train_x2, val_x2 = x2[train_idx], x2[val_idx]
    train_y, val_y = y[train_idx], y[val_idx]
    val_set = list(zip(val_x1, val_x2, val_y))

    # 🔄 Oversample minority classes
    train = list(zip(train_x1, train_x2, train_y))
    train = oversample_minority_classes(train)

    # ✅ Re-unpack after oversampling
    train_x1, train_x2, train_y = [], [], []
    for marlin, openface, label in train:
        train_x1.append(marlin)
        train_x2.append(openface)
        train_y.append(label)

    train_x1, train_x2, train_y = map(np.array, (train_x1, train_x2, train_y))


    # Split features again from oversampled list
    def split_features(data_set):
        x1, x2, y = [], [], []
        for marlin, openface, label in data_set:
            x1.append(marlin)
            x2.append(openface)
            y.append(label)

        x1 = np.array(x1)
        x2 = np.array(x2)
        y = np.array(y)

        if len(x1.shape) == 2:
            x1 = x1[:, np.newaxis, :]

        return x1, x2, y

    train_x1, train_x2, train_y = split_features(train)
    val_x1, val_x2, val_y = split_features(val_set)
    test_x1, test_x2, test_y = split_features(train)  # temp placeholder

    print("✅ Train MARLIN shape:", train_x1.shape) 
    print("✅ Train OpenFace shape:", train_x2.shape)
    print("Training label distribution:", np.bincount(train_y))

    # Convert to numpy arrays
    train_x1, train_x2, train_y = map(np.array, (train_x1, train_x2, train_y))

    train_x1 = add_noise(train_x1, std=0.07)
    train_x2 = add_noise(train_x2, std=0.07)

    val_x1, val_x2, val_y = map(np.array, (val_x1, val_x2, val_y))
    test_x1, test_x2, test_y = map(np.array, (test_x1, test_x2, test_y))

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_y),
        y=train_y
    )
    class_weights_dict = dict(enumerate(class_weights))
    print("Class Weights:", class_weights_dict)

    # Optionally cap very high weights

    print("train stats:")
    print(train_x1.shape, train_y.shape)
    print(train_x2.shape, train_y.shape)

    if val:
        print("val stats:")
        print(val_x1.shape, val_y.shape)
        print(val_x2.shape, val_y.shape)

    print("test stats:")
    print(test_x1.shape, test_y.shape)
    print(test_x2.shape, test_y.shape)

    model = build_fusion_model(hidden_dim=128, dropout_rate=0.3, num_heads=2, num_transformer_layers=2)

    model.compile(
        loss=SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy", F1Score()]
    )


    checkpoint_path = "checkpoints/" + model_name + ".epoch{epoch:02d}-acc{val_accuracy:.2f}.keras"

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )

    checkpoint_callback = ModelCheckpoint(
        filepath='checkpoints/fusion.epoch{epoch:02d}-acc{val_accuracy:.2f}.keras',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    callbacks = [
        checkpoint_callback,
        lr_scheduler,
    ]
    if val:
        callbacks.append(F1ScoreCallback([val_x1, val_x2, val_y]))

    checkpoint_dir = 'checkpoints'
    resume = True

    if resume and os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.keras')]
        if checkpoints:
            latest_checkpoint = max(
                checkpoints,
                key=lambda x: int(x.split("epoch")[1].split("-")[0])
            )
            model_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"🔁 Resuming from checkpoint: {model_path}")
            model = load_model(
                model_path,
                custom_objects={
                    "F1Score": F1Score
                }
            )
            initial_epoch = int(latest_checkpoint.split("epoch")[1].split("-")[0])
        else:
            print("⚠️ No checkpoint found. Training from scratch.")
            initial_epoch = 0
    else:
        print("🚀 Starting fresh training run.")
        initial_epoch = 0



    batch_size = 32  # or whatever size you want (try 16, 32, or 64)
    epochs = 200     # since this was also missing earlier


    model.fit (
        x=[train_x1, train_x2],
        y=train_y,
        validation_data=([val_x1, val_x2], val_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights_dict,
        verbose=1,
        initial_epoch=initial_epoch  # <-- added
    )


    y_pred_train = model.predict([train_x1, train_x2])
    print("Train prediction sample:", y_pred_train[:5])
    print("Train label sample:", train_y[:5])


    model.load_weights(f"checkpoints/{get_best_weights_by_val_acc()}")

    print("Evaluating on train set:")
    model.evaluate([train_x1, train_x2], train_y)

    print("Evaluating on valid set:")
    model.evaluate([val_x1, val_x2], val_y)

    print("Evaluating on test set:")
    model.evaluate([test_x1, test_x2], test_y)

    # Generate predictions on validation set
    y_pred_val = model.predict([val_x1, val_x2])
    y_pred_val = np.argmax(y_pred_val, axis=1)

    # Classification report
    print("Classification report (val): ")
    print(classification_report(val_y, y_pred_val))

    cm = confusion_matrix(val_y, y_pred_val)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3], yticklabels=[0,1,2,3])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix on Validation Set")
    plt.tight_layout()
    plt.show()

    
# import sys
if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    train(config.FUSION, val=True)