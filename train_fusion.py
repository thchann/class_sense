import numpy as np
import tensorflow as tf
import os
import math
import keras
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import load_model
from keras import Input, Model
from keras.layers import Dense, Dropout, GlobalAveragePooling1D, Concatenate, LayerNormalization, MultiHeadAttention
from keras.losses import SparseCategoricalCrossentropy
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedShuffleSplit
from data_prep import data_loader_fusion
import config

def cosine_annealing(epoch, lr, T_max=200, eta_min=1e-6):
    return eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / T_max)) / 2

def build_fusion_model(hidden_dim=128, dropout_rate=0.4, num_heads=2, num_transformer_layers=1):
    marlin_input = Input(shape=(1, 768), name="marlin_input")
    x1 = LayerNormalization()(marlin_input)
    x1 = GlobalAveragePooling1D()(x1)
    x1 = Dense(256, activation='relu')(x1)
    x1 = Dropout(dropout_rate)(x1)
    x1 = Dense(hidden_dim, activation='relu')(x1)

    openface_input = Input(shape=(9, 768), name="openface_input")
    x2 = LayerNormalization()(openface_input)
    for _ in range(num_transformer_layers):
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=64)(x2, x2)
        x2 = LayerNormalization()(x2 + attn_out)
    x2 = GlobalAveragePooling1D()(x2)
    x2 = Dense(256, activation='relu')(x2)
    x2 = Dropout(dropout_rate)(x2)
    x2 = Dense(hidden_dim, activation='relu')(x2)

    fused = Concatenate()([x1, x2])
    fused = Dense(hidden_dim, activation='relu')(fused)
    fused = Dropout(dropout_rate)(fused)
    output = Dense(4, activation='softmax')(fused)

    return Model(inputs=[marlin_input, openface_input], outputs=output)

def oversample_minority_classes(train_data, target_distribution=1000):
    from collections import defaultdict
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

def train(model_name):
    (x1, x2, y), _ = data_loader_fusion(model_name, val=False)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(splitter.split(x1, y))

    train_x1, val_x1 = x1[train_idx], x1[val_idx]
    train_x2, val_x2 = x2[train_idx], x2[val_idx]
    train_y, val_y = y[train_idx], y[val_idx]

    train = list(zip(train_x1, train_x2, train_y))
    train = oversample_minority_classes(train)
    train_x1, train_x2, train_y = map(np.array, zip(*[(m, o, l) for m, o, l in train]))

    if len(train_x1.shape) == 2:
        train_x1 = train_x1[:, np.newaxis, :]
    if len(val_x1.shape) == 2:
        val_x1 = val_x1[:, np.newaxis, :]

    class_weights_dict = dict(enumerate(class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(train_y), y=train_y
    )))

    model = build_fusion_model()
    model.compile(
        loss=SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )

    checkpoint_callback = ModelCheckpoint(
        filepath='checkpoints/fusion.epoch29-acc.0.80.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    lr_scheduler = LearningRateScheduler(cosine_annealing, verbose=1)

    model.fit(
        x=[train_x1, train_x2],
        y=train_y,
        validation_data=([val_x1, val_x2], val_y),
        epochs=200,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=[checkpoint_callback, lr_scheduler],
        verbose=1
    )

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    train(config.FUSION)
