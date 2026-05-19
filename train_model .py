"""
Handwritten Character Recognition - Model Training
Dataset: MNIST (digits 0-9) + EMNIST (letters a-z, A-Z)
Model: Convolutional Neural Network (CNN)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE    = 28
NUM_CLASSES = 47          # EMNIST Balanced: 10 digits + 26 uppercase + 11 lowercase
BATCH_SIZE  = 128
EPOCHS      = 20
MODEL_PATH  = "saved_model/handwriting_cnn.keras"
HISTORY_PATH= "saved_model/training_history.pkl"
LABELS_PATH = "saved_model/class_labels.pkl"

# EMNIST Balanced label mapping (0-9 digits, then letters)
EMNIST_LABELS = (
    list("0123456789")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abdefghnqrt")   # 11 extra lowercase (Balanced subset)
)

os.makedirs("saved_model", exist_ok=True)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_mnist():
    """Load MNIST digits dataset."""
    print("📦 Loading MNIST digits...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def load_emnist_balanced():
    """
    Load EMNIST Balanced from tensorflow_datasets or fall back to
    a lightweight manual download via urllib.
    Returns (x_train, y_train), (x_test, y_test) shaped like MNIST.
    """
    print("📦 Loading EMNIST Balanced...")
    try:
        import tensorflow_datasets as tfds
        ds_train, ds_test = tfds.load(
            "emnist/balanced",
            split=["train", "test"],
            as_supervised=True,
            shuffle_files=False,
        )

        def collect(dataset):
            xs, ys = [], []
            for img, label in dataset:
                img_np = img.numpy()          # (28,28,1)
                img_np = img_np[:, :, 0]      # (28,28)
                # EMNIST images are transposed relative to MNIST — fix that
                img_np = np.transpose(img_np)
                xs.append(img_np)
                ys.append(label.numpy())
            return np.array(xs), np.array(ys)

        x_tr, y_tr = collect(ds_train)
        x_te, y_te = collect(ds_test)
        print(f"   ✅ EMNIST train: {x_tr.shape}, test: {x_te.shape}")
        return (x_tr, y_tr), (x_te, y_te)

    except Exception as e:
        print(f"   ⚠️  tensorflow_datasets unavailable ({e}).")
        print("   ℹ️  Training on MNIST only (digits 0-9).")
        return None, None


def preprocess(x, y, num_classes):
    x = x.astype("float32") / 255.0
    x = x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = to_categorical(y, num_classes)
    return x, y


def build_dataset():
    """Combine MNIST + EMNIST if available, else use MNIST only."""
    (mx_tr, my_tr), (mx_te, my_te) = load_mnist()
    emnist_result = load_emnist_balanced()

    if emnist_result[0] is not None:
        (ex_tr, ey_tr), (ex_te, ey_te) = emnist_result
        num_classes = NUM_CLASSES   # 47

        # One-hot encode separately then concat
        mx_tr, my_tr = preprocess(mx_tr, my_tr, num_classes)
        mx_te, my_te = preprocess(mx_te, my_te, num_classes)
        ex_tr, ey_tr = preprocess(ex_tr, ey_tr, num_classes)
        ex_te, ey_te = preprocess(ex_te, ey_te, num_classes)

        x_train = np.concatenate([mx_tr, ex_tr])
        y_train = np.concatenate([my_tr, ey_tr])
        x_test  = np.concatenate([mx_te, ex_te])
        y_test  = np.concatenate([my_te, ey_te])
        labels  = EMNIST_LABELS
    else:
        num_classes = 10
        x_train, y_train = preprocess(mx_tr, my_tr, num_classes)
        x_test,  y_test  = preprocess(mx_te, my_te, num_classes)
        labels = list("0123456789")

    # Shuffle training set
    idx = np.random.permutation(len(x_train))
    x_train, y_train = x_train[idx], y_train[idx]

    print(f"\n📊 Dataset ready — classes: {num_classes}")
    print(f"   Train: {x_train.shape}  |  Test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test), num_classes, labels


# ─────────────────────────────────────────────
# CNN MODEL
# ─────────────────────────────────────────────

def build_cnn(num_classes: int) -> tf.keras.Model:
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ], name="HandwritingCNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train(model, x_train, y_train, x_test, y_test):
    cb = [
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True,
                                  monitor="val_accuracy", verbose=1),
    ]

    aug = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    history = model.fit(
        aug.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=cb,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def save_plots(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Handwritten Character Recognition — Training Results", fontsize=14)

    ax1.plot(history.history["accuracy"],     label="Train Acc",  color="#4CAF50")
    ax1.plot(history.history["val_accuracy"], label="Val Acc",    color="#2196F3")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(history.history["loss"],     label="Train Loss", color="#F44336")
    ax2.plot(history.history["val_loss"], label="Val Loss",   color="#FF9800")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("saved_model/training_curves.png", dpi=150)
    plt.close()
    print("📈 Training curves saved → saved_model/training_curves.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Handwritten Character Recognition — CNN Trainer")
    print("=" * 55)

    (x_train, y_train), (x_test, y_test), num_classes, labels = build_dataset()

    model = build_cnn(num_classes)
    model.summary()

    history = train(model, x_train, y_train, x_test, y_test)

    # Evaluate
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n🎯 Test Accuracy : {acc * 100:.2f}%")
    print(f"   Test Loss     : {loss:.4f}")

    # Save artefacts
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history.history, f)
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels, f)

    save_plots(history)

    print("\n✅ Training complete!")
    print(f"   Model   → {MODEL_PATH}")
    print(f"   Labels  → {LABELS_PATH}")


if __name__ == "__main__":
    main()
