import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from models.topo import SegNet
from datareader import *
from loss import dice_bce_loss, r_squared
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_visible_devices(gpus, 'GPU')
    print(f"Detected {len(gpus)} GPU(s), memory growth enabled")
else:
    print("No GPU detected, using CPU")


strategy = tf.distribute.experimental.CentralStorageStrategy()
print("Number of replicas:", strategy.num_replicas_in_sync)


root_dir = r"../data"
all_folds = [f"fold_{i}" for i in range(5)]
test_fold = "fold_4"
cv_folds = [f for f in all_folds if f != test_fold]


class EpochLogger(Callback):
    def __init__(self, log_file='epoch_log.txt'):
        super().__init__()
        self.log_file = log_file

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        mae = logs.get('mae')
        val_mae = logs.get('val_mae')
        r2 = logs.get('r_squared')
        val_r2 = logs.get('val_r_squared')

        if np.isnan(val_loss) or np.isinf(val_loss):
            raise ValueError(f"Epoch {epoch + 1}: Validation loss is NaN or Inf")

        lr = float(self.model.optimizer.lr.numpy())
        duration = time.time() - self.epoch_start_time

        with open(self.log_file, 'a') as f:
            f.write(
                f"Epoch {epoch + 1}, "
                f"Loss: {loss:.4f}, Val_Loss: {val_loss:.4f}, "
                f"MAE: {mae:.4f}, Val_MAE: {val_mae:.4f}, "
                f"R2: {r2:.4f}, Val_R2: {val_r2:.4f}, "
                f"LR: {lr:.6f}, Time: {duration:.2f}s\n"
            )


for fold_idx in range(4):
    val_fold = cv_folds[fold_idx]
    train_folds = [f for f in cv_folds if f != val_fold]

    print(f"\nFold {fold_idx + 1}")
    print("Train folds:", train_folds)
    print("Validation fold:", val_fold)

    train_files = []
    for fold in train_folds:
        pre_files = sorted(tf.io.gfile.glob(os.path.join(root_dir, fold, 'pre', '*.tfrecord')))
        post_files = sorted(tf.io.gfile.glob(os.path.join(root_dir, fold, 'post', '*.tfrecord')))
        train_files += list(zip(pre_files, post_files))

    val_pre = sorted(tf.io.gfile.glob(os.path.join(root_dir, val_fold, 'pre', '*.tfrecord')))
    val_post = sorted(tf.io.gfile.glob(os.path.join(root_dir, val_fold, 'post', '*.tfrecord')))

    train_pre, train_post = zip(*train_files)
    val_pre, val_post = val_pre, val_post

    batch_size = 8 * strategy.num_replicas_in_sync
    train_loader = dynamic_dataset_loader(train_pre, train_post, batch_size)
    val_loader = dynamic_dataset_loader(val_pre, val_post, batch_size)

    with strategy.scope():
        model = SegNet((256, 256, 36))
        model.compile(
            optimizer=tfa.optimizers.AdamW(
                learning_rate=3e-4,
                weight_decay=1e-4,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss=dice_bce_loss,
            metrics=['mae', r_squared]
        )

    save_path = os.path.join("MODEL", f"segnet_fold{fold_idx + 1}.h5")

    callbacks = [
        ModelCheckpoint(
            filepath=save_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=120,
            restore_best_weights=True
        ),
        EpochLogger(log_file='segnet.txt')
    ]

    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=120,
        callbacks=callbacks
    )

    best_epoch = np.argmin(history.history['val_loss'])

    print("Best epoch:", best_epoch + 1)
    print("Train loss:", history.history['loss'][best_epoch])
    print("Val loss:", history.history['val_loss'][best_epoch])
    print("Train MAE:", history.history['mae'][best_epoch])
    print("Val MAE:", history.history['val_mae'][best_epoch])
    print("Train R2:", history.history['r_squared'][best_epoch])
    print("Val R2:", history.history['val_r_squared'][best_epoch])
    print("Model saved to:", save_path)
