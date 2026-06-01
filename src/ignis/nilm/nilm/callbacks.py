"""
Custom Keras callbacks for NILM training.
"""

import json
import logging
import os
from datetime import datetime

import redis
from tensorflow.keras import callbacks


logger = logging.getLogger(__name__)


class RedisTrainingCallback(callbacks.Callback):
    """
    Custom Keras callback that publishes training events to Redis Pub/Sub.

    Events published:
    - training_start: When training begins
    - epoch_start: At the beginning of each epoch
    - epoch_end: At the end of each epoch with metrics
    - batch_update: Every N batches with current metrics
    - training_complete: When training finishes

    Messages are published to Redis channel 'training:logs' for consumption
    by WebSocket endpoints.
    """

    def __init__(self, model_name, total_epochs, batch_update_freq=10):
        """
        Args:
            model_name: Model name identifier (format: linkya_model_<timestamp>)
            total_epochs: Total number of epochs to train
            batch_update_freq: Publish batch updates every N batches
        """
        super().__init__()
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.batch_update_freq = batch_update_freq
        self.redis_client = None
        self.channel = "training:logs"
        self.current_epoch = 0
        self.training_start_time = None

        # Initialize Redis connection
        try:
            redis_host = os.environ.get("REDIS_HOST", "redis")
            redis_port = int(os.environ.get("REDIS_PORT", 6379))
            print(f"[RedisCallback] Connecting to Redis at {redis_host}:{redis_port}")
            self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            print("[RedisCallback]  Connected to Redis")
            logger.info(f"RedisTrainingCallback connected to {redis_host}:{redis_port}")
        except Exception as e:
            print(f"[RedisCallback]  Failed to connect to Redis: {e}")
            logger.warning(f"RedisTrainingCallback: Could not connect to Redis: {e}")
            self.redis_client = None

    def _publish(self, event_type, data):
        """Publish event to Redis Pub/Sub channel"""
        if not self.redis_client:
            print(f"[RedisCallback] Cannot publish {event_type}: no Redis client")
            return

        try:
            message = {
                "event": event_type,
                "model_name": self.model_name,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
            }
            result = self.redis_client.publish(self.channel, json.dumps(message))
            print(f"[RedisCallback] Published {event_type} to {result} subscribers")
        except Exception as e:
            print(f"[RedisCallback] Failed to publish {event_type}: {e}")
            logger.error(f"Failed to publish to Redis: {e}")

    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        print("[RedisCallback] on_train_begin called")
        self.training_start_time = datetime.utcnow()
        self._publish(
            "training_start",
            {
                "total_epochs": self.total_epochs,
                "message": f"Starting training for model {self.model_name}",
            },
        )

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch"""
        self.current_epoch = epoch + 1
        print(f"[RedisCallback] on_epoch_begin - Epoch {self.current_epoch}/{self.total_epochs}")
        self._publish(
            "epoch_start",
            {
                "epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "progress": round((self.current_epoch / self.total_epochs) * 100, 1),
            },
        )

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        logs = logs or {}

        # Calculate ETA
        elapsed = (datetime.utcnow() - self.training_start_time).total_seconds()
        eta_seconds = (elapsed / self.current_epoch) * (self.total_epochs - self.current_epoch)

        self._publish(
            "epoch_end",
            {
                "epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "metrics": {k: float(v) for k, v in logs.items()},
                "progress": round((self.current_epoch / self.total_epochs) * 100, 1),
                "elapsed_seconds": round(elapsed, 1),
                "eta_seconds": round(eta_seconds, 1),
            },
        )

    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch"""
        # Only publish every N batches to avoid flooding
        if batch % self.batch_update_freq == 0:
            logs = logs or {}
            self._publish(
                "batch_update",
                {
                    "epoch": self.current_epoch,
                    "batch": batch,
                    "metrics": {k: float(v) for k, v in logs.items()},
                },
            )

    def on_train_end(self, logs=None):
        """Called at the end of training"""
        logs = logs or {}
        elapsed = (datetime.utcnow() - self.training_start_time).total_seconds()

        self._publish(
            "training_complete",
            {
                "epochs_completed": self.current_epoch,
                "final_metrics": {k: float(v) for k, v in logs.items()},
                "total_duration_seconds": round(elapsed, 1),
                "message": f"Training completed for model {self.model_name}",
            },
        )
