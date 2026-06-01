"""
Sequence-to-Point Multi-Output NILM model.
"""

import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import text
from tensorflow import keras
from tensorflow.keras import callbacks, layers, models

from ..callbacks import RedisTrainingCallback
from ..layers import MultiHeadAttentionLayer
from ..losses import asymmetric_loss, focal_loss_fixed
from ..preprocessing import Seq2PointPreprocessor


logger = logging.getLogger(__name__)


class Seq2PointMultiOutputModel:
    """
    Modèle Sequence-to-Point avec Multi-Output architecture.

    Architecture:
    1. Input aggregate power (séquence temporelle)
    2. Feature extraction (Conv1D + GRU/LSTM)
    3. Multi-Head Attention (pour patterns simultanés)
    4. Output branches: une par appareil

    Avantages :
    - Désagrégation simultanée de N appliances
    - Détection native de chevauchements temporels
    - Pas besoin de conditioning (one-hot encoding)
    - Architecture simple et efficace
    """

    def __init__(self, appliance_ids, appliance_names, sequence_length=599, model_type="gru"):
        self.appliance_ids = appliance_ids
        self.appliance_names = appliance_names
        self.sequence_length = sequence_length if sequence_length % 2 == 1 else sequence_length - 1
        self.model_type = model_type
        self.num_appliances = len(appliance_ids)
        self.model = None
        self.preprocessor = Seq2PointPreprocessor(self.sequence_length)
        self.history = None
        self.use_gpu = self._configure_device()

        # Mapping appareil ID -> index
        self.appliance_id_to_idx = {app_id: idx for idx, app_id in enumerate(appliance_ids)}
        self.appliance_idx_to_id = {idx: app_id for app_id, idx in self.appliance_id_to_idx.items()}

    def _configure_device(self):
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Device: GPU ({len(gpus)} disponible(s))")
            return True
        logger.info("💻 Device: CPU")
        return False

    def build_model(self):
        """
        Construit le modèle Multi-Output avec attention.

        Architecture:
        - Input: aggregate_power (sequence_length, 1)
        - Conv1D: extraction de features locales
        - GRU/LSTM: extraction de features temporelles
        - Multi-Head Attention: patterns simultanés
        - Dense shared: features communes
        - Outputs: N branches (une par appareil)
        """
        # Input
        aggregate_input = layers.Input(shape=(self.sequence_length, 1), name="aggregate_power")

        # Conv1D pour features locales
        x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu", name="conv1d_1")(aggregate_input)
        x = layers.MaxPooling1D(pool_size=2, name="pool_1")(x)

        # Recurrent layers pour features temporelles
        if self.model_type == "gru":
            x = layers.GRU(128, return_sequences=True, name="gru_1")(x)
            x = layers.Dropout(0.2)(x)
            x = layers.GRU(64, return_sequences=True, name="gru_2")(x)
            x = layers.Dropout(0.2)(x)
        elif self.model_type == "lstm":
            x = layers.LSTM(128, return_sequences=True, name="lstm_1")(x)
            x = layers.Dropout(0.2)(x)
            x = layers.LSTM(64, return_sequences=True, name="lstm_2")(x)
            x = layers.Dropout(0.2)(x)
        else:
            raise ValueError(f"Type de modèle inconnu: {self.model_type}")

        # Multi-Head Attention pour capturer patterns simultanés
        x = MultiHeadAttentionLayer(num_heads=4, key_dim=16, name="multi_head_attention")(x)

        # Flatten pour dense layers
        x = layers.Flatten(name="flatten")(x)

        # Shared dense layers
        shared = layers.Dense(128, activation="relu", name="shared_dense_1")(x)
        shared = layers.Dropout(0.2)(shared)
        shared = layers.Dense(64, activation="relu", name="shared_dense_2")(shared)
        shared = layers.Dropout(0.1)(shared)

        # Output branches (une par appareil)
        # Utilise l'ID de l'appareil pour les noms de layers
        # pour que le modèle reste valide si l'utilisateur renomme
        outputs = []
        output_names = []
        for i, (app_id, app_name) in enumerate(zip(self.appliance_ids, self.appliance_names)):
            # Utiliser l'ID de l'appareil pour le nom du layer
            output_name = f"output_appliance_{app_id}"
            output_names.append(output_name)

            # Branch spécifique à l'appareil
            branch = layers.Dense(32, activation="relu", name=f"branch_appliance_{app_id}")(shared)
            output = layers.Dense(1, activation="linear", name=output_name)(branch)
            outputs.append(output)

        # Build model
        model = models.Model(inputs=aggregate_input, outputs=outputs, name=f"s2p_multioutput_{self.model_type}")

        # Compile avec loss asymétrique pour chaque output
        losses = {name: asymmetric_loss for name in output_names}
        metrics_dict = {name: ["mae", "mse"] for name in output_names}

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=losses, metrics=metrics_dict)

        logger.info(
            f" Modèle S2P-MultiOutput {self.model_type.upper()} "
            f"construit:\n"
            f"   - {self.num_appliances} appliances: "
            f"{self.appliance_names}\n"
            f"   - Séquence: {self.sequence_length}\n"
            f"   - Architecture: Multi-Output + Multi-Head Attention\n"
            f"   - Loss: asymmetric (FP penalty=1.5)"
        )

        return model

    def train(self, all_signatures, model_name, epochs=30, batch_size=32, validation_split=0.15, use_feedback=True, fine_tune=False):
        """
        Entraîne le modèle Multi-Output.

        Args:
            all_signatures: Dict[appliance_id, List[signature]]
            model_name: Name of the model
            use_feedback: Utiliser détections invalidées comme négatifs
            fine_tune: Continue entraînement modèle existant

        Returns:
            Dictionnaire de métriques
        """
        is_fine_tuning = fine_tune and self.model is not None
        if is_fine_tuning:
            logger.info("🔄 Fine-tuning du modèle Multi-Output existant")
            learning_rate = 0.0001
            epochs = min(epochs, 15)
        else:
            logger.info("🆕 Entraînement from-scratch Multi-Output")
            learning_rate = 0.001

        # Préparer les données
        X_aggregate = []
        y_outputs = {idx: [] for idx in range(self.num_appliances)}
        timestamps = []
        class_counts = {idx: 0 for idx in range(self.num_appliances)}

        for appliance_id, signatures in all_signatures.items():
            if appliance_id not in self.appliance_id_to_idx:
                logger.warning(f"Appareil {appliance_id} inconnu, ignoré")
                continue

            app_idx = self.appliance_id_to_idx[appliance_id]

            for sig in signatures:
                # Optimisation: utiliser les données déjà chargées
                if sig.get("raw_data"):
                    aggregate_power = np.array([d["papp"] for d in sig["raw_data"]], dtype=np.float32)
                    if sig.get("is_negative", False):
                        appliance_power = np.zeros(len(aggregate_power), dtype=np.float32)
                    else:
                        appliance_power = aggregate_power.copy()
                else:
                    # Fallback (ne devrait plus arriver souvent)
                    aggregate_power, appliance_power = self._load_signature_data_static(sig)

                if aggregate_power is None or len(aggregate_power) < self.sequence_length:
                    continue

                # Créer les séquences
                X, y = self.preprocessor.create_sequences(aggregate_power, appliance_power, stride=10)

                if len(X) > 0:
                    X_aggregate.append(X)

                    # Pour chaque appareil, créer target
                    for other_idx in range(self.num_appliances):
                        if other_idx == app_idx:
                            # Cet appareil: utiliser la vraie consommation
                            y_outputs[other_idx].append(y)
                            class_counts[other_idx] += len(y)
                        else:
                            # Autres appliances: zéro
                            y_outputs[other_idx].append(np.zeros_like(y))

                    # Timestamps
                    sig_start = sig["start_time"]
                    sig_timestamp = sig_start.timestamp() if hasattr(sig_start, "timestamp") else sig_start
                    timestamps.extend([sig_timestamp] * len(X))

        # Ajouter exemples négatifs si demandé
        if use_feedback:
            negative_count = self._add_negative_examples_multioutput(X_aggregate, y_outputs, timestamps, class_counts)
            if negative_count > 0:
                logger.info(f"{negative_count} negative examples added " f"(Multi-Output)")

        # Concaténer
        if not X_aggregate:
            logger.error("Aucune donnée pour entraînement Multi-Output")
            return {}

        X = np.concatenate(X_aggregate, axis=0)
        y_dict = {idx: np.concatenate(y_outputs[idx], axis=0) for idx in range(self.num_appliances)}

        # Calculer class weights (inverse proportionnel)
        total_samples = len(X)
        class_weights = {}
        for idx in range(self.num_appliances):
            count = class_counts[idx]
            if count > 0:
                weight = total_samples / (self.num_appliances * count)
                class_weights[idx] = weight
            else:
                class_weights[idx] = 1.0

        logger.info("Class weights:")
        for idx, app_id in self.appliance_idx_to_id.items():
            app_name = self.appliance_names[idx]
            logger.info(f"{app_name}: {class_weights[idx]:.2f} " f"({class_counts[idx]} samples)")

        # Ajuster scalers (from-scratch seulement)
        if not is_fine_tuning:
            logger.info("Ajustement scalers (from-scratch)")
            self.preprocessor.input_scaler.fit(X.reshape(-1, 1))
            # Pour multi-output, on normalise avec toutes les cibles
            all_y = np.concatenate([y_dict[idx] for idx in range(self.num_appliances)])
            self.preprocessor.target_scaler.fit(all_y.reshape(-1, 1))
            self.preprocessor.fitted = True
        else:
            logger.info("Réutilisation scalers existants (fine-tuning)")

        # Normaliser
        X_scaled, _ = self.preprocessor.transform(X)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        y_scaled_dict = {}
        for idx in range(self.num_appliances):
            y_scaled = self.preprocessor.target_scaler.transform(y_dict[idx].reshape(-1, 1)).flatten()
            y_scaled_dict[idx] = y_scaled

        # Time Series Cross-Validation
        timestamps_array = np.array(timestamps)
        sorted_indices = np.argsort(timestamps_array)

        tscv = TimeSeriesSplit(n_splits=5)
        all_splits = list(tscv.split(sorted_indices))
        train_idx, val_idx = all_splits[-1]

        idx_train = sorted_indices[train_idx]
        idx_val = sorted_indices[val_idx]

        logger.info(
            f"Dernier fold utilisé: {len(idx_train)} train / " f"{len(idx_val)} val " f"({100 * len(idx_val) / len(sorted_indices):.1f}% récent)"
        )

        X_train = X_scaled[idx_train]
        X_val = X_scaled[idx_val]

        y_train_dict = {}
        y_val_dict = {}
        safe_output_names = []

        for idx, (app_id, app_name) in enumerate(zip(self.appliance_ids, self.appliance_names)):
            # Utiliser l'ID de l'appareil pour le nom du layer
            output_name = f"output_appliance_{app_id}"
            safe_output_names.append(output_name)

            y_train_dict[output_name] = y_scaled_dict[idx][idx_train]
            y_val_dict[output_name] = y_scaled_dict[idx][idx_val]

        # Debug: vérifier les types et shapes
        logger.info(f"y_train_dict keys: {list(y_train_dict.keys())}")
        for key, val in y_train_dict.items():
            logger.info(
                f"   {key}: type={type(val)}, "
                f"shape={val.shape if hasattr(val, 'shape') else 'N/A'}, "
                f"dtype={val.dtype if hasattr(val, 'dtype') else 'N/A'}"
            )

        # Convertir les dictionnaires en listes dans l'ordre des outputs
        # Keras accepte mieux les listes pour les multi-output models
        y_train_list = [y_train_dict[name] for name in safe_output_names]
        y_val_list = [y_val_dict[name] for name in safe_output_names]

        # Construire ou ajuster modèle
        if not is_fine_tuning:
            self.model = self.build_model()
        else:
            logger.info(f"Ajustement learning rate: {learning_rate}")
            self.model.optimizer.learning_rate.assign(learning_rate)

        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ]

        # Redis real-time training logs callback
        redis_callback = RedisTrainingCallback(model_name=model_name, total_epochs=epochs, batch_update_freq=10)
        callbacks_list.append(redis_callback)
        logger.info("Real-time logs → channel 'training:logs'")

        # Entraînement
        self.history = self.model.fit(
            X_train, y_train_list, validation_data=(X_val, y_val_list), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=1
        )

        logger.info("Multi-Output training terminé")

        # Métriques
        metrics = {
            "epochs_trained": len(self.history.history["loss"]),
            "train_loss": float(self.history.history["loss"][-1]),
            "val_loss": float(self.history.history["val_loss"][-1]),
            "appliances": self.appliance_names,
            "architecture": "MultiOutput",
            "class_weights": {self.appliance_names[idx]: float(class_weights[idx]) for idx in range(self.num_appliances)},
        }

        # Métriques par appareil
        for idx, (app_id, app_name) in enumerate(zip(self.appliance_ids, self.appliance_names)):
            # Utiliser l'ID de l'appareil pour retrouver les métriques
            output_name = f"output_appliance_{app_id}"
            mae_key = f"{output_name}_mae"
            if mae_key in self.history.history:
                metrics[f"{app_name}_train_mae"] = float(self.history.history[mae_key][-1])
                metrics[f"{app_name}_val_mae"] = float(self.history.history[f"val_{mae_key}"][-1])

        return metrics

    def predict(self, aggregate_power, stride=1):
        """
        Prédit la consommation pour TOUS les appliances simultanément.

        Args:
            aggregate_power: Série agrégée
            stride: Pas de fenêtre glissante

        Returns:
            Dict[appliance_id, predictions]
        """
        if self.model is None:
            raise ValueError("Modèle Multi-Output non entraîné/chargé")

        # Créer fenêtres glissantes
        X = self.preprocessor.create_prediction_windows(aggregate_power, stride=stride)

        if len(X) == 0:
            return {app_id: np.zeros_like(aggregate_power) for app_id in self.appliance_ids}

        # Normaliser
        X_scaled, _ = self.preprocessor.transform(X)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        # Prédiction (liste de N outputs)
        predictions_scaled_list = self.model.predict(X_scaled, batch_size=32, verbose=0)

        # Dénormaliser et reconstruire pour chaque appareil
        result = {}
        half_window = self.sequence_length // 2

        for idx, app_id in enumerate(self.appliance_ids):
            predictions_scaled = predictions_scaled_list[idx]
            predictions = self.preprocessor.target_scaler.inverse_transform(predictions_scaled).flatten()

            # Post-traitement
            predictions = np.maximum(predictions, 0)

            # Reconstruction signal complet
            signal = np.zeros(len(aggregate_power))

            for i, pred in enumerate(predictions):
                sig_idx = i * stride + half_window
                if sig_idx < len(signal):
                    signal[sig_idx] = pred

            # Interpolation si stride > 1
            if stride > 1:
                from scipy.interpolate import interp1d

                indices = np.arange(0, len(signal), stride)
                indices = np.minimum(indices, len(signal) - 1)
                values = signal[indices]
                f = interp1d(indices, values, kind="linear", fill_value="extrapolate")
                signal = f(np.arange(len(signal)))

            result[app_id] = signal

        return result

    @staticmethod
    def _load_signature_data_static(signature):
        """Charge les données d'une signature (méthode statique)."""
        from ignis.nilm.database import db_manager

        try:
            with db_manager.engine.connect() as conn:
                query = text(
                    """
                    SELECT time, papp
                    FROM linky_realtime
                    WHERE time >= :start_time AND time <= :end_time
                    ORDER BY time
                """
                )

                result = conn.execute(query, {"start_time": signature["start_time"], "end_time": signature["end_time"]})

                aggregate_power = [row[1] for row in result if row[1] is not None]

                if not aggregate_power:
                    return None, None

                aggregate_power = np.array(aggregate_power, dtype=np.float32)

                if signature.get("is_negative", False):
                    appliance_power = np.zeros(len(aggregate_power), dtype=np.float32)
                else:
                    appliance_power = aggregate_power.copy()

                return aggregate_power, appliance_power
        except Exception as e:
            logger.error(f"Erreur chargement données signature: {e}")
            return None, None

    def _add_negative_examples_multioutput(self, X_aggregate, y_outputs, timestamps, class_counts):
        """Ajoute exemples négatifs pour Multi-Output."""
        negative_count = 0
        negative_sigs = self._load_negative_signatures()

        for appliance_id, signatures in negative_sigs.items():
            if appliance_id not in self.appliance_id_to_idx:
                continue

            app_idx = self.appliance_id_to_idx[appliance_id]

            for sig in signatures:
                aggregate = self._load_aggregate_data(sig["start_time"], sig["end_time"])

                if aggregate is None or len(aggregate) < self.sequence_length:
                    continue

                zero_target = np.zeros(len(aggregate), dtype=np.float32)

                X, y = self.preprocessor.create_sequences(aggregate, zero_target, stride=15)

                if len(X) > 0:
                    X_aggregate.append(X)

                    # Pour chaque appareil
                    for other_idx in range(self.num_appliances):
                        y_outputs[other_idx].append(np.zeros_like(y))

                    # Compter pour l'appareil concerné
                    class_counts[app_idx] += len(y)

                    sig_start = sig["start_time"]
                    sig_timestamp = sig_start.timestamp() if hasattr(sig_start, "timestamp") else sig_start
                    timestamps.extend([sig_timestamp] * len(X))
                    negative_count += 1

        return negative_count

    def _load_negative_signatures(self):
        """Charge les signatures négatives depuis la base."""
        from ignis.nilm.database import db_manager

        negative_sigs = {}
        try:
            with db_manager.get_session() as session:
                query = text(
                    """
                    SELECT id, appliance_id, start_time, end_time
                    FROM nilm_signatures
                    WHERE is_negative = TRUE
                    ORDER BY created_at DESC
                """
                )
                result = session.execute(query)

                for row in result:
                    app_id = row[1]
                    if app_id not in negative_sigs:
                        negative_sigs[app_id] = []
                    negative_sigs[app_id].append({"id": row[0], "appliance_id": app_id, "start_time": row[2], "end_time": row[3]})
        except Exception as e:
            logger.error(f"Erreur chargement signatures négatives: {e}")

        return negative_sigs

    def _load_aggregate_data(self, start_time, end_time):
        """Charge les données agrégées pour une période."""
        from ignis.nilm.database import db_manager

        try:
            with db_manager.engine.connect() as conn:
                query = text(
                    """
                    SELECT papp
                    FROM linky_realtime
                    WHERE time >= :start_time AND time <= :end_time
                    ORDER BY time
                """
                )
                result = conn.execute(query, {"start_time": start_time, "end_time": end_time})
                data = [row[0] for row in result if row[0] is not None]

                if not data:
                    return None

                return np.array(data, dtype=np.float32)
        except Exception as e:
            logger.error(f"Erreur chargement données agrégées: {e}")
            return None

    def save(self, filepath, metadata=None):
        """Sauvegarde le modèle Multi-Output."""
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")

        self.model.save(filepath)
        logger.info(f"Multi-Output model saved: {filepath}")

        meta = {
            "architecture": "MultiOutput",
            "model_type": self.model_type,
            "sequence_length": self.sequence_length,
            "num_appliances": self.num_appliances,
            "appliance_ids": self.appliance_ids,
            "appliance_names": self.appliance_names,
            "appliance_id_to_idx": self.appliance_id_to_idx,
        }
        if metadata:
            meta.update(metadata)

        meta_path = filepath.replace(".keras", ".metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Metadata saved: {meta_path}")

    def load(self, filepath):
        """Charge le modèle Multi-Output."""
        custom_objects = {
            "MultiHeadAttentionLayer": MultiHeadAttentionLayer,
            "asymmetric_loss": asymmetric_loss,
            "focal_loss_fixed": focal_loss_fixed,
        }
        self.model = keras.models.load_model(filepath, custom_objects=custom_objects)
        logger.info(f"Multi-Output model loaded: {filepath}")

        meta_path = filepath.replace(".keras", ".metadata.json")
        if Path(meta_path).exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)

            self.appliance_ids = meta["appliance_ids"]
            self.appliance_names = meta["appliance_names"]
            self.num_appliances = meta["num_appliances"]
            self.appliance_id_to_idx = {int(k): v for k, v in meta["appliance_id_to_idx"].items()}
            self.appliance_idx_to_id = {v: int(k) for k, v in self.appliance_id_to_idx.items()}
            logger.info(f"📋 Métadonnées chargées: {self.num_appliances} appliances")
