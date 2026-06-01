"""
Data preprocessing for Sequence-to-Point NILM models.
"""

import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


logger = logging.getLogger(__name__)


class Seq2PointPreprocessor:
    """Preprocessing pour modèle Sequence-to-Point"""

    def __init__(self, sequence_length=599):
        """
        Args:
            sequence_length: Longueur de la fenêtre d'entrée (impair pour point central)
        """
        # Forcer impair pour avoir un point central
        self.sequence_length = sequence_length if sequence_length % 2 == 1 else sequence_length - 1
        self.input_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.fitted = False

    def create_sequences(self, aggregate_power, appliance_power, stride=1):
        """
        Crée des séquences pour l'entraînement Sequence-to-Point

        Args:
            aggregate_power: Consommation totale (agrégée)
            appliance_power: Consommation de l'appareil cible
            stride: Pas de déplacement de la fenêtre

        Returns:
            Tuple (X: séquences d'entrée, y: valeurs cibles au point central)
        """
        if len(aggregate_power) != len(appliance_power):
            raise ValueError("aggregate_power et appliance_power doivent avoir la même longueur")

        if len(aggregate_power) < self.sequence_length:
            logger.warning(f"Séquence trop courte ({len(aggregate_power)} < {self.sequence_length})")
            return np.array([]), np.array([])

        X, y = [], []
        half_window = self.sequence_length // 2

        for i in range(half_window, len(aggregate_power) - half_window, stride):
            # Fenêtre d'entrée centrée sur i
            window = aggregate_power[i - half_window : i + half_window + 1]

            # Cible : consommation de l'appareil au point central
            target = appliance_power[i]

            X.append(window)
            y.append(target)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return X, y

    def create_prediction_windows(self, aggregate_power, stride=1):
        """
        Creates sliding windows for prediction (inference mode).

        Args:
            aggregate_power: Aggregate consumption data
            stride: Window step size

        Returns:
            Array of windows for prediction
        """
        if len(aggregate_power) < self.sequence_length:
            logger.warning(f"Sequence too short ({len(aggregate_power)} < {self.sequence_length})")
            return np.array([])

        X = []
        half_window = self.sequence_length // 2

        for i in range(half_window, len(aggregate_power) - half_window, stride):
            window = aggregate_power[i - half_window : i + half_window + 1]
            X.append(window)

        return np.array(X, dtype=np.float32)

    def fit(self, aggregate_power, appliance_power):
        """
        Ajuste les scalers sur les données d'entraînement

        Args:
            aggregate_power: Consommation totale
            appliance_power: Consommation de l'appareil
        """
        # Fit scaler pour l'entrée (agrégat)
        self.input_scaler.fit(aggregate_power.reshape(-1, 1))

        # Fit scaler pour la cible (appareil)
        self.target_scaler.fit(appliance_power.reshape(-1, 1))

        self.fitted = True
        logger.info(
            f"Scalers ajustés: input=[{aggregate_power.min():.1f}, {aggregate_power.max():.1f}], "
            f"target=[{appliance_power.min():.1f}, {appliance_power.max():.1f}]"
        )

    def transform(self, X, y=None):
        """
        Transforme les données avec les scalers

        Args:
            X: Séquences d'entrée
            y: Valeurs cibles (optionnel)

        Returns:
            Tuple (X transformé, y transformé ou None)
        """
        if not self.fitted:
            raise ValueError("Le preprocessor doit être ajusté avant transformation (fit)")

        # Normaliser X (chaque séquence indépendamment)
        X_scaled = np.zeros_like(X)
        for i in range(len(X)):
            X_scaled[i] = self.input_scaler.transform(X[i].reshape(-1, 1)).flatten()

        # Normaliser y si fourni
        y_scaled = None
        if y is not None:
            y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()

        return X_scaled, y_scaled

    def inverse_transform_target(self, y_scaled):
        """
        Inverse la transformation pour les prédictions

        Args:
            y_scaled: Valeurs normalisées

        Returns:
            Valeurs originales
        """
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
