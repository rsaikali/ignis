"""
Appliance state detection using clustering.
"""

import logging

import numpy as np
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)


class ApplianceStateDetector:
    """Détecteur d'états/cycles pour un appareil"""

    def __init__(self, n_states=5):
        """
        Args:
            n_states: Nombre d'états à détecter (par défaut 5: off, low, medium, high, peak)
        """
        self.n_states = n_states
        self.kmeans = None
        self.state_thresholds = None

    def fit(self, power_values):
        """
        Entraîne le détecteur d'états sur des données de consommation

        Args:
            power_values: Array de valeurs de puissance

        Returns:
            Self pour chaînage
        """
        if len(power_values) < self.n_states:
            logger.warning(f"Pas assez de données pour {self.n_states} états, réduction automatique")
            self.n_states = max(2, len(power_values) // 10)

        # Reshape pour KMeans
        power_reshaped = power_values.reshape(-1, 1)

        # Clustering pour identifier les états
        self.kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
        self.kmeans.fit(power_reshaped)

        # Calculer les seuils entre états (centres triés)
        centers = sorted(self.kmeans.cluster_centers_.flatten())
        self.state_thresholds = centers

        logger.info(f"États détectés: {len(centers)} niveaux = {[f'{c:.1f}W' for c in centers]}")
        return self

    def predict_states(self, power_values):
        """
        Prédit les états pour une séquence de consommation

        Args:
            power_values: Array de valeurs de puissance

        Returns:
            Tuple (états prédits, cycles détectés)
        """
        if self.kmeans is None:
            raise ValueError("Le détecteur doit être entraîné avant prédiction (fit)")

        # Prédire les états
        power_reshaped = power_values.reshape(-1, 1)
        states = self.kmeans.predict(power_reshaped)

        # Détecter les cycles (transitions d'états)
        cycles = self._detect_cycles(states, power_values)

        return states, cycles

    def _detect_cycles(self, states, power_values):
        """
        Détecte les cycles/phases dans les transitions d'états

        Args:
            states: Array d'états prédits
            power_values: Valeurs de puissance correspondantes

        Returns:
            Liste de cycles détectés avec métadonnées
        """
        cycles = []
        current_state = states[0]
        start_idx = 0

        for i in range(1, len(states)):
            # Transition d'état détectée
            if states[i] != current_state:
                # Enregistrer le cycle précédent
                if i - start_idx >= 10:  # Minimum 10 points (10 secondes)
                    cycle_power = power_values[start_idx:i]
                    cycles.append(
                        {
                            "state": int(current_state),
                            "start_idx": int(start_idx),
                            "end_idx": int(i),
                            "duration_seconds": int(i - start_idx),
                            "avg_power": float(np.mean(cycle_power)),
                            "max_power": float(np.max(cycle_power)),
                            "energy_wh": float(np.sum(cycle_power) / 3600),  # Wh (1Hz = 1s)
                        }
                    )

                # Nouveau cycle
                current_state = states[i]
                start_idx = i

        # Dernier cycle
        if len(states) - start_idx >= 10:
            cycle_power = power_values[start_idx:]
            cycles.append(
                {
                    "state": int(current_state),
                    "start_idx": int(start_idx),
                    "end_idx": int(len(states)),
                    "duration_seconds": int(len(states) - start_idx),
                    "avg_power": float(np.mean(cycle_power)),
                    "max_power": float(np.max(cycle_power)),
                    "energy_wh": float(np.sum(cycle_power) / 3600),
                }
            )

        return cycles
