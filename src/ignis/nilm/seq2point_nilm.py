"""
Modèle Sequence-to-Point NILM pour désagrégation d'appliances concurrents
et détection de cycles complexes.

Architecture basée sur LSTM/GRU avec mécanisme d'attention pour :
- Désagrégation : prédit la consommation individuelle de chaque appareil
- Détection d'états : identifie les différentes phases/cycles (chauffage, lavage, etc.)
- Appareils concurrents : gère plusieurs appliances fonctionnant simultanément
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
from sqlalchemy import text

from .config import settings
from .database import db_manager
from .nilm.detectors import ChangePointPatternDetector
from .nilm.models import Seq2PointMultiOutputModel


logger = logging.getLogger(__name__)


class Seq2PointNILMManager:
    """Gestionnaire de modèles S2P NILM avec architecture Multi-Output"""

    def __init__(self):
        self.model_type = os.getenv("NILM_MODEL_TYPE", "gru").lower()
        # Architecture: 'multioutput'
        self.architecture = os.getenv("NILM_ARCHITECTURE", "multioutput").lower()

        # Créer le répertoire des modèles
        Path(settings.nilm_model_path).mkdir(parents=True, exist_ok=True)

        # Détecteur hybride change point + pattern matching
        self.change_point_detector = ChangePointPatternDetector(
            min_power_change=settings.nilm_min_power_threshold, min_duration=settings.nilm_min_duration_seconds
        )
        logger.info("Change Point Pattern Detector initialisé")

        # Modèle Multi-Output
        self.multioutput_model = None

        logger.info(f"Architecture: {self.architecture.upper()}, " f"Type: {self.model_type.upper()}")

    def load_model(self, model_path):
        """
        Charge un modèle existant for fine-tuning.

        Args:
            model_path: Chemin vers le modèle .keras à charger
        """
        try:
            # Charger les métadonnées
            metadata_path = Path(model_path).with_suffix(".metadata.json")
            if not metadata_path.exists():
                raise ValueError(f"Métadonnées introuvables: {metadata_path}")

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            appliance_ids = metadata.get("appliance_ids", [])
            appliance_names = metadata.get("appliance_names", [])
            sequence_length = metadata.get("sequence_length", settings.effective_sequence_length)
            architecture = metadata.get("architecture", "MultiOutput")

            logger.info(f"Loading model {architecture}...")

            if architecture.lower() == "multioutput":
                self.multioutput_model = Seq2PointMultiOutputModel(
                    appliance_ids=appliance_ids, appliance_names=appliance_names, sequence_length=sequence_length, model_type=self.model_type
                )
                self.multioutput_model.load(model_path)
                self.architecture = "multioutput"
            else:
                raise ValueError(f"Architecture {architecture} not supported. " f"Only 'MultiOutput' is available.")

            logger.info(f"Model {architecture} loaded: {model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def train_all_appliances(self, model_name, fine_tune=False):
        """
        Entraîne le modèle sur tous les appliances (Multi-Output).

        Args:
            model_name: Name of the model (format: linkya_model_<timestamp>)
            fine_tune: Si True, continue l'entraînement du modèle existant

        Returns:
            Dictionnaire global de métriques
        """
        try:
            with db_manager.get_session() as session:
                query = """
                    SELECT DISTINCT a.id, a.name, COUNT(s.id) as num_signatures
                    FROM nilm_appliances a
                    JOIN nilm_signatures s ON s.appliance_id = a.id
                    GROUP BY a.id, a.name
                    HAVING COUNT(s.id) >= 2
                    ORDER BY a.name
                """
                appliances = session.execute(text(query)).fetchall()

            if len(appliances) < 1:
                logger.error("Aucun appareil avec assez de signatures (minimum 2)")
                return {"error": "insufficient_data", "min_appliances": 1}

            appliance_ids = [row[0] for row in appliances]
            appliance_names = [row[1] for row in appliances]

            # Charger les signatures
            all_signatures = {}
            with db_manager.get_session() as session:
                for appliance_id in appliance_ids:
                    query = """
                        SELECT id, appliance_id, start_time, end_time
                        FROM nilm_signatures
                        WHERE appliance_id = :appliance_id
                        ORDER BY created_at
                    """
                    result = session.execute(text(query), {"appliance_id": appliance_id})
                    all_signatures[appliance_id] = [dict(row._mapping) for row in result]

            # Charger les profiles pour change point detector
            logger.info("Loading signature profiles...")
            for appliance_id, signatures in all_signatures.items():
                app_idx = appliance_ids.index(appliance_id)
                appliance_name = appliance_names[app_idx]
                for sig in signatures:
                    # Charger les données de signature
                    agg, app_pwr = Seq2PointMultiOutputModel._load_signature_data_static(sig)
                    if app_pwr is None or len(app_pwr) == 0:
                        continue

                    duration = int((sig["end_time"] - sig["start_time"]).total_seconds())

                    self.change_point_detector.add_signature_profile(
                        appliance_id=appliance_id, appliance_name=appliance_name, power_sequence=app_pwr, duration=duration, signature_id=sig["id"]
                    )

            total_profiles = sum(len(data["profiles"]) for data in (self.change_point_detector.signature_profiles.values()))
            logger.info(f"{len(self.change_point_detector.signature_profiles)} " f"appliances, {total_profiles} profiles")

            # Entraîner avec l'architecture choisie
            if self.architecture == "multioutput":
                logger.info("Multi-Output training " "(outputs parallèles + attention)")

                # Créer ou réutiliser modèle Multi-Output
                if fine_tune and self.multioutput_model is not None:
                    logger.info("Reusing Multi-Output model " "for fine-tuning")
                else:
                    self.multioutput_model = Seq2PointMultiOutputModel(
                        appliance_ids, appliance_names, sequence_length=settings.effective_sequence_length, model_type=self.model_type
                    )

                metrics = self.multioutput_model.train(all_signatures, model_name, epochs=30, batch_size=32, use_feedback=True, fine_tune=fine_tune)

                if not metrics:
                    logger.error("Multi-Output training impossible " "(données insuffisantes)")
                    return {"error": "insufficient_training_data"}

                model_path = Path(settings.nilm_model_path) / (f"{model_name}.keras")
                self.multioutput_model.save(str(model_path), metadata=metrics)
                architecture_name = "MultiOutput"

            # Formater la réponse pour compatibilité frontend
            return {
                "model_name": model_name,
                "model_type": f"{architecture_name}-{self.model_type}",
                "architecture": architecture_name,
                "num_appliances": len(appliance_ids),
                "model_path": str(model_path),
                "appliances": [
                    {
                        "id": appliance_ids[i],
                        "name": appliance_names[i],
                        "num_signatures": len(all_signatures[appliance_ids[i]]),
                        "metrics": {
                            "train_mae": metrics.get("train_mae"),
                            "val_mae": metrics.get("val_mae"),
                            "train_mse": metrics.get("train_mae", 0) ** 2,
                            "val_mse": metrics.get("val_mae", 0) ** 2,
                            "train_loss": metrics.get("train_loss"),
                            "val_loss": metrics.get("val_loss"),
                            "epochs_trained": metrics.get("epochs_trained"),
                        },
                    }
                    for i in range(len(appliance_ids))
                ],
            }

        except Exception as e:
            logger.error(f"Erreur entraînement global: {e}", exc_info=True)
            return {"error": str(e)}

    def _filter_against_negative_signatures(self, detections):
        """
        Filtre les détections qui ressemblent aux signatures négatives.

        Une signature négative est créée quand l'utilisateur invalide
        une détection. On compare durée, puissance moyenne et énergie
        pour rejeter les false positives similaires.

        Args:
            detections: Liste de détections à filtrer

        Returns:
            Liste de détections filtrées (sans les false positives)
        """
        if not detections:
            return []

        # Charger les signatures négatives depuis la base
        negative_sigs = {}
        try:
            with db_manager.engine.connect() as conn:
                query = text(
                    """
                    SELECT
                        cs.id,
                        cs.appliance_id,
                        cs.start_time,
                        cs.end_time,
                        (
                            SELECT AVG(papp)
                            FROM linky_realtime
                            WHERE time >= cs.start_time
                              AND time <= cs.end_time
                        ) as avg_power,
                        (
                            SELECT SUM(papp) / 3600.0
                            FROM linky_realtime
                            WHERE time >= cs.start_time
                              AND time <= cs.end_time
                        ) as energy_wh
                    FROM nilm_signatures cs
                    WHERE is_negative = TRUE
                    ORDER BY created_at DESC
                """
                )

                result = conn.execute(query)

                for row in result:
                    sig_id, app_id, start_t, end_t, avg_pwr, energy = row
                    duration = int((end_t - start_t).total_seconds())

                    if app_id not in negative_sigs:
                        negative_sigs[app_id] = []

                    negative_sigs[app_id].append(
                        {
                            "id": sig_id,
                            "duration_seconds": duration,
                            "avg_power": float(avg_pwr) if avg_pwr else 0.0,
                            "energy_wh": float(energy) if energy else 0.0,
                        }
                    )

                total_negs = sum(len(s) for s in negative_sigs.values())
                if total_negs > 0:
                    logger.info(f"Filtrage contre {total_negs} signatures négatives")
        except Exception as e:
            logger.error(f"Erreur chargement signatures négatives: {e}")
            return detections  # Retourner sans filtrer si erreur

        # Filtrer les détections
        filtered = []
        rejected_count = 0

        for det in detections:
            app_id = det["appliance_id"]
            negs = negative_sigs.get(app_id, [])

            if not negs:
                # Pas de signatures négatives pour cet appareil
                filtered.append(det)
                continue

            is_false_positive = False

            # DEBUG: Log de la détection à analyser
            logger.info(f"Analyzing detection: {det['duration_seconds']}s, " f"{det['avg_power']:.1f}W, {det.get('energy_wh', 0):.1f}Wh")

            for neg in negs:
                # Critère 1: Durée similaire (±50% car change points)
                duration_ratio = det["duration_seconds"] / neg["duration_seconds"] if neg["duration_seconds"] > 0 else 0

                # DEBUG: Log détaillé de la comparaison
                logger.debug(
                    f"  vs signature négative #{neg['id']}: " f"{neg['duration_seconds']:.0f}s, " f"{neg['avg_power']:.1f}W, {neg['energy_wh']:.1f}Wh"
                )
                logger.debug(f"Ratios: durée={duration_ratio:.2f}, " f"seuils=[0.50, 1.50]")

                if not (0.50 <= duration_ratio <= 1.50):
                    logger.debug("✗ Durée hors limite")
                    continue

                logger.debug("✓ Durée OK")

                # Critère 2: Puissance moyenne similaire (±15% assoupli)
                if neg["avg_power"] > 0:
                    power_ratio = det["avg_power"] / neg["avg_power"]
                    logger.debug(f"Puissance: ratio={power_ratio:.2f}, " f"seuils=[0.85, 1.15]")
                    if not (0.85 <= power_ratio <= 1.15):
                        logger.debug("✗ Puissance hors limite")
                        continue
                    logger.debug("✓ Puissance OK")

                # Critère 3: Énergie similaire (±20% assoupli)
                det_energy = det.get("energy_wh", 0)
                if neg["energy_wh"] > 0 and det_energy > 0:
                    energy_ratio = det_energy / neg["energy_wh"]
                    logger.debug(f"Énergie: ratio={energy_ratio:.2f}, " f"seuils=[0.80, 1.20]")
                    if not (0.80 <= energy_ratio <= 1.20):
                        logger.debug("✗ Énergie hors limite")
                        continue
                    logger.debug("✓ Énergie OK")

                # Tous les critères correspondent → faux positif
                is_false_positive = True
                logger.debug(
                    f" Détection rejetée (similaire à signature "
                    f"négative #{neg['id']}): {det.get('appliance_name')} - "
                    f"{det['duration_seconds']}s, "
                    f"{det['avg_power']:.1f}W"
                )
                break

            if not is_false_positive:
                filtered.append(det)
            else:
                rejected_count += 1

        if rejected_count > 0:
            logger.info(f"Filtering complete: {rejected_count} false positives " f"rejected, {len(filtered)} detections kept")

        return filtered

    def _load_signature_profiles(self):
        """
        Charge les profiles de signatures depuis la base de données.
        Inclut les données morphologiques si available.

        Utilisé pour le pattern matching dans la détection.
        """
        import json as json_module

        with db_manager.get_session() as session:
            # Récupérer les appliances actifs avec leurs signatures
            appliances_query = """
                SELECT DISTINCT appliance_id, ca.name
                FROM nilm_signatures cs
                JOIN nilm_appliances ca ON cs.appliance_id = ca.id
            """
            appliances = session.execute(text(appliances_query)).fetchall()

            for appliance_id, appliance_name in appliances:
                # Récupérer signatures avec morphology_analysis
                sig_query = """
                    SELECT
                        id,
                        start_time,
                        end_time,
                        power_data,
                        morphology_analysis
                    FROM nilm_signatures
                    WHERE appliance_id = :appliance_id
                      AND is_negative = FALSE
                    ORDER BY created_at
                """
                signatures = session.execute(text(sig_query), {"appliance_id": appliance_id}).fetchall()

                for row in signatures:
                    sig_id = row[0]
                    start_time = row[1]
                    end_time = row[2]
                    power_data_json = row[3]
                    morphology_json = row[4]

                    # Charger power_data depuis JSON ou linky_realtime
                    appliance_power = None

                    if power_data_json:
                        # Utiliser power_data stocké
                        try:
                            power_data = json_module.loads(power_data_json)
                            appliance_power = np.array(power_data.get("values", []))
                        except Exception as e:
                            logger.warning(f"Erreur lecture power_data " f"sig {sig_id}: {e}")

                    # Fallback: charger depuis linky_realtime
                    if appliance_power is None or len(appliance_power) == 0:
                        signature = {"id": sig_id, "appliance_id": appliance_id, "start_time": start_time, "end_time": end_time}
                        aggregate_power, appliance_power = Seq2PointMultiOutputModel._load_signature_data_static(signature)

                    if appliance_power is None or len(appliance_power) == 0:
                        continue

                    duration = int((end_time - start_time).total_seconds())

                    # Parser morphology_analysis
                    morphology = None
                    if morphology_json:
                        try:
                            morphology = json_module.loads(morphology_json)
                        except Exception as e:
                            logger.warning(f"Erreur lecture morphology " f"sig {sig_id}: {e}")

                    # Ajouter le profil avec morphologie
                    self.change_point_detector.add_signature_profile(
                        appliance_id=appliance_id,
                        appliance_name=appliance_name,
                        power_sequence=appliance_power,
                        duration=duration,
                        signature_id=sig_id,
                        morphology=morphology,
                    )

        total_profiles = sum(len(data["profiles"]) for data in self.change_point_detector.signature_profiles.values())
        logger.info(f"Profils chargés: " f"{len(self.change_point_detector.signature_profiles)} " f"appliances, {total_profiles} profiles")

    def disaggregate(self, start_time, end_time):
        """
        Désagrège la consommation totale pour tous les appliances.
        Utilise l'architecture Multi-Output avec détection hybride.

        Args:
            start_time: Début de la période
            end_time: Fin de la période

        Returns:
            Liste de détections par appareil
        """
        if self.multioutput_model is None:
            logger.error("Aucun modèle Multi-Output chargé pour la désagrégation")
            return []

        # Charger les profiles de signatures si nécessaire
        if not self.change_point_detector.signature_profiles:
            logger.info("Chargement des profiles de signatures pour détection...")
            self._load_signature_profiles()

        try:
            # Charger la consommation totale
            with db_manager.get_session() as session:
                query = """
                    SELECT time, papp
                    FROM linky_realtime
                    WHERE time >= :start_time AND time <= :end_time
                    ORDER BY time
                """
                result = session.execute(text(query), {"start_time": start_time, "end_time": end_time})
                data = result.fetchall()
                if not data:
                    logger.warning("Aucune donnée pour désagrégation")
                    return []
                timestamps = [row[0] for row in data]
                aggregate_power = np.array([row[1] for row in data], dtype=np.float32)
            logger.info(f"Désagrégation sur {len(aggregate_power)} points")

            ##########################################################
            # APPROCHE HYBRIDE : Change Point Detection + Pattern Matching
            ##########################################################

            logger.info("=== Détection Hybride " "(Change Point + Pattern Matching) ===")

            # Étape 1 : Détecter les change points dans l'agrégé
            change_points = self.change_point_detector.detect_change_points(aggregate_power)

            if not change_points:
                logger.warning("Aucun change point détecté")
                return []

            # Étape 2 : Extraire les patterns entre les change points
            patterns = self.change_point_detector.extract_patterns(aggregate_power, change_points)

            if not patterns:
                logger.warning("Aucun pattern extrait")
                return []

            # Étape 3 : Matcher chaque pattern avec les profiles de signatures
            detections = []
            for pattern_data in patterns:
                match_result = self.change_point_detector.match_pattern(pattern_data, pattern_morphology=pattern_data.get("morphology"))

                if match_result:
                    (appliance_id, appliance_name, matched_signature_id, confidence) = match_result

                    # Mapper les indices vers les timestamps
                    start_idx = pattern_data["start_idx"]
                    end_idx = pattern_data["end_idx"]

                    if start_idx < len(timestamps) and end_idx <= len(timestamps):
                        detection = {
                            "appliance_id": appliance_id,
                            "appliance_name": appliance_name,
                            "signature_id": matched_signature_id,
                            "start_time": timestamps[start_idx],
                            "end_time": timestamps[min(end_idx, len(timestamps) - 1)],
                            "duration_seconds": pattern_data["duration"],
                            "avg_power": pattern_data["avg_power"],
                            "max_power": pattern_data["max_power"],
                            "energy_wh": pattern_data["energy_wh"],
                            "confidence_score": float(confidence),
                            "features": {"detection_method": ("change_point_pattern_matching"), "change_point_based": True},
                        }
                        if matched_signature_id is not None:
                            detection["features"]["matched_signature_id"] = int(matched_signature_id)
                            detection["features"]["matching"] = {"score": float(confidence), "method": "duration_power_shape_combined"}
                        detections.append(detection)

                        logger.info(
                            f"Pattern matché: {appliance_name} - "
                            f"{pattern_data['duration']}s - "
                            f"{pattern_data['avg_power']:.1f}W - "
                            f"confiance {confidence:.2%}"
                        )

            logger.info(f"Total détections avant filtrage: {len(detections)}")

            #  NOUVEAU: Filtrer contre les signatures négatives
            detections = self._filter_against_negative_signatures(detections)

            #  NOUVEAU: Filtrer par seuil de confiance minimum
            min_confidence = 0.40  # 40% de confiance minimum (assoupli)
            before_conf_filter = len(detections)
            detections = [d for d in detections if d.get("confidence_score", 0) >= min_confidence]
            if before_conf_filter > len(detections):
                logger.info(
                    f"Filtrage confiance: " f"{before_conf_filter - len(detections)} " f"détections rejetées (confiance < {min_confidence:.0%})"
                )

            logger.info(f"Total détections après filtrage: {len(detections)}")
            return detections

        except Exception as e:
            logger.error(f"Erreur désagrégation: {e}", exc_info=True)
            return []

    def _merge_consecutive_cycles(self, cycles, max_gap_seconds=120):
        """
        Fusionne les cycles consécutifs séparés par moins de max_gap_seconds.
        Générique : fonctionne pour tous les appliances.

        Args:
            cycles: Liste de cycles détectés par KMeans
            max_gap_seconds: Gap maximal en secondes pour fusionner deux cycles

        Returns:
            Liste de cycles fusionnés
        """
        if not cycles or len(cycles) == 0:
            return []

        # Trier les cycles par start_idx
        sorted_cycles = sorted(cycles, key=lambda c: c["start_idx"])

        merged = []
        current_merged = sorted_cycles[0].copy()

        for i in range(1, len(sorted_cycles)):
            cycle = sorted_cycles[i]

            # Calculer le gap entre la fin du cycle fusionné actuel et le début du prochain
            gap = cycle["start_idx"] - current_merged["end_idx"]

            if gap <= max_gap_seconds:
                # Fusionner : étendre le cycle actuel
                current_merged["end_idx"] = cycle["end_idx"]
                current_merged["duration_seconds"] = current_merged["end_idx"] - current_merged["start_idx"]
                # Recalculer avg_power et max_power (moyenne pondérée)
                # Note: on garde la max_power la plus élevée
                current_merged["max_power"] = max(current_merged["max_power"], cycle["max_power"])
                # Pour avg_power, on fait une moyenne simple (approximation)
                current_merged["avg_power"] = (current_merged["avg_power"] + cycle["avg_power"]) / 2
                # Sommer l'énergie
                current_merged["energy_wh"] = current_merged["energy_wh"] + cycle["energy_wh"]
            else:
                # Gap trop grand : sauvegarder le cycle fusionné actuel et commencer un nouveau
                merged.append(current_merged)
                current_merged = cycle.copy()

        # Ajouter le dernier cycle fusionné
        merged.append(current_merged)

        return merged

    def _find_active_segments(self, active_mask, timestamps, predictions, min_duration):
        """
        Trouve les segments actifs dans les prédictions, en détectant les gaps
        pour fragmenter les longues périodes en cycles individuels.

        Args:
            active_mask: Masque booléen des prédictions actives
            timestamps: Timestamps correspondants
            predictions: Prédictions de puissance
            min_duration: Durée minimale en secondes

        Returns:
            Liste de segments actifs
        """
        segments = []

        # Padding pour gérer les indices
        half_window = (settings.effective_sequence_length - 1) // 2

        # Paramètres de détection de gaps (périodes inactives entre deux cycles)
        # Un gap est détecté si la puissance reste < 20% du threshold pendant min_gap_duration
        # Pour un ballon d'eau chaude (3500W), un gap = puissance < 100W
        gap_threshold = settings.nilm_min_power_threshold * 0.2  # 20% du seuil (= 100W avec threshold=500W)
        min_gap_duration = 120  # 2 minutes minimum pour considérer un vrai gap (fin de chauffe)

        in_segment = False
        start_idx = 0
        gap_start = None

        for i in range(len(active_mask)):
            current_power = predictions[i] if i < len(predictions) else 0

            if active_mask[i] and not in_segment:
                # Début d'un nouveau segment
                in_segment = True
                start_idx = i
                gap_start = None

            elif in_segment:
                # Dans un segment actif
                if current_power < gap_threshold:
                    # Puissance faible, début potentiel d'un gap
                    if gap_start is None:
                        gap_start = i
                    elif (i - gap_start) >= min_gap_duration:
                        # Gap confirmé : fin du segment actuel
                        duration = gap_start - start_idx

                        if duration >= min_duration:
                            # Enregistrer le segment avant le gap
                            orig_start = start_idx + half_window
                            orig_end = gap_start + half_window

                            if orig_start < len(timestamps) and orig_end <= len(timestamps):
                                segment_predictions = predictions[start_idx:gap_start]

                                segment = {
                                    "start_time": timestamps[orig_start],
                                    "end_time": timestamps[min(orig_end, len(timestamps) - 1)],
                                    "duration_seconds": duration,
                                    "avg_power": float(np.mean(segment_predictions)),
                                    "max_power": float(np.max(segment_predictions)),
                                    "energy_wh": float(np.sum(segment_predictions) / 3600),
                                    "confidence_score": (
                                        float(np.mean(segment_predictions) / np.max(predictions)) if np.max(predictions) > 0 else 0.0
                                    ),
                                }
                                segments.append(segment)

                        # Réinitialiser pour chercher le prochain segment
                        in_segment = False
                        start_idx = i
                        gap_start = None
                else:
                    # Puissance élevée, réinitialiser le compteur de gap
                    gap_start = None

                # Vérifier aussi si on sort du masque actif (cas standard)
                if not active_mask[i]:
                    duration = i - start_idx

                    if duration >= min_duration:
                        orig_start = start_idx + half_window
                        orig_end = i + half_window

                        if orig_start < len(timestamps) and orig_end <= len(timestamps):
                            segment_predictions = predictions[start_idx:i]

                            segment = {
                                "start_time": timestamps[orig_start],
                                "end_time": timestamps[min(orig_end, len(timestamps) - 1)],
                                "duration_seconds": duration,
                                "avg_power": float(np.mean(segment_predictions)),
                                "max_power": float(np.max(segment_predictions)),
                                "energy_wh": float(np.sum(segment_predictions) / 3600),
                                "confidence_score": (float(np.mean(segment_predictions) / np.max(predictions)) if np.max(predictions) > 0 else 0.0),
                            }
                            segments.append(segment)

                    in_segment = False
                    gap_start = None

        # Dernier segment si actif
        if in_segment:
            duration = len(active_mask) - start_idx
            if duration >= min_duration:
                orig_start = start_idx + half_window
                orig_end = len(active_mask) + half_window

                if orig_start < len(timestamps):
                    segment_predictions = predictions[start_idx:]

                    segment = {
                        "start_time": timestamps[orig_start],
                        "end_time": timestamps[min(orig_end, len(timestamps) - 1)],
                        "duration_seconds": duration,
                        "avg_power": float(np.mean(segment_predictions)),
                        "max_power": float(np.max(segment_predictions)),
                        "energy_wh": float(np.sum(segment_predictions) / 3600),
                        "confidence_score": (float(np.mean(segment_predictions) / np.max(predictions)) if np.max(predictions) > 0 else 0.0),
                    }
                    segments.append(segment)

        return segments
