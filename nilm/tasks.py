"""
Tâches Celery pour nilm-service
"""

import logging
import os
import warnings
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import redis


# Filtrer les warnings Celery root avant les imports
os.environ["C_FORCE_ROOT"] = "true"
warnings.filterwarnings("ignore", message=".*superuser privileges.*")

from celery import Celery  # noqa: E402
from celery.schedules import crontab  # noqa: E402


# Configuration du logger (doit être avant les imports locaux)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LOCAL_TIMEZONE = ZoneInfo(os.environ.get("TZ", "Europe/Paris"))

from .config import settings  # noqa: E402
from .database import db_manager  # noqa: E402
from .seq2point_nilm import Seq2PointNILMManager  # noqa: E402


logger.info("Sequence-to-Point (S2P) mode enabled")
nilm_manager = Seq2PointNILMManager()

# Vérifier la disponibilité du GPU (sauf si CUDA_VISIBLE_DEVICES est vide)
if os.environ.get("CUDA_VISIBLE_DEVICES", "0") != "":
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            logger.info(f"GPU available: {len(gpus)}")
            for gpu in gpus:
                logger.info(f"- {gpu.name}")
            # Activer la croissance mémoire dynamique
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            logger.warning("No GPU detected, using CPU")
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
else:
    # Beat n'a pas besoin de GPU, importer TensorFlow sans init GPU
    logger.info("CPU-only mode (scheduler)")
    import tensorflow as tf

# Initialisation de Celery
celery_app = Celery("nilm_tasks", broker=settings.celery_broker_url, backend=settings.celery_result_backend)

# Configuration Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 heure max par tâche
    worker_pool="solo",  # Pour éviter les conflits avec TensorFlow
    task_routes={"train_nilm_model": {"queue": "nilm"}, "detect_nilm_appliances": {"queue": "nilm"}, "add_nilm_signature": {"queue": "nilm"}},
)

# Redis client pour WebSocket real-time updates
try:
    redis_client = redis.from_url(settings.celery_broker_url, decode_responses=True)
    logger.info("Redis client initialized for detection updates")
except Exception as e:
    logger.warning(f"Redis client init failed: {e}")
    redis_client = None


@celery_app.task(name="train_nilm_model", bind=True)
def train_nilm_model(self, min_signatures=2, generation=0):
    """
    Entraîne un nouveau modèle NILM (S2P) - remplace l'ancien modèle

    Args:
        self: Instance de la tâche Celery (bind=True)
        min_signatures: Nombre minimum de signatures par appareil
        generation: Numéro de génération du training (pour éviter doublons)

    Returns:
        Statut et métriques de l'entraînement
    """
    try:
        import json
        import time
        from pathlib import Path

        from sqlalchemy import text

        # Vérification préventive : cette génération est-elle obsolète ?
        if redis_client and generation > 0:
            try:
                current_generation = redis_client.get("nilm:training:generation")
                if current_generation:
                    current_gen = int(current_generation)
                    if generation < current_gen:
                        logger.warning(f"Tâche génération {generation} annulée : " f"génération actuelle = {current_gen}")
                        return {
                            "status": "cancelled",
                            "message": f"Training obsolète (gen {generation} < {current_gen})",
                            "task_id": self.request.id,
                            "generation": generation,
                            "current_generation": current_gen,
                        }
            except Exception as e:
                logger.warning(f"Cannot check generation: {e}")

        logger.info(f"Training generation {generation} started (task {self.request.id})")

        start_time = time.time()

        # Générer le nom du modèle avec timestamp
        timestamp = datetime.now(LOCAL_TIMEZONE).strftime("%Y%m%d_%H%M%S")
        model_name = f"linkya_model_{timestamp}"

        logger.info(f"� Début de l'entraînement du modèle {model_name}...")

        # Entraîner le modèle from-scratch (pas de fine-tuning)
        metrics = nilm_manager.train_all_appliances(model_name, fine_tune=False)
        if "error" in metrics:
            return {"status": "error", "message": metrics.get("error"), "details": metrics}

        # Préparer les infos pour la base
        num_appliances = metrics.get("num_appliances", 0)
        total_signatures = sum(app["num_signatures"] for app in metrics.get("appliances", []))
        architecture = {
            "type": f"S2P-MULTI-{nilm_manager.model_type.upper()}",
            "sequence_length": settings.effective_sequence_length,
            "num_appliances": num_appliances,
            "model_type": nilm_manager.model_type,
            "appliances": metrics.get("appliances", []),
        }
        model_type_str = f"S2P-MULTI-{nilm_manager.model_type.upper()}"

        # Calculer la durée
        training_duration = int(time.time() - start_time)
        completed_at = datetime.now(LOCAL_TIMEZONE)

        # Supprimer l'ancien modèle s'il existe
        with db_manager.get_session() as session:
            old_models = session.execute(text("SELECT id, model_path FROM nilm_models")).fetchall()

            for old_model in old_models:
                old_id, old_path = old_model
                # Supprimer le fichier modèle
                if old_path and Path(old_path).exists():
                    try:
                        Path(old_path).unlink()
                        logger.info(f"Model file deleted: {old_path}")
                    except Exception as e:
                        logger.warning(f"Cannot delete {old_path}: {e}")

                # Supprimer metadata
                metadata_path = Path(str(old_path).replace(".keras", ".metadata.json"))
                if metadata_path.exists():
                    try:
                        metadata_path.unlink()
                    except Exception as e:
                        logger.warning(f"Cannot delete metadata: {e}")

            # Supprimer toutes les entrées de la table
            session.execute(text("DELETE FROM nilm_models"))
            session.commit()

            logger.info("Old model deleted")

        # Sauvegarder le nouveau modèle en base
        with db_manager.get_session() as session:
            query = text(
                """
                INSERT INTO nilm_models
                (model_name, model_type, architecture, training_date,
                 num_signatures, num_classes, metrics, model_path,
                 training_duration_seconds)
                VALUES
                (:model_name, :model_type, cast(:architecture as jsonb),
                 :training_date, :num_signatures, :num_classes,
                 cast(:metrics as jsonb), :model_path,
                 :training_duration_seconds)
                RETURNING id
            """
            )

            session.execute(
                query,
                {
                    "model_name": model_name,
                    "model_type": model_type_str,
                    "architecture": json.dumps(architecture),
                    "training_date": completed_at,
                    "num_signatures": total_signatures,
                    "num_classes": num_appliances,
                    "metrics": json.dumps(metrics, default=str),
                    "model_path": metrics.get("model_path", f"{settings.nilm_model_path}/{model_name}.keras"),
                    "training_duration_seconds": training_duration,
                },
            )

            session.commit()

        logger.info(f"Model {model_name} trained in {training_duration}s - {num_appliances} appliances, {total_signatures} signatures")

        return {
            "status": "success",
            "model_name": model_name,
            "model_type": model_type_str,
            "num_signatures": total_signatures,
            "num_appliances": num_appliances,
            "metrics": metrics,
            "training_duration_seconds": training_duration,
            "timestamp": completed_at.isoformat(),
        }

    except Exception as e:
        logger.error(f"Erreur entraînement: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@celery_app.task(name="detect_nilm_appliances")
def detect_nilm_appliances(hours=None, min_confidence=0.3):
    """
    Détecte et désagrège les appliances (S2P)

    Args:
        hours: Nombre d'heures à analyser (défaut: depuis config)
        min_confidence: Seuil de confiance minimal

    Returns:
        Statut et nombre de détections
    """
    try:
        import json

        from sqlalchemy import text

        # Vérifier qu'un modèle est disponible
        with db_manager.engine.connect() as conn:
            result = conn.execute(text("SELECT model_name, model_type FROM nilm_models " "ORDER BY training_date DESC LIMIT 1"))
            row = result.first()

            if not row:
                message = "Aucun modèle disponible"
                logger.warning(message)
                return {"status": "skipped", "message": message}

            active_model = row.model_name

        # Charger le modèle Multi-Output si pas déjà chargé
        model_path = os.path.join(settings.nilm_model_path, f"{active_model}.keras")
        if nilm_manager.multioutput_model is None:
            if os.path.exists(model_path):
                logger.info(f"Chargement du modèle: {active_model}")
                nilm_manager.load_model(model_path)
            else:
                message = f"Fichier modèle introuvable: {model_path}"
                logger.error(message)
                return {"status": "error", "message": message}

        # Définir la période d'analyse
        from datetime import timezone

        end_time = datetime.now(timezone.utc)

        # Si hours est None, analyser toute la période disponible
        if hours is None:
            with db_manager.engine.connect() as conn:
                result = conn.execute(text("SELECT MIN(time) as min_date " "FROM linky_realtime"))
                row = result.first()
                if row and row.min_date:
                    start_time = row.min_date
                    total_hours = int((end_time - start_time).total_seconds() / 3600)
                    logger.info(f"Détection sur TOUTE la période disponible " f"({total_hours}h - depuis {start_time})...")
                else:
                    # Fallback sur la config par défaut si pas de données
                    hours = settings.nilm_detection_period_hours or 240
                    start_time = end_time - timedelta(hours=hours)
                    logger.info(f"Détection sur les {hours} dernières heures...")
        else:
            start_time = end_time - timedelta(hours=hours)
            logger.info(f"Détection sur les {hours} dernières heures...")

        # Multi-Output disaggregation
        logger.info(f"Multi-Output disaggregation: {start_time} -> {end_time}")

        # Publish detection_start event to Redis
        if redis_client:
            try:
                message = json.dumps(
                    {
                        "event": "detection_start",
                        "data": {"model_name": active_model, "start_time": start_time.isoformat(), "end_time": end_time.isoformat()},
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                redis_client.publish("detections:updates", message)
                logger.info("Published detection_start to Redis")
            except Exception as e:
                logger.error(f"Failed to publish detection_start to Redis: {e}")

        # Vérifier qu'un modèle Multi-Output est chargé
        if nilm_manager.multioutput_model is None:
            logger.error("Aucun modèle Multi-Output chargé")
            return {"status": "error", "message": "Aucun modèle Multi-Output disponible. Veuillez entraîner un modèle."}

        # Désagrégation
        events = nilm_manager.disaggregate(start_time, end_time)

        if not events:
            logger.info("Aucun événement détecté")
            return {"status": "success", "num_detections": 0, "period": {"start": start_time.isoformat(), "end": end_time.isoformat()}}

        logger.info(f"{len(events)} events to process")

        # Sauvegarder les détections en base
        num_saved = 0
        num_updated = 0
        num_skipped = 0

        with db_manager.get_session() as session:
            for event in events:
                # Récupérer l'appliance_id (déjà présent en S2P)
                appliance_id = event.get("appliance_id")

                # Si pas d'appliance_id, chercher par nom (mode legacy)
                if not appliance_id:
                    result = session.execute(text("SELECT id FROM nilm_appliances " "WHERE name = :name LIMIT 1"), {"name": event["appliance_name"]})
                    row = result.first()
                    appliance_id = row.id if row else None

                if not appliance_id:
                    logger.warning(f"Appareil inconnu: {event['appliance_name']}")
                    num_skipped += 1
                    continue

                # Vérifier chevauchements pour cet appareil
                check_overlap_query = text(
                    """
                    SELECT id, confidence_score, start_time, end_time
                    FROM nilm_detections
                    WHERE appliance_id = :appliance_id
                    AND (
                        (:start_time <= end_time AND :end_time >= start_time)
                    )
                    ORDER BY confidence_score DESC
                    LIMIT 1
                """
                )

                overlap_result = session.execute(
                    check_overlap_query, {"appliance_id": appliance_id, "start_time": event["start_time"], "end_time": event["end_time"]}
                )
                existing = overlap_result.first()

                if existing:
                    # Une détection superposée existe
                    existing_id = existing.id
                    existing_confidence = float(existing.confidence_score)
                    new_confidence = float(event["confidence_score"])

                    if new_confidence > existing_confidence:
                        # Mettre à jour la détection existante avec de meilleures données
                        update_query = text(
                            """
                            UPDATE nilm_detections
                            SET start_time = :start_time,
                                end_time = :end_time,
                                avg_power = :avg_power,
                                energy_consumed = :energy_consumed,
                                confidence_score = :confidence_score,
                                prediction_class = :prediction_class,
                                features = :features,
                                created_at = NOW()
                            WHERE id = :detection_id
                        """
                        )

                        # Préparer features selon le mode
                        features_data = event.get("features", {})
                        if "probabilities" in event:
                            features_data["probabilities"] = event["probabilities"]

                        session.execute(
                            update_query,
                            {
                                "detection_id": existing_id,
                                "start_time": event["start_time"],
                                "end_time": event["end_time"],
                                "avg_power": event["avg_power"],
                                "energy_consumed": event.get("energy_consumed", event.get("energy_wh", 0)),
                                "confidence_score": event["confidence_score"],
                                "prediction_class": event.get("prediction_class"),
                                "features": json.dumps(features_data),
                            },
                        )
                        num_updated += 1
                        logger.info(f"Détection #{existing_id} mise à jour " f"(confiance {existing_confidence:.2%} → {new_confidence:.2%})")
                    else:
                        # Ignorer la nouvelle détection (moins bonne confiance)
                        num_skipped += 1
                        logger.debug(
                            f"Détection ignorée pour {event['appliance_name']} " f"(confiance {new_confidence:.2%} ≤ {existing_confidence:.2%})"
                        )
                else:
                    # Aucune détection superposée, insérer une nouvelle
                    # Préparer features
                    features_data = event.get("features", {})
                    if "probabilities" in event:
                        features_data["probabilities"] = event["probabilities"]

                    insert_query = text(
                        """
                        INSERT INTO nilm_detections
                        (appliance_id, start_time, end_time,
                         confidence_score, prediction_class,
                         features, created_at)
                        VALUES
                        (:appliance_id, :start_time, :end_time,
                         :confidence_score, :prediction_class,
                         :features, NOW())
                        RETURNING id
                    """
                    )

                    result = session.execute(
                        insert_query,
                        {
                            "appliance_id": appliance_id,
                            "start_time": event["start_time"],
                            "end_time": event["end_time"],
                            "confidence_score": event["confidence_score"],
                            "prediction_class": event.get("prediction_class"),
                            "features": json.dumps(features_data),
                        },
                    )
                    detection_id = result.scalar()
                    num_saved += 1
                    logger.debug(
                        f" Nouvelle détection: {event['appliance_name']} " f"- {event.get('duration_seconds', 0)}s " f"- {event['avg_power']:.1f}W"
                    )

                    # Publish detection to Redis for WebSocket streaming
                    if redis_client:
                        try:
                            message = json.dumps(
                                {
                                    "event": "new_detection",
                                    "data": {
                                        "id": detection_id,
                                        "appliance_id": appliance_id,
                                        "name": event["appliance_name"],
                                        "start_time": event["start_time"].isoformat(),
                                        "end_time": event["end_time"].isoformat(),
                                        "avg_power": event["avg_power"],
                                        "energy_consumed": event.get("energy_consumed", event.get("energy_wh", 0)),
                                        "confidence_score": event["confidence_score"],
                                        "prediction_class": event.get("prediction_class"),
                                    },
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )
                            redis_client.publish("detections:updates", message)
                            logger.debug(f"Published detection #{detection_id} to Redis")
                        except Exception as e:
                            logger.error(f"Failed to publish detection to Redis: {e}")

        total_processed = num_saved + num_updated + num_skipped
        logger.info(f"Détections traitées: {total_processed} " f"(new: {num_saved}, updated: {num_updated}, " f"ignorées: {num_skipped})")

        result = {
            "status": "success",
            "num_detections": num_saved,
            "num_updated": num_updated,
            "num_skipped": num_skipped,
            "total_processed": total_processed,
            "model_name": active_model,
            "period": {"start": start_time.isoformat(), "end": end_time.isoformat()},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Publish detection_complete event to Redis
        if redis_client:
            try:
                message = json.dumps({"event": "detection_complete", "data": result, "timestamp": datetime.utcnow().isoformat()})
                redis_client.publish("detections:updates", message)
                logger.info(f"Published detection_complete to Redis " f"({num_saved} new, {num_updated} updated)")
            except Exception as e:
                logger.error(f"Failed to publish detection_complete: {e}")

        return result

    except Exception as e:
        logger.error(f"Erreur lors de la détection: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@celery_app.task(name="add_nilm_signature")
def add_nilm_signature(appliance_name, start_time_str, end_time_str, is_negative=False):
    """
    Ajoute une signature manuelle soumise par l'utilisateur

    Args:
        appliance_name: Nom de l'appareil
        start_time_str: Timestamp de début (ISO format)
        end_time_str: Timestamp de fin (ISO format)
        is_negative: True si c'est une signature négative (faux positif)

    Returns:
        Statut et ID de la signature créée
    """
    try:
        logger.info(f"Ajout de signature pour {appliance_name}...")

        # Parser les dates
        start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))

        # Vérifier la durée
        duration = (end_time - start_time).total_seconds()
        if duration < settings.nilm_min_duration_seconds:
            return {"status": "error", "message": f"Durée trop courte (minimum {settings.nilm_min_duration_seconds}s)"}

        from sqlalchemy import text

        # Trouver ou créer l'appareil
        with db_manager.get_session() as session:
            result = session.execute(text("SELECT id FROM nilm_appliances WHERE name = :name LIMIT 1"), {"name": appliance_name})
            row = result.first()

            if row:
                appliance_id = row.id
            else:
                # Créer un nouvel appareil
                result = session.execute(
                    text(
                        """
                        INSERT INTO nilm_appliances
                        (name, created_at, updated_at)
                        VALUES (:name, NOW(), NOW())
                        RETURNING id
                    """
                    ),
                    {"name": appliance_name},
                )
                appliance_id = result.scalar()
                logger.info(f"Appareil créé: {appliance_name} (ID: {appliance_id})")

        # Ajouter la signature
        try:
            signature_id = db_manager.add_signature(appliance_id=appliance_id, start_time=start_time, end_time=end_time, is_negative=is_negative)
        except ValueError as ve:
            # Erreur de validation (chevauchement, etc.)
            logger.warning(f"Validation échouée: {ve}")
            return {"status": "error", "error_type": "validation", "message": str(ve)}

        if not signature_id:
            return {"status": "error", "message": "Échec de l'ajout de la signature"}

        logger.info(f"Signature {signature_id} ajoutée avec succès")

        # Déclencher un réentraînement si assez de signatures
        signatures = db_manager.get_all_signatures()
        if len(signatures) >= 10 and len(signatures) % 5 == 0:
            train_nilm_model.delay()
            logger.info("Réentraînement du modèle déclenché")

        return {
            "status": "success",
            "signature_id": signature_id,
            "appliance_id": appliance_id,
            "appliance_name": appliance_name,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Erreur lors de l'ajout de signature: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


# Configuration des tâches périodiques
celery_app.conf.beat_schedule = {
    "train-nilm-model": {"task": "train_nilm_model", "schedule": crontab(hour=f"*/{settings.nilm_training_interval_hours}", minute=0)},
    "detect-nilm-appliances": {
        "task": "detect_nilm_appliances",
        "schedule": settings.nilm_detection_interval_minutes * 60.0,
        "kwargs": {"hours": 2, "min_confidence": 0.6},  # Analyse 2h pour couvrir les cycles HC/HP
    },
}
