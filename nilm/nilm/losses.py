"""
Custom loss functions for NILM training.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="custom_losses")
def focal_loss_fixed(y_true=None, y_pred=None, gamma=2.0, alpha=0.25):
    """
    Focal Loss pour se concentrer sur les exemples difficiles.

    Réduit la perte pour les exemples bien classifiés et augmente
    pour les difficiles. Utile pour rejeter les false positives.

    Args:
        y_true: Valeurs réelles (None si utilisé comme constructeur)
        y_pred: Valeurs prédites (None si utilisé comme constructeur)
        gamma: Facteur de modulation (2.0 par défaut).
               Plus élevé = focus sur difficiles
        alpha: Facteur de balance (0.25 par défaut).
               Poids relatif des classes

    Returns:
        Perte calculée si y_true et y_pred fournis, sinon fonction de perte
    """
    # Si y_true et y_pred sont fournis, calculer la perte directement
    if y_true is not None and y_pred is not None:
        # MAE de base
        mae = tf.abs(y_true - y_pred)

        # Normaliser les erreurs pour focal
        # Pour les prédictions proches de la vérité, p_t sera proche de 1
        # Pour les erreurs importantes, p_t sera proche de 0
        max_error = tf.reduce_max(mae) + 1e-7
        p_t = 1.0 - (mae / max_error)

        # Focal term: (1 - p_t)^gamma
        # Quand p_t proche de 1 (bonne prédiction): focal_weight ~0
        # Quand p_t proche de 0 (mauvaise prédiction): focal_weight ~1
        focal_weight = tf.pow(1.0 - p_t, gamma)

        # Pénaliser plus les false positives (y_true=0 mais y_pred>0)
        false_positive_mask = tf.cast(y_true < 0.1, tf.float32)
        alpha_weight = 1.0 + alpha * false_positive_mask

        # Loss finale
        loss = alpha_weight * focal_weight * mae

        return tf.reduce_mean(loss)

    # Sinon retourner une fonction de perte paramétrée
    def loss_fn(y_true, y_pred):
        # MAE de base
        mae = tf.abs(y_true - y_pred)

        # Normaliser les erreurs pour focal
        # Pour les prédictions proches de la vérité, p_t sera proche de 1
        # Pour les erreurs importantes, p_t sera proche de 0
        max_error = tf.reduce_max(mae) + 1e-7
        p_t = 1.0 - (mae / max_error)

        # Focal term: (1 - p_t)^gamma
        # Quand p_t proche de 1 (bonne prédiction): focal_weight ~0
        # Quand p_t proche de 0 (mauvaise prédiction): focal_weight ~1
        focal_weight = tf.pow(1.0 - p_t, gamma)

        # Pénaliser plus les false positives (y_true=0 mais y_pred>0)
        false_positive_mask = tf.cast(y_true < 0.1, tf.float32)
        alpha_weight = 1.0 + alpha * false_positive_mask

        # Loss finale
        loss = alpha_weight * focal_weight * mae

        return tf.reduce_mean(loss)

    return loss_fn


@tf.keras.utils.register_keras_serializable(package="custom_losses")
def asymmetric_loss(y_true=None, y_pred=None, false_positive_penalty=1.5):
    """
    Loss asymétrique qui pénalise plus les false positives.

    Quand y_true=0 (appareil éteint ou signature négative),
    les erreurs de prédiction sont pénalisées davantage.

    Args:
        y_true: Valeurs réelles (None si utilisé comme constructeur)
        y_pred: Valeurs prédites (None si utilisé comme constructeur)
        false_positive_penalty: Multiplicateur pour les false positives
                                (défaut: 1.5, réduit de 2.5 pour être moins agressif)

    Returns:
        Perte calculée si y_true et y_pred fournis, sinon fonction de perte
    """
    # Si y_true et y_pred sont fournis, calculer la perte directement
    if y_true is not None and y_pred is not None:
        mae = tf.abs(y_true - y_pred)

        # Détecter où y_true est proche de 0 (OFF ou négatif)
        is_negative = tf.cast(y_true < 0.1, tf.float32)

        # Pénaliser plus les erreurs quand l'appareil devrait être OFF
        weight = 1.0 + (false_positive_penalty - 1.0) * is_negative

        weighted_mae = weight * mae

        return tf.reduce_mean(weighted_mae)

    # Sinon retourner une fonction de perte paramétrée
    def loss_fn(y_true, y_pred):
        mae = tf.abs(y_true - y_pred)

        # Détecter où y_true est proche de 0 (OFF ou négatif)
        is_negative = tf.cast(y_true < 0.1, tf.float32)

        # Pénaliser plus les erreurs quand l'appareil devrait être OFF
        weight = 1.0 + (false_positive_penalty - 1.0) * is_negative

        weighted_mae = weight * mae

        return tf.reduce_mean(weighted_mae)

    return loss_fn
