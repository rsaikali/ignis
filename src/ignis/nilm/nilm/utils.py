"""
Utility functions for NILM processing.
"""

import re


def normalize_name_for_tensorflow(name):
    """
    Normalise un nom pour être compatible avec TensorFlow/Keras.
    Les noms de scope TensorFlow doivent correspondre au pattern: ^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$

    Args:
        name: Nom à normaliser

    Returns:
        Nom normalisé (espaces → underscores, caractères spéciaux supprimés)
    """
    # Remplacer les espaces par des underscores
    normalized = name.replace(" ", "_")
    # Remplacer les apostrophes par rien
    normalized = normalized.replace("'", "")
    # Garder uniquement les caractères alphanumériques, points, underscores, slashes, tirets
    normalized = re.sub(r"[^A-Za-z0-9._/\->]", "", normalized)
    # S'assurer que le nom commence par une lettre, chiffre ou point
    if normalized and not re.match(r"^[A-Za-z0-9.]", normalized):
        normalized = "appliance_" + normalized
    return normalized if normalized else "unknown_appliance"
