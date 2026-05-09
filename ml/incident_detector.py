# import random
# import numpy as np
# import torch
# from ml.autoencoder_incident import TrafficAutoencoder, detect_incident_autoencoder

# # Synthetic tiny autoencoder model
# auto_model = TrafficAutoencoder(input_dim=2)
# auto_model.eval()

# def incident_from_autoencoder(queue_length, speed):
#     sample = np.array([queue_length, speed], dtype=np.float32)
#     incident = detect_incident_autoencoder(auto_model, sample, threshold=0.15)
#     return incident


# def simulate_synthetic_incident():
#     """
#     Returns True/False + reason string
#     """
#     if random.random() < 0.35:
#         return True, "Random Synthetic Incident"
#     return False, "No Incident"


"""
Simplified incident detector compatible with the original project.

This file provides placeholder implementations for all incident
detection functions expected by the simulator, allowing the app
to run even if the deep learning models are not present.
"""

import random


# ==========================================================
# Individual incident detection functions
# ==========================================================

def incident_from_autoencoder(*args, **kwargs):
    """Placeholder autoencoder-based detector."""
    return False


def incident_from_cnn_lstm(*args, **kwargs):
    """Placeholder CNN-LSTM detector."""
    return False


def incident_from_yolo(*args, **kwargs):
    """Placeholder YOLO detector."""
    return False


# ==========================================================
# Synthetic incident generator
# ==========================================================

def simulate_synthetic_incident(probability=1.0):
    """
    Randomly generates a synthetic incident.

    Returns
    -------
    tuple
        (incident_detected, reason)
    """

    incident = random.random() < probability

    if incident:
        return True, "Synthetic traffic incident detected"
    else:
        return False, "No incident detected"

# ==========================================================
# Main incident detection API
# ==========================================================

def detect_incident(sample_dict=None, model_name="autoencoder"):
    """
    Main incident detection function used by the simulator.

    Parameters
    ----------
    sample_dict : dict, optional
        Input traffic features.
    model_name : str
        Model name ('autoencoder', 'cnn_lstm', 'yolo', 'synthetic').

    Returns
    -------
    bool
        True if incident detected, otherwise False.
    """

    model_name = model_name.lower()

    if model_name == "autoencoder":
        return incident_from_autoencoder(sample_dict)
    elif model_name == "cnn_lstm":
        return incident_from_cnn_lstm(sample_dict)
    elif model_name == "yolo":
        return incident_from_yolo(sample_dict)
    elif model_name == "synthetic":
        return simulate_synthetic_incident()
    else:
        return False