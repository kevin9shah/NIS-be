"""
What-If Simulation Engine
==========================

Allows users to explore how varying network traffic parameters affects
anomaly detection probability. Generates a cartesian product of parameter
combinations, preprocesses each, runs through the model and QTTA, and
returns results suitable for heatmap visualization.

Use cases:
- Stress testing detection thresholds under varying attack intensities
- Understanding model sensitivity to packet size and traffic volume
- Validating QTTA behavior under different network conditions
"""

import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, Any, List, Optional
from core.qtta import QTTAThreshold
import logging

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    What-if simulation engine for network anomaly detection.

    Generates synthetic network conditions by varying selected parameters,
    then runs each through the preprocessing → prediction → QTTA pipeline
    to assess anomaly detection behavior.
    """

    def run(
        self,
        model: Any,
        preprocessor: Any,
        qtta_params: Dict[str, float],
        base_sample: pd.Series,
        param_grid: Dict[str, List],
    ) -> Dict[str, Any]:
        """
        Run simulation across cartesian product of parameter combinations.

        Args:
            model: Trained classifier with predict_proba
            preprocessor: Fitted NetworkPreprocessor
            qtta_params: QTTA configuration {base_threshold, alpha, d}
            base_sample: A single row from the dataset to use as template
            param_grid: Parameters to vary, e.g.:
                {
                    "packet_count_5s": [5, 10, 20, 50, 100, 200],
                    "mean_packet_size": [64, 128, 256, 512, 1024, 1500]
                }

        Returns:
            Dict with:
                results: List of individual simulation results
                heatmap_data: {x, y, z} for heatmap rendering
        """
        # Initialize fresh QTTA for simulation
        qtta = QTTAThreshold(
            base_threshold=qtta_params.get("base_threshold", 0.5),
            alpha=qtta_params.get("alpha", 0.3),
            d=qtta_params.get("d", 1.0),
        )

        # Get parameter names and values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        results = []

        # Generate cartesian product of all parameter combinations
        for combo in product(*param_values):
            # Clone base sample
            sample = base_sample.copy()

            # Override with simulation parameters
            params_dict = {}
            for name, value in zip(param_names, combo):
                sample[name] = value
                params_dict[name] = value

            # Create single-row DataFrame for preprocessing
            sample_df = pd.DataFrame([sample])

            try:
                # Preprocess (transform only — use fitted scaler)
                processed = preprocessor.transform(sample_df)

                # Drop label if present
                if "label" in processed.columns:
                    processed = processed.drop(columns=["label"])

                # Get anomaly probability from model
                proba = model.predict_proba(processed)[:, 1][0]

                # Run through QTTA
                is_anomaly, threshold, T = qtta.update(float(proba))

                result = {**params_dict}
                result["anomaly_prob"] = round(float(proba), 4)
                result["qtta_threshold"] = round(float(threshold), 4)
                result["is_anomaly"] = bool(is_anomaly)
                results.append(result)

            except Exception as e:
                logger.warning(f"Simulation failed for {params_dict}: {e}")
                result = {**params_dict}
                result["anomaly_prob"] = 0.0
                result["qtta_threshold"] = qtta_params.get("base_threshold", 0.5)
                result["is_anomaly"] = False
                results.append(result)

        # Build heatmap data if exactly 2 parameters
        heatmap_data = {}
        if len(param_names) == 2:
            x_name, y_name = param_names[1], param_names[0]
            x_vals = sorted(set(r[x_name] for r in results))
            y_vals = sorted(set(r[y_name] for r in results))

            z_matrix = []
            for y_val in y_vals:
                row = []
                for x_val in x_vals:
                    matching = [r for r in results
                                if r[x_name] == x_val and r[y_name] == y_val]
                    prob = matching[0]["anomaly_prob"] if matching else 0.0
                    row.append(prob)
                z_matrix.append(row)

            heatmap_data = {
                "x": x_vals,
                "y": y_vals,
                "z": z_matrix,
            }

        return {
            "results": results,
            "heatmap_data": heatmap_data,
        }
