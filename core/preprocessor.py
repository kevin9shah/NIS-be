"""
Network Traffic Preprocessor
=============================

Feature engineering pipeline for network traffic data. Handles:
1. Deduplication
2. Missing value imputation (median for numeric, mode for categorical)
3. Outlier clipping via IQR method (1.5× IQR)
4. Feature normalization with StandardScaler

Supports flexible datasets with any numeric features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Dataset-agnostic: automatically detect features
# REQUIRED_FEATURES is now optional - all numeric columns will be used
REQUIRED_FEATURES = []  # Empty - auto-detect from dataset

# Columns subject to IQR outlier clipping
IQR_CLIP_COLUMNS = [
    "packet_size",
    "inter_arrival_time",
    "mean_packet_size",
    "spectral_entropy",
    "frequency_band_energy",
]

# High-risk destination ports commonly associated with attacks
HIGH_RISK_PORTS = {22, 23, 3389, 4444, 6667}
# Low-risk destination ports for standard web/DNS traffic
LOW_RISK_PORTS = {80, 443, 53}


class NetworkPreprocessor:
    """
    Feature engineering pipeline for network traffic anomaly detection.

    In fit_transform mode, learns statistics (scaler params, medians, modes)
    from training data. In transform mode, applies learned transforms to
    new data for inference.
    """

    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.medians: dict = {}
        self.modes: dict = {}
        self._is_fitted: bool = False

    def _validate_columns(self, df: pd.DataFrame) -> List[str]:
        """Check if dataset has required columns. Empty list means all numeric columns are OK."""
        # If REQUIRED_FEATURES is empty, use all numeric columns
        if not REQUIRED_FEATURES:
            # Dataset-agnostic mode: use all numeric columns except label
            return []
        
        # Otherwise check for specific features (legacy mode)
        missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
        return missing

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        after = len(df)
        if before != after:
            logger.info(f"Dropped {before - after} duplicate rows")
        return df

    def _impute_missing(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Impute missing values:
        - Numeric columns: median imputation
        - Categorical columns: mode imputation
        """
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                if fit:
                    self.medians[col] = float(df[col].median())
                fill_val = self.medians.get(col, 0.0)
                df[col] = df[col].fillna(fill_val)
            else:
                if fit:
                    self.modes[col] = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else ""
                fill_val = self.modes.get(col, "")
                df[col] = df[col].fillna(fill_val)
        return df

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip extreme outliers using IQR method (3.0× IQR).
        Uses a wider fence (3.0× instead of 1.5×) to preserve anomaly signal:
        anomalies are often extreme values, so aggressive clipping destroys
        the very features needed for minority-class detection.
        Only clips the most pathological values (> 3 IQR from quartiles).
        """
        for col in IQR_CLIP_COLUMNS:
            if col not in df.columns:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3.0 * IQR
            upper = Q3 + 3.0 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally create engineered features if dataset has the required columns.
        
        For datasets like CIC-IDS 2017, this is a no-op since features are already
        well-designed. For synthetic datasets, this adds domain-specific features.
        """
        # Only try to engineer features if we have the expected columns
        if "spectral_entropy" not in df.columns or "frequency_band_energy" not in df.columns:
            # Dataset doesn't have expected columns - skip feature engineering
            # (already has good features from real network data)
            logger.info("Skipping feature engineering - dataset has different feature set")
            return df
        
        # Original feature engineering for specific datasets
        # a) Entropy-energy ratio
        df["entropy_energy_ratio"] = df["spectral_entropy"] / (df["frequency_band_energy"] + 1e-6)

        # b) High-frequency burst detection
        if "packet_count_5s" in df.columns:
            max_count = df["packet_count_5s"].max()
            burst_threshold = 0.35 if max_count <= 1.0 else 50
            df["is_high_freq_burst"] = (df["packet_count_5s"] > burst_threshold).astype(int)

        # c) Log-transformed inter-arrival time
        if "inter_arrival_time" in df.columns:
            df["log_inter_arrival"] = np.log1p(df["inter_arrival_time"])

        # d) Packet count anomaly: deviation from mean
        if "packet_count_5s" in df.columns:
            mean_pkt_count = df["packet_count_5s"].mean()
            df["packet_anomaly_score"] = np.abs(df["packet_count_5s"] - mean_pkt_count)

        # e) Entropy anomaly: deviation from median
        if "spectral_entropy" in df.columns:
            median_entropy = df["spectral_entropy"].median()
            df["entropy_anomaly"] = np.abs(df["spectral_entropy"] - median_entropy)

        # --- Interaction features for better class separation ---
        # Anomalies tend to combine: short inter-arrival + non-standard port + high pack rate

        # f) Burst × port-risk: high packet count to a non-standard port (attack indicator)
        if "packet_count_5s" in df.columns and "dst_port" in df.columns:
            # Normalize dst_port to [0,1] range for the product to be meaningful
            port_norm = df["dst_port"] / (df["dst_port"].max() + 1e-6)
            df["burst_port_risk"] = df["packet_count_5s"] * (1 - port_norm)  # high pkt + low std port

        # g) Inter-arrival × dst_port: very short IAT to high port numbers = scan/DDoS signature
        if "inter_arrival_time" in df.columns and "dst_port" in df.columns:
            port_norm = df["dst_port"] / (df["dst_port"].max() + 1e-6)
            iat_norm = df["inter_arrival_time"] / (df["inter_arrival_time"].max() + 1e-6)
            df["iat_port_product"] = (1 - iat_norm) * port_norm  # short IAT + high dst_port

        # h) Packet size × spectral entropy: anomalies often have unusual payload entropy
        if "packet_size" in df.columns:
            pkt_norm = df["packet_size"] / (df["packet_size"].max() + 1e-6)
            df["packet_entropy_product"] = pkt_norm * df["spectral_entropy"]

        # i) Burst × entropy: rapid bursts with unusual entropy = likely attack traffic
        if "packet_count_5s" in df.columns:
            df["burst_entropy"] = df["packet_count_5s"] * df["spectral_entropy"]

        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Full preprocessing pipeline for training data.

        Steps:
        1. Validate columns
        2. Drop duplicates
        3. Impute missing values (learn medians/modes)
        4. Clip outliers via IQR
        5. Engineer features
        6. Fit StandardScaler and transform

        Args:
            df: Raw training DataFrame

        Returns:
            Tuple of (processed DataFrame, list of feature names)

        Raises:
            ValueError: If required columns are missing
        """
        df = df.copy()

        # Validate required columns exist
        missing = self._validate_columns(df)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Separate label if present
        label = None
        if "label" in df.columns:
            label = df["label"].copy()
            df = df.drop(columns=["label"])

        # Auto-detect usable columns if REQUIRED_FEATURES is empty.
        # Include numeric AND boolean columns (booleans are cast to 0/1 int).
        # Pandas' select_dtypes(np.number) drops bool columns — explicitly
        # include them here to preserve features like protocol_type_TCP, tcp_flags_SYN.
        if not REQUIRED_FEATURES:
            bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
            if bool_cols:
                df[bool_cols] = df[bool_cols].astype(int)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            keep_cols = numeric_cols
            logger.info(f"Auto-detected {len(keep_cols)} columns ({len(bool_cols)} bool→int cast)")
        else:
            # Use specified features (legacy mode for specific datasets)
            keep_cols = [c for c in REQUIRED_FEATURES if c in df.columns]
            extra_cols_in_data = [c for c in df.columns if c not in REQUIRED_FEATURES]
            if extra_cols_in_data:
                logger.info(f"Dropping extra columns: {extra_cols_in_data}")
        
        df = df[keep_cols].copy()

        # Step 1-3: Clean data
        df = self._drop_duplicates(df)
        df = self._impute_missing(df, fit=True)
        df = self._clip_outliers(df)

        # Step 4: Feature engineering
        df = self._engineer_features(df)

        # Step 5: Scale numeric features
        self.feature_names = list(df.columns)
        self.scaler = StandardScaler()
        scaled_values = self.scaler.fit_transform(df[self.feature_names])
        df_scaled = pd.DataFrame(scaled_values, columns=self.feature_names, index=df.index)

        # Re-attach label
        if label is not None:
            # Align label index with deduplicated df
            if len(label) != len(df_scaled):
                label = label.iloc[df.index].reset_index(drop=True)
            df_scaled["label"] = label.values[:len(df_scaled)]

        self._is_fitted = True
        return df_scaled, self.feature_names

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply saved preprocessing transforms for inference.

        Uses the scaler and statistics learned during fit_transform.

        Args:
            df: Raw inference DataFrame

        Returns:
            Processed DataFrame

        Raises:
            RuntimeError: If fit_transform hasn't been called
            ValueError: If required columns are missing
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")

        df = df.copy()

        missing = self._validate_columns(df)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        label = None
        if "label" in df.columns:
            label = df["label"].copy()
            df = df.drop(columns=["label"])

        # Use same column selection as fit_transform.
        # Cast boolean columns to int first so they aren't dropped by select_dtypes.
        if not REQUIRED_FEATURES:
            bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
            if bool_cols:
                df[bool_cols] = df[bool_cols].astype(int)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            keep_cols = numeric_cols
        else:
            # Use specified features
            keep_cols = [c for c in REQUIRED_FEATURES if c in df.columns]
        
        df = df[keep_cols].copy()

        df = self._impute_missing(df, fit=False)
        df = self._clip_outliers(df)
        df = self._engineer_features(df)

        # Ensure same column order as training
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_names]
        scaled_values = self.scaler.transform(df)
        df_scaled = pd.DataFrame(scaled_values, columns=self.feature_names, index=df.index)

        if label is not None:
            df_scaled["label"] = label.values[:len(df_scaled)]

        return df_scaled
