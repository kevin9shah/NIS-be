"""
Network Traffic Preprocessor
=============================

Feature engineering pipeline for network traffic data. Handles:
1. Deduplication
2. Missing value imputation (median for numeric, mode for categorical)
3. Outlier clipping via IQR method (1.5× IQR)
4. Engineered features:
   - packet_size_ratio: relative packet size vs rolling mean
   - port_risk_score: risk classification based on destination port
   - entropy_energy_ratio: spectral entropy / frequency band energy
   - is_high_freq_burst: binary flag for high-frequency burst traffic
   - log_inter_arrival: log-transformed inter-arrival time
5. StandardScaler normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# The expected columns from the specification
REQUIRED_FEATURES = [
    "packet_size",
    "inter_arrival_time",
    "src_port",
    "dst_port",
    "packet_count_5s",
    "mean_packet_size",
    "spectral_entropy",
    "frequency_band_energy",
    "protocol_type_TCP",
]

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
        """Check for missing required columns. Returns list of missing."""
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
        Clip outliers using IQR method (1.5× IQR).
        For each target column: values below Q1 - 1.5*IQR or above
        Q3 + 1.5*IQR are clipped to those bounds.
        """
        for col in IQR_CLIP_COLUMNS:
            if col not in df.columns:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw columns:

        a) packet_size_ratio = packet_size / (mean_packet_size + 1e-6)
           Captures relative packet size — large ratios indicate abnormal packets.

        b) port_risk_score:
           - dst_port ∈ {22, 23, 3389, 4444, 6667} → 1.0 (high risk)
           - dst_port ∈ {80, 443, 53} → 0.0 (low risk)
           - else → 0.5 (medium risk)
           Encodes known risky services (SSH, Telnet, RDP, backdoors, IRC).

        c) entropy_energy_ratio = spectral_entropy / (frequency_band_energy + 1e-6)
           High entropy with low energy can indicate encrypted C2 channels.

        d) is_high_freq_burst = 1 if packet_count_5s > 50 else 0
           Binary flag for possible DDoS or brute-force burst patterns.
           Note: if packet_count_5s is normalized (0-1), we use threshold 0.35
           which corresponds to ~50/140 of the max range.

        e) log_inter_arrival = log1p(inter_arrival_time)
           Log transform to reduce skewness of timing features.
        """
        # a) Relative packet size
        df["packet_size_ratio"] = df["packet_size"] / (df["mean_packet_size"] + 1e-6)

        # b) Port risk classification
        def _port_risk(port):
            if port in HIGH_RISK_PORTS:
                return 1.0
            elif port in LOW_RISK_PORTS:
                return 0.0
            else:
                return 0.5
        df["port_risk_score"] = df["dst_port"].apply(_port_risk)

        # c) Entropy-energy ratio
        df["entropy_energy_ratio"] = df["spectral_entropy"] / (df["frequency_band_energy"] + 1e-6)

        # d) High-frequency burst detection
        # Handle both raw counts and normalized values
        max_count = df["packet_count_5s"].max()
        if max_count <= 1.0:
            # Data appears normalized; 50 out of ~140 max ≈ 0.35
            burst_threshold = 0.35
        else:
            burst_threshold = 50
        df["is_high_freq_burst"] = (df["packet_count_5s"] > burst_threshold).astype(int)

        # e) Log-transformed inter-arrival time
        df["log_inter_arrival"] = np.log1p(df["inter_arrival_time"])

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

        # Drop extra columns not needed for modeling
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
