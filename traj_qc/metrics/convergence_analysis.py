"""
Convergence analysis metrics for trajectory quality assessment.

This module provides metrics for analyzing trajectory convergence
and clustering for conformational analysis.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..core.base_metric import BaseMetric


class RMSDConvergenceMetric(BaseMetric):
    """
    RMSD convergence analysis metric.

    Analyzes RMSD convergence over time to assess trajectory stability
    and identify stable periods.
    """

    def __init__(self, selection: str = "protein and name CA", 
                 convergence_threshold: float = 0.5,
                 convergence_window: int = 50):
        """
        Initialize RMSD convergence metric.

        Args:
            selection: Atom selection string
            convergence_threshold: RMSD threshold for convergence
            convergence_window: Window size for convergence analysis
        """
        super().__init__(
            name="rmsd_convergence",
            description="RMSD convergence analysis over time"
        )
        self.selection = selection
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window

    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """
        Validate input data for convergence analysis.

        Args:
            trajectory_data: Dictionary containing trajectory data

        Returns:
            True if input is valid, False otherwise
        """
        required_keys = ['n_frames', 'reader']
        return all(key in trajectory_data for key in required_keys)

    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate RMSD convergence metrics.

        Args:
            trajectory_data: Dictionary containing trajectory data

        Returns:
            Dictionary containing convergence analysis results
        """
        if not self.validate_input(trajectory_data):
            raise ValueError("Invalid trajectory data for convergence analysis")

        n_frames = trajectory_data['n_frames']
        reader = trajectory_data['reader']

        # Calculate RMSD values for convergence analysis
        rmsd_values = self._calculate_rmsd_series(trajectory_data, reader)
        
        # Analyze convergence
        convergence_analysis = self._analyze_convergence(rmsd_values)
        
        # Identify stable periods
        stable_periods = self._identify_stable_periods(rmsd_values)
        
        # Calculate convergence quality score
        convergence_score = self._calculate_convergence_score(rmsd_values, convergence_analysis)

        self.results = {
            'rmsd_values': rmsd_values,
            'selection': self.selection,
            'convergence_threshold': self.convergence_threshold,
            'convergence_window': self.convergence_window,
            'convergence_analysis': convergence_analysis,
            'stable_periods': stable_periods,
            'convergence_score': convergence_score,
            'n_frames': n_frames
        }

        self.is_calculated = True
        return self.results

    def _calculate_rmsd_series(self, trajectory_data: Dict[str, Any], reader: str) -> np.ndarray:
        """Calculate RMSD series for convergence analysis."""
        n_frames = trajectory_data['n_frames']
        rmsd_values = []

        if reader == 'MDAnalysis':
            universe = trajectory_data['universe']
            selection = universe.select_atoms(self.selection)
            reference_coords = selection.positions

            for frame_idx in range(n_frames):
                universe.trajectory[frame_idx]
                current_coords = selection.positions
                
                # Align and calculate RMSD
                aligned_coords = self._align_coordinates(current_coords, reference_coords)
                rmsd = self._calculate_rmsd(aligned_coords, reference_coords)
                rmsd_values.append(rmsd)

        elif reader == 'MDTraj':
            trajectory = trajectory_data['trajectory']
            selection = trajectory.topology.select(self.selection)
            
            if selection is None or len(selection) == 0:
                raise ValueError(f"No atoms found for selection: {self.selection}")

            reference_coords = trajectory.xyz[0, selection]

            for frame_idx in range(n_frames):
                current_coords = trajectory.xyz[frame_idx, selection]
                
                # Align and calculate RMSD
                aligned_coords = self._align_coordinates(current_coords, reference_coords)
                rmsd = self._calculate_rmsd(aligned_coords, reference_coords)
                rmsd_values.append(rmsd)

        return np.array(rmsd_values)

    def _analyze_convergence(self, rmsd_values: np.ndarray) -> Dict[str, Any]:
        """Analyze RMSD convergence over time."""
        n_frames = len(rmsd_values)
        window_size = min(self.convergence_window, n_frames // 2)
        
        if window_size < 10:
            return {
                'converged': False,
                'reason': 'Insufficient frames for convergence analysis',
                'convergence_frame': None,
                'final_rmsd': float(rmsd_values[-1])
            }

        # Calculate rolling mean and std
        rolling_mean = []
        rolling_std = []
        
        for i in range(window_size, n_frames):
            window_data = rmsd_values[i-window_size:i]
            rolling_mean.append(np.mean(window_data))
            rolling_std.append(np.std(window_data))

        rolling_mean = np.array(rolling_mean)
        rolling_std = np.array(rolling_std)

        # Check for convergence
        converged = False
        convergence_frame = None
        
        for i, (mean_val, std_val) in enumerate(zip(rolling_mean, rolling_std)):
            if std_val < self.convergence_threshold and mean_val < self.convergence_threshold * 2:
                converged = True
                convergence_frame = i + window_size
                break

        # Calculate convergence statistics
        if converged:
            stable_rmsd = rmsd_values[convergence_frame:]
            convergence_stats = {
                'mean_stable_rmsd': float(np.mean(stable_rmsd)),
                'std_stable_rmsd': float(np.std(stable_rmsd)),
                'rmsd_drift': float(np.mean(stable_rmsd) - np.mean(rmsd_values[:convergence_frame]))
            }
        else:
            convergence_stats = {
                'mean_stable_rmsd': None,
                'std_stable_rmsd': None,
                'rmsd_drift': None
            }

        return {
            'converged': converged,
            'convergence_frame': convergence_frame,
            'convergence_threshold': self.convergence_threshold,
            'final_rmsd': float(rmsd_values[-1]),
            'rolling_mean': rolling_mean.tolist(),
            'rolling_std': rolling_std.tolist(),
            'convergence_stats': convergence_stats
        }

    def _identify_stable_periods(self, rmsd_values: np.ndarray) -> List[Dict[str, Any]]:
        """Identify stable periods in the trajectory."""
        stable_periods = []
        n_frames = len(rmsd_values)
        
        # Find periods where RMSD is below threshold
        below_threshold = rmsd_values < self.convergence_threshold
        
        if not np.any(below_threshold):
            return stable_periods

        # Find continuous stable periods
        start_idx = None
        for i in range(n_frames):
            if below_threshold[i] and start_idx is None:
                start_idx = i
            elif not below_threshold[i] and start_idx is not None:
                # End of stable period
                period_length = i - start_idx
                if period_length >= 10:  # Minimum stable period
                    stable_periods.append({
                        'start_frame': start_idx,
                        'end_frame': i - 1,
                        'length': period_length,
                        'mean_rmsd': float(np.mean(rmsd_values[start_idx:i])),
                        'std_rmsd': float(np.std(rmsd_values[start_idx:i]))
                    })
                start_idx = None

        # Handle case where trajectory ends in stable period
        if start_idx is not None:
            period_length = n_frames - start_idx
            if period_length >= 10:
                stable_periods.append({
                    'start_frame': start_idx,
                    'end_frame': n_frames - 1,
                    'length': period_length,
                    'mean_rmsd': float(np.mean(rmsd_values[start_idx:])),
                    'std_rmsd': float(np.std(rmsd_values[start_idx:]))
                })

        return stable_periods

    def _calculate_convergence_score(self, rmsd_values: np.ndarray, 
                                   convergence_analysis: Dict[str, Any]) -> float:
        """Calculate overall convergence quality score (0-1)."""
        if convergence_analysis['converged']:
            # Base score for convergence
            score = 0.7
            
            # Bonus for early convergence
            convergence_frame = convergence_analysis['convergence_frame']
            early_convergence_bonus = max(0, 0.3 * (1 - convergence_frame / len(rmsd_values)))
            score += early_convergence_bonus
            
            # Bonus for stability
            if convergence_analysis['convergence_stats']['std_stable_rmsd'] is not None:
                stability_bonus = max(0, 0.2 * (1 - convergence_analysis['convergence_stats']['std_stable_rmsd'] / self.convergence_threshold))
                score += stability_bonus
        else:
            # Score based on how close to convergence
            min_std = np.min(convergence_analysis['rolling_std']) if convergence_analysis['rolling_std'] else float('inf')
            if min_std < float('inf'):
                score = max(0, 0.5 * (1 - min_std / self.convergence_threshold))
            else:
                score = 0.0

        return min(1.0, max(0.0, score))

    def _align_coordinates(self, coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
        """Align coordinates using Kabsch algorithm."""
        # Center coordinates
        coords1_centered = coords1 - np.mean(coords1, axis=0)
        coords2_centered = coords2 - np.mean(coords2, axis=0)
        
        # Calculate covariance matrix
        H = coords1_centered.T @ coords2_centered
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Rotation matrix
        R = Vt.T @ U.T
        
        # Apply rotation
        aligned_coords = coords1_centered @ R
        
        # Translate back
        aligned_coords += np.mean(coords2, axis=0)
        
        return aligned_coords

    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        """Calculate RMSD between two coordinate sets."""
        return np.sqrt(np.mean(np.sum((coords1 - coords2)**2, axis=1)))


class TrajectoryClusteringMetric(BaseMetric):
    """
    Trajectory clustering metric for conformational analysis.

    Performs clustering analysis on trajectory frames to identify
    distinct conformational states and assess sampling quality.
    """

    def __init__(self, n_clusters: int = 5, 
                 clustering_method: str = "kmeans",
                 selection: str = "protein and name CA",
                 min_cluster_size: int = 10):
        """
        Initialize trajectory clustering metric.

        Args:
            n_clusters: Number of clusters to identify
            clustering_method: Clustering algorithm to use
            selection: Atom selection string
            min_cluster_size: Minimum size for valid clusters
        """
        super().__init__(
            name="trajectory_clustering",
            description="Trajectory clustering for conformational analysis"
        )
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.selection = selection
        self.min_cluster_size = min_cluster_size

    def validate_input(self, trajectory_data: Dict[str, Any]) -> bool:
        """
        Validate input data for clustering analysis.

        Args:
            trajectory_data: Dictionary containing trajectory data

        Returns:
            True if input is valid, False otherwise
        """
        required_keys = ['n_frames', 'reader']
        if not all(key in trajectory_data for key in required_keys):
            return False
        
        # Check if we have enough frames for clustering
        if trajectory_data['n_frames'] < self.n_clusters * self.min_cluster_size:
            return False
            
        return True

    def calculate(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate trajectory clustering metrics.

        Args:
            trajectory_data: Dictionary containing trajectory data

        Returns:
            Dictionary containing clustering analysis results
        """
        if not self.validate_input(trajectory_data):
            raise ValueError("Invalid trajectory data for clustering analysis")

        n_frames = trajectory_data['n_frames']
        reader = trajectory_data['reader']

        # Extract features for clustering
        features = self._extract_clustering_features(trajectory_data, reader)
        
        # Perform clustering
        clustering_results = self._perform_clustering(features)
        
        # Analyze cluster quality
        cluster_quality = self._analyze_cluster_quality(clustering_results, features)
        
        # Identify representative structures
        representative_structures = self._identify_representatives(clustering_results, features)

        self.results = {
            'n_clusters': self.n_clusters,
            'clustering_method': self.clustering_method,
            'selection': self.selection,
            'n_frames': n_frames,
            'clustering_results': clustering_results,
            'cluster_quality': cluster_quality,
            'representative_structures': representative_structures
        }

        self.is_calculated = True
        return self.results

    def _extract_clustering_features(self, trajectory_data: Dict[str, Any], reader: str) -> np.ndarray:
        """Extract features for clustering analysis."""
        n_frames = trajectory_data['n_frames']
        features = []

        if reader == 'MDAnalysis':
            universe = trajectory_data['universe']
            selection = universe.select_atoms(self.selection)
            n_atoms = len(selection)

            for frame_idx in range(n_frames):
                universe.trajectory[frame_idx]
                coords = selection.positions.flatten()
                features.append(coords)

        elif reader == 'MDTraj':
            trajectory = trajectory_data['trajectory']
            selection = trajectory.topology.select(self.selection)
            
            if selection is None or len(selection) == 0:
                raise ValueError(f"No atoms found for selection: {self.selection}")

            for frame_idx in range(n_frames):
                coords = trajectory.xyz[frame_idx, selection].flatten()
                features.append(coords)

        features = np.array(features)
        
        # Normalize features
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        
        return features

    def _perform_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """Perform clustering analysis on features."""
        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
        except ImportError:
            raise ImportError("scikit-learn is required for clustering analysis")

        if self.clustering_method.lower() == "kmeans":
            clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        elif self.clustering_method.lower() == "hierarchical":
            clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
        else:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")

        # Perform clustering
        cluster_labels = clusterer.fit_predict(features)
        
        # Calculate quality metrics
        silhouette_avg = silhouette_score(features, cluster_labels) if self.n_clusters > 1 else 0
        calinski_harabasz = calinski_harabasz_score(features, cluster_labels) if self.n_clusters > 1 else 0

        # Calculate cluster sizes
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        cluster_sizes = dict(zip(unique_labels, counts))

        return {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_sizes': cluster_sizes,
            'silhouette_score': float(silhouette_avg),
            'calinski_harabasz_score': float(calinski_harabasz),
            'clusterer': clusterer
        }

    def _analyze_cluster_quality(self, clustering_results: Dict[str, Any], 
                                features: np.ndarray) -> Dict[str, Any]:
        """Analyze the quality of clustering results."""
        cluster_labels = np.array(clustering_results['cluster_labels'])
        cluster_sizes = clustering_results['cluster_sizes']
        
        # Check cluster size distribution
        min_size = min(cluster_sizes.values())
        max_size = max(cluster_sizes.values())
        size_ratio = min_size / max_size if max_size > 0 else 0
        
        # Check for balanced clusters
        balanced_clusters = size_ratio > 0.3
        
        # Calculate intra-cluster variance
        intra_cluster_variances = []
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 1:
                cluster_features = features[cluster_mask]
                variance = np.var(cluster_features, axis=0).mean()
                intra_cluster_variances.append(variance)
        
        avg_intra_cluster_variance = np.mean(intra_cluster_variances) if intra_cluster_variances else 0
        
        # Overall quality score
        quality_score = (
            clustering_results['silhouette_score'] * 0.4 +
            (1 - avg_intra_cluster_variance / np.var(features)) * 0.3 +
            size_ratio * 0.3
        )
        quality_score = max(0, min(1, quality_score))

        return {
            'balanced_clusters': balanced_clusters,
            'size_ratio': float(size_ratio),
            'min_cluster_size': min_size,
            'max_cluster_size': max_size,
            'avg_intra_cluster_variance': float(avg_intra_cluster_variance),
            'overall_quality_score': float(quality_score),
            'silhouette_threshold_met': clustering_results['silhouette_score'] > 0.3
        }

    def _identify_representatives(self, clustering_results: Dict[str, Any], 
                                features: np.ndarray) -> Dict[int, int]:
        """Identify representative structures for each cluster."""
        cluster_labels = np.array(clustering_results['cluster_labels'])
        representative_structures = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                cluster_features = features[cluster_mask]
                
                # Find frame closest to cluster center
                cluster_center = np.mean(cluster_features, axis=0)
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                representative_idx = np.argmin(distances)
                
                # Convert back to original frame index
                original_indices = np.where(cluster_mask)[0]
                representative_frame = original_indices[representative_idx]
                
                representative_structures[cluster_id] = int(representative_frame)
        
        return representative_structures 