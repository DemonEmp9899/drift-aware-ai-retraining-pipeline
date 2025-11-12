# src/drift_monitor.py
"""
Drift Monitor - Detects embedding, behavior, and accuracy drift
==============================================================
Uses statistical methods to detect when AI model performance degrades.
"""

import os
import json
import ast
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


def _parse_embedding_value(val: Any) -> Optional[np.ndarray]:
    """
    Robustly parse an embedding retrieved from DB.
    Accepts:
      - list/tuple/np.ndarray -> returns np.array(dtype=float32)
      - JSON string -> json.loads -> list -> np.array
      - Python-list string (literal) -> ast.literal_eval -> list -> np.array
      - comma-separated numbers in brackets -> fallback parse
    Returns None if val is None or empty.
    """
    if val is None:
        return None

    # If already a numpy array
    if isinstance(val, np.ndarray):
        try:
            return val.astype(np.float32)
        except Exception:
            return np.array(val.tolist(), dtype=np.float32)

    # Already a sequence (list/tuple)
    if isinstance(val, (list, tuple)):
        try:
            return np.array(val, dtype=np.float32)
        except Exception:
            try:
                return np.array([float(x) for x in val], dtype=np.float32)
            except Exception:
                raise TypeError(f"Cannot coerce embedding list elements to float: {val}")

    # If string, try JSON then ast.literal_eval then fallback split
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return None

        # Try JSON first
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (list, tuple)):
                return np.array(parsed, dtype=np.float32)
        except Exception:
            pass

        # Try Python literal eval
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return np.array(parsed, dtype=np.float32)
        except Exception:
            pass

        # Fallback: remove brackets and split by comma
        s2 = s.strip("[]() ")
        if s2:
            try:
                parts = [p.strip() for p in s2.split(",") if p.strip()]
                return np.array([float(x) for x in parts], dtype=np.float32)
            except Exception:
                pass

    # Unknown type -> raise to surface issue
    raise TypeError(f"Unsupported embedding value type: {type(val)}")


class DriftMonitor:
    """
    Monitors AI model for three types of drift:
    1. Embedding Drift - Distribution of input embeddings changes
    2. Behavior Drift - Refusal/toxicity rates spike
    3. Accuracy Drift - Model performance degrades
    """

    def __init__(self):
        """Initialize drift monitor with database connection and thresholds."""
        # Initialize Supabase client
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )

        # Load thresholds from environment
        self.emb_drift_threshold = float(os.getenv("EMBEDDING_DRIFT_THRESHOLD", "0.15"))
        self.refusal_threshold = float(os.getenv("REFUSAL_RATE_THRESHOLD", "0.05"))
        self.toxicity_threshold = float(os.getenv("TOXICITY_RATE_THRESHOLD", "0.02"))
        self.accuracy_threshold = float(os.getenv("ACCURACY_DROP_THRESHOLD", "0.05"))

        print(f"✓ DriftMonitor initialized")
        print(f"  Embedding drift threshold: {self.emb_drift_threshold}")
        print(f"  Refusal rate threshold: {self.refusal_threshold}")
        print(f"  Toxicity rate threshold: {self.toxicity_threshold}")
        print(f"  Accuracy drop threshold: {self.accuracy_threshold}")

    def get_embeddings(self, days_ago: int, emb_type: str = "query") -> List[np.ndarray]:
        """
        Fetch embeddings from database for specified time period.

        Args:
            days_ago: Number of days back from now
            emb_type: Type of embedding ('query' or 'document')

        Returns:
            List of NumPy arrays (each embedding), or empty list if none
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_ago)

            response = self.supabase.table("embeddings_log")\
                .select("embedding")\
                .eq("type", emb_type)\
                .gte("timestamp", cutoff_date.isoformat())\
                .execute()

            if not response.data:
                return []

            parsed_embeddings: List[np.ndarray] = []
            for item in response.data:
                raw = item.get("embedding")
                try:
                    emb = _parse_embedding_value(raw)
                    if emb is not None:
                        parsed_embeddings.append(emb)
                except Exception as e:
                    print(f"   ⚠️ Skipping malformed embedding row: {e}")

            return parsed_embeddings

        except Exception as e:
            print(f"❌ Error fetching embeddings: {e}")
            return []

    def detect_embedding_drift(self) -> Tuple[bool, float, Dict]:
        """
        Detect embedding drift using multiple statistical methods:
        1. Centroid distance - How far mean vectors have moved
        2. Variance change - How spread out data is
        3. Cluster analysis - How cluster structure has changed

        Returns:
            Tuple of (is_drifted, drift_score, details_dict)
        """
        print("\n🔍 Checking Embedding Drift...")

        # Get baseline (14 days back) vs recent (last 7 days)
        baseline_vectors = self.get_embeddings(days_ago=14, emb_type="query")
        recent_vectors = self.get_embeddings(days_ago=7, emb_type="query")

        # Check if we have enough data
        if len(baseline_vectors) < 10 or len(recent_vectors) < 10:
            print("   ⚠️ Not enough data for drift detection")
            print(f"   Baseline: {len(baseline_vectors)}, Recent: {len(recent_vectors)}")
            return False, 0.0, {
                "reason": "insufficient_data",
                "baseline_samples": len(baseline_vectors),
                "recent_samples": len(recent_vectors)
            }

        # Ensure all embeddings have the same dimensionality
        try:
            dim = baseline_vectors[0].shape[0]
            # Filter only vectors with matching dim
            baseline_vectors = [v for v in baseline_vectors if v.shape[0] == dim]
            recent_vectors = [v for v in recent_vectors if v.shape[0] == dim]
        except Exception as e:
            print(f"   ❌ Error determining embedding dimension: {e}")
            return False, 0.0, {"error": str(e)}

        if len(baseline_vectors) < 5 or len(recent_vectors) < 5:
            print("   ⚠️ Not enough consistent-dimension embeddings for drift detection")
            return False, 0.0, {
                "reason": "insufficient_consistent_embeddings",
                "baseline_samples": len(baseline_vectors),
                "recent_samples": len(recent_vectors)
            }

        # Stack into matrices
        try:
            baseline_mat = np.vstack(baseline_vectors).astype(np.float32)
            recent_mat = np.vstack(recent_vectors).astype(np.float32)
        except Exception as e:
            print(f"   ❌ Failed to stack embeddings: {e}")
            return False, 0.0, {"error": f"stack_error: {e}"}

        # METHOD 1: Centroid Distance
        baseline_centroid = baseline_mat.mean(axis=0)
        recent_centroid = recent_mat.mean(axis=0)

        # Calculate Euclidean distance
        centroid_distance = np.linalg.norm(baseline_centroid - recent_centroid)

        # Normalize by square root of embedding dimension
        normalized_distance = centroid_distance / np.sqrt(len(baseline_centroid))

        print(f"   📊 Centroid distance: {normalized_distance:.6f}")

        # METHOD 2: Variance Change
        baseline_variance = np.var(baseline_mat, axis=0).mean()
        recent_variance = np.var(recent_mat, axis=0).mean()
        variance_ratio = recent_variance / (baseline_variance + 1e-10)

        print(f"   📊 Variance ratio: {variance_ratio:.6f}")

        # METHOD 3: Cluster Analysis
        cluster_shift = 0.0
        try:
            # Reduce dimensionality for clustering (safeguard n_components)
            n_components = min(50, baseline_mat.shape[1], baseline_mat.shape[0] - 1)
            if n_components <= 0:
                n_components = min(baseline_mat.shape[1], 1)

            pca = PCA(n_components=n_components)
            baseline_reduced = pca.fit_transform(baseline_mat)
            recent_reduced = pca.transform(recent_mat)

            # Choose number of clusters (safeguard)
            n_clusters = min(5, max(2, baseline_mat.shape[0] // 3))
            if n_clusters >= baseline_reduced.shape[0]:
                n_clusters = max(2, baseline_reduced.shape[0] // 2)

            kmeans_base = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_recent = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

            kmeans_base.fit(baseline_reduced)
            kmeans_recent.fit(recent_reduced)

            centroid_shift = np.linalg.norm(
                kmeans_base.cluster_centers_ - kmeans_recent.cluster_centers_
            ) / n_clusters

            print(f"   📊 Cluster shift: {centroid_shift:.6f}")

            cluster_shift = float(centroid_shift)

        except Exception as e:
            print(f"   ⚠️ Clustering analysis failed: {e}")
            cluster_shift = 0.0

        # CALCULATE FINAL DRIFT SCORE
        # Weighted combination of all methods
        drift_score = (
            0.5 * normalized_distance +
            0.3 * abs(variance_ratio - 1.0) +
            0.2 * cluster_shift
        )

        # Check if drift exceeds threshold
        is_drifted = drift_score > self.emb_drift_threshold

        # Prepare details
        details = {
            "centroid_distance": float(normalized_distance),
            "variance_ratio": float(variance_ratio),
            "cluster_shift": float(cluster_shift),
            "drift_score": float(drift_score),
            "baseline_samples": int(baseline_mat.shape[0]),
            "recent_samples": int(recent_mat.shape[0]),
            "threshold": float(self.emb_drift_threshold)
        }

        # Print result
        if is_drifted:
            print(f"   ⚠️ DRIFT DETECTED! Score: {drift_score:.6f} > {self.emb_drift_threshold}")
        else:
            print(f"   ✓ No drift. Score: {drift_score:.6f} <= {self.emb_drift_threshold}")

        return is_drifted, drift_score, details

    def detect_behavior_drift(self) -> Tuple[bool, Dict]:
        """
        Detect behavior drift by monitoring refusal and toxicity rates.

        Returns:
            Tuple of (is_drifted, details_dict)
        """
        print("\n🔍 Checking Behavior Drift...")

        try:
            # Get recent interactions (last 7 days)
            cutoff_date = datetime.now() - timedelta(days=7)

            response = self.supabase.table("interaction_log")\
                .select("refusal_flag, toxicity_flag")\
                .gte("timestamp", cutoff_date.isoformat())\
                .execute()

            if not response.data or len(response.data) < 10:
                print("   ⚠️ Not enough interaction data")
                return False, {
                    "reason": "insufficient_data",
                    "sample_count": len(response.data) if response.data else 0
                }

            # Calculate rates
            total = len(response.data)
            refusals = sum(1 for x in response.data if x.get("refusal_flag"))
            toxic = sum(1 for x in response.data if x.get("toxicity_flag"))

            refusal_rate = refusals / total
            toxicity_rate = toxic / total

            print(f"   📊 Refusal rate: {refusal_rate:.2%} (threshold: {self.refusal_threshold:.2%})")
            print(f"   📊 Toxicity rate: {toxicity_rate:.2%} (threshold: {self.toxicity_threshold:.2%})")

            # Check if either rate exceeds threshold
            is_drifted = (
                refusal_rate > self.refusal_threshold or
                toxicity_rate > self.toxicity_threshold
            )

            details = {
                "refusal_rate": float(refusal_rate),
                "toxicity_rate": float(toxicity_rate),
                "total_interactions": int(total),
                "refusal_count": int(refusals),
                "toxic_count": int(toxic),
                "refusal_threshold": float(self.refusal_threshold),
                "toxicity_threshold": float(self.toxicity_threshold)
            }

            if is_drifted:
                reasons = []
                if refusal_rate > self.refusal_threshold:
                    reasons.append(f"High refusals ({refusal_rate:.2%})")
                if toxicity_rate > self.toxicity_threshold:
                    reasons.append(f"High toxicity ({toxicity_rate:.2%})")
                print(f"   ⚠️ BEHAVIOR DRIFT DETECTED! {', '.join(reasons)}")
            else:
                print(f"   ✓ Behavior normal")

            return is_drifted, details

        except Exception as e:
            print(f"   ❌ Error checking behavior: {e}")
            return False, {"error": str(e)}

    def detect_accuracy_drop(self) -> Tuple[bool, Dict]:
        """
        Detect accuracy drop via user feedback scores.
        Compares recent vs baseline performance.

        Returns:
            Tuple of (is_dropped, details_dict)
        """
        print("\n🔍 Checking Accuracy...")

        try:
            # Compare last 7 days vs previous 7 days
            recent_cutoff = datetime.now() - timedelta(days=7)
            baseline_cutoff = datetime.now() - timedelta(days=14)

            # Get recent feedback scores
            recent = self.supabase.table("interaction_log")\
                .select("user_feedback_score")\
                .gte("timestamp", recent_cutoff.isoformat())\
                .not_.is_("user_feedback_score", "null")\
                .execute()

            # Get baseline feedback scores
            baseline = self.supabase.table("interaction_log")\
                .select("user_feedback_score")\
                .gte("timestamp", baseline_cutoff.isoformat())\
                .lt("timestamp", recent_cutoff.isoformat())\
                .not_.is_("user_feedback_score", "null")\
                .execute()

            if not recent.data or not baseline.data:
                print("   ⚠️ Not enough feedback data")
                return False, {
                    "reason": "insufficient_feedback",
                    "recent_samples": len(recent.data) if recent.data else 0,
                    "baseline_samples": len(baseline.data) if baseline.data else 0
                }

            # Calculate average scores
            recent_scores = [float(x["user_feedback_score"]) for x in recent.data]
            baseline_scores = [float(x["user_feedback_score"]) for x in baseline.data]

            recent_avg = float(np.mean(recent_scores))
            baseline_avg = float(np.mean(baseline_scores))
            accuracy_drop = baseline_avg - recent_avg

            print(f"   📊 Baseline accuracy: {baseline_avg:.2f}/5")
            print(f"   📊 Recent accuracy: {recent_avg:.2f}/5")
            print(f"   📊 Drop: {accuracy_drop:.2f} (threshold: {self.accuracy_threshold:.2f})")

            # Check if drop exceeds threshold
            is_dropped = accuracy_drop > self.accuracy_threshold

            details = {
                "baseline_accuracy": float(baseline_avg),
                "recent_accuracy": float(recent_avg),
                "accuracy_drop": float(accuracy_drop),
                "baseline_samples": int(len(baseline_scores)),
                "recent_samples": int(len(recent_scores)),
                "threshold": float(self.accuracy_threshold)
            }

            if is_dropped:
                print(f"   ⚠️ ACCURACY DROP DETECTED!")
            else:
                print(f"   ✓ Accuracy stable or improving")

            return is_dropped, details

        except Exception as e:
            print(f"   ❌ Error checking accuracy: {e}")
            return False, {"error": str(e)}

    def log_drift_event(self, drift_type: str, drift_score: float,
                       threshold: float, action_taken: str, details: Dict):
        """
        Log drift detection event to database.

        Args:
            drift_type: Type of drift detected
            drift_score: Numerical drift score
            threshold: Threshold that was exceeded
            action_taken: Action triggered in response
            details: Additional details dictionary
        """
        try:
            data = {
                "drift_type": drift_type,
                "drift_score": float(drift_score),
                "threshold": float(threshold),
                "action_taken": action_taken,
                "details": details
            }
            self.supabase.table("drift_events").insert(data).execute()
            print(f"   ✓ Logged drift event: {drift_type}")
        except Exception as e:
            print(f"   ❌ Failed to log drift event: {e}")

    def run_full_check(self) -> Dict[str, any]:
        """
        Run all drift checks and return comprehensive results.

        Returns:
            Dictionary with timestamp and list of detected drifts
        """
        print("\n" + "="*70)
        print(f"{'DRIFT MONITORING PIPELINE':^70}")
        print(f"{'Started: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^70}")
        print("="*70)

        results = {
            "timestamp": datetime.now().isoformat(),
            "drifts_detected": [],
            "details": {}
        }

        # CHECK 1: Embedding Drift
        emb_drift, emb_score, emb_details = self.detect_embedding_drift()
        results["details"]["embedding"] = emb_details

        if emb_drift:
            results["drifts_detected"].append("embedding")
            self.log_drift_event(
                "embedding", emb_score, self.emb_drift_threshold,
                "reindex_documents", emb_details
            )

        # CHECK 2: Behavior Drift
        behav_drift, behav_details = self.detect_behavior_drift()
        results["details"]["behavior"] = behav_details

        if behav_drift:
            results["drifts_detected"].append("behavior")
            behav_score = max(
                behav_details.get("refusal_rate", 0),
                behav_details.get("toxicity_rate", 0)
            )
            self.log_drift_event(
                "behavior", behav_score, self.refusal_threshold,
                "retrain_model", behav_details
            )

        # CHECK 3: Accuracy Drift
        acc_drift, acc_details = self.detect_accuracy_drop()
        results["details"]["accuracy"] = acc_details

        if acc_drift:
            results["drifts_detected"].append("accuracy")
            self.log_drift_event(
                "accuracy", acc_details.get("accuracy_drop", 0),
                self.accuracy_threshold, "retrain_model", acc_details
            )

        # Print summary
        print("\n" + "="*70)
        print(f"{'MONITORING COMPLETE':^70}")
        print("="*70)
        print(f"\n✓ Drifts detected: {len(results['drifts_detected'])}")
        if results['drifts_detected']:
            print(f"  Types: {', '.join(results['drifts_detected'])}")
        else:
            print(f"  System is healthy - no drift detected")
        print()

        return results


# Testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DRIFT MONITOR - TEST MODE")
    print("="*70 + "\n")

    monitor = DriftMonitor()
    results = monitor.run_full_check()

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Drifts detected: {results['drifts_detected']}")
    print(f"\nDetails:")
    for drift_type, details in results['details'].items():
        print(f"\n{drift_type.upper()}:")
        for key, value in details.items():
            print(f"  {key}: {value}")

    print("\n✅ Drift monitor test complete!\n")
