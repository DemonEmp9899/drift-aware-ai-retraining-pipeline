"""
Metrics Exporter - Export metrics to Prometheus
==============================================
Exposes metrics endpoint for Prometheus to scrape.
"""

import os
import time
from datetime import datetime, timedelta
from prometheus_client import start_http_server, Gauge, Counter, Info
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


class MetricsExporter:
    """
    Exports drift detection and model performance metrics to Prometheus.
    
    Metrics include:
    - Embedding drift score
    - Refusal and toxicity rates
    - Model accuracy
    - API costs
    - Retrain events
    """
    
    def __init__(self, port: int = 8000):
        """
        Initialize metrics exporter.
        
        Args:
            port: Port number for metrics server
        """
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )
        self.port = port
        
        # Define Prometheus metrics
        self.embedding_drift_score = Gauge(
            'embedding_drift_score',
            'Current embedding drift score (0-1 scale)'
        )
        
        self.refusal_rate = Gauge(
            'model_refusal_rate',
            'Percentage of queries refused by model'
        )
        
        self.toxicity_rate = Gauge(
            'model_toxicity_rate',
            'Percentage of toxic outputs'
        )
        
        self.model_accuracy = Gauge(
            'model_accuracy_score',
            'Current model accuracy from user feedback (0-5 scale)'
        )
        
        self.total_interactions = Counter(
            'total_interactions',
            'Total number of interactions logged'
        )
        
        self.retrain_events = Counter(
            'retrain_events_total',
            'Number of retraining events triggered'
        )
        
        self.reindex_events = Counter(
            'reindex_events_total',
            'Number of re-indexing events triggered'
        )
        
        self.total_cost_usd = Gauge(
            'openai_total_cost_usd',
            'Total cumulative API cost in USD'
        )
        
        self.avg_tokens_per_query = Gauge(
            'avg_tokens_per_query',
            'Average tokens used per query'
        )
        
        self.model_info = Info(
            'model_version',
            'Current active model version information'
        )
        
        # Track last update
        self.last_update = datetime.now()
        
        print(f"✓ MetricsExporter initialized on port {port}")
    
    def update_metrics(self):
        """Fetch latest data from Supabase and update all metrics."""
        try:
            update_start = time.time()
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"\n[{timestamp}] 🔄 Updating metrics...")
            
            # Get recent period (last 24 hours)
            cutoff = datetime.now() - timedelta(days=1)
            
            # === INTERACTION METRICS ===
            interactions = self.supabase.table("interaction_log")\
                .select("*")\
                .gte("timestamp", cutoff.isoformat())\
                .execute()
            
            if interactions.data:
                data = interactions.data
                total = len(data)
                
                # Refusal rate
                refusals = sum(1 for x in data if x.get("refusal_flag"))
                refusal_rate = refusals / total if total > 0 else 0
                self.refusal_rate.set(refusal_rate)
                
                # Toxicity rate
                toxic = sum(1 for x in data if x.get("toxicity_flag"))
                toxic_rate = toxic / total if total > 0 else 0
                self.toxicity_rate.set(toxic_rate)
                
                # Average accuracy (from feedback)
                feedback_scores = [
                    x["user_feedback_score"] for x in data 
                    if x.get("user_feedback_score") is not None
                ]
                if feedback_scores:
                    avg_accuracy = sum(feedback_scores) / len(feedback_scores)
                    self.model_accuracy.set(avg_accuracy)
                    print(f"   📊 Accuracy: {avg_accuracy:.2f}/5 (from {len(feedback_scores)} feedbacks)")
                
                # Token usage
                total_tokens = sum(x.get("tokens_used", 0) for x in data)
                if total > 0:
                    avg_tokens = total_tokens / total
                    self.avg_tokens_per_query.set(avg_tokens)
                    print(f"   📊 Avg tokens/query: {avg_tokens:.0f}")
                
                # Cost metrics
                total_cost = sum(x.get("cost_usd", 0) for x in data)
                self.total_cost_usd.set(total_cost)
                print(f"   💰 Total cost (24h): ${total_cost:.4f}")
                
                print(f"   ✓ Interactions: {total}, Refusals: {refusals} ({refusal_rate:.1%}), Toxic: {toxic} ({toxic_rate:.1%})")
            
            # === DRIFT EVENTS ===
            drift_events = self.supabase.table("drift_events")\
                .select("*")\
                .gte("timestamp", cutoff.isoformat())\
                .execute()
            
            if drift_events.data:
                # Count event types
                retrains = sum(
                    1 for x in drift_events.data 
                    if "retrain" in x.get("action_taken", "").lower()
                )
                reindexes = sum(
                    1 for x in drift_events.data 
                    if "reindex" in x.get("action_taken", "").lower()
                )
                
                # Update counters (increment by new events since last update)
                if retrains > 0:
                    self.retrain_events.inc(retrains)
                if reindexes > 0:
                    self.reindex_events.inc(reindexes)
                
                # Get latest embedding drift score
                embedding_drifts = [
                    x for x in drift_events.data 
                    if x.get("drift_type") == "embedding"
                ]
                if embedding_drifts:
                    # Get most recent drift score
                    latest_drift = embedding_drifts[-1]
                    drift_score = latest_drift.get("drift_score", 0)
                    self.embedding_drift_score.set(drift_score)
                    print(f"   📈 Embedding drift: {drift_score:.4f}")
                
                print(f"   ✓ Drift events (24h): {len(drift_events.data)} (Retrains: {retrains}, Reindexes: {reindexes})")
            
            # === MODEL VERSION ===
            active_model = self.supabase.table("model_versions")\
                .select("version_name, accuracy, avg_tokens")\
                .eq("is_active", True)\
                .execute()
            
            if active_model.data:
                model_data = active_model.data[0]
                self.model_info.info({
                    'version': model_data.get("version_name", "unknown"),
                    'accuracy': str(model_data.get("accuracy", 0)),
                    'avg_tokens': str(model_data.get("avg_tokens", 0))
                })
                print(f"   ✓ Active model: {model_data.get('version_name')}")
            
            update_time = time.time() - update_start
            print(f"   ⏱️  Update completed in {update_time:.2f}s")
            
            self.last_update = datetime.now()
            
        except Exception as e:
            print(f"   ❌ Error updating metrics: {e}")
            import traceback
            traceback.print_exc()
    
    def start_server(self, update_interval: int = 60):
        """
        Start Prometheus metrics server and update loop.
        
        Args:
            update_interval: Seconds between metric updates (default: 60)
        """
        # Start HTTP server for Prometheus to scrape
        start_http_server(self.port)
        
        print("\n" + "="*70)
        print(f"{'🚀 METRICS SERVER STARTED':^70}")
        print("="*70)
        print(f"\n📊 Prometheus endpoint: http://localhost:{self.port}/metrics")
        print(f"🔄 Update interval: {update_interval}s")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n💡 Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        # Initial update
        self.update_metrics()
        
        # Continuous update loop
        try:
            while True:
                time.sleep(update_interval)
                self.update_metrics()
                
        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print(f"{'👋 METRICS SERVER STOPPED':^70}")
            print("="*70)
            print(f"Last update: {self.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total runtime: {(datetime.now() - self.last_update).total_seconds():.0f}s")
            print("="*70 + "\n")
    
    def get_current_metrics(self) -> dict:
        """
        Get current metric values as dictionary (for testing/debugging).
        
        Returns:
            Dictionary of current metric values
        """
        return {
            "embedding_drift_score": self.embedding_drift_score._value.get(),
            "refusal_rate": self.refusal_rate._value.get(),
            "toxicity_rate": self.toxicity_rate._value.get(),
            "model_accuracy": self.model_accuracy._value.get(),
            "avg_tokens_per_query": self.avg_tokens_per_query._value.get(),
            "total_cost_usd": self.total_cost_usd._value.get(),
        }


# Main execution
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PROMETHEUS_PORT", "8000"))
    
    # Create and start metrics exporter
    exporter = MetricsExporter(port=port)
    
    # Update every 30 seconds (configurable)
    exporter.start_server(update_interval=30)
