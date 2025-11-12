# src/triggers.py
"""
TriggerActions

Provides a class-based interface expected by run_pipeline.py:

 - TriggerActions()
 - execute_drift_response(drifts_detected: list[str]) -> dict

drifts_detected should be a list of strings describing which drifts were found,
for example: ["embedding", "behavior", "accuracy"] or more detailed tags.

This module:
 - logs drift events to Supabase (if SUPABASE_URL & SUPABASE_KEY present)
 - executes corrective actions (stubs provided)
 - returns a summary dict with actions taken
"""

import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Optional Supabase client (only used if env vars present)
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

load_dotenv()
logger = logging.getLogger("TriggerActions")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # use service_role for backend scripts in dev
USE_DB = bool(SUPABASE_URL and SUPABASE_KEY and create_client is not None)

if USE_DB:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    if not create_client:
        logger.warning("supabase package not found; DB logging disabled.")
    elif not SUPABASE_URL or not SUPABASE_KEY:
        logger.info("SUPABASE_URL/KEY not configured; DB logging disabled.")


class TriggerActions:
    """
    Class wrapper expected by run_pipeline.py
    """

    def __init__(self):
        # thresholds can be customised via env if you want
        self.embedding_action = os.getenv("EMBEDDING_ACTION", "reindex")  # reserved (unused in this simple adapter)
        self._init_ts = datetime.utcnow().isoformat()
        logger.info("TriggerActions initialized at %s", self._init_ts)

    # ---------------------
    # Public API
    # ---------------------
    def execute_drift_response(self, drifts_detected):
        """
        drifts_detected: list[str] - e.g. ["embedding", "behavior", "accuracy"]
        Returns:
            dict with summary: {"actions_taken": [...], "notes": "..."}
        """
        if not isinstance(drifts_detected, (list, tuple)):
            drifts_detected = [drifts_detected]

        logger.info("execute_drift_response called with drifts: %s", drifts_detected)
        actions_taken = []
        notes = []

        # Basic mapping of drift types -> actions
        for drift in drifts_detected:
            d = drift.lower()
            if "embed" in d or "embedding" in d:
                logger.info("Handling embedding drift...")
                self._log_event("embedding_drift_detected", {"drift": d})
                # action: reindex + refresh retrieval
                self.reindex_documents()
                self.refresh_retrieval()
                actions_taken.append("reindex_documents")
                actions_taken.append("refresh_retrieval")
                notes.append("Re-indexed documents and refreshed retrieval index.")

            elif "behav" in d or "behavior" in d or "refusal" in d or "toxic" in d:
                logger.info("Handling behavior drift (refusal/toxicity)...")
                self._log_event("behavior_drift_detected", {"drift": d})
                # attempt minimal intervention first: adjust prompt/policy
                self.update_system_prompt("Adjust system prompt to reduce unnecessary refusals / clarify policy")
                actions_taken.append("update_system_prompt")
                notes.append("Updated system prompt as first-line remediation.")
                # if toxicity explicitly included, also schedule fine-tune
                if "toxic" in d or "toxicity" in d:
                    self.fine_tune_model()
                    actions_taken.append("fine_tune_model")
                    notes.append("Scheduled fine-tune to improve safety.")

            elif "accur" in d or "accuracy" in d or "perf" in d:
                logger.info("Handling accuracy drop...")
                self._log_event("accuracy_drop_detected", {"drift": d})
                # retrain / fine-tune
                self.fine_tune_model()
                actions_taken.append("fine_tune_model")
                notes.append("Triggered fine-tune to recover accuracy.")

            else:
                # Unknown drift label: log and be conservative
                logger.info("Unknown drift type '%s' - logging only", d)
                self._log_event("unknown_drift_detected", {"drift": d})
                notes.append(f"Unknown drift type: {d}")

        # Log a summary event
        self._log_event("actions_taken", {"actions": actions_taken, "notes": notes})

        summary = {"actions_taken": actions_taken, "notes": notes}
        logger.info("execute_drift_response summary: %s", summary)
        return summary

    # ---------------------
    # Action implementations (stubs - replace with your infra)
    # ---------------------
    def reindex_documents(self):
        """
        Re-run your ingestion pipeline and regenerate embeddings.
        Replace this body with a call to your real ingestion process.
        """
        logger.info("ACTION: reindex_documents() - start")
        # Example: run a local script or HTTP call to your ingestion endpoint:
        # subprocess.run(["python", "scripts/reindex.py"], check=True)
        time.sleep(0.8)
        logger.info("ACTION: reindex_documents() - done")

    def refresh_retrieval(self):
        """
        Refresh in-memory indexes or rebuild vector indices.
        """
        logger.info("ACTION: refresh_retrieval() - start")
        # Example: refresh a Faiss index, or call your retrieval service API
        time.sleep(0.4)
        logger.info("ACTION: refresh_retrieval() - done")

    def fine_tune_model(self):
        """
        Kick off a fine-tune job or train a LoRA adapter.
        In production, queue a job and return; here we simulate.
        """
        logger.info("ACTION: fine_tune_model() - starting (simulated)")
        # Example: queue job via Celery, or call cloud training API
        time.sleep(1.5)
        logger.info("ACTION: fine_tune_model() - scheduled/completed (simulated)")

    def update_system_prompt(self, new_prompt: str):
        """
        Update system prompt or safety/policy config. Replace with your deployment logic.
        """
        preview = (new_prompt[:200] + "...") if len(new_prompt) > 200 else new_prompt
        logger.info("ACTION: update_system_prompt() - preview: %s", preview)
        # e.g., write to config file, push to S3, or call config API
        time.sleep(0.2)
        logger.info("ACTION: update_system_prompt() - applied (simulated)")

    # ---------------------
    # DB logging helper
    # ---------------------
    def _log_event(self, event_type: str, details: dict):
        """
        Log an event to Supabase drift_events table (if available), otherwise prints.
        details can be any JSON-serializable dict.
        """
        payload = {
            "event_type": event_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        if supabase:
            try:
                res = supabase.table("drift_events").insert(payload).execute()
                logger.info("Logged event to DB: %s", event_type)
            except Exception as e:
                logger.exception("Failed to write drift event to Supabase: %s", e)
                logger.info("Event (local): %s %s", event_type, details)
        else:
            # fallback - print to console (useful for dev)
            logger.info("DRIFT EVENT (mock): %s %s", event_type, details)

# End of file
