from flask import (
    Flask, render_template, request, redirect, url_for, flash
)
import joblib
import sqlite3
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
app.secret_key = "change-this-demo-secret"

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "emergency.db"
MODEL_FILE = BASE_DIR / "emergency_model.pkl"

# Load the trained NLP model
model = joblib.load(MODEL_FILE)

VALID_STATUSES = ["New", "Under Review", "Assigned", "Resolved"]


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message TEXT NOT NULL,
                location TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'New',
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def determine_priority(message, category):
    """
    Demonstration-only keyword rules.
    This does not reliably measure real emergency severity.
    Every report should be reviewed by a responsible person.
    """
    text = message.lower()

    urgent_phrases = [
        "trapped",
        "unconscious",
        "not breathing",
        "severe bleeding",
        "people are injured",
        "life threatening",
        "immediate danger",
        "smoke is spreading",
        "house is on fire",
        "caught fire",
        "difficulty breathing",
        "urgent medical",
        "rising flood water"
    ]

    if any(phrase in text for phrase in urgent_phrases):
        return "High - Verify immediately"

    if category in ["fire", "medical", "flood"]:
        return "Needs urgent human review"

    return "Needs human review"


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        message = request.form.get("message", "").strip()
        location = request.form.get("location", "").strip()

        if not message or not location:
            result = {
                "error": "Please enter both the emergency description and location."
            }
            return render_template("index.html", result=result)

        # Predict the emergency category
        category = model.predict([message])[0]

        # Assign a review label using demo rules
        priority = determine_priority(message, category)

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save the report to the database
        with get_db_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO incidents
                (message, location, category, priority, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                message,
                location,
                category,
                priority,
                "New",
                created_at
            ))

            incident_id = cursor.lastrowid
            conn.commit()

        result = {
            "id": incident_id,
            "message": message,
            "location": location,
            "category": category,
            "priority": priority,
            "status": "New",
            "created_at": created_at
        }

    return render_template("index.html", result=result)


@app.route("/dashboard")
def dashboard():
    with get_db_connection() as conn:
        incidents = conn.execute("""
            SELECT * FROM incidents
            ORDER BY id DESC
        """).fetchall()

        total = conn.execute(
            "SELECT COUNT(*) FROM incidents"
        ).fetchone()[0]

        category_counts = conn.execute("""
            SELECT category, COUNT(*) AS count
            FROM incidents
            GROUP BY category
            ORDER BY count DESC
        """).fetchall()

        status_counts = conn.execute("""
            SELECT status, COUNT(*) AS count
            FROM incidents
            GROUP BY status
        """).fetchall()

    return render_template(
        "dashboard.html",
        incidents=incidents,
        total=total,
        category_counts=category_counts,
        status_counts=status_counts,
        statuses=VALID_STATUSES
    )


@app.route("/update/<int:incident_id>", methods=["POST"])
def update_incident(incident_id):
    new_status = request.form.get("status", "").strip()

    if new_status not in VALID_STATUSES:
        flash("Invalid incident status.")
        return redirect(url_for("dashboard"))

    with get_db_connection() as conn:
        cursor = conn.execute("""
            UPDATE incidents
            SET status = ?
            WHERE id = ?
        """, (new_status, incident_id))
        conn.commit()

    if cursor.rowcount == 0:
        flash("Incident not found.")
    else:
        flash(f"Incident #{incident_id} status updated to {new_status}.")

    return redirect(url_for("dashboard"))


@app.route("/health")
def health():
    return {
        "status": "running",
        "model_loaded": MODEL_FILE.exists()
    }


initialize_db()

if __name__ == "__main__":
    # Local development only. Do not deploy with debug=True.
    app.run(debug=True)

