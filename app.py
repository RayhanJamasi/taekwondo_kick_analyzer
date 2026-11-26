import json
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path

import cv2
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
)

from main import run_pose_detection


# Finding this app.py file first, then building the data paths from its folder
app = Flask(__name__)
data_dir = Path(__file__).resolve().parent / "data"
database_path = data_dir / "sessions.db"
session_lock = threading.Lock()
current_session = None


# Making the data folder and the table if they aren't there already
def init_db():
    data_dir.mkdir(exist_ok=True)
    with closing(sqlite3.connect(database_path)) as db:
        with db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    kick_count INTEGER NOT NULL,
                    right_kicks INTEGER NOT NULL,
                    left_kicks INTEGER NOT NULL,
                    kick_results TEXT NOT NULL
                )
            """)


# Getting the most recent session results
def get_history():
    with closing(sqlite3.connect(database_path)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT session_id, started_at, ended_at, kick_count, right_kicks, left_kicks
            FROM sessions
            ORDER BY ended_at DESC
            LIMIT 20
        """).fetchall()
    return [dict(row) for row in rows]


def get_recording_datetime(folder):
    # Kick folder names end with a timestamp and the kicking side
    try:
        timestamp, side = folder.name.rsplit("_", 1)
        recorded_at = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None

    if side not in ("left", "right"):
        return None
    return recorded_at


def get_recording_folders():
    # Looking through folders that hold old clips or new camera-session folders
    for folder in data_dir.iterdir():
        if not folder.is_dir():
            continue
        if folder.name == "sessions":
            # New clips are inside a camera-session folder, so go one level deeper
            for session_folder in folder.iterdir():
                if session_folder.is_dir():
                    for kick_folder in session_folder.iterdir():
                        if kick_folder.is_dir():
                            # yield gives back one folder at a time as the loop finds it
                            yield kick_folder
        else:
            # Old clips were saved directly inside data
            yield folder


# Getting the dates that have saved sessions or kicks
def get_session_dates():
    # Asking SQLite for the dates when a camera session finished
    with closing(sqlite3.connect(database_path)) as db:
        session_dates = db.execute(
            "SELECT DISTINCT substr(ended_at, 1, 10) FROM sessions"
        ).fetchall()

    # A set keeps each date only once
    dates = {row[0] for row in session_dates}
    # Adding dates from the kick folders too, including older clips
    for folder in get_recording_folders():
        recorded_at = get_recording_datetime(folder)
        if recorded_at is not None:
            dates.add(recorded_at.strftime("%Y-%m-%d"))

    # Showing the newest date first in the date dropdown
    return sorted(dates, reverse=True)


# Grouping saved kicks by camera session for this date
def get_recorded_kicks(selected_date):
    groups = {}
    # Checking each saved kick folder and keeping only the chosen date
    for folder in get_recording_folders():
        recorded_at = get_recording_datetime(folder)
        metrics_path = folder / "metrics.json"
        if recorded_at is None or recorded_at.strftime("%Y-%m-%d") != selected_date:
            continue
        if not metrics_path.is_file():
            continue

        with metrics_path.open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)

        if folder.parent.parent == data_dir / "sessions":
            session_time = folder.parent.name[:19]
            try:
                session_time = datetime.strptime(
                    session_time, "%Y-%m-%d_%H-%M-%S"
                ).strftime("%Y-%m-%d %I:%M %p")
            except ValueError:
                session_time = folder.parent.name
            group_name = f"Camera session {session_time} ({folder.parent.name[-8:]})"
        else:
            group_name = "Older kicks (not grouped)"

        # Adding this kick to the list for its camera session
        groups.setdefault(group_name, []).append({
            "clip_id": folder.relative_to(data_dir).as_posix(),
            "recorded_at": recorded_at.strftime("%I:%M:%S %p"),
            "metrics": metrics,
            "has_video": (folder / "kick.mp4").is_file(),
        })

    # Turning the grouped lists into the shape the page uses
    return [
        {
            "name": group_name,
            "kicks": sorted(kicks, key=lambda kick: kick["clip_id"], reverse=True),
        }
        for group_name, kicks in groups.items()
    ]


# Getting the web sessions that finished on this date
def get_sessions_for_date(selected_date):
    with closing(sqlite3.connect(database_path)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("""
            SELECT started_at, kick_count, right_kicks, left_kicks
            FROM sessions
            WHERE substr(ended_at, 1, 10) = ?
            ORDER BY ended_at DESC
        """, (selected_date,)).fetchall()

    return [dict(row) for row in rows]


# Saving the finished camera session in SQLite
def save_session(session):
    with session["lock"]:
        values = (
            session["id"],
            session["started_at"],
            session["ended_at"],
            session["kick_count"],
            session["right_kicks"],
            session["left_kicks"],
            json.dumps(session["kick_results"]),
        )

    with closing(sqlite3.connect(database_path)) as db:
        with db:
            db.execute("""
                INSERT INTO sessions
                (session_id, started_at, ended_at, kick_count, right_kicks, left_kicks, kick_results)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, values)


# Running the camera analysis in the background
def run_session(session):
    camera = None
    try:
        # Opening the computer's default camera
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Could not open the camera. Check that it is connected and not in use.")

        run_pose_detection(
            show_coords=True,
            # These lambdas pass each new frame/result to the helper below
            frame_callback=lambda frame: update_frame(session, frame),
            metrics_callback=lambda metrics: update_metrics(session, metrics),
            stop_event=session["stop_event"],
            display=False,
            camera=camera,
            recording_dir=session["recording_dir"],
        )
    except Exception as error:
        with session["lock"]:
            session["error"] = str(error)
    finally:
        # Releasing the camera when analysis ends, even if an error happened
        if camera is not None:
            camera.release()

        # Copying the session values while the worker thread isn't changing them
        with session["lock"]:
            session["running"] = False
            session["ended_at"] = datetime.now().isoformat(timespec="seconds")

        try:
            save_session(session)
        except sqlite3.Error as error:
            with session["lock"]:
                session["error"] = f"Could not save session history: {error}"


# Converting a camera frame so the browser can show it
def update_frame(session, frame):
    # Turning the OpenCV image into JPG bytes the browser can display
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Could not encode the camera frame.")

    with session["lock"]:
        session["latest_frame"] = encoded.tobytes()
        session["frame_number"] += 1


# Updating the session totals when a kick finishes
def update_metrics(session, metrics):
    # Keeping the newest kick and updating the totals for this session
    with session["lock"]:
        session["latest_metrics"] = metrics
        session["kick_results"].append(metrics)
        session["kick_count"] += 1
        if metrics["side"] == "right":
            session["right_kicks"] += 1
        else:
            session["left_kicks"] += 1


# Showing the analyzer page and recent sessions
@app.route("/")
def index():
    return render_template("index.html", history=get_history())


# Showing saved kicks for the selected date
@app.route("/sessions")
def sessions():
    dates = get_session_dates()
    # Use the date from the link, or start with the newest one
    selected_date = request.args.get("date", dates[0] if dates else "")
    if selected_date and selected_date not in dates:
        abort(404)

    kick_groups = get_recorded_kicks(selected_date) if selected_date else []
    kick_count = sum(len(group["kicks"]) for group in kick_groups)
    session_history = get_sessions_for_date(selected_date) if selected_date else []
    return render_template(
        "sessions.html",
        dates=dates,
        selected_date=selected_date,
        kick_groups=kick_groups,
        kick_count=kick_count,
        sessions=session_history,
    )


# Opening or deleting one saved kick
@app.route("/recordings/<path:clip_id>", methods=["GET", "DELETE"])
def recording(clip_id):
    # Making sure the requested clip stays inside the data folder
    recording_dir = (data_dir / clip_id).resolve()
    recorded_at = get_recording_datetime(recording_dir)
    try:
        relative_path = recording_dir.relative_to(data_dir.resolve())
    except ValueError:
        abort(404)

    if recorded_at is None or len(relative_path.parts) not in (1, 3):
        abort(404)
    if len(relative_path.parts) == 3 and relative_path.parts[0] != "sessions":
        abort(404)

    if request.method == "DELETE":
        if not recording_dir.is_dir():
            abort(404)
        # Removing this kick's video and metrics
        shutil.rmtree(recording_dir)
        return jsonify({"deleted": True})

    return send_from_directory(
        recording_dir,
        "kick.mp4",
        mimetype="video/mp4",
        as_attachment=request.args.get("download") == "1",
        download_name=f"{recording_dir.name}.mp4",
    )


# Starting a new camera session
@app.route("/start", methods=["POST"])
def start():
    global current_session

    with session_lock:
        if current_session is not None and current_session["running"]:
            return jsonify({"error": "Analysis is already running."}), 409

        # Giving this camera session its own ID and folder
        session_id = uuid.uuid4().hex
        session_started = datetime.now()
        current_session = {
            "id": session_id,
            "started_at": session_started.isoformat(timespec="seconds"),
            "ended_at": None,
            "running": True,
            # Keeping all the kicks from this camera session in one folder
            "recording_dir": data_dir / "sessions" / (
                f"{session_started.strftime('%Y-%m-%d_%H-%M-%S')}_{session_id[:8]}"
            ),
            "stop_event": threading.Event(),
            "lock": threading.Lock(),
            "latest_frame": None,
            "frame_number": 0,
            "latest_metrics": None,
            "kick_results": [],
            "kick_count": 0,
            "right_kicks": 0,
            "left_kicks": 0,
            "error": None,
        }
        current_session["thread"] = threading.Thread(
            target=run_session,
            args=(current_session,),
            daemon=True,
        )
        current_session["thread"].start()

    return jsonify({"session_id": current_session["id"]})


# Stopping the camera session
@app.route("/stop", methods=["POST"])
def stop():
    with session_lock:
        session = current_session

    if session is None or not session["running"]:
        return jsonify({"error": "Analysis is not running."}), 409

    # Telling the camera loop to stop, then waiting for it to finish
    session["stop_event"].set()
    session["thread"].join(timeout=10)
    if session["thread"].is_alive():
        return jsonify({"error": "Analysis is still stopping."}), 202
    return jsonify({"stopped": True})


# Sending the latest analysis results to the page
@app.route("/status")
def status():
    with session_lock:
        session = current_session

    if session is None:
        return jsonify({"running": False, "latest_metrics": None})

    # Reading shared session values safely while the camera thread updates them
    with session["lock"]:
        return jsonify({
            "running": session["running"],
            "latest_metrics": session["latest_metrics"],
            "kick_count": session["kick_count"],
            "right_kicks": session["right_kicks"],
            "left_kicks": session["left_kicks"],
            "error": session["error"],
        })


# Sending the saved session history to the page
@app.route("/history")
def history():
    return jsonify(get_history())


# Sending processed camera frames to the browser
@app.route("/video_feed")
def video_feed():
    session_id = request.args.get("session_id")
    with session_lock:
        session = current_session

    if session is None or session["id"] != session_id:
        abort(404)

    def frames():
        last_frame_number = -1
        while True:
            # Getting the newest frame and checking if the camera is still running
            with session["lock"]:
                frame = session["latest_frame"]
                frame_number = session["frame_number"]
                running = session["running"]

            if frame is not None and frame_number != last_frame_number:
                last_frame_number = frame_number
                # Sending this JPG frame as part of the live browser feed
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            elif not running:
                break
            else:
                time.sleep(0.03)

    return Response(
        stream_with_context(frames()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
