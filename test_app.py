import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import app as web_app


class TestWebApp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "data"
        self.data_dir.mkdir()
        self.database_path = self.data_dir / "sessions.db"
        self.old_data_dir = web_app.data_dir
        self.old_database_path = web_app.database_path
        web_app.data_dir = self.data_dir
        web_app.database_path = self.database_path
        web_app.init_db()
        web_app.current_session = None
        web_app.app.config["TESTING"] = True
        self.client = web_app.app.test_client()

    def tearDown(self):
        if web_app.current_session is not None:
            web_app.current_session["stop_event"].set()
            web_app.current_session["thread"].join(timeout=2)
        web_app.data_dir = self.old_data_dir
        web_app.database_path = self.old_database_path
        self.temp_dir.cleanup()

    # Checking that a session saves its results and releases the camera
    def test_analysis_saves_session_and_releases_camera(self):
        camera = Mock()
        camera.isOpened.return_value = True
        analysis_started = threading.Event()
        recording_dirs = []

        def fake_analysis(**kwargs):
            recording_dirs.append(kwargs["recording_dir"])
            kwargs["frame_callback"](np.zeros((2, 2, 3), dtype=np.uint8))
            kwargs["metrics_callback"]({
                "side": "right",
                "max_knee_angle": 165,
                "max_kick_height": 0.4,
                "duration": 0.5,
                "feedback": {
                    "height": "Good height!",
                    "knee": "Nice knee extension!",
                    "speed": "Oh good job, you're fast!",
                },
            })
            analysis_started.set()
            kwargs["stop_event"].wait(timeout=2)

        with patch.object(web_app.cv2, "VideoCapture", return_value=camera):
            with patch.object(web_app, "run_pose_detection", side_effect=fake_analysis):
                response = self.client.post("/start")
                self.assertEqual(response.status_code, 200)
                self.assertTrue(analysis_started.wait(timeout=2))

                status = self.client.get("/status").get_json()
                self.assertEqual(status["kick_count"], 1)
                self.assertEqual(status["latest_metrics"]["side"], "right")

                feed = self.client.get(
                    f"/video_feed?session_id={response.get_json()['session_id']}",
                    buffered=False,
                )
                self.assertIn(b"Content-Type: image/jpeg", next(feed.response))
                feed.close()

                response = self.client.post("/stop")
                self.assertEqual(response.status_code, 200)

        self.assertTrue(camera.release.called)
        self.assertEqual(recording_dirs[0].parent, self.data_dir / "sessions")
        history = self.client.get("/history").get_json()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["right_kicks"], 1)
        self.assertEqual(history[0]["left_kicks"], 0)

    # Checking that the analyzer page opens
    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Session history", response.data)

    # Checking that older saved kicks still show in the archive
    def test_date_page_shows_saved_video_and_download_link(self):
        clip_id = "2026-11-25_12-00-00_right"
        clip_dir = self.data_dir / clip_id
        clip_dir.mkdir()
        with (clip_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
            json.dump({
                "side": "right",
                "max_knee_angle": 165,
                "max_kick_height": 0.4,
                "duration": 0.5,
                "feedback": {
                    "height": "Good height!",
                    "knee": "Nice knee extension!",
                    "speed": "Oh good job, you're fast!",
                },
            }, metrics_file)
        (clip_dir / "kick.mp4").write_bytes(b"test video")

        response = self.client.get("/sessions?date=2026-11-25")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"right side", response.data)
        self.assertIn(b"Open video", response.data)

        response = self.client.get(f"/recordings/{clip_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"test video")
        response.close()

        response = self.client.get(f"/recordings/{clip_id}?download=1")
        self.assertIn("attachment", response.headers["Content-Disposition"])
        response.close()

    # Checking that grouped kicks can be viewed and deleted
    def test_grouped_recording_can_be_viewed_and_deleted(self):
        clip_dir = self.data_dir / "sessions" / "2026-11-25_12-00-00_ab12cd34" / "2026-11-25_12-00-05_left"
        clip_dir.mkdir(parents=True)
        with (clip_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
            json.dump({
                "side": "left",
                "max_knee_angle": 165,
                "max_kick_height": 0.4,
                "duration": 0.5,
                "feedback": {
                    "height": "Good height!",
                    "knee": "Nice knee extension!",
                    "speed": "Oh good job, you're fast!",
                },
            }, metrics_file)
        (clip_dir / "kick.mp4").write_bytes(b"test video")

        response = self.client.get("/sessions?date=2026-11-25")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Camera session 2026-11-25 12:00 PM (ab12cd34)", response.data)

        clip_id = "sessions/2026-11-25_12-00-00_ab12cd34/2026-11-25_12-00-05_left"
        self.assertIn(f'href="/recordings/{clip_id}"'.encode(), response.data)
        response = self.client.get(f"/recordings/{clip_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"test video")
        response.close()

        response = self.client.delete(f"/recordings/{clip_id}")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(clip_dir.exists())

    # Making sure invalid paths cannot access or delete recordings
    def test_recording_route_rejects_invalid_clip_path(self):
        response = self.client.get("/recordings/..%2F..%2FREADME.md")
        self.assertEqual(response.status_code, 404)

        response = self.client.delete("/recordings/sessions")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
