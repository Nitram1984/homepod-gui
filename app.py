#!/usr/bin/env python3
"""Kleine lokale HomePod GUI auf Basis von http.server."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import base64
import binascii
import hmac
import ssl
from urllib.parse import unquote
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


def _atvremote_candidates() -> list[str]:
    candidates: list[str] = []
    env_bin = os.environ.get("ATVREMOTE_BIN", "").strip()
    if env_bin:
        candidates.append(env_bin)
    for candidate in (
        shutil.which("atvremote"),
        str(Path.home() / ".local" / "bin" / "atvremote"),
        "/usr/local/bin/atvremote",
        "/usr/bin/atvremote",
        "atvremote",
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _resolve_atvremote_bin() -> tuple[str, bool]:
    for candidate in _atvremote_candidates():
        if "/" in candidate:
            if Path(candidate).exists():
                return candidate, True
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved, True
    fallback = os.environ.get("ATVREMOTE_BIN", "").strip() or "atvremote"
    return fallback, False


ATVREMOTE_BIN, ATVREMOTE_EXISTS = _resolve_atvremote_bin()
DEFAULT_IP = os.environ.get("HOMEPOD_IP", "")
DEFAULT_NAME = os.environ.get("HOMEPOD_NAME", "HomePod")
DEFAULT_TV_IP = os.environ.get("APPLETV_IP", "")
DEFAULT_TV_NAME = os.environ.get("APPLETV_NAME", "Apple TV")
DEFAULT_HOST = os.environ.get("HOMEPOD_GUI_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("HOMEPOD_GUI_PORT", "8788"))
DEFAULT_AUTH_USER = os.environ.get("HOMEPOD_GUI_AUTH_USER", "").strip()
DEFAULT_AUTH_PASSWORD = os.environ.get("HOMEPOD_GUI_AUTH_PASSWORD", "")
DEFAULT_TLS_CERT = Path(
    os.environ.get("HOMEPOD_GUI_TLS_CERT", str(Path.home() / ".config" / "homepod-gui" / "cert.pem"))
).expanduser()
DEFAULT_TLS_KEY = Path(
    os.environ.get("HOMEPOD_GUI_TLS_KEY", str(Path.home() / ".config" / "homepod-gui" / "key.pem"))
).expanduser()
UPLOADS_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_BYTES = int(os.environ.get("MEDIA_UPLOAD_MAX_BYTES", str(512 * 1024 * 1024)))
SCREENCAST_DIR = BASE_DIR / "screencast"
SCREENCAST_HTTP_PORT = int(os.environ.get("SCREENCAST_HTTP_PORT", "8911"))
SCREENCAST_DEFAULT_SIZE = os.environ.get("SCREENCAST_DEFAULT_SIZE", "1280x720")
SCREENCAST_DEFAULT_FPS = int(os.environ.get("SCREENCAST_DEFAULT_FPS", "22"))
SCREENCAST_DEFAULT_DISPLAY = os.environ.get(
    "SCREENCAST_DEFAULT_DISPLAY",
    os.environ.get("DISPLAY", ":0.0"),
)
SCREENCAST_HOST_IP = os.environ.get("SCREENCAST_HOST_IP", "").strip()

ALLOWED_MEDIA_EXTENSIONS = {
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".wav",
    ".ogg",
    ".opus",
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
}

REMOTE_ACTIONS = {
    "play_pause": "play_pause",
    "play": "play",
    "pause": "pause",
    "stop": "stop",
    "next": "next",
    "previous": "previous",
    "skip_forward": "skip_forward",
    "skip_backward": "skip_backward",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "back": "menu",
    "select": "select",
    "menu": "menu",
    "home": "home",
    "home_hold": "home_hold",
    "top_menu": "top_menu",
    "control_center": "control_center",
    "volume_up": "volume_up",
    "volume_down": "volume_down",
}

TV_SERVICES = {
    "netflix": {"label": "Netflix", "bundle_id": "com.netflix.Netflix"},
    "amazon": {"label": "Prime Video", "bundle_id": "com.amazon.aiv.AIVApp"},
    "joyn": {"label": "Joyn", "bundle_id": "de.prosiebensat1digital.seventv"},
    "spotify": {"label": "Spotify", "bundle_id": "com.spotify.client"},
}


def _ensure_self_signed_cert(cert_path: Path, key_path: Path) -> None:
    if cert_path.exists() and key_path.exists():
        return

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "openssl",
        "req",
        "-x509",
        "-nodes",
        "-newkey",
        "rsa:2048",
        "-days",
        "3650",
        "-subj",
        "/CN=localhost",
        "-keyout",
        str(key_path),
        "-out",
        str(cert_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=20)
    except FileNotFoundError as exc:
        raise RuntimeError("openssl ist nicht installiert") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("openssl timeout beim Erstellen des TLS-Zertifikats") from exc

    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip() or "unbekannter openssl Fehler"
        raise RuntimeError(f"TLS-Zertifikat konnte nicht erstellt werden: {stderr}")

    os.chmod(key_path, 0o600)
    os.chmod(cert_path, 0o644)


@dataclass
class DeviceConfig:
    ip: str
    name: str


class ConfigStore:
    def __init__(self, ip: str, name: str) -> None:
        self._lock = threading.Lock()
        self._config = DeviceConfig(ip=ip, name=name)

    def get(self) -> DeviceConfig:
        with self._lock:
            return DeviceConfig(ip=self._config.ip, name=self._config.name)

    def set(self, ip: str | None, name: str | None) -> DeviceConfig:
        with self._lock:
            if ip is not None:
                self._config.ip = ip
            if name is not None:
                self._config.name = name
            return DeviceConfig(ip=self._config.ip, name=self._config.name)


class UploadHistoryStore:
    def __init__(self, history_file: Path, max_items: int = 60) -> None:
        self._lock = threading.Lock()
        self._history_file = history_file
        self._max_items = max_items
        self._items: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self._history_file.exists():
            return
        try:
            raw = json.loads(self._history_file.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._items = [item for item in raw if isinstance(item, dict)]
        except Exception:  # noqa: BLE001
            self._items = []

    def _save(self) -> None:
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        self._history_file.write_text(
            json.dumps(self._items[: self._max_items], ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def add(self, entry: dict[str, Any]) -> None:
        with self._lock:
            path = str(entry.get("path", ""))
            self._items = [item for item in self._items if str(item.get("path", "")) != path]
            self._items.insert(0, entry)
            self._items = self._items[: self._max_items]
            self._save()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._items]


class StreamManager:
    def __init__(
        self,
        on_complete: Callable[[int, str | None, str], None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._started_at: float | None = None
        self._path: str | None = None
        self._last_exit_code: int | None = None
        self._last_log: str = ""
        self._on_complete = on_complete

    def _watch(self, proc: subprocess.Popen[str]) -> None:
        stdout, stderr = proc.communicate()
        combined = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part)
        finished_path: str | None = None
        should_callback = False
        with self._lock:
            if self._proc is proc:
                finished_path = self._path
                self._proc = None
                self._last_exit_code = proc.returncode
                self._last_log = combined[-4000:]
                self._started_at = None
                should_callback = True
        if should_callback and self._on_complete is not None:
            try:
                self._on_complete(proc.returncode, finished_path, combined[-4000:])
            except Exception:
                pass

    def _stop_process(self, proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

    def stop_local_process(self) -> None:
        with self._lock:
            proc = self._proc
        if proc is None:
            return
        self._stop_process(proc)
        with self._lock:
            if self._proc is proc:
                self._proc = None
                self._started_at = None

    def start(self, command: list[str], path: str) -> None:
        with self._lock:
            old_proc = self._proc
        if old_proc is not None:
            self._stop_process(old_proc)

        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        with self._lock:
            self._proc = proc
            self._started_at = time.time()
            self._path = path
            self._last_exit_code = None
            self._last_log = ""

        watcher = threading.Thread(target=self._watch, args=(proc,), daemon=True)
        watcher.start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            active = self._proc is not None and self._proc.poll() is None
            return {
                "active": active,
                "path": self._path,
                "started_at": self._started_at,
                "last_exit_code": self._last_exit_code,
                "last_log": self._last_log,
            }


class QueueManager:
    def __init__(self, start_stream: Callable[[str], None]) -> None:
        self._lock = threading.Lock()
        self._items: list[str] = []
        self._current: str | None = None
        self._enabled = False
        self._start_stream = start_stream

    def _snapshot_unlocked(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "current": self._current,
            "items": list(self._items),
            "count": len(self._items),
        }

    def _start_next_locked(self) -> str | None:
        if not self._items:
            self._enabled = False
            self._current = None
            return None
        next_path = self._items.pop(0)
        self._current = next_path
        self._enabled = True
        return next_path

    def add(self, path: str) -> dict[str, Any]:
        with self._lock:
            self._items.append(path)
        return self.snapshot()

    def clear(self) -> dict[str, Any]:
        with self._lock:
            self._items = []
            self._enabled = False
            self._current = None
        return self.snapshot()

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
            self._current = None

    def start(self) -> dict[str, Any]:
        next_path: str | None = None
        with self._lock:
            if self._current is not None:
                self._enabled = True
            else:
                next_path = self._start_next_locked()
        if next_path is not None:
            try:
                self._start_stream(next_path)
            except Exception:
                with self._lock:
                    self._current = None
                    self._enabled = False
        return self.snapshot()

    def skip_to_next(self) -> dict[str, Any]:
        with self._lock:
            self._current = None
            next_path = self._start_next_locked()
        if next_path is not None:
            try:
                self._start_stream(next_path)
            except Exception:
                with self._lock:
                    self._current = None
                    self._enabled = False
        return self.snapshot()

    def on_stream_complete(self, _returncode: int, finished_path: str | None, _log: str) -> None:
        with self._lock:
            if not self._enabled:
                return
            if finished_path is not None and self._current is not None and finished_path != self._current:
                return
            self._current = None
            next_path = self._start_next_locked()
        if next_path is None:
            return
        try:
            self._start_stream(next_path)
        except Exception:
            with self._lock:
                self._current = None
                self._enabled = False

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()


class ScreenCastManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ffmpeg_proc: subprocess.Popen[Any] | None = None
        self._http_proc: subprocess.Popen[Any] | None = None
        self._started_at: float | None = None
        self._local_ip: str | None = None
        self._url: str | None = None
        self._display: str | None = None
        self._size: str | None = None
        self._fps: int | None = None
        self._last_error: str = ""

    def _stop_process(self, proc: subprocess.Popen[Any]) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

    def _log_tail(self) -> str:
        path = SCREENCAST_DIR / "screencast.log"
        if not path.exists():
            return ""
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
        return text[-1600:].strip()

    def _detect_local_ip(self, tv_ip: str) -> str:
        if SCREENCAST_HOST_IP:
            return SCREENCAST_HOST_IP

        candidates = [(tv_ip, 7000), ("8.8.8.8", 80)]
        for host, port in candidates:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.connect((host, port))
                    ip = sock.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return ip
            except OSError:
                continue
        raise RuntimeError("lokale IP fuer Screen-Stream konnte nicht erkannt werden")

    def _snapshot_unlocked(self) -> dict[str, Any]:
        ffmpeg_active = self._ffmpeg_proc is not None and self._ffmpeg_proc.poll() is None
        http_active = self._http_proc is not None and self._http_proc.poll() is None
        active = ffmpeg_active and http_active

        if self._ffmpeg_proc is not None and not ffmpeg_active and not self._last_error:
            self._last_error = f"ffmpeg beendet (code {self._ffmpeg_proc.returncode})"
        if self._http_proc is not None and not http_active and not self._last_error:
            self._last_error = f"http server beendet (code {self._http_proc.returncode})"

        return {
            "active": active,
            "ffmpeg_active": ffmpeg_active,
            "http_active": http_active,
            "url": self._url,
            "local_ip": self._local_ip,
            "http_port": SCREENCAST_HTTP_PORT,
            "display": self._display,
            "size": self._size,
            "fps": self._fps,
            "started_at": self._started_at,
            "last_error": self._last_error,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            ffmpeg_proc = self._ffmpeg_proc
            http_proc = self._http_proc
            self._ffmpeg_proc = None
            self._http_proc = None
            self._started_at = None
            self._local_ip = None
            self._url = None
            self._display = None
            self._size = None
            self._fps = None
            self._last_error = ""

        if ffmpeg_proc is not None:
            self._stop_process(ffmpeg_proc)
        if http_proc is not None:
            self._stop_process(http_proc)
        return self.status()

    def start(
        self,
        tv_config: DeviceConfig,
        launch_play_url: Callable[[str], dict[str, Any]],
        display: str | None = None,
        size: str | None = None,
        fps: int | None = None,
    ) -> dict[str, Any]:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg wurde nicht gefunden. Bitte installieren.")

        selected_display = (display or SCREENCAST_DEFAULT_DISPLAY).strip() or ":0.0"
        selected_size = (size or SCREENCAST_DEFAULT_SIZE).strip()
        if not re.fullmatch(r"[0-9]{3,5}x[0-9]{3,5}", selected_size):
            raise RuntimeError("ungueltige Aufloesung (erwartet z.B. 1280x720)")

        selected_fps = fps if fps is not None else SCREENCAST_DEFAULT_FPS
        selected_fps = max(8, min(60, int(selected_fps)))

        local_ip = self._detect_local_ip(tv_config.ip)
        stream_url = f"http://{local_ip}:{SCREENCAST_HTTP_PORT}/stream.m3u8"

        self.stop()

        SCREENCAST_DIR.mkdir(parents=True, exist_ok=True)
        for pattern in ("*.m3u8", "*.ts"):
            for file in SCREENCAST_DIR.glob(pattern):
                file.unlink(missing_ok=True)
        log_path = SCREENCAST_DIR / "screencast.log"
        log_path.write_text("", encoding="utf-8")

        http_proc: subprocess.Popen[Any] | None = None
        ffmpeg_proc: subprocess.Popen[Any] | None = None

        try:
            with log_path.open("a", encoding="utf-8") as log_handle:
                http_proc = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "http.server",
                        str(SCREENCAST_HTTP_PORT),
                        "--bind",
                        "0.0.0.0",
                    ],
                    cwd=str(SCREENCAST_DIR),
                    stdout=log_handle,
                    stderr=log_handle,
                )

            time.sleep(0.35)
            if http_proc.poll() is not None:
                details = compact_error(self._log_tail() or "http server konnte nicht starten")
                raise RuntimeError(f"screen http server fehler: {details}")

            output_path = SCREENCAST_DIR / "stream.m3u8"
            ffmpeg_cmd = [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "warning",
                "-f",
                "x11grab",
                "-video_size",
                selected_size,
                "-framerate",
                str(selected_fps),
                "-i",
                selected_display,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-tune",
                "zerolatency",
                "-pix_fmt",
                "yuv420p",
                "-g",
                str(max(12, selected_fps * 2)),
                "-sc_threshold",
                "0",
                "-f",
                "hls",
                "-hls_time",
                "1",
                "-hls_list_size",
                "4",
                "-hls_flags",
                "delete_segments+append_list+omit_endlist",
                "-hls_allow_cache",
                "0",
                "-y",
                str(output_path),
            ]

            with log_path.open("a", encoding="utf-8") as log_handle:
                ffmpeg_proc = subprocess.Popen(
                    ffmpeg_cmd,
                    cwd=str(SCREENCAST_DIR),
                    stdout=log_handle,
                    stderr=log_handle,
                )

            ready = False
            deadline = time.time() + 8
            while time.time() < deadline:
                if ffmpeg_proc.poll() is not None:
                    break
                if output_path.exists() and output_path.stat().st_size > 0:
                    ready = True
                    break
                time.sleep(0.25)

            if not ready:
                details = compact_error(self._log_tail() or "ffmpeg konnte keinen HLS Stream erzeugen")
                hint = "Hinweis: bei Wayland kann x11grab scheitern."
                raise RuntimeError(f"screen capture fehlgeschlagen: {details}. {hint}")

            launch_warning = ""
            launch_detail = ""
            launch_result = launch_play_url(stream_url)
            launch_stdout = launch_result.get("stdout", "")
            launch_stderr = launch_result.get("stderr", "")
            if launch_result["returncode"] != 0:
                raw_error = "\n".join(
                    part for part in [launch_result["stderr"], launch_result["stdout"]] if part
                ).strip()
                if is_playback_info_500_error(raw_error):
                    launch_warning = (
                        "Apple TV meldet playback-info 500; Stream wurde trotzdem gestartet."
                    )
                    launch_stdout = ""
                    launch_stderr = ""
                    launch_detail = compact_error(raw_error, max_len=260)
                else:
                    error_text = compact_error(raw_error or "play_url auf Apple TV fehlgeschlagen")
                    raise RuntimeError(f"Apple TV konnte Stream nicht starten: {error_text}")

            with self._lock:
                self._ffmpeg_proc = ffmpeg_proc
                self._http_proc = http_proc
                self._started_at = time.time()
                self._local_ip = local_ip
                self._url = stream_url
                self._display = selected_display
                self._size = selected_size
                self._fps = selected_fps
                self._last_error = ""

            snapshot = self.status()
            snapshot["launch_returncode"] = launch_result["returncode"]
            snapshot["launch_stdout"] = launch_stdout
            snapshot["launch_stderr"] = launch_stderr
            if launch_warning:
                snapshot["launch_warning"] = launch_warning
            if launch_detail:
                snapshot["launch_detail"] = launch_detail
            return snapshot
        except Exception as exc:
            if ffmpeg_proc is not None:
                self._stop_process(ffmpeg_proc)
            if http_proc is not None:
                self._stop_process(http_proc)
            with self._lock:
                self._last_error = compact_error(str(exc), max_len=360)
            raise


CONFIG = ConfigStore(ip=DEFAULT_IP, name=DEFAULT_NAME)
TV_CONFIG = ConfigStore(ip=DEFAULT_TV_IP, name=DEFAULT_TV_NAME)
HISTORY = UploadHistoryStore(UPLOADS_DIR / "history.json")


def start_stream_path(path: str) -> None:
    STREAMS.start(atv_command_parts([f"stream_file={path}"]), path=path)


QUEUE = QueueManager(start_stream_path)
STREAMS = StreamManager(on_complete=QUEUE.on_stream_complete)
SCREENCAST = ScreenCastManager()


def atv_command_parts_for(config: DeviceConfig, args: list[str]) -> list[str]:
    command: list[str] = [ATVREMOTE_BIN]
    ip = config.ip.strip()
    name = config.name.strip()
    if ip:
        command.extend(["-s", ip])
    if name:
        command.extend(["-n", name])
    command.extend(args)
    return command


def atv_command_parts(args: list[str]) -> list[str]:
    return atv_command_parts_for(CONFIG.get(), args)


def atv_tv_command_parts(args: list[str]) -> list[str]:
    return atv_command_parts_for(TV_CONFIG.get(), args)


def _run_command(command: list[str], timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        missing = exc.filename or command[0]
        return {
            "command": command,
            "returncode": 127,
            "stdout": "",
            "stderr": f"command not found: {missing}",
        }
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        detail = f"timeout after {timeout}s"
        if err:
            detail = f"{detail} ({err})"
        return {
            "command": command,
            "returncode": 124,
            "stdout": out,
            "stderr": detail,
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def run_atv(args: list[str], timeout: int = 12) -> dict[str, Any]:
    config = CONFIG.get()
    if not config.ip.strip():
        return {
            "command": atv_command_parts(args),
            "returncode": 2,
            "stdout": "",
            "stderr": "homepod ip ist nicht gesetzt",
        }
    return _run_command(atv_command_parts_for(config, args), timeout=timeout)


def run_atv_tv(args: list[str], timeout: int = 12) -> dict[str, Any]:
    config = TV_CONFIG.get()
    if not config.ip.strip():
        return {
            "command": atv_tv_command_parts(args),
            "returncode": 2,
            "stdout": "",
            "stderr": "appletv ip ist nicht gesetzt",
        }
    return _run_command(atv_command_parts_for(config, args), timeout=timeout)


def run_scan(timeout: int = 18) -> dict[str, Any]:
    return _run_command([ATVREMOTE_BIN, "-t", "8", "scan"], timeout=timeout)


def check_connection(target: str) -> dict[str, Any]:
    def _single(label: str, cfg: DeviceConfig, runner: Any) -> dict[str, Any]:
        try:
            result = runner(["device_info"], timeout=12)
        except subprocess.TimeoutExpired:
            return {
                "target": label,
                "name": cfg.name,
                "ip": cfg.ip,
                "connected": False,
                "stdout": "",
                "stderr": "connection timeout",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "target": label,
                "name": cfg.name,
                "ip": cfg.ip,
                "connected": False,
                "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}",
            }

        return {
            "target": label,
            "name": cfg.name,
            "ip": cfg.ip,
            "connected": result["returncode"] == 0,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
        }

    normalized = target.strip().lower()
    if normalized not in {"homepod", "appletv", "all"}:
        raise ValueError("unknown connect target")

    if normalized == "homepod":
        home = _single("homepod", CONFIG.get(), run_atv)
        return {"target": "homepod", "connected": home["connected"], "homepod": home}

    if normalized == "appletv":
        tv = _single("appletv", TV_CONFIG.get(), run_atv_tv)
        return {"target": "appletv", "connected": tv["connected"], "appletv": tv}

    home = _single("homepod", CONFIG.get(), run_atv)
    tv = _single("appletv", TV_CONFIG.get(), run_atv_tv)
    return {
        "target": "all",
        "connected": home["connected"] and tv["connected"],
        "homepod": home,
        "appletv": tv,
    }


def sanitize_filename(name: str) -> str:
    base = Path(name).name.replace("\x00", "").strip()
    if not base:
        base = f"upload_{int(time.time())}.bin"
    safe = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    if not safe:
        safe = f"upload_{int(time.time())}.bin"
    return safe


def unique_upload_path(base_dir: Path, filename: str) -> Path:
    candidate = base_dir / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for idx in range(2, 10000):
        attempt = base_dir / f"{stem}_{idx}{suffix}"
        if not attempt.exists():
            return attempt
    return base_dir / f"{stem}_{int(time.time())}{suffix}"


def parse_playing(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def playing_signature(text: str) -> str:
    parsed = parse_playing(text)
    preferred = [
        parsed.get("Identifier", "").strip(),
        parsed.get("Title", "").strip(),
        parsed.get("Device state", "").strip(),
        parsed.get("Media type", "").strip(),
    ]
    compact = "|".join([item for item in preferred if item])
    if compact:
        return compact
    return str(text).strip()


def extract_output_device_id(result: dict[str, Any]) -> str:
    blobs = [str(result.get("stdout", "")), str(result.get("stderr", ""))]
    line_pattern = re.compile(r"^[0-9A-Fa-f-]{12,64}$")
    uuid_pattern = re.compile(
        r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
    )

    for blob in blobs:
        lines = [line.strip() for line in blob.splitlines() if line.strip()]
        for line in reversed(lines):
            if line_pattern.fullmatch(line):
                return line
            match = uuid_pattern.search(line)
            if match:
                return match.group(0)

    joined = "\n".join(blobs)
    match = uuid_pattern.search(joined)
    if match:
        return match.group(0)
    return ""


def is_playback_info_500_error(text: str) -> bool:
    lowered = str(text).lower()
    if not lowered:
        return False
    return "playback-info" in lowered and "code 500" in lowered


def compact_error(text: str, max_len: int = 320) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    priority_markers = [
        "HTTP/1.1 method GET failed with code 500",
        "Authentication error",
        "Connection Authorization Required",
        "PairingError:",
        "TimeoutError:",
        "ConnectionLostError:",
        "no response to",
        "connection was lost",
        "ERROR [",
    ]
    for marker in priority_markers:
        for line in lines:
            if marker in line:
                return line if len(line) <= max_len else f"{line[: max_len - 3]}..."

    ignore_prefixes = (
        "Traceback",
        "File ",
        "^",
        "return await ",
        "ret = await ",
        "value = await ",
        "await ",
        "...<",
        ">>>",
    )
    for line in lines:
        if any(line.startswith(prefix) for prefix in ignore_prefixes):
            continue
        return line if len(line) <= max_len else f"{line[: max_len - 3]}..."

    first = lines[0]
    return first if len(first) <= max_len else f"{first[: max_len - 3]}..."


class Handler(SimpleHTTPRequestHandler):
    auth_user = DEFAULT_AUTH_USER or None
    auth_password = DEFAULT_AUTH_PASSWORD or None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        print(f"[http] {self.address_string()} - {format % args}")

    @classmethod
    def _auth_enabled(cls) -> bool:
        return bool(cls.auth_user and cls.auth_password)

    def _send_unauthorized(self) -> None:
        if self.path.startswith("/api/"):
            body = json.dumps({"ok": False, "error": "Authentifizierung erforderlich"}).encode("utf-8")
            self.send_response(401)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("WWW-Authenticate", 'Basic realm="homepod-gui", charset="UTF-8"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = b"Authentifizierung erforderlich\n"
        self.send_response(401)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("WWW-Authenticate", 'Basic realm="homepod-gui", charset="UTF-8"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _is_authorized(self) -> bool:
        if not self._auth_enabled():
            return True

        header = self.headers.get("Authorization", "").strip()
        if not header.lower().startswith("basic "):
            return False

        token = header[6:].strip()
        try:
            decoded = base64.b64decode(token, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            return False

        user, sep, password = decoded.partition(":")
        if sep != ":":
            return False
        if self.auth_user is None or self.auth_password is None:
            return False
        return hmac.compare_digest(user, self.auth_user) and hmac.compare_digest(
            password, self.auth_password
        )

    def _require_auth(self) -> bool:
        if self._is_authorized():
            return True
        self._send_unauthorized()
        return False

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _error(self, message: str, status: int = 400) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def do_GET(self) -> None:  # noqa: N802
        if not self._require_auth():
            return

        if not self.path.startswith("/api/"):
            return super().do_GET()

        if self.path == "/api/health":
            cfg = CONFIG.get()
            self._send_json(
                {
                    "ok": True,
                    "atvremote_bin": ATVREMOTE_BIN,
                    "atvremote_exists": ATVREMOTE_EXISTS,
                    "ip": cfg.ip,
                    "name": cfg.name,
                }
            )
            return

        if self.path == "/api/config":
            cfg = CONFIG.get()
            self._send_json({"ok": True, "ip": cfg.ip, "name": cfg.name})
            return

        if self.path == "/api/tv_config":
            cfg = TV_CONFIG.get()
            self._send_json({"ok": True, "ip": cfg.ip, "name": cfg.name})
            return

        if self.path == "/api/stream_status":
            self._send_json({"ok": True, **STREAMS.status()})
            return

        if self.path == "/api/screen_status":
            self._send_json({"ok": True, **SCREENCAST.status()})
            return

        if self.path == "/api/upload_history":
            self._send_json({"ok": True, "items": HISTORY.list()})
            return

        if self.path == "/api/queue":
            self._send_json({"ok": True, **QUEUE.snapshot()})
            return

        if self.path == "/api/volume":
            result = run_atv(["volume"])
            if result["returncode"] != 0:
                self._error(result["stderr"] or result["stdout"] or "volume failed", status=502)
                return
            try:
                level = float(result["stdout"])
            except ValueError:
                level = None
            self._send_json({"ok": True, "level": level, "raw": result["stdout"]})
            return

        if self.path == "/api/playing":
            result = run_atv(["playing"])
            if result["returncode"] != 0:
                self._error(result["stderr"] or result["stdout"] or "playing failed", status=502)
                return
            self._send_json(
                {"ok": True, "raw": result["stdout"], "parsed": parse_playing(result["stdout"])}
            )
            return

        if self.path == "/api/features":
            result = run_atv(["features"], timeout=25)
            if result["returncode"] != 0:
                self._error(result["stderr"] or result["stdout"] or "features failed", status=502)
                return
            self._send_json({"ok": True, "raw": result["stdout"]})
            return

        if self.path == "/api/device_info":
            result = run_atv(["device_info"], timeout=25)
            if result["returncode"] != 0:
                self._error(result["stderr"] or result["stdout"] or "device_info failed", status=502)
                return
            self._send_json({"ok": True, "raw": result["stdout"]})
            return

        if self.path == "/api/scan":
            result = run_scan(timeout=20)
            if result["returncode"] != 0:
                self._error(result["stderr"] or result["stdout"] or "scan failed", status=502)
                return
            self._send_json({"ok": True, "raw": result["stdout"]})
            return

        self._error("unknown endpoint", status=404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_auth():
            return

        if not self.path.startswith("/api/"):
            self._error("unknown endpoint", status=404)
            return

        if self.path == "/api/upload_media":
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError:
                self._error("ungueltige content length")
                return

            if length <= 0:
                self._error("leerer upload")
                return

            if length > MAX_UPLOAD_BYTES:
                self._error(
                    f"datei zu gross (max {MAX_UPLOAD_BYTES} bytes)",
                    status=413,
                )
                return

            raw_filename = self.headers.get("X-Filename", "")
            filename = sanitize_filename(unquote(raw_filename))
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_MEDIA_EXTENSIONS:
                self._error("dateityp nicht erlaubt")
                return

            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            target = unique_upload_path(UPLOADS_DIR, filename)

            received = 0
            try:
                with target.open("wb") as handle:
                    remaining = length
                    while remaining > 0:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        handle.write(chunk)
                        received += len(chunk)
                        remaining -= len(chunk)
            except Exception as exc:  # noqa: BLE001
                if target.exists():
                    target.unlink(missing_ok=True)
                self._error(f"upload fehlgeschlagen: {type(exc).__name__}: {exc}", status=500)
                return

            if received != length:
                target.unlink(missing_ok=True)
                self._error("upload unvollstaendig", status=400)
                return

            media_type = "video" if suffix in {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi"} else "audio"
            history_entry = {
                "path": str(target),
                "filename": target.name,
                "size": received,
                "uploaded_at": int(time.time()),
                "media_type": media_type,
            }
            HISTORY.add(history_entry)

            self._send_json(
                {
                    "ok": True,
                    "path": str(target),
                    "filename": target.name,
                    "size": received,
                    "media_type": media_type,
                }
            )
            return

        try:
            data = self._read_json()
        except json.JSONDecodeError:
            self._error("invalid json")
            return

        if self.path == "/api/config":
            raw_ip = data.get("ip")
            raw_name = data.get("name")
            ip = str(raw_ip).strip() if raw_ip is not None else None
            name = str(raw_name).strip() if raw_name is not None else None
            if ip is not None and not ip:
                self._error("ip darf nicht leer sein")
                return
            if name is not None and not name:
                self._error("name darf nicht leer sein")
                return
            cfg = CONFIG.set(ip=ip, name=name)
            self._send_json({"ok": True, "ip": cfg.ip, "name": cfg.name})
            return

        if self.path == "/api/tv_config":
            raw_ip = data.get("ip")
            raw_name = data.get("name")
            ip = str(raw_ip).strip() if raw_ip is not None else None
            name = str(raw_name).strip() if raw_name is not None else None
            if ip is not None and not ip:
                self._error("tv ip darf nicht leer sein")
                return
            if name is not None and not name:
                self._error("tv name darf nicht leer sein")
                return
            cfg = TV_CONFIG.set(ip=ip, name=name)
            self._send_json({"ok": True, "ip": cfg.ip, "name": cfg.name})
            return

        if self.path == "/api/volume":
            level_raw = data.get("level")
            try:
                level = float(level_raw)
            except (TypeError, ValueError):
                self._error("level muss eine Zahl sein")
                return
            if level < 0 or level > 100:
                self._error("level muss zwischen 0 und 100 liegen")
                return
            result = run_atv([f"set_volume={level:.1f}"])
            if result["returncode"] != 0:
                self._error(result["stderr"] or result["stdout"] or "set volume failed", status=502)
                return
            self._send_json({"ok": True, "level": level})
            return

        if self.path == "/api/play":
            path_raw = data.get("path")
            path = str(path_raw).strip() if path_raw is not None else ""
            if not path:
                self._error("path darf nicht leer sein")
                return

            if not (path.startswith("http://") or path.startswith("https://")):
                if not Path(path).is_file():
                    self._error("Datei nicht gefunden")
                    return

            QUEUE.disable()
            STREAMS.start(atv_command_parts([f"stream_file={path}"]), path=path)
            self._send_json({"ok": True, "message": "stream gestartet", "path": path})
            return

        if self.path == "/api/stop":
            stop_result = run_atv(["stop"], timeout=8)
            QUEUE.disable()
            STREAMS.stop_local_process()
            self._send_json(
                {
                    "ok": True,
                    "returncode": stop_result["returncode"],
                    "stdout": stop_result["stdout"],
                    "stderr": stop_result["stderr"],
                }
            )
            return

        if self.path == "/api/screen_start":
            raw_display = data.get("display")
            raw_size = data.get("size")
            raw_fps = data.get("fps")

            display = str(raw_display).strip() if raw_display is not None else None
            size = str(raw_size).strip() if raw_size is not None else None

            fps: int | None = None
            if raw_fps is not None and str(raw_fps).strip() != "":
                try:
                    fps = int(str(raw_fps).strip())
                except (TypeError, ValueError):
                    self._error("fps muss eine ganze Zahl sein")
                    return

            tv_cfg = TV_CONFIG.get()
            pre_playing_result = run_atv_tv(["playing"], timeout=8)

            tv_output_id: str | None = None
            output_set_result: dict[str, Any] | None = None
            output_id_result = run_atv_tv(["output_device_id"], timeout=10)
            if output_id_result["returncode"] == 0:
                candidate = extract_output_device_id(output_id_result)
                if candidate:
                    tv_output_id = candidate
                    output_set_result = run_atv_tv(
                        [f"set_output_devices={candidate}"],
                        timeout=12,
                    )

            def launch_play_url(url: str) -> dict[str, Any]:
                return run_atv_tv([f"play_url={url}"], timeout=25)

            try:
                snapshot = SCREENCAST.start(
                    tv_config=tv_cfg,
                    launch_play_url=launch_play_url,
                    display=display,
                    size=size,
                    fps=fps,
                )
            except RuntimeError as exc:
                self._error(str(exc), status=502)
                return

            if tv_output_id is not None:
                snapshot["target_output_device_id"] = tv_output_id
            if output_set_result is not None:
                snapshot["output_set_returncode"] = output_set_result["returncode"]
                snapshot["output_set_stdout"] = output_set_result["stdout"]
                snapshot["output_set_stderr"] = output_set_result["stderr"]

            post_playing_result = run_atv_tv(["playing"], timeout=10)
            post_playing_text = post_playing_result.get("stdout", "")
            snapshot["tv_playing_after"] = post_playing_text

            if snapshot.get("launch_returncode", 0) != 0:
                pre_sig = playing_signature(pre_playing_result.get("stdout", ""))
                post_sig = playing_signature(post_playing_text)
                unchanged = bool(pre_sig and post_sig and pre_sig == post_sig)
                stopped_or_unknown = "Device state: Stopped" in post_playing_text or not post_sig
                if post_playing_result.get("returncode", 1) != 0 or unchanged or stopped_or_unknown:
                    SCREENCAST.stop()
                    detail = (
                        str(snapshot.get("launch_detail", "")).strip()
                        or str(snapshot.get("launch_stderr", "")).strip()
                    )
                    message = "Apple TV hat den Bildschirmstream nicht uebernommen."
                    if "code 500" in detail:
                        message += (
                            " AirPlay-Autorisierung auf Apple TV pruefen: "
                            "Einstellungen > AirPlay & HomeKit > Zugriff."
                        )
                    if detail:
                        message = f"{message} Detail: {detail}"
                    self._error(message, status=502)
                    return

            self._send_json({"ok": True, **snapshot})
            return

        if self.path == "/api/screen_stop":
            snapshot = SCREENCAST.stop()
            tv_stop = run_atv_tv(["stop"], timeout=10)
            self._send_json(
                {
                    "ok": True,
                    **snapshot,
                    "tv_returncode": tv_stop["returncode"],
                    "tv_stdout": tv_stop["stdout"],
                    "tv_stderr": tv_stop["stderr"],
                }
            )
            return

        if self.path == "/api/open_system_mirror":
            mirror_bin = shutil.which("gnome-network-displays")
            if mirror_bin is None:
                self._error(
                    "gnome-network-displays wurde nicht gefunden. Bitte installieren.",
                    status=500,
                )
                return

            env = os.environ.copy()
            if not env.get("DISPLAY") and SCREENCAST_DEFAULT_DISPLAY:
                env["DISPLAY"] = SCREENCAST_DEFAULT_DISPLAY

            try:
                proc = subprocess.Popen(
                    [mirror_bin],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    start_new_session=True,
                )
            except Exception as exc:  # noqa: BLE001
                self._error(f"Mirror-Tool konnte nicht gestartet werden: {exc}", status=500)
                return

            self._send_json(
                {"ok": True, "pid": proc.pid, "command": [mirror_bin], "display": env.get("DISPLAY")}
            )
            return

        if self.path == "/api/queue/add":
            path_raw = data.get("path")
            path = str(path_raw).strip() if path_raw is not None else ""
            if not path:
                self._error("path darf nicht leer sein")
                return
            if not (path.startswith("http://") or path.startswith("https://")):
                if not Path(path).is_file():
                    self._error("Datei nicht gefunden")
                    return
            snapshot = QUEUE.add(path)
            self._send_json({"ok": True, **snapshot})
            return

        if self.path == "/api/queue/start":
            snapshot = QUEUE.start()
            self._send_json({"ok": True, **snapshot})
            return

        if self.path == "/api/queue/next":
            STREAMS.stop_local_process()
            snapshot = QUEUE.skip_to_next()
            self._send_json({"ok": True, **snapshot})
            return

        if self.path == "/api/queue/clear":
            snapshot = QUEUE.clear()
            self._send_json({"ok": True, **snapshot})
            return

        if self.path == "/api/remote":
            raw_action = data.get("action")
            action = str(raw_action).strip() if raw_action is not None else ""
            command = REMOTE_ACTIONS.get(action)
            if command is None:
                self._error("unbekannte action")
                return

            raw_target = data.get("target")
            target = str(raw_target).strip().lower() if raw_target is not None else ""
            if target not in {"", "homepod", "appletv"}:
                self._error("unbekanntes target")
                return

            if not target:
                target = "appletv"

            runner = run_atv_tv if target == "appletv" else run_atv
            result = runner([command], timeout=10)
            if result["returncode"] != 0:
                self._error(result["stderr"] or result["stdout"] or f"{action} failed", status=502)
                return
            self._send_json(
                {
                    "ok": True,
                    "action": action,
                    "target": target,
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                }
            )
            return

        if self.path == "/api/command":
            raw_command = data.get("command")
            command_text = str(raw_command).strip() if raw_command is not None else ""
            if not command_text:
                self._error("command darf nicht leer sein")
                return
            try:
                args = shlex.split(command_text)
            except ValueError as exc:
                self._error(f"ungueltiger command: {exc}")
                return
            if not args:
                self._error("ungueltiger command")
                return
            if any(arg.startswith("-") for arg in args):
                self._error("optionen mit '-' sind hier nicht erlaubt")
                return

            timeout = 20
            if "timeout" in data:
                try:
                    timeout = int(data.get("timeout"))
                except (TypeError, ValueError):
                    self._error("timeout muss eine Zahl sein")
                    return
                timeout = max(3, min(timeout, 90))

            result = run_atv(args, timeout=timeout)
            if result["returncode"] != 0:
                self._error(result["stderr"] or result["stdout"] or "command failed", status=502)
                return
            self._send_json(
                {
                    "ok": True,
                    "command": command_text,
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                }
            )
            return

        if self.path == "/api/service_launch":
            raw_service = data.get("service")
            service = str(raw_service).strip().lower() if raw_service is not None else ""
            entry = TV_SERVICES.get(service)
            if entry is None:
                self._error("unbekannter service")
                return

            result = run_atv_tv([f"launch_app={entry['bundle_id']}"], timeout=15)
            if result["returncode"] != 0:
                self._error(
                    result["stderr"] or result["stdout"] or f"launch {service} failed",
                    status=502,
                )
                return

            cfg = TV_CONFIG.get()
            self._send_json(
                {
                    "ok": True,
                    "service": service,
                    "label": entry["label"],
                    "bundle_id": entry["bundle_id"],
                    "target_ip": cfg.ip,
                    "target_name": cfg.name,
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                }
            )
            return

        if self.path == "/api/connect":
            raw_target = data.get("target")
            target = str(raw_target).strip() if raw_target is not None else "all"
            try:
                result = check_connection(target)
            except ValueError:
                self._error("unbekanntes target")
                return
            self._send_json({"ok": True, **result})
            return

        self._error("unknown endpoint", status=404)


def main() -> int:
    if bool(DEFAULT_AUTH_USER) ^ bool(DEFAULT_AUTH_PASSWORD):
        print("Fehler: Fuer Basic Auth muessen User und Passwort gesetzt sein.")
        return 2

    try:
        _ensure_self_signed_cert(DEFAULT_TLS_CERT, DEFAULT_TLS_KEY)
    except RuntimeError as exc:
        print(f"Fehler: {exc}")
        return 2

    if not DEFAULT_TLS_CERT.exists() or not DEFAULT_TLS_KEY.exists():
        print("Fehler: TLS-Zertifikat oder Key fehlt.")
        return 2

    Handler.auth_user = DEFAULT_AUTH_USER or None
    Handler.auth_password = DEFAULT_AUTH_PASSWORD or None

    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), Handler)
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
    ssl_context.load_cert_chain(certfile=str(DEFAULT_TLS_CERT), keyfile=str(DEFAULT_TLS_KEY))
    server.socket = ssl_context.wrap_socket(server.socket, server_side=True)

    print(f"HomePod GUI laeuft auf https://{DEFAULT_HOST}:{DEFAULT_PORT}")
    print(f"ATVREMOTE_BIN={ATVREMOTE_BIN}")
    print(f"TLS_CERT={DEFAULT_TLS_CERT}")
    if Handler._auth_enabled():
        print(f"Auth: aktiviert (User={Handler.auth_user})")
    else:
        print("Auth: deaktiviert")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStoppe Server...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
