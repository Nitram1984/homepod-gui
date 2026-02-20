# HomePod GUI

Lokale Web-GUI fuer HomePod-Steuerung ueber `atvremote`.

## Start

```bash
cd /home/maddin/homepod-gui
python3 app.py
```

Dann im Browser:

```text
http://127.0.0.1:8788
```

## Optional: Startwerte per Env setzen

```bash
HOMEPOD_IP="192.168.178.183" \
HOMEPOD_NAME="Wohnzimmer" \
python3 app.py
```

## Funktionen

- HomePod-IP und Name speichern
- Lautstaerke lesen und setzen
- Datei/URL streamen (`stream_file`)
- Musik- und Video-Dateien direkt aus der GUI hochladen (`Medien hinzufuegen`)
- Upload-Historie mit Schnellaktionen (`In Feld`, `+ Queue`, `Play`)
- Playlist/Queue (`Zur Playlist`, `Playlist Start`, `Naechster`, `Playlist leeren`)
- Stop-Befehl senden
- Wiedergabe-Status anzeigen
- `ATV Remote`-Buttons (Play/Pause, Skip, Menu/Home, Vol +/-)
- `ATV Tools`: `features`, `device_info`, `scan`
- Direkte `atvremote`-Kommandos ueber Eingabefeld
- Service-Buttons (`Netflix`, `Amazon`, `Joyn`, `Spotify`) starten direkt auf Apple TV
- Eigene Apple-TV-Zielkonfiguration (`Apple TV IP` + `Apple TV Name`) in der GUI
- Menue-Buttons `HomePod verbinden`, `Apple TV verbinden`, `Alle verbinden` fuer aktive Verbindungspruefung
- Bildschirm-Stream vom Laptop auf Apple TV (`Screen Start/Stop/Status`) via `play_url` + HLS

## Hinweis Bildschirm-Stream

- Benoetigt `ffmpeg` auf dem Laptop
- Nutzt standardmaessig X11-Quelle `:0.0` (bei Wayland kann das scheitern)
- Stream wird lokal unter `http://<deine-laptop-ip>:8911/stream.m3u8` bereitgestellt

## Persoenliche Bewertung (Stand: 20. Februar 2026)

Meine persoenliche Bewertung fuer dieses Projekt: **8.6/10**.

Staerken:
- Sehr schneller Praxisnutzen: HomePod + Apple TV Kontrolle in einer einzigen, lokalen GUI
- Gute Funktionsbreite (Playback, Queue, App-Start, Voice-Commands, Screen-Optionen)
- Sinnvoll fuer Heimnetz-Setups ohne Cloud-Zwang

Was als naechstes den groessten Sprung bringt:
- Stabileres AirPlay-Handling bei `play_url`-Sonderfaellen (z. B. `playback-info 500`)
- Optionales Pairing/Connection-Wizard direkt in der GUI
- Robusteres Mirroring-Fallback fuer unterschiedliche Linux-Desktop-Umgebungen
