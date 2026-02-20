# HomePod GUI

Lokale Web-GUI fuer HomePod-Steuerung ueber `atvremote`.

## Start

```bash
cd /pfad/zu/homepod-gui
python3 app.py
```

Dann im Browser:

```text
http://localhost:8788
```

## Optional: Startwerte per Env setzen

```bash
HOMEPOD_IP="DEINE_HOMEPOD_IP" \
HOMEPOD_NAME="HomePod" \
APPLETV_IP="DEINE_APPLETV_IP" \
APPLETV_NAME="Apple TV" \
python3 app.py
```

Hinweis: `DEINE_HOMEPOD_IP` und `DEINE_APPLETV_IP` sind nur Platzhalter. Bitte durch die echten IP-Adressen deiner Geraete ersetzen.

## Hinweis zu Marken und Nutzung

- `Apple`, `HomePod`, `Apple TV`, `AirPlay` und App-Namen wie `Netflix`, `Amazon`, `Spotify` sind Marken ihrer jeweiligen Inhaber.
- Dieses Projekt ist ein inoffizielles Community-Tool und steht in keiner Verbindung zu Apple.
- Nutze das Tool nur mit deinen eigenen Geraeten, in deinem eigenen Netzwerk und im Rahmen der jeweils gueltigen Nutzungsbedingungen.

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
- Hinweis: `<deine-laptop-ip>` ist ein Platzhalter und muss durch die echte IP-Adresse deines Laptops ersetzt werden.

## Persoenliche Bewertung (Stand: 20. Februar 2026)

Diese Bewertung wurde persoenlich von **AIDRAX (KI-Assistent)** abgegeben.

Meine persoenliche Bewertung fuer dieses Projekt: **8.6/10**.

Staerken:
- Sehr schneller Praxisnutzen: HomePod + Apple TV Kontrolle in einer einzigen, lokalen GUI
- Gute Funktionsbreite (Playback, Queue, App-Start, Voice-Commands, Screen-Optionen)
- Sinnvoll fuer Heimnetz-Setups ohne Cloud-Zwang

Was als naechstes den groessten Sprung bringt:
- Stabileres AirPlay-Handling bei `play_url`-Sonderfaellen (z. B. `playback-info 500`)
- Optionales Pairing/Connection-Wizard direkt in der GUI
- Robusteres Mirroring-Fallback fuer unterschiedliche Linux-Desktop-Umgebungen
