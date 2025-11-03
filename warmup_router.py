from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uvicorn.protocols.utils import ClientDisconnected
import json, math, time, queue, subprocess, threading, asyncio
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import essentia
import essentia.standard as es
from starlette.websockets import WebSocketState
from sqlalchemy.orm import Session
from app.models import WarmupSession 
from app.helper import decode_access_token
import logging

router = APIRouter()
logger = logging.getLogger("uvicorn.info")
# ======================= PARÁMETROS =======================
SR = 44100
FRAME_SIZE = 2048
HOP_SIZE = 512                  
PITCH_CENTS_TOL = 50            
MIN_NOTE_TIME = 0.10            
ONSET_ENERGY_THRESH = 2.0       
TIMEOUT_PER_NOTE = 2.0          
USE_BPM = False
BPM = 80

CONF_THRESH = 0.60              
NOISE_GATE_DB_OFFSET = 12.0     
MIN_IOI = 0.15                  

MIN_F0, MAX_F0 = 35.0, 220.0

TUNING = {"E": 41.2034, "A": 55.0, "D": 73.4162, "G": 97.9989}

@dataclass
class TabNote:
    string: str
    fret: int
    beats: float = 1.0

def _ordinal_traste_es(fret: int) -> str:
    if fret == 0: return "al aire"
    mapa = {
        1: "1er traste", 2: "2do traste", 3: "3er traste", 4: "4to traste", 5: "5to traste",
        6: "6to traste", 7: "7mo traste", 8: "8vo traste", 9: "9no traste", 10: "10mo traste"
    }
    return mapa.get(fret, f"{fret}° traste")

def _label_pos(string: str, fret: int) -> str:
    return f"{_ordinal_traste_es(fret)} cuerda {string.upper()}"

def freq_from_string_fret(string: str, fret: int) -> float:
    base = TUNING[string.upper()]
    return base * (2.0 ** (fret / 12.0))

def cents_error(f_detect: float, f_target: float) -> float:
    if f_detect <= 0 or f_target <= 0:
        return 1e9
    return 1200.0 * math.log2(f_detect / f_target)

def fold_to_target_octave(f0: float, f_target: float) -> float:
    if f0 <= 0 or f_target <= 0:
        return 0.0
    while f0 > 1.9 * f_target:
        f0 /= 2.0
    while f0 < 0.55 * f_target:
        f0 *= 2.0
    return f0


class PulseCapture:
    def __init__(self, q_audio: queue.Queue, source_name: Optional[str] = None,
                 samplerate: int = SR, channels: int = 1, frame_bytes: int = FRAME_SIZE*2):
        self.q = q_audio
        self.source_name = source_name
        self.samplerate = samplerate
        self.channels = channels
        self.frame_bytes = frame_bytes  
        self.proc = None
        self._stop = threading.Event()
        self.thread = None

    def start(self):
        cmd = ["parec", "--format=s16le", f"--rate={self.samplerate}", f"--channels={self.channels}"]
        if self.source_name:
            cmd += ["--device", self.source_name]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        buf = bytearray()
        stream = self.proc.stdout
        need = self.frame_bytes
        while not self._stop.is_set():
            chunk = stream.read(need - len(buf))
            if not chunk:
                time.sleep(0.005)
                continue
            buf.extend(chunk)
            if len(buf) >= need:
                frame_bytes = bytes(buf[:need])
                del buf[:need]
                x = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                self.q.put(x)

    def stop(self):
        self._stop.set()
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass

# ========= Evaluador =========
class WarmUpEvaluator:
    def __init__(self, tab: List[TabNote], pulse_source: Optional[str] = "input", samplerate: int = SR):
        self.tab = tab
        self.idx = 0
        self.results = []
        self.q_audio = queue.Queue()
        self.samplerate = int(samplerate)

        self.window = es.Windowing(type='hann')
        self.spectrum = es.Spectrum()
        self.pitch = es.PitchYinFFT(frameSize=FRAME_SIZE, sampleRate=self.samplerate)
        self.rms = es.RMS()

        self.prev_rms = 0.0
        self.prev_pitch = 0.0
        self.in_note_window = False
        self.note_window_start = 0.0
        self.last_onset_time = 0.0
        self.current_target_time = None
        self.f0_buf = deque(maxlen=5)

        self.capture = PulseCapture(self.q_audio, source_name=pulse_source, samplerate=self.samplerate)

    def _frame_to_pitch(self, frame) -> float:
        w = self.window(frame)
        spec = self.spectrum(w)
        f0, conf = self.pitch(spec)
        if conf < CONF_THRESH:
            return 0.0
        if f0 < MIN_F0 or f0 > 2*MAX_F0:
            return 0.0
        return float(f0)

    def _detect_onset(self, f0, rms, now) -> bool:
        rms_db = 20 * math.log10(max(rms, 1e-12))
        if hasattr(self, "noise_gate_db") and rms_db < self.noise_gate_db:
            self.prev_rms = rms
            if f0 > 0: self.prev_pitch = f0
            return False

        if (now - self.last_onset_time) < MIN_IOI:
            self.prev_rms = rms
            if f0 > 0: self.prev_pitch = f0
            return False

        prev_db = 20 * math.log10(max(self.prev_rms, 1e-12))
        energy_jump = (rms_db - prev_db) > ONSET_ENERGY_THRESH

        pitch_jump = False
        if self.prev_pitch > 0 and f0 > 0:
            cents = abs(1200.0 * math.log2((f0 + 1e-9) / (self.prev_pitch + 1e-9)))
            pitch_jump = cents > 80

        onset = energy_jump or pitch_jump
        self.prev_rms = rms
        if f0 > 0: self.prev_pitch = f0
        if onset:
            self.last_onset_time = now
        return onset

    def start(self):
        logger.info(">> Warm-Up (PulseAudio). Presioná Ctrl+C para salir.\n")
        self.capture.start()

        logger.info("Calibrando ruido (1 s)...")
        calib = []
        t0 = time.time()
        while time.time() - t0 < 1.0:
            try:
                chunk = self.q_audio.get(timeout=0.5)
            except queue.Empty:
                continue
            for off in range(0, len(chunk) - FRAME_SIZE + 1, FRAME_SIZE):
                frame = chunk[off:off+FRAME_SIZE].astype(np.float32)
                calib.append(self.rms(frame))
        if calib:
            rms_db_base = 20 * math.log10(max(float(np.median(calib)), 1e-12))
            self.noise_gate_db = rms_db_base + NOISE_GATE_DB_OFFSET
        else:
            rms_db_base = -60.0
            self.noise_gate_db = -48.0
        logger.info(f"Noise gate: {self.noise_gate_db:.1f} dB (base {rms_db_base:.1f} dB)")

        start_time = time.time()
        if USE_BPM and self.tab:
            self.current_target_time = start_time + 60.0 / BPM * self.tab[0].beats

        try:
            while self.idx < len(self.tab):
                try:
                    chunk = self.q_audio.get(timeout=0.5)
                except queue.Empty:
                    self._maybe_timeout(start_time)
                    continue

                for off in range(0, len(chunk) - FRAME_SIZE + 1, HOP_SIZE):
                    frame = chunk[off: off + FRAME_SIZE].astype(np.float32)
                    now = time.time()

                    f0_raw = self._frame_to_pitch(frame)
                    rms = self.rms(frame)

                    f_target = self._current_target_freq()
                    if f0_raw > 0 and f_target:
                        f0_raw = fold_to_target_octave(f0_raw, f_target)
                    if f0_raw > 0:
                        self.f0_buf.append(f0_raw)
                    f0 = float(np.median(self.f0_buf)) if self.f0_buf else 0.0

                    onset = self._detect_onset(f0, rms, now)
                    if onset and not self.in_note_window:
                        self.in_note_window = True
                        self.note_window_start = now

                    self._try_validate_current(f0, now, start_time)
                    self._maybe_timeout(start_time)

            self.capture.stop()
            logger.info("\nResumen:")
            ok = sum(1 for r in self.results if r[1])
            logger.info(f"Aciertos {ok}/{len(self.results)}")
        except KeyboardInterrupt:
            self.capture.stop()
            logger.info("\nInterrumpido por el usuario.")

    def _current_target_freq(self) -> Optional[float]:
        if self.idx >= len(self.tab):
            return None
        n = self.tab[self.idx]
        return freq_from_string_fret(n.string, n.fret)

    def _try_validate_current(self, f0, now, start_time):
        if not self.in_note_window or self.idx >= len(self.tab) or f0 <= 0:
            return
        f_target = self._current_target_freq()
        if not f_target:
            return
        cents = cents_error(f0, f_target)
        if abs(cents) <= PITCH_CENTS_TOL:
            if (now - self.note_window_start) >= MIN_NOTE_TIME:
                self._mark_result(True, f0, cents, now - start_time)
                self._advance_index(now, start_time)

    def _maybe_timeout(self, start_time):
        if self.idx >= len(self.tab):
            return
        now = time.time()
        if self.in_note_window and (now - self.note_window_start) > TIMEOUT_PER_NOTE:
            self._mark_result(False, 0.0, 1e9, now - start_time)
            self._advance_index(now, start_time)
            return
        if USE_BPM and self.current_target_time is not None and now > self.current_target_time:
            self._mark_result(False, 0.0, 1e9, now - start_time)
            self._advance_index(now, start_time)

    def _advance_index(self, now, start_time):
        self.idx += 1
        self.in_note_window = False
        self.note_window_start = 0.0
        if USE_BPM and self.idx < len(self.tab):
            dt = 60.0 / BPM * self.tab[self.idx].beats
            self.current_target_time = now + dt

        if self.idx < len(self.tab):
            tgt = self.tab[self.idx]
            logger.info(f"→ Siguiente: {_label_pos(tgt.string, tgt.fret)} (pos {self.idx+1}/{len(self.tab)})")
        else:
            logger.info("✔️  Warm-Up finalizado.")

    def _mark_result(self, ok: bool, f0: float, cents: float, t_event: float):
        n = self.tab[self.idx]
        msg = "✅ Correcto" if ok else "❌ Incorrecto"
        tgt_hz = freq_from_string_fret(n.string, n.fret)
        etiqueta = _label_pos(n.string, n.fret)
        if ok:
            logger.info(f"{msg}  esperado {etiqueta} ~{tgt_hz:.1f} Hz | detectado {f0:.1f} Hz ({cents:+.1f} cents)")
        else:
            logger.info(f"{msg}  esperado {etiqueta} ~{tgt_hz:.1f} Hz (sin match)")
        self.results.append((self.idx, ok, f0, cents, t_event))


def load_tab(path: str) -> List[TabNote]:
    data = json.load(open(path, "r", encoding="utf-8"))
    return [TabNote(d["string"], int(d["fret"]), float(d.get("beats", 1.0))) for d in data]


NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def f_to_note_name(f: float, a4: float = 440.0) -> str:
    if f <= 0: return "—"
    m = 69.0 + 12.0 * math.log2(f / a4)
    m_ = int(round(m))
    name = NOTE_NAMES[m_ % 12]
    octv = int(m_ // 12) - 1
    return f"{name}{octv}"

OPEN_MIDI = {"E": 40, "A": 45, "D": 50, "G": 55}

def midi_to_string_fret(midi: int) -> Optional[TabNote]:
    for string in ["G", "D", "A", "E"]:
        fret = midi - OPEN_MIDI[string]
        if 0 <= fret <= 24:
            return TabNote(string=string, fret=int(fret), beats=1.0)
    best = None
    best_err = 1e9
    for string in ["G", "D", "A", "E"]:
        fret = midi - OPEN_MIDI[string]
        fret_clamped = max(0, min(24, fret))
        midi_back = OPEN_MIDI[string] + fret_clamped
        err = abs(midi - midi_back)
        if err < best_err:
            best_err = err
            best = TabNote(string=string, fret=int(fret_clamped), beats=1.0)
    return best

async def send_pitch(ws: WebSocket, f0: float, f_target: Optional[float], correct: Optional[bool]):
    if ws.client_state != WebSocketState.CONNECTED:
        return
    cents = None
    if f0 > 0 and f_target and f_target > 0:
        cents = round(cents_error(f0, f_target))
    msg = {
        "event": "PITCH",
        "note": f_to_note_name(f0),
        "hz": round(f0, 2) if f0 > 0 else None,
        "cents": cents,
        "confidence": None,
        "correct": bool(correct) if correct is not None else None
    }
    try:
        await ws.send_text(json.dumps(msg))
    except (WebSocketDisconnect, ClientDisconnected, AttributeError):
        pass

def parse_tab_payload(notes_payload) -> List[TabNote]:
    tab = []
    if not isinstance(notes_payload, list) or len(notes_payload) == 0:
        return tab
    for item in notes_payload:
        if isinstance(item, int):
            midi = item
            beats = 1.0
        elif isinstance(item, dict):
            midi = int(item.get("midi"))
            beats = float(item.get("beats", 1.0))
        else:
            continue
        tn = midi_to_string_fret(midi)
        if tn:
            tn.beats = beats
            tab.append(tn)
    return tab



def _user_id_from_token(token: Optional[str]) -> Optional[int]:
    if not token:
        return None
    try:
        payload = decode_access_token(token)
        sub = payload.get("sub")
        return int(sub) if sub is not None else None
    except Exception:
        return None

def _open_db() -> Tuple[Optional[Session], callable]:
    try:
        from app.helper import get_db as _get_db
        gen = _get_db()
        db = next(gen)
        def _close():
            try:
                gen.close()
            except Exception:
                pass
        return db, _close
    except Exception:
            return None, (lambda: None)  

def _persist_warmup_session(
    db: Session,
    *,
    user_id: Optional[int],
    tablature_id: Optional[int],
    results: list[tuple],     
    duration_seconds: float,
) -> Optional[int]:
    if not user_id:
        return None
    total_steps = len(results)
    if total_steps <= 0:
        return None

    flags = [(True if r[1] else False) for r in results]
    correct_steps   = sum(1 for ok in flags if ok)
    timeouts        = sum(1 for (_i, ok, f0, _c, _t) in results if (not ok and (float(f0) == 0.0)))
    incorrect_steps = max(0, total_steps - correct_steps - timeouts)
    performance = (correct_steps / total_steps) * 100.0

    row = WarmupSession(
        user_id=user_id,
        tablature_id=tablature_id,
        duration_seconds=float(duration_seconds),
        performance_score=round(performance, 2),
        total_steps=total_steps,
        correct_steps=correct_steps,
        incorrect_steps=incorrect_steps,
        timeouts=timeouts,
        results_flags=flags,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row.id

# ======================= WebSocket =======================
@router.websocket("/ws/warmup")
async def ws_warmup(ws: WebSocket):
    await ws.accept()

    token_qs = ws.query_params.get("token")
    user_id: Optional[int] = _user_id_from_token(token_qs)

    tablature_id: Optional[int] = None

    sample_rate = SR
    evaluator = WarmUpEvaluator(tab=[], pulse_source="input", samplerate=sample_rate)

    audio_q = deque()
    stop_flag = False
    tab_ready_event = asyncio.Event()

    async def recv_loop():
        nonlocal sample_rate, evaluator, stop_flag, user_id, tablature_id
        try:
            while True:
                msg = await ws.receive()
                if "text" in msg and msg["text"] is not None:
                    try:
                        data = json.loads(msg["text"])
                    except Exception:
                        continue

                    t = data.get("type")

                    if t == "hello":
                        sr = int(data.get("sampleRate", SR))
                        sample_rate = sr
                        evaluator = WarmUpEvaluator(tab=evaluator.tab, pulse_source="input", samplerate=sample_rate)
                        if not user_id and isinstance(data.get("token"), str):
                            user_id = _user_id_from_token(data["token"])

                    elif t == "auth":
                        token = data.get("token")
                        if token and not user_id:
                            user_id = _user_id_from_token(token)

                    elif t == "set_tab":
                        notes_payload = data.get("notes", [])
                        new_tab = parse_tab_payload(notes_payload)
                        evaluator = WarmUpEvaluator(tab=new_tab, pulse_source="input", samplerate=sample_rate)
                        tablature_id = data.get("tablatureId") or data.get("tablature_id")
                        try:
                            tablature_id = int(tablature_id) if tablature_id is not None else None
                        except Exception:
                            tablature_id = None

                        if len(evaluator.tab) > 0:
                            tab_ready_event.set()

                elif "bytes" in msg and msg["bytes"] is not None:
                    buf = msg["bytes"]
                    x = np.frombuffer(buf, dtype=np.float32)
                    if x.ndim == 1 and x.size > 0:
                        audio_q.append(x)

                elif msg.get("type") == "websocket.disconnect":
                    break

        except WebSocketDisconnect:
            pass
        finally:
            stop_flag = True

    async def analysis_loop():
        nonlocal stop_flag, evaluator

        calib = []
        t_calib_start = time.time()
        while (time.time() - t_calib_start) < 1.0 and not stop_flag:
            if not audio_q:
                await asyncio.sleep(0.005)
                continue
            chunk = audio_q.popleft()
            for off in range(0, len(chunk) - FRAME_SIZE + 1, FRAME_SIZE):
                frame = chunk[off:off+FRAME_SIZE].astype(np.float32)
                calib.append(evaluator.rms(frame))
        if calib:
            rms_db_base = 20 * math.log10(max(float(np.median(calib)), 1e-12))
            evaluator.noise_gate_db = rms_db_base + NOISE_GATE_DB_OFFSET
        else:
            rms_db_base = -60.0
            evaluator.noise_gate_db = -48.0

        try:
            await asyncio.wait_for(tab_ready_event.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1008)
            return

        start_time = time.time()
        if USE_BPM and evaluator.tab:
            evaluator.current_target_time = start_time + 60.0 / BPM * evaluator.tab[0].beats

        while not stop_flag:
            if evaluator.idx >= len(evaluator.tab) and len(evaluator.tab) > 0:
                closer = (lambda: None)
                try:
                    if evaluator.results and user_id:
                        db, closer = _open_db()
                        if db is not None:
                            last_t = evaluator.results[-1][4] if evaluator.results else 0.0
                            _persist_warmup_session(
                                db,
                                user_id=user_id,
                                tablature_id=tablature_id,
                                results=evaluator.results,
                                duration_seconds=float(last_t),
                            )
                except Exception:
                    pass
                finally:
                    try:
                        closer()
                    except Exception:
                        pass

                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(json.dumps({"event": "DONE"}))
                    await ws.close()
                return

            if not audio_q:
                prev_idx = evaluator.idx
                prev_len = len(evaluator.results)

                evaluator._maybe_timeout(start_time)

                if (len(evaluator.results) > prev_len) and (evaluator.idx > prev_idx):
                    ok = bool(evaluator.results[-1][1])
                    await send_pitch(ws, 0.0, None, correct=ok)

                await asyncio.sleep(0.005)
                continue

            chunk = audio_q.popleft()
            for off in range(0, len(chunk) - FRAME_SIZE + 1, HOP_SIZE):
                frame = chunk[off: off + FRAME_SIZE].astype(np.float32)
                now = time.time()

                f0_raw = evaluator._frame_to_pitch(frame)
                rms = evaluator.rms(frame)

                f_target = evaluator._current_target_freq()
                if f0_raw > 0 and f_target:
                    f0_raw = fold_to_target_octave(f0_raw, f_target)
                if f0_raw > 0:
                    evaluator.f0_buf.append(f0_raw)
                f0 = float(np.median(evaluator.f0_buf)) if evaluator.f0_buf else 0.0

                onset = evaluator._detect_onset(f0, rms, now)
                if onset and not evaluator.in_note_window:
                    evaluator.in_note_window = True
                    evaluator.note_window_start = now 

                prev_idx = evaluator.idx
                prev_results_len = len(evaluator.results)

                evaluator._try_validate_current(f0, now, start_time)
                evaluator._maybe_timeout(start_time)

                correct_flag = None
                if len(evaluator.results) > prev_results_len and evaluator.idx > prev_idx:
                    ok = bool(evaluator.results[-1][1])
                    correct_flag = ok

                await send_pitch(ws, f0, f_target, correct=correct_flag)

                if stop_flag or ws.client_state != WebSocketState.CONNECTED:
                    return

    recv_task = asyncio.create_task(recv_loop())
    analysis_task = asyncio.create_task(analysis_loop())
    done, pending = await asyncio.wait({recv_task, analysis_task}, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    if ws.client_state == WebSocketState.CONNECTED:
        await ws.close()
