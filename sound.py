"""Procedural sound design (numpy, no audio files): soft sine/FM tones, filtered noise, a small
convolution reverb and stereo panning by on-screen position."""
import numpy as np
import pygame as pg

SR = 44100
_rng = np.random.default_rng(7)


# ------------------------------------------------------------------ DSP helpers
def _t(dur):
    return np.arange(int(SR * dur)) / SR


def env(dur, attack=0.004, decay=0.1, sustain=0.0, release=None):
    """Attack then exponential decay (time constant `decay` seconds)."""
    t = _t(dur)
    a = np.clip(t / attack, 0, 1) if attack else np.ones_like(t)
    e = a * (sustain + (1 - sustain) * np.exp(-t / decay))
    if release:
        e *= np.clip((dur - t) / release, 0, 1)
    return e


def sine(freq, dur, phase=0.0):
    f = np.broadcast_to(np.asarray(freq, dtype=float), (int(SR * dur),))
    return np.sin(phase + np.cumsum(2 * np.pi * f / SR))


def glide(f0, f1, dur, curve=4.0):
    t = _t(dur) / dur
    return f1 + (f0 - f1) * np.exp(-curve * t)


def noise(dur):
    return _rng.uniform(-1, 1, int(SR * dur))


def _fft_filter(x, lo=None, hi=None, order=2):
    n = 1 << int(np.ceil(np.log2(len(x) * 2)))
    X = np.fft.rfft(x, n)
    f = np.fft.rfftfreq(n, 1 / SR)
    H = np.ones_like(f)
    if hi:
        H /= np.sqrt(1 + (f / hi) ** (2 * order))
    if lo:
        H *= 1 / np.sqrt(1 + (lo / np.maximum(f, 1e-3)) ** (2 * order))
    return np.fft.irfft(X * H, n)[:len(x)]


def lowpass(x, hz, order=2):
    return _fft_filter(x, hi=hz, order=order)


def highpass(x, hz, order=2):
    return _fft_filter(x, lo=hz, order=order)


def bandpass(x, lo, hi, order=2):
    return _fft_filter(x, lo=lo, hi=hi, order=order)


def reverb(x, time=0.6, mix=0.25, tone=5000):
    ir_len = int(SR * time)
    ir = lowpass(noise(time) * np.exp(-np.arange(ir_len) / (SR * time / 5)), tone)
    ir /= np.sqrt((ir ** 2).sum()) + 1e-9
    n = 1 << int(np.ceil(np.log2(len(x) + ir_len)))
    wet = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)[:len(x) + ir_len]
    out = np.zeros(len(x) + ir_len)
    out[:len(x)] += x * (1 - mix)
    out += wet * mix * 0.6
    return out


def pad(x, dur):
    out = np.zeros(int(SR * dur))
    out[:min(len(x), len(out))] = x[:len(out)]
    return out


def seq(parts):
    """parts: list of (start_seconds, signal) mixed together."""
    end = max(int(s * SR) + len(sig) for s, sig in parts)
    out = np.zeros(end)
    for s, sig in parts:
        i = int(s * SR)
        out[i:i + len(sig)] += sig
    return out


def note(freq, dur, bright=0.3, decay=0.25, attack=0.006):
    """Soft bell/pluck: sine + gentle FM overtone."""
    t = _t(dur)
    mod = np.sin(2 * np.pi * freq * 2 * t) * bright * np.exp(-t / (decay * 0.5))
    return np.sin(2 * np.pi * freq * t + mod) * env(dur, attack, decay)


def normalize(x, peak):
    m = np.abs(x).max()
    return x * (peak / m) if m > 0 else x


# ------------------------------------------------------------------ the sounds
def _build():
    s = {}
    # player shot: soft laser "pew" - falling sine with a hint of FM, dark filtered click
    body = sine(glide(1500, 340, 0.11, 5), 0.11) * env(0.11, 0.002, 0.035)
    click = highpass(noise(0.012), 2500) * env(0.012, 0.0005, 0.003)
    s["shoot"] = normalize(lowpass(body + 0.25 * pad(click, 0.11), 6000), 0.22)
    body = sine(glide(900, 220, 0.12, 5), 0.12) * env(0.12, 0.002, 0.04)
    s["shoot_ai"] = normalize(lowpass(body, 3500), 0.15)
    # you hit the enemy: bright two-partial "tink" + small reverb (rewarding)
    tink = note(1568, 0.25, 0.4, 0.07) + 0.5 * note(2349, 0.25, 0.2, 0.05)
    s["hit"] = normalize(lowpass(reverb(tink, 0.4, 0.2, 3500), 5000), 0.2)
    # you got hit: deep punch (kick-style pitch drop) + muffled crunch
    kick = sine(glide(190, 48, 0.28, 9), 0.28) * env(0.28, 0.001, 0.09)
    crunch = lowpass(noise(0.1), 900) * env(0.1, 0.001, 0.025)
    s["hurt"] = normalize(np.tanh(1.6 * (kick + 0.6 * pad(crunch, 0.28))), 0.45)
    # death: big low boom with a long airy tail
    boom = sine(glide(120, 32, 0.9, 5), 0.9) * env(0.9, 0.002, 0.25)
    rumble = lowpass(noise(0.9), 500) * env(0.9, 0.003, 0.3)
    s["death"] = normalize(reverb(np.tanh(1.4 * (boom + 0.7 * rumble)), 1.0, 0.3, 2500), 0.5)
    # dodge: airy whoosh
    w = lowpass(bandpass(noise(0.22), 500, 2200), 3000) * np.sin(np.pi * np.clip(_t(0.22) / 0.22, 0, 1)) ** 2
    s["dodge"] = normalize(w, 0.07)
    # UI
    s["hover"] = normalize(note(2400, 0.05, 0.0, 0.012), 0.05)
    s["select"] = normalize(reverb(seq([(0, note(880, 0.18, 0.3, 0.06)), (0.06, note(1318.5, 0.25, 0.3, 0.08))]),
                                   0.5, 0.25), 0.14)
    s["back"] = normalize(reverb(seq([(0, note(1046.5, 0.15, 0.2, 0.05)), (0.05, note(784, 0.2, 0.2, 0.06))]),
                                 0.4, 0.2), 0.11)
    # match flow
    s["count"] = normalize(reverb(note(740, 0.3, 0.2, 0.09), 0.6, 0.3), 0.13)
    chord = sum(note(f, 1.0, 0.25, 0.35, 0.02) for f in (523.3, 659.3, 784.0, 1046.5))
    s["go"] = normalize(reverb(chord, 0.9, 0.3), 0.16)
    arp = seq([(i * 0.09, note(f, 0.7, 0.3, 0.22, 0.01)) for i, f in enumerate((523.3, 659.3, 784.0, 1046.5, 1318.5))])
    s["win"] = normalize(reverb(arp, 1.2, 0.35), 0.17)
    down = seq([(i * 0.16, lowpass(note(f, 0.8, 0.1, 0.3, 0.02), 2500)) for i, f in enumerate((440.0, 349.2, 293.7, 220.0))])
    s["lose"] = normalize(reverb(down, 1.2, 0.35, 3000), 0.17)
    rise = bandpass(noise(0.6), 400, 4000) * (_t(0.6) / 0.6) ** 2 * env(0.6, 0.0, 10, release=0.05)
    s["wave"] = normalize(reverb(seq([(0, 0.5 * rise), (0.55, chord[:int(SR * 0.8)] * 0.8)]), 0.9, 0.3), 0.16)
    # pickup: bright rising two-note chime
    s["pickup"] = normalize(reverb(seq([(0, note(1046.5, 0.2, 0.5, 0.06)), (0.07, note(1568, 0.35, 0.5, 0.1))]),
                                   0.6, 0.3), 0.16)
    # shield absorbs a hit: glassy ping
    ping = note(2093, 0.3, 0.9, 0.05) + 0.4 * note(3136, 0.3, 0.5, 0.04)
    s["shield"] = normalize(lowpass(reverb(ping, 0.5, 0.25, 4000), 5500), 0.14)
    # heavy weapons
    body = sine(glide(700, 90, 0.2, 6), 0.2) * env(0.2, 0.001, 0.06)
    zap = lowpass(noise(0.2), 1500) * env(0.2, 0.001, 0.03)
    s["shoot_rail"] = normalize(np.tanh(2 * (body + 0.4 * zap)), 0.24)
    burst = lowpass(noise(0.12), 1800) * env(0.12, 0.001, 0.03)
    burst = burst + 0.6 * sine(glide(260, 80, 0.12, 6), 0.12) * env(0.12, 0.001, 0.05)
    s["shoot_shotgun"] = normalize(np.tanh(1.5 * burst), 0.26)
    return s


class Sound:
    def __init__(self, enabled=True, volume=0.7):
        self.enabled = enabled
        self.volume = volume
        self.sounds = {}
        try:
            if not pg.mixer.get_init():
                pg.mixer.pre_init(SR, -16, 2, 512)
                pg.mixer.init(SR, -16, 2, 512)
            freq, _, self.ch = pg.mixer.get_init()
            for k, wav in _build().items():
                if freq != SR:
                    idx = np.linspace(0, len(wav) - 1, int(len(wav) * freq / SR))
                    wav = np.interp(idx, np.arange(len(wav)), wav)
                pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
                if self.ch > 1:
                    pcm = np.repeat(pcm[:, None], self.ch, 1)
                self.sounds[k] = pg.mixer.Sound(buffer=np.ascontiguousarray(pcm).tobytes())
            pg.mixer.set_num_channels(32)
        except Exception as e:  # no audio device: run silently
            print("sound disabled:", e)
            self.sounds = {}

    def play(self, name, x=None, gain=1.0):
        """x: horizontal screen position for stereo panning (None = centre)."""
        if not (self.enabled and name in self.sounds and self.volume > 0):
            return
        ch = self.sounds[name].play()
        if ch is None:
            return
        v = self.volume * gain
        if x is None:
            ch.set_volume(v, v)
        else:
            pan = float(np.clip(x / 800, 0, 1))
            ch.set_volume(v * min(1, 2 * (1 - pan)) ** 0.5, v * min(1, 2 * pan) ** 0.5)
