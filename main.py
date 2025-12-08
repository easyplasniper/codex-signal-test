"""Simple real-time audio recorder with countdown and visualization.

This script uses the default microphone input to capture audio, displays a
real-time time-domain waveform with a stop button, and after recording ends it
shows the full time-domain and frequency-domain plots for the captured signal.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button


@dataclass
class RecorderConfig:
    """Configuration for the audio recorder."""

    samplerate: int = 44_100
    channels: int = 1
    blocksize: int = 1_024
    display_window_seconds: float = 3.0
    countdown_seconds: int = 3


@dataclass
class RecorderState:
    """Runtime state of the recorder."""

    running: bool = False
    stop_requested: bool = False
    buffer_queue: queue.Queue[np.ndarray] = field(default_factory=queue.Queue)
    recorded_blocks: List[np.ndarray] = field(default_factory=list)


class AudioRecorder:
    """Capture audio blocks and provide data for live visualization."""

    def __init__(self, config: RecorderConfig) -> None:
        self.config = config
        self.state = RecorderState()
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:  # type: ignore[override]
        if status:
            print(f"Stream status: {status}")
        with self._lock:
            if self.state.running and not self.state.stop_requested:
                data_copy = indata.copy().reshape(-1)
                self.state.buffer_queue.put(data_copy)
                self.state.recorded_blocks.append(data_copy)

    def start(self) -> None:
        """Start audio capture."""
        if self.state.running:
            return
        self.state.running = True
        self._stream = sd.InputStream(
            samplerate=self.config.samplerate,
            channels=self.config.channels,
            blocksize=self.config.blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop audio capture and close the stream."""
        with self._lock:
            self.state.stop_requested = True
            self.state.running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def collect_full_signal(self) -> np.ndarray:
        """Return the concatenated recording as a 1-D numpy array."""
        if not self.state.recorded_blocks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.state.recorded_blocks)


class LivePlotter:
    """Manage the real-time waveform display and stop button."""

    def __init__(self, recorder: AudioRecorder) -> None:
        self.recorder = recorder
        self.config = recorder.config
        self.live_buffer = np.zeros(
            int(self.config.samplerate * self.config.display_window_seconds),
            dtype=np.float32,
        )
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(bottom=0.2)
        self.line, = self.ax.plot(self.live_buffer)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, len(self.live_buffer))
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title("实时声信号波形（点击停止按钮结束录制）")

        stop_ax = plt.axes([0.45, 0.05, 0.1, 0.075])
        self.stop_button = Button(stop_ax, "停止收集")
        self.stop_button.on_clicked(self._handle_stop)

        self.animation = FuncAnimation(
            self.fig,
            self._update_plot,
            interval=50,
            blit=False,
        )

    def _handle_stop(self, _event) -> None:
        print("Stop requested by user.")
        self.recorder.stop()
        plt.close(self.fig)

    def _update_plot(self, _frame):
        while not self.recorder.state.buffer_queue.empty():
            new_data = self.recorder.state.buffer_queue.get_nowait()
            shift_len = len(new_data)
            self.live_buffer = np.roll(self.live_buffer, -shift_len)
            self.live_buffer[-shift_len:] = new_data
        self.line.set_ydata(self.live_buffer)
        return self.line,

    def show(self) -> None:
        plt.show()


def countdown(seconds: int) -> None:
    """Display a simple countdown before recording starts."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    text = ax.text(0.5, 0.5, "", ha="center", va="center", fontsize=28)
    for remaining in range(seconds, 0, -1):
        text.set_text(f"录制将在 {remaining} 秒后开始")
        plt.pause(1)
    text.set_text("开始录制！")
    plt.pause(0.8)
    plt.close(fig)


def plot_full_signal(signal: np.ndarray, samplerate: int) -> None:
    """Plot the full time-domain and frequency-domain representations."""
    if signal.size == 0:
        print("No audio was recorded.")
        return

    time_axis = np.arange(signal.size) / samplerate
    freq_domain = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1 / samplerate)
    magnitude = np.abs(freq_domain)

    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8))

    ax_time.plot(time_axis, signal)
    ax_time.set_title("整段信号的时域波形")
    ax_time.set_xlabel("时间 (秒)")
    ax_time.set_ylabel("幅值")

    ax_freq.plot(freqs, magnitude)
    ax_freq.set_title("整段信号的频域幅度谱")
    ax_freq.set_xlabel("频率 (Hz)")
    ax_freq.set_ylabel("幅度")
    ax_freq.set_xlim(0, samplerate / 2)

    plt.tight_layout()
    plt.show()


def main() -> None:
    config = RecorderConfig()
    print("欢迎使用实时声信号采集与显示程序！")
    print("请确保麦克风已就绪。")
    countdown(config.countdown_seconds)

    recorder = AudioRecorder(config)
    live_plotter = LivePlotter(recorder)

    recorder.start()
    print("开始采集。点击窗口中的‘停止收集’按钮结束录制。")
    live_plotter.show()

    recorder.stop()
    full_signal = recorder.collect_full_signal()
    if full_signal.size > 0:
        duration = full_signal.size / config.samplerate
        print(f"录制完成，时长约 {duration:.2f} 秒。")
    plot_full_signal(full_signal, config.samplerate)


if __name__ == "__main__":
    main()
