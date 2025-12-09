"""Simple real-time audio recorder with countdown and visualization.

This script uses the default microphone input to capture audio, displays a
real-time time-domain waveform with a stop button, and after recording ends it
shows the full time-domain and frequency-domain plots for the captured signal.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import Button


@dataclass
class RecorderConfig:
    """Configuration for the audio recorder."""

    # 采样率（每秒采集多少个点），与声卡设置一致即可
    samplerate: int = 44_100
    # 录音通道数，1 表示单声道，2 表示立体声
    channels: int = 1
    # 回调每次推送的采样点数，越小延迟越低但 CPU 占用越高
    blocksize: int = 1_024
    # 实时显示窗口长度（秒），决定滚动波形可看到的时间范围
    display_window_seconds: float = 3.0
    # 开始录制前的倒计时秒数
    countdown_seconds: int = 3


@dataclass
class RecorderState:
    """Runtime state of the recorder."""

    # 标记录音是否开始
    running: bool = False
    # 标记是否收到停止指令（按钮事件设置）
    stop_requested: bool = False
    # 实时回调把采样数据放入队列，绘图线程从这里取数据
    buffer_queue: queue.Queue[np.ndarray] = field(default_factory=queue.Queue)
    # 录制的所有块依次累加在列表，结束后拼成完整信号
    recorded_blocks: List[np.ndarray] = field(default_factory=list)


@lru_cache(maxsize=1)
def get_chinese_font() -> FontProperties | None:
    """Return a font with Chinese glyph support if available.

    The helper first checks several common Chinese fonts and then falls back to
    scanning every available font for one that supports Chinese characters.
    This avoids missing-glyph boxes (口口口) when the system has a different
    Chinese font installed from the preferred list.
    """

    # 避免负号显示为方块
    plt.rcParams["axes.unicode_minus"] = False

    # 0) 优先使用用户通过环境变量指定的字体，确保可控性
    custom_font_path = os.getenv("RECORDER_FONT_PATH")
    if custom_font_path and os.path.exists(custom_font_path):
        try:
            font_manager.fontManager.addfont(custom_font_path)
            font_prop = FontProperties(fname=custom_font_path)
            plt.rcParams["font.family"] = [font_prop.get_name()] + plt.rcParams.get(
                "font.family", []
            )
            return font_prop
        except Exception:
            print("警告：指定的 RECORDER_FONT_PATH 无法加载，尝试使用系统字体。")
    preferred_fonts = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
        "PingFang SC",
        "Hiragino Sans GB",
        "SimSun",
        "Microsoft JhengHei",
    ]

    available = font_manager.fontManager.ttflist
    available_names = {font.name for font in available}

    # 1) 优先尝试常见中文字体名称（命中率最高）
    for font_name in preferred_fonts:
        if font_name in available_names:
            plt.rcParams["font.family"] = [font_name] + plt.rcParams.get(
                "font.family", []
            )
            return FontProperties(family=font_name)

    # 2) 兜底：遍历系统字体，找到任意支持中文字符的字体
    sample_text = "中文测试"
    for font in available:
        try:
            if font_manager.get_font(font.fname).supports_text(sample_text):
                font_properties = FontProperties(fname=font.fname)
                plt.rcParams["font.family"] = [font_properties.get_name()] + plt.rcParams.get(
                    "font.family", []
                )
                return font_properties
        except Exception:
            # Ignore fonts that Matplotlib cannot load.
            continue

    print(
        "提示：未找到可用的中文字体，文本可能显示为方块。"
        "请安装 Noto Sans CJK、思源黑体、黑体/雅黑等字体后重试。"
    )
    return None


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
                # 复制数据避免被 sounddevice 重用，reshape(-1) 将多通道展开成一维
                data_copy = indata.copy().reshape(-1)
                # 实时绘图线程从队列取数据显示
                self.state.buffer_queue.put(data_copy)
                # 录制结束后把所有块拼接成完整信号
                self.state.recorded_blocks.append(data_copy)

    def start(self) -> None:
        """Start audio capture."""
        if self.state.running:
            return
        # 标记为运行中，回调函数才会保存数据
        self.state.running = True
        self._stream = sd.InputStream(
            samplerate=self.config.samplerate,
            channels=self.config.channels,
            blocksize=self.config.blocksize,
            callback=self._callback,
        )
        # 打开麦克风输入流，sounddevice 会在后台线程持续触发 _callback
        self._stream.start()

    def stop(self) -> None:
        """Stop audio capture and close the stream."""
        with self._lock:
            self.state.stop_requested = True
            self.state.running = False
        if self._stream is not None:
            # 停止并关闭底层音频流，释放设备资源
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def collect_full_signal(self) -> np.ndarray:
        """Return the concatenated recording as a 1-D numpy array."""
        if not self.state.recorded_blocks:
            return np.array([], dtype=np.float32)
        # 将连续的块拼接成完整录音（1 维数组）
        return np.concatenate(self.state.recorded_blocks)


class LivePlotter:
    """Manage the real-time waveform display and stop button."""

    def __init__(self, recorder: AudioRecorder) -> None:
        self.recorder = recorder
        self.config = recorder.config
        # 用固定长度的一维数组保存“窗口”内的最新数据，超出的数据向左滚动
        self.live_buffer = np.zeros(
            int(self.config.samplerate * self.config.display_window_seconds),
            dtype=np.float32,
        )
        font_props = get_chinese_font()
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        plt.subplots_adjust(bottom=0.2)
        # line 是实时波形曲线句柄，后续直接修改其 y 数据即可刷新图像
        self.line, = self.ax.plot(self.live_buffer)
        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, len(self.live_buffer))
        self.ax.set_xlabel("Sample", fontproperties=font_props)
        self.ax.set_ylabel("Amplitude", fontproperties=font_props)
        self.ax.set_title("实时声信号波形（点击停止按钮结束录制）", fontproperties=font_props)

        stop_ax = plt.axes([0.45, 0.05, 0.1, 0.075])
        self.stop_button = Button(stop_ax, "停止收集")
        if font_props is not None:
            self.stop_button.label.set_fontproperties(font_props)
        # 按钮点击后触发 _handle_stop，通知录音结束
        self.stop_button.on_clicked(self._handle_stop)

        self.animation = FuncAnimation(
            self.fig,
            self._update_plot,
            interval=50,  # 每 50 ms 刷新一次画面（约 20 FPS）
            blit=False,
        )

    def _handle_stop(self, _event) -> None:
        print("Stop requested by user.")
        self.recorder.stop()
        plt.close(self.fig)

    def _update_plot(self, _frame):
        # 消耗队列里累计的所有新数据，更新滚动窗口
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
    font_props = get_chinese_font()
    fig, ax = plt.subplots(figsize=(6, 3))
    # 只显示文字，不需要坐标轴
    ax.axis("off")
    text = ax.text(
        0.5, 0.5, "", ha="center", va="center", fontsize=28, fontproperties=font_props
    )
    for remaining in range(seconds, 0, -1):
        # 每秒更新提示文本
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

    font_props = get_chinese_font()
    time_axis = np.arange(signal.size) / samplerate
    # 对录音做快速傅里叶变换，得到频域幅度谱
    freq_domain = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1 / samplerate)
    magnitude = np.abs(freq_domain)

    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(10, 8))

    ax_time.plot(time_axis, signal)
    ax_time.set_title("整段信号的时域波形", fontproperties=font_props)
    ax_time.set_xlabel("时间 (秒)", fontproperties=font_props)
    ax_time.set_ylabel("幅值", fontproperties=font_props)

    ax_freq.plot(freqs, magnitude)
    ax_freq.set_title("整段信号的频域幅度谱", fontproperties=font_props)
    ax_freq.set_xlabel("频率 (Hz)", fontproperties=font_props)
    ax_freq.set_ylabel("幅度", fontproperties=font_props)
    ax_freq.set_xlim(0, samplerate / 2)

    plt.tight_layout()
    plt.show()


def main() -> None:
    config = RecorderConfig()
    print("欢迎使用实时声信号采集与显示程序！")
    print("请确保麦克风已就绪。")
    # 简单的可视化倒计时，给用户准备时间
    countdown(config.countdown_seconds)

    recorder = AudioRecorder(config)
    live_plotter = LivePlotter(recorder)

    # 打开麦克风开始采集，并启动实时波形界面
    recorder.start()
    print("开始采集。点击窗口中的‘停止收集’按钮结束录制。")
    live_plotter.show()

    # 用户点击停止后才会回到这里
    recorder.stop()
    full_signal = recorder.collect_full_signal()
    if full_signal.size > 0:
        duration = full_signal.size / config.samplerate
        print(f"录制完成，时长约 {duration:.2f} 秒。")
    # 展示完整录音的时域与频域曲线
    plot_full_signal(full_signal, config.samplerate)


if __name__ == "__main__":
    main()
