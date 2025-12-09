# 实时声信号采集与可视化

该程序使用默认麦克风采集音频，提供倒计时提醒、实时时域波形显示，以及录制结束后的时域与频域展示。

## 功能
- 录制前 3 秒倒计时提示界面。
- 录制过程中实时显示声信号时域波形。
- 窗口下方按钮手动停止采集。
- 录制结束后展示整段信号的时域和频域（幅度谱）。

## 运行
```bash
pip install matplotlib sounddevice numpy
python main.py
```

运行后根据窗口提示完成录制和停止操作。

### 字体显示为方块的解决方法
- 确保系统已安装支持中文的字体（如 Noto Sans CJK、思源黑体、黑体/雅黑 等）。
- 如果系统字体列表与默认优先级不匹配，程序会自动扫描所有字体以找到支持中文的字体。
- 仍无法正常显示时，可将字体文件路径写入环境变量 `RECORDER_FONT_PATH` 后再运行：

  ```bash
  RECORDER_FONT_PATH=/path/to/your/ChineseFont.ttf python main.py
  ```
