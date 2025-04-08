# 说话人分割识别系统
## 环境配置
## 基础环境
Python 3.10
CUDA 11.8
nvidia-driver-535
nvidia-utils-535

## CUDA 配置
确保系统已正确安装CUDA 11.8，可通过以下命令验证：

nvcc --version

## 项目功能
本项目是一个说话人分割识别系统，可以：
从音频文件中自动识别不同的说话人
输出每个说话人的发言时间段
支持上传文件、麦克风录音或URL方式输入音频
提供多种不同的说话人嵌入模型选择


## 运行项目
python app.py

## 使用方法

选择说话人嵌入框架（Speaker embedding frameworks）
支持多种中文模型3D-Speaker

选择说话人分割模型（Speaker segmentation model）
可选pyannote或reverb模型

设置说话人数量参数
如果设置为0，则使用自动聚类

设置聚类阈值
仅当说话人数量设为0时有效


通过以下方式之一输入音频：
上传音频文件
使用麦克风录制
提供音频URL
点击"Submit for speaker diarization"按钮开始处理
输出解释
处理完成后，系统会显示：
每个说话人的发言时间段（开始时间 -- 结束时间 说话人ID）
音频时长、处理时间和实时因子(RTF)

