# pyCinemetircsV1.0-Crew
在进度条基础上增加了演职员表的字幕提取

# 修改细节
1、在algorithms中添加了CrewEasyOcr.py用于识别演职员表

2、在algorithms/easyocr_utils/character中添加了德语、西班牙语、法语、日语、韩语的语言库

3、更改了ui/control.py，将subtitle按钮暂时用于演职员表识别（SubtitleEasyOcr->CrewEasyOcr）
