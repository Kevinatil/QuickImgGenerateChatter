## 各环节支持配置
- LLM
    - DeepSeek-R1 8B
- 图像生成
    - SANA1.5 1.6B
    - flux-dev
    - flux-schnell


## 部署DeepSeek-R1 8B

```shell
brew install ollama # mac
curl -fsSL https://ollama.com/install.sh | sh # linux

ollama serve # 打开服务才能通过openai接口调用

ollama run deepseek-r1:8b # 自动下载模型，下载完成后kill掉
```

## 部署flux

[flux本地安装](https://github.com/black-forest-labs/flux?tab=readme-ov-file#local-installation)