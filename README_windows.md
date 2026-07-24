# window 本地部署指导

## AMD版本
依赖：
 Visual Studio Build Tools（C++ 编译工具链）

可选：
    AMD版本GPU加速 Vulkan SDK（AMD显卡）

安装：llama_cpp_python（AMD GPU版本）
依赖 Vulkan SDK、Visual Studio Build Tools（C++ 编译工具链）
```
$env:CMAKE_ARGS = "-DGGML_VULKAN=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir
```

## Nvidia版本
NULL


## 模型转换

```
# 1. 导出视觉投影层 mmproj（架构为 clip，不能单独推理）
python convert_hf_to_gguf.py "D:\workdir2\mano-skill\model" --mmproj --outtype f16 `
    --outfile D:\workdir2\mano-skill\model\Mano-CUA-2.0-4B-mmproj-F16.gguf

# 2. 导出文本模型主体（36 层 qwen3vl backbone，398 tensors）
python convert_hf_to_gguf.py "D:\workdir2\mano-skill\model" --outtype f16 `
    --outfile D:\workdir2\mano-skill\model\Mano-CUA-2.0-4B-F16.gguf
```

运行
~~~
python3 vla.py run "查看微信的第一个消息什么内容？" --local --model-path D:\workdir2\mano-skill\model
~~~