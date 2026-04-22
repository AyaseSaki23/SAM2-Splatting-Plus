# Geo-Semantic-Splatting

Geo-Semantic-Splatting 是一个基于 Nerfstudio / Splatfacto 的几何-语义联合重建项目，包含从视频抽帧、SAM2 标注与推理，到 Stage1 几何重建、Stage2 语义注入与可视化的完整流程。

## 主要功能

- 视频抽帧与数据整理
- 交互式 prompts 标注
- SAM2 批量推理生成 ID mask
- Nerfstudio Splatfacto Stage1 训练
- Stage2 语义训练、几何-语义对齐与 object culling
- TensorBoard / viewer 结果查看

## 目录结构

```text
Geo-Semantic-Splatting/
├─ app/                 # 演示入口
├─ configs/             # 配置文件 main
├─ data/                # 原始数据、标注与处理结果
├─ outputs/             # Stage1 输出（已忽略）
├─ outputs_stage2/      # Stage2 输出（已忽略）
├─ sam2/                # SAM2 代码/本地依赖
├─ scripts/             # 训练与辅助脚本
├─ src/                 # 核心代码
└─ requirements.txt
```

## 环境安装

建议使用 Python 3.11 + CUDA 环境。

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

如果你使用的是本地 SAM2 / Nerfstudio 依赖，请确保它们已正确安装并可导入。

## 1. 抽帧

把视频转成连续帧，输出通常放到：

```text
data/processed/scene_01/images
```

可先查看脚本帮助：

```powershell
.\venv\Scripts\python.exe src\preprocess\extract_frames.py --help
```

## 2. 标注 prompts

生成 prompts JSON：

```powershell
.\venv\Scripts\python.exe src\preprocess\make_prompts_json.py `
  --images-dir data\processed\scene_01\images `
  --output-json data\prompts\scene_01_prompts.json
```

建议：

- 每个物体在多帧上标注
- 尽量同时提供 box 和正样本点
- 背景如果需要参与训练，可单独作为 `obj_id=0`
- `label_name` 主要用于可读性

## 3. SAM2 推理

根据 prompts 生成每帧 ID mask：

```powershell
.\venv\Scripts\python.exe src\preprocess\sam2_inference.py `
  --images-dir data\processed\scene_01\images `
  --output-dir data\processed\scene_01\sam2_out `
  --sam2-checkpoint checkpoints\sam2_hiera_large.pt `
  --model-config sam2_hiera_l `
  --prompts-json data\prompts\scene_01_prompts.json `
  --threshold 0.0 `
  --save-overlays `
  --overwrite
```

输出一般包括：

- `id_masks/`：每帧的类别 ID mask
- `overlays/`：可视化叠加图
- `summary.json`：推理摘要

## 4. Stage1 训练

先进行基础几何/外观重建：

```powershell
ns-train splatfacto --data data\processed\scene_01
```

训练完成后，结果一般在：

```text
outputs\scene_01\splatfacto\YYYY-MM-DD_HHMMSS\
```

## 5. Stage2 语义训练

在 Stage1 checkpoint 基础上继续做语义注入：

```powershell
.\venv\Scripts\python.exe scripts\train_stage2_semantic.py `
  --data data\processed\scene_01 `
  --load-checkpoint outputs\scene_01\splatfacto\YYYY-MM-DD_HHMMSS\nerfstudio_models\step-000029999.ckpt `
  --semantic-masks-dir data\processed\scene_01\sam2_out\id_masks `
  --num-classes 4 `
  --semantic-dim 16 `
  --lambda-sem 0.1 `
  --max-num-iterations 2000 `
  --freeze-geometry `
  --vis tensorboard `
  --output-dir outputs_stage2
```

说明：

- `--load-checkpoint` 指向 Stage1 checkpoint
- `--max-num-iterations` 是本次训练的终止步数
- `--output-dir` 建议单独放到 `outputs_stage2`

## 6. 查看结果

### TensorBoard

```powershell
tensorboard --logdir outputs_stage2
```

### Viewer

如果训练目录里有 `config_viewer.yml`，可直接打开：

```powershell
.\venv\Scripts\python.exe -m nerfstudio.scripts.viewer.run_viewer --load-config 
```

如果只有 `config.yml`，可以复制一份并将 `method_name` 改成 `splatfacto` 作为 viewer 配置。

## 7. 常见输出文件

- `semantic_stats_latest.log`：语义统计日志
- `config.yml`：训练配置
- `config_viewer.yml`：viewer 配置
- `nerfstudio_models/step-*.ckpt`：训练 checkpoint

