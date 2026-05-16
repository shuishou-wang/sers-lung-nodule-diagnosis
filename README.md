# SERS 多模态融合肺结节诊断系统

基于表面增强拉曼光谱（SERS）+ 临床肿瘤标志物的肺结节良恶性智能辅助诊断系统。

## 技术路线

```
SERS光谱 → airPLS基线校正 → SG(11,3)平滑 → Z-Score归一化 → PCA(14维)
                                                              ↓
临床标志物(CEA+SCC+NSE) → 分组中位数插补 → Z-Score归一化 ────→ 特征拼接(17维)
                                                              ↓
                                                        SMOTE(k=3) → 随机森林(50,3)
```

## 性能指标（LOOCV, N=49）

| 指标 | 数值 |
|------|:----:|
| 准确率 (Acc) | 87.8% |
| AUC | 0.891 |
| 灵敏度 (Sens) | 86.7% |
| 特异度 (Spec) | 89.5% |
| F1 | 89.7% |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key 和密码
```

### 3. 准备模型文件

将以下文件放入 `模型/` 目录：
- `pipeline.pkl` — 训练好的 sklearn Pipeline
- `train_data_cache.npz` — 训练数据缓存（用于可视化）
- `demo_cases.json` — 演示案例数据

### 4. 启动

```bash
cd 代码
python app.py
```

打开 http://localhost:5000，使用 `.env` 中配置的账号密码登录。

## LLM 后端

支持两种后端，在 `app.py` 中通过 `LLM_BACKEND` 切换：

| 后端 | 模型 | 速度 | 费用 |
|------|------|:--:|:--:|
| SiliconFlow | Qwen2.5-7B | ~8s | 免费额度 |
| Ollama 本地 | Qwen2.5:3B | ~8s | 完全免费 |

## 文献验证的拉曼生物标志物

| 峰位 (cm⁻¹) | 归属 | 生物机制 |
|:-----------:|------|------|
| 643 | 酪氨酸 C-C 扭转 | RTK 通路过度激活 |
| 822 | 酪氨酸环呼吸 | 肿瘤微环境重塑 |
| 1004 ★ | 苯丙氨酸环呼吸 | Warburg 效应 |
| 1126 | 蛋白 C-N 伸缩 | 细胞膜合成↑ |
| 1655 ★ | 酰胺I (α-螺旋) | 癌蛋白过表达 |
| 1675 ★ | 酰胺I (β-折叠) | CAF 活化 |

## 目录结构

```
├── 代码/
│   ├── app.py                  # Flask Web 应用
│   ├── pipeline_utils.py       # 预处理 + 推理管线
│   └── train_save_pipeline.py  # 模型训练脚本
├── 模型/
│   ├── pipeline_metadata.json  # 超参数记录
│   ├── demo_cases.json         # 演示案例
│   └── loocv_roc.json          # ROC 曲线数据
├── web/
│   └── templates/
│       ├── index.html          # 主界面
│       └── login.html          # 登录页
├── requirements.txt
├── .env.example
└── .gitignore
```

## 免责声明

本系统为**辅助诊断工具**，仅供临床参考，不能替代病理活检等金标准诊断方法。

## License

MIT
