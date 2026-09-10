# E1 结论：C1「依赖不可共存」不成立

日期：2026-09-09 · uv 0.9.8 (Win) / 0.9.9 (WSL) · Python 3.11 · Linux 侧为 WSL2 Ubuntu 22.04

## 一句话

设计文档的 C1 —— 「训练栈与环境栈在同一个 Python 环境里不可共存」——
**经三轮实验证否**（元数据层、Linux 安装层、环境×环境矩阵）。
第四轮量化了唯一幸存的差别。以此为地基的定位需要重估。

## 前三轮：证否 C1

### 第一轮：训练栈 × 环境，元数据层（`run.sh`）

8 组基线全部 RESOLVED（工具与各侧约束无问题），3 组冲突配对中只有 1 组失败。

| 配对 | 结果 |
|---|---|
| d4rl × reinflow | RESOLVED ✗ |
| robomimic × dppo | RESOLVED ✗ |
| **libero × openpi** | **FAILED ✓** |

唯一失败的原因：**lerobot 钉死 `gymnasium==0.29.1`，env-client 要求 `>=1.2.0`**。
这是策略侧上游包的一个普通 pin，向 lerobot 提 PR 放宽即可解决 ——
恰恰说明它不是结构性矛盾。

### 第二轮：训练栈 × 环境，安装层（`run-install.sh`，Linux）

| 案例 | 结果 |
|---|---|
| solo-training | INSTALLED (101s, 92 包) |
| solo-envclient | INSTALLED (3s, 30 包) |
| solo-env-d4rl | INSTALLED (20s, 69 包) |
| **union-d4rl × reinflow** | **INSTALLED (144 包) ✗** |

关键否证：union 一次装成 144 个包，同时含

```
torch==2.7.1  d4rl==1.1  mujoco-py==2.1.2.14  cython==0.29.37
gymnasium==1.3.0  numpy==2.3.2  reinflow==1.0.0
```

**现代训练栈与带 `mujoco-py` + `cython<3` 的老仿真栈确实能装在一起。**

robomimic 一组前两次因 harness 缺陷失败，不构成证据；第三次修正后闭合：

1. `CMake must be installed` —— 缺系统工具（harness 错）
2. 装了 cmake 4.4.3 后 `egl-probe` 构建失败（harness 错）
3. 补上仓库自带的
   `[tool.uv] extra-build-variables = { "egl-probe" = { CMAKE_POLICY_VERSION_MINIMUM = "3.5" } }`
   后 —— **两组全部 INSTALLED**

| 案例 | 结果 |
|---|---|
| solo-env-robomimic-fixed | INSTALLED (135s, 111 包) |
| **union-robomimic × dppo-fixed** | **INSTALLED (150 包) ✗** |

union 同时装入
`torch==2.7.1  robomimic==0.3.0  robosuite==1.4.1  mujoco-py==2.1.2.14
cython==0.29.37  d4rl==1.1  dppo==0.8.0  egl-probe==1.0.2`。

**安装层最终 0/2 支持 C1。**

### 第三轮：环境 × 环境，元数据层（`run-env-matrix.sh`）

这才是 benchmark 论文的真实处境 —— 评测矩阵要同时立起多个环境家族。

| 案例 | 结果 |
|---|---|
| 5 个家族各自单独 | 全部 RESOLVED |
| 5 组跨家族两两配对 | 全部 RESOLVED |
| **五家族一次全上** | **RESOLVED ✗** |

0/6 支持。这条路也不通。

### 前三轮总账

| 提问方式 | 支持 C1 |
|---|---|
| 训练栈 × 环境（元数据） | 1/3 |
| 训练栈 × 环境（安装，Linux） | **0/2** |
| 环境 × 环境（元数据） | 0/6 |

## 第四轮：部署足迹（`run-footprint.sh`，Linux 实测）

C1 死了之后，唯一还站得住的差别不是「能不能装」，而是「每台产生 rollout 的机器
要背多少东西、需不需要 GPU」。这决定实验室工作站、CI runner、机器人本体计算机
能不能当 rollout 源。实测（uv 0.9.9，WSL2 Ubuntu 22.04）：

| 案例 | 角色 | 包数 | 磁盘 | CUDA |
|---|---|---|---|---|
| envclient-bare | rollout | 30 | **224M** | 否 |
| envclient-d4rl | rollout | 69 | **1008M** | 否 |
| envclient-robomimic | rollout | 111 | **7.2G** | **是（16 wheel）** |
| monolith-d4rl | 两者 | 142 | 6.5G | 是 |
| monolith-robomimic | 两者 | 150 | 7.4G | 是 |
| training-only | trainer | 111 | 5.6G | 是 |

### 结论：足迹优势**取决于环境包本身**

| 环境 | env client | 单体 | 优势 |
|---|---|---|---|
| dummy / classic / atari | 224M，免 GPU | 6.5G，需 GPU | **29×，且免 GPU** ✓ |
| d4rl | 1008M，免 GPU | 6.5G，需 GPU | **6.4×，且免 GPU** ✓ |
| **robomimic** | **7.2G，需 GPU** | 7.4G，需 GPU | **≈0** ✗ |

`envclient-robomimic` 自己拖进了 `torch==2.14.0` 与 16 个 nvidia wheel ——
robomimic 本身就是深度学习库，自带策略学习代码。对这类环境，
「环境侧无需 GPU」的说法不成立。

**因此「env client 无 torch」这句只对 plugrl-env-client 的直接依赖成立，
不能推广到所有环境家族。** 论文若用足迹论证，必须按环境分别陈述，
并明确点出 robomimic 这类反例 —— 审稿人会自己发现。

## 定位重估：三个选项

### A. 转向「部署足迹」（数据现成，但有反例）

主张改为：*单体框架要求每台产生 rollout 的机器都托管完整训练栈；
PlugRL 对纯仿真环境只要求 224M–1G 且不需要 GPU。*

- 优点：第四轮数据直接支持，最好的一档是 29× 且免 GPU；与 E6、E7 天然衔接
- 缺点：**robomimic 是反例**（7.2G 且需 GPU，与单体几乎无差别）；
  且这更像工程贡献而非研究贡献，顶会说服力存疑

### B. 保留互操作性框架，换支点

不谈「装不上」，谈**跨语言与跨网络**：线格式与语言无关，真机控制器可用
C++/ROS 实现客户端 —— 这是 Ray placement group 类设计做不到的。
需要 E7 真机或至少一个非 Python 客户端的演示来支撑。

### C. 回到吞吐路线（`features/robocasa` 改变了前提，但只到单机）

该分支的 `ray_inference.py` 已实现多 GPU 分片推理 + 粘性路由 + 每 worker
独立模型副本 + 学习后同步权重，**学习不阻塞推理**。此前判断「打不过」是基于
main 的 WebSocket 路径，对这条路径不成立。

**但它目前只支持单机多卡**：`cli_ray.py:100-107` 用
`available_gpus = torch.cuda.device_count()`（本地设备数）作为上限，
请求超过本地 GPU 数即 `raise ValueError`。即使 `ray.init()` 连上了多节点集群，
推理 worker 数仍被本地卡数卡死。要做多机扩展性曲线，这里必须先改。

需要：合并分支 + 解除单机上限 + 多机实测 + 与 RLinf/RL-VLA³ 正面对比。
作者是 `nothingbutbut`，合并需其参与。

## 建议

**在选定支点之前，不要按现有 C1 推进 C2/C3。** 本轮实验已经完成了它的使命 ——
在投入四个月实验之前否掉了一个错误前提，代价是一天。

## 复现

```bash
bash run.sh                                    # 第一轮（元数据）
wsl bash wsl-bootstrap.sh                      # 第二轮（Linux 安装层）
bash run-env-matrix.sh                         # 第三轮（环境×环境）
wsl bash wsl-footprint.sh                      # 第四轮（部署足迹）
```

日志分别在 `results/`、`results-install/`、`results-matrix/`、`results-footprint/`。
依赖 SHA 全部钉死在各脚本顶部（2026-09-09 解析）。
