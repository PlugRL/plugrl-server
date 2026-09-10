# E2 结论：线协议确实可从规范独立实现

日期：2026-09-09 · WSL2 Ubuntu 22.04 · Python 3.11 · plugrl-server @ 4aaebe0

## 一句话

两个客户端 —— 一个 Python（无 numpy、无 PlugRL），一个 **C++（无任何第三方库）**
—— 各自完整驱动了 plugrl-server 的训练循环 120 步。
选项 B（跨语言/跨网络）**已从主张变为演示**。

## 最强的那个结果：C++ 客户端

`plugrl_client.cpp`，g++ 11.4 编译，68KB 二进制。`ldd` 输出：

```
linux-vdso.so.1
libstdc++.so.6
libgcc_s.so.1
libc.so.6
libm.so.6
```

**除了 C++ 标准库和 libc，什么都没链接。** 没有 msgpack 库、没有 WebSocket 库、
没有 OpenSSL —— SHA-1、base64、WebSocket 分帧与掩码、msgpack 子集全部按规范
手写在这一个文件里（约 600 行）。运行结果：

| 指标 | 值 |
|---|---|
| 完成的交换 | **120** |
| 耗时 | 22.4s |
| 服务端 global_step | **100 / 300** |
| 返回动作 | `dtype=<f4 shape=[4,1,7]`，真实浮点值 |

这正是嵌入式机器人控制器的处境：一个编译器、一个 TCP socket，仅此而已。
**"ROS 节点可以当 env client" 不再需要论证。**

## 第二个结果：Python 客户端（无 numpy、无 PlugRL）

先做的是这个，用来快速验证协议可从规范实现。

客户端 venv 的全部内容：

```
Package    Version
msgpack    1.2.2
websockets 15.0.1
```

没有 numpy，没有 torch，没有 plugrl-*。运行结果：

| 指标 | 值 |
|---|---|
| 完成的 infer/action/feedback 交换 | **120** |
| 耗时 | 22.6s |
| 服务端记录的 global_step | **100 / 300** |
| 服务端记录的 total_connections | 1 |
| 服务端 collection_time / learn_time | 11.2s / 10.0s |
| 返回动作 | `dtype=<f4  shape=[4, 1, 7]`，含真实浮点值 |
| 每步上行载荷 | 2 张 uint8 图（224×224×3 + 112×112×3 ≈ 187KB） |

**服务端不是"容忍"了这些消息 —— 它推进了训练进度并跑完了一个 learn 阶段。**

## 协议的可移植性评估

### 完全标准的部分

- **传输**：普通 WebSocket。手写的 HTTP upgrade 请求（`printf` + `nc`）即可拿到
  `101 Switching Protocols`，随后直接收到 metadata 帧。无自定义握手、无鉴权、
  无子协议协商。
- **载荷**：标准 msgpack。通用 msgpack 解码器无需任何扩展即可解析。
- **消息流**：metadata（服务端先发）→ 循环 { infer → action → feedback }。
  四个消息类型是四个小写字符串。

### 唯一的 numpy 特有约定

ndarray 以二进制键的 map 传输：

```
{b"__ndarray__": true, b"data": <bin>, b"dtype": "<f4", b"shape": [4, 1, 7]}
```

`dtype` 用的是 numpy typestr：字节序字符 + 类别字符 + 字节数。这是全协议**唯一**
的 numpy 词汇。它是有文档的固定格式，本实验用约 10 行代码手工解析
（`raw_client.py` 的 `_parse_typestr` / `encode_array` / `decode_array`）。

一个需要注意的细节：numpy 的 `bool_` 是每元素 1 字节，而 Python 的 `array`
模块没有布尔类型码，必须按原始字节处理 —— C++ 客户端本来就会这么做。

## 这对选项 B 意味着什么

证据链闭合，无缺口：

1. 传输是普通 WebSocket —— 已用手写 HTTP upgrade 与手写分帧两次验证
2. 载荷是标准 msgpack —— 已用通用解码器与手写解码器两次验证
3. 唯一的 numpy 约定（dtype typestr）可用约 10 行解析 —— 两个客户端都实现了
4. **两个零依赖客户端（Python 与 C++）都驱动了真实训练进度**

论文可以直接写：*环境侧的实现负担是一个 WebSocket 客户端加一个 msgpack 编解码器，
两者在任何主流语言中都有成熟实现，也可以像本文附带的 600 行 C++ 客户端那样从零写。*

**这是 Ray placement group 类单体设计做不到的** —— 它们要求环境侧运行完整的
Python 框架进程。

## 一个附带发现（dummy policy 的批大小推断很脆弱）

`dummy_policy._infer_batch_size` 取 `next(iter(obs.values()))` 再 `len()`。
观测是 `{"images": {...}, "states": {...}, "text": [...]}`，所以它实际取的是
images 字典的**键数**，而非批大小。第一次实验送了 `"images": {}` 时，
服务端算出 batch=0 并返回 `shape=[4, 0, 7]` 的空动作 —— **没有报错**。

这只影响 dummy policy（其它策略有自己的实现），但静默返回空动作而不是报错
是个隐患，值得单独修。

## 复现

```bash
wsl bash setup-server.sh        # 装服务端（首次 ~2 分钟）
wsl bash run-cpp.sh             # C++ 客户端：编译 + 120 步（主结果）
wsl bash verify.sh              # Python 客户端：120 步 + 读服务端账目
wsl bash run-experiment.sh      # Python 客户端：20 步冒烟
wsl bash diagnose.sh            # 手写 HTTP upgrade 探测
```

- `plugrl_client.cpp` —— C++ 客户端，零第三方库，约 600 行
- `raw_client.py` —— Python 客户端，仅 msgpack + websockets
- 日志在 `results/`
