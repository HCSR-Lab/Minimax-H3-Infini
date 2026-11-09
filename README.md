# H3-BSA: Bridge-Synchronized Alignment for MiniMax-H3

**基于跨框架张量交接同步的 MiniMax-H3 生成一致性修复**

## 摘要

MiniMax-H3 的跨框架推理涉及 PyTorch 与 InfiniCore 之间的 GPU 张量交接。共享张量指针并不建立异步计算之间的执行依赖，可能使消费端读取尚未完成写入的数据。本文采用设备同步约束张量交接顺序，并保留 InfiniLM 条件编码器与 H3 Transformer 的原生执行入口。该方法命名为 **Bridge-Synchronized Alignment for MiniMax-H3（H3-BSA）**。在既有八卡海光 DCU、10 步 FL2VA 实验中，H3-BSA 相对 B-legacy 和 B-pytorch 的平均视频 SSIM 分别为 95.154% 和 93.089%。这些结果用于评估生成一致性；本文尚未建立同条件性能加速结论。

## 1. H3-BSA 的编译与使用

目标环境：八卡海光 DCU（gfx936，64 GiB/卡），Python 3.11、厂商 PyTorch 2.9.0+das.opt1.dtk2604、DTK/HIP、xmake。以下命令在仓库根目录执行。

**构建所需源码已直接放在 `sources/`，无需解压或恢复备份。** 其中 InfiniLM、InfiniCore、ninetoothed 来自服务器源码导出，B-pytorch 使用的完整固定版本 Diffusers 源码位于 `sources/diffusers/`。设置环境后即可按下文编译。

服务器导出的 11,424 个文件已完成校验，当前目录仅保留构建源码和相关证据；安装态源码副本、源码压缩包、模板备份与评分临时文件已移出提交目录。源码逐文件记录见 [server-source-files.json](manifests/server-source-files.json)，导出摘要见 [server-source-snapshot.json](manifests/server-source-snapshot.json)。历史记录中的安装态文件路径用于追溯，不代表这些副本仍需随源码交付。

**当前源码是 2026-11-07 服务器当前源码，包含 H3-BSA 实验之后的修改；下文 SSIM 和耗时是历史 H3-BSA 结果，尚未用此源码重新验收。** [sources.json](manifests/sources.json) 记录原始上游版本，不能代替本次服务器修改后的源码。

**当前状态：编译步骤根据官方构建文件与历史服务器命令整理，尚未在新的干净环境完整验收。完整依赖安装包和工具链获取流程仍需补齐。**

### 1.1 环境与依赖

先激活已准备好厂商 Torch、Diffusers、Transformers 等依赖的独立环境。环境版本见 [环境清单](manifests/vendor-environment.json)；[服务器包记录](manifests/server_installed.freeze.txt)仅供核对，不能直接替代完整安装锁。

```bash
# 改成实际路径；权重和图片使用赛事提供的文件。
export PYTHON=/absolute/path/to/infini-venv/bin/python
export MODEL_PATH=/absolute/path/to/MiniMax-H3/diffusers
export FIRST_FRAME=/absolute/path/to/Golde33443.jpg

# 本地 sources/ 已就绪，不需要重新下载源码。
export PROJECT_ROOT="$PWD"
export INFINILM_SOURCE="$PROJECT_ROOT/sources/InfiniLM"
export INFINI_ROOT="$PROJECT_ROOT/install"
export PATH="$(dirname "$PYTHON"):$PATH"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

从本次服务器包恢复后，不要再次应用 Flash 补丁。若要另外构建原始上游版本，请在新的仓库目录执行 `python3 scripts/prepare_sources.py`；它下载固定版本及子模块并应用 Flash 补丁，不安装 Python 依赖。

准备厂商 Flash Attention `2.6.1+das.opt1.dtk2604.torch290` 的 **cp311** wheel，校验并解压后设置：

```bash
export FLASH_ROOT=/absolute/path/to/unpacked-wheel
export FLASH_ATTN_2_CUDA_SO="$FLASH_ROOT/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so"
export FA_SO_PATH="$FLASH_ATTN_2_CUDA_SO"
export LD_LIBRARY_PATH="$FLASH_ROOT:$FLASH_ROOT/flash_attn/lib:$LD_LIBRARY_PATH"
```

wheel 获取索引和 SHA256 见[环境清单](manifests/vendor-environment.json)。当前链接配置仅在独立 `libflash_attention.so` 存在时链接它，cp311 厂商 wheel 使用实际的 `flash_attn_2_cuda…so`。**编译和运行时均需保持上述变量有效，并保留解压目录；编译不会将厂商 Flash 库打包进程序。**

### 1.2 编译

```bash
# 根据节点 CPU 和内存调整并发数。
export BUILD_JOBS=14
export XMAKE_GLOBALDIR="$PROJECT_ROOT/build/xmake-global"
export XMAKE_PKG_INSTALLDIR="$PROJECT_ROOT/build/xmake-packages"
export XMAKE_PKG_CACHEDIR="$PROJECT_ROOT/build/xmake-cache"
mkdir -p "$INFINI_ROOT" "$XMAKE_GLOBALDIR" "$XMAKE_PKG_INSTALLDIR" "$XMAKE_PKG_CACHEDIR"

cd "$PROJECT_ROOT/sources/InfiniCore"
xmake f -y -cv -m release --hygon-dcu=y --hygon-arch=gfx936 \
  --cuda=/opt/dtk/cuda/cuda --ccl=y --aten=y --flash-attn="$FLASH_ROOT"
xmake build -j "$BUILD_JOBS"
xmake install
xmake build -j "$BUILD_JOBS" _infinicore
xmake install _infinicore

cd "$PROJECT_ROOT/sources/InfiniLM"
xmake f -y -cv -m release
xmake build -j "$BUILD_JOBS" _infinilm
xmake install _infinilm

# 将本次构建的 Python 包安装到上面指定的独立环境。
# 官方 setup.py 会调用 xmake；安装前保持上述环境变量有效。
"$PYTHON" -m pip install --no-deps --no-build-isolation "$PROJECT_ROOT/sources/InfiniCore"
"$PYTHON" -m pip install --no-deps --no-build-isolation "$PROJECT_ROOT/sources/InfiniLM"
cd "$PROJECT_ROOT"
```

`/opt/dtk/cuda/cuda` 为历史目标环境路径，其他机器应改为实际 DTK 路径。构建使用独立安装前缀，不向共享环境安装。依赖解析、子模块完整性和最终运行库仍需在干净机器验收，详细环境与验收状态见本文说明。

### 1.3 使用

```bash
# 执行本文的 H3-BSA 实现：InfiniLM 官方非 Legacy 入口，加张量交接同步。
bash scripts/run.sh infini runs/h3-bsa

# 已生成 B-legacy 后，运行官方 FL2VA 对比脚本。
bash scripts/compare.sh runs/legacy/generated.mp4 \
  runs/h3-bsa/generated.mp4 runs/compare-legacy
```

固定参数：`A dog falls asleep`、seed 42、10 steps、576×384、124 帧、24 fps。图片 SHA256 在 [case.json](manifests/case.json)，运行前自动检查。结果保存在指定目录的 `generated.mp4`、`run.log`、`environment.json` 和 `video.sha256`；每次使用新的输出目录。

## 2. 对照实现：B-pytorch 与 B-legacy

| 参考路径 | 来源 | 运行方式 |
|---|---|---|
| B-pytorch | 固定 Diffusers 官方 MiniMax-H3 FL2VA 示例 | 官方 pipeline，配置 group offload |
| B-legacy | 固定 InfiniLM 官方 `examples/minimax_h3.py` | 使用 `--legacy` |

### 2.1 B-pytorch：构建与使用

B-pytorch 不需要编译 InfiniCore、InfiniLM。使用独立的厂商 PyTorch 环境，并从固定源码安装 Diffusers；Transformers 等依赖需按目标环境准备。

```bash
export PYTHON=/absolute/path/to/pytorch-venv/bin/python
# 完整固定版本源码已随本目录保存在 sources/diffusers/。
"$PYTHON" -m pip install --no-deps --no-build-isolation ./sources/diffusers
bash scripts/run.sh pytorch runs/pytorch
```

这里执行的是[官方示例的配置版本](references/pytorch_fl2va_group_offload.py)：设置本地权重、赛事图片、prompt 和固定生成参数，采用官方 group offload API。原始示例与[配置差异](patches/pytorch-example-configuration.patch)均保留，不另写采样和生成算法。

### 2.2 B-legacy：构建与使用

复用上面的 Infini 环境及固定 InfiniLM 源码，不需要再编译一套。切回 Infini 环境解释器：

```bash
export PYTHON=/absolute/path/to/infini-venv/bin/python
bash scripts/run.sh legacy runs/legacy
```

脚本调用官方入口，明确添加 `--legacy`，条件编码器使用 0–3 卡，DiT 使用 4–7 卡。三条路径顺序运行，使用相同的权重、图片和生成参数。

生成两个参考视频后可分别比较：

```bash
bash scripts/compare.sh runs/legacy/generated.mp4 \
  runs/h3-bsa/generated.mp4 runs/compare-legacy
bash scripts/compare.sh runs/pytorch/generated.mp4 \
  runs/h3-bsa/generated.mp4 runs/compare-pytorch
```

## 3. 方法：跨框架张量交接同步

### 3.1 问题：共享内存地址与计算完成是两个条件

MiniMax-H3 的生成链路由输入处理、条件编码、迭代去噪和视频/音频解码组成。PyTorch 与 InfiniCore 之间需要交接设备张量，因此除了数值运算本身，还必须保证生产端写入与消费端读取的顺序。

源码中的 [`infinicore.from_torch`](sources/InfiniCore/python/infinicore/tensor.py) 通过 `data_ptr()` 取得 PyTorch 张量地址，再调用 `_infinicore.from_blob` 创建共享底层存储的张量，并保存原张量引用。该过程避免了数据复制，保存引用也有助于维持存储生命周期；但这一层 Python 包装没有显式等待 PyTorch 生产端的异步计算完成。主机端已取得指针，不意味着设备端已经完成对该地址的写入。

若生产端计算与消费端读取之间缺少执行依赖，消费端就可能读到尚未完成更新的数据。对迭代去噪而言，早期条件或中间状态的偏差还可能随后续迭代传播。因此，输出差异可能远大于通常的浮点舍入误差。这里描述的是代码与异步执行机制支持的风险；尚未通过逐张量事件追踪确定每一次实际发生竞争的读写。

### 3.2 方法：在张量交接边界建立完成关系

H3-BSA 将同步放在公共交接入口，而不是分别修改每个模型层。对每个进入 `from_torch` 的 GPU 张量，先等待其所在设备上的已有工作完成，再调用原始转换函数。核心顺序为：

```text
PyTorch 提交生产端计算
        ↓
torch.cuda.synchronize(tensor.device)
        ↓
原始 infinicore.from_torch(tensor)
        ↓
InfiniCore 后续计算消费该张量
```

实现位于 [`scripts/run_h3_bsa.py`](scripts/run_h3_bsa.py)。包装器先保存原始 `infinicore.from_torch`，再替换包级入口与 `infinicore.tensor.from_torch`，覆盖这两种模块访问路径。CPU 张量直接交给原始函数；GPU 张量执行同步后再转换。随后通过 `runpy.run_path` 调用官方 `examples/minimax_h3.py`，由官方非 Legacy 分支继续完成条件编码、Transformer 去噪和解码。

同步对象由 `tensor.device` 决定，不固定为某一张卡。这里采用设备级等待，覆盖该设备已提交的工作；不是只等待当前流，也不是所有设备之间的一次全局屏障。该实现针对 Torch → InfiniCore 的交接，并不声称覆盖任意反向交接、提前绑定的函数别名或所有跨设备依赖。

### 3.3 为什么较小的改动能够显著改善 SSIM

H3-BSA 复用既有模型权重、条件编码器、Transformer、采样流程和解码器。高 SSIM 所依赖的模型计算来自这些已有实现；同步修复的作用，是让跨框架交接满足数据已经就绪这一前提。若主要偏差来自错误的读写顺序，那么修复关键边界即可显著改变最终输出，无需以大量新增模型代码换取一致性。该解释不意味着“同步本身提高模型生成能力”，也不意味着更复杂输入下必然得到相同分数。

在固定输入、seed、步数与分辨率的历史前后对照中，未加入交接同步的实现与 H3-BSA 得到以下结果。下表为原始评分数据的汇总，完整控制台输出见第 4.3 节。

| 比较对象 | 未加入交接同步的实现 | H3-BSA |
|---|---:|---:|
| 相对 B-pytorch 的平均视频 SSIM | 18.013% | 93.089% |
| 相对 B-legacy 的平均视频 SSIM | 18.009% | 95.154% |

历史运行日志记录了 **126 次 GPU 张量交接，其中 20 次在检查时发现当前 PyTorch 流尚未完成**。这说明交接发生时确实存在未完成的异步工作，与执行依赖缺失的解释相符。但当前流未完成不等于该张量必然尚未就绪，也不能把这 20 次直接认定为 20 次错误读取。结合评分变化，现有证据支持交接同步是该用例中的有效修复；完整因果定位仍需要事件追踪与更严格的受控实验。原始证据见[同步前评分](results/before-bridge-sync-original.json)和[运行日志](results/h3-bsa-generation-original.log)。

### 3.4 探索过程：精度调整并未带来一致的收益

H3-BSA 的确定经历了运行依赖修复、交接同步验证和数值路径探测，并非直接选定一次修改就完成。首先解决厂商 Flash 库布局导致的链接与加载问题，使原生链路能够运行；随后用交接同步恢复生成一致性；在此基础上，继续探测 VAE 数据类型、注意力精度、AdaLN 运算顺序及其组合是否能够进一步改善结果。库能够加载、算子能够执行与整条生成链路数值一致，是不同层次的验收目标。

以下为固定 FL2VA 用例下已有候选的历史结果。后续数值候选均保留交接同步；表中的低分表示未优于本用例的 H3-BSA 视频结果，不意味着这些精度策略在所有输入上无效。原始评分及失败记录保存在 [探索实验记录目录](results/exploration/)。

| 探测方案 | 相对 B-pytorch 的视频 SSIM | 相对 B-legacy 的视频 SSIM | 相对 B-legacy 的音频 PCM 余弦 |
|---|---:|---:|---:|
| 未加入交接同步 | 18.013% | 18.009% | 0.0118 |
| **H3-BSA：交接同步** | **93.089%** | **95.154%** | **0.8834** |
| 交接同步 + 保留 VAE FP32 参数 | 88.724% | 87.078% | 0.8881 |
| 交接同步 + DiT MHA FP32 | 89.327% | 91.383% | 0.7506 |
| 交接同步 + AdaLN 运算顺序对齐 | 89.015% | 89.679% | 0.9498 |
| 交接同步 + 完整精度组合 | 88.252% | 90.281% | 0.7403 |
| 交接同步 + 音频输出 AdaLN 定向调整 | 未完成有效评分 | 未完成有效评分 | 未完成有效评分 |

**VAE 参数精度探测。** 该候选保留视频和音频 VAE 的 FP32 参数，视频 SSIM 却下降。检查实现后发现，它没有关闭解码外层的 FP16 autocast，还改变了首帧编码路径。因此，“参数保持 FP32”并不等同于“完整 FP32 解码”，不能据此认定真实 FP32 解码无效，也不能把该候选视为只改变末端解码精度的严格单因素实验。它提示后续实验必须同时审计参数、输入、autocast 和编码/解码路径。

**注意力计算精度探测。** 将 DiT MHA 的计算调整为 FP32 后，两条视频对照均未超过 H3-BSA，音频余弦也下降。提升局部精度并不自动保证最终输出更接近参考：参考路径本身具有特定的算子实现、归约顺序与舍入位置，改变其中一处会改变迭代过程。现有结果支持“局部升精度没有在本用例带来端到端收益”，尚不足以定位是哪一次归约或舍入导致降分。

**AdaLN 运算顺序探测。** 候选将相关计算调整为参考形式 `x * (1 + scale) + shift`。实数意义下相近的表达式，在有限精度下可能因运算顺序与舍入位置不同而产生差异。该候选音频余弦提高至 0.9498，但两条视频 SSIM 均下降，且音频仍未达到 0.95 的项目目标。这说明视频与音频指标可能出现权衡，不能仅因一个指标改善就替换整个生成方案。

**完整精度组合探测。** 在局部调整之外，还迁移了 DiT MHA FP32、Qwen 静态因果注意力 FP32、AdaLN 顺序对齐和关闭 autocast 的真实 FP32 视频解码组合，保留原始 BF16 首帧编码与音频 VAE。该候选经过编译问题修复后完成生成与评分，但最终视频、音频指标仍未优于 H3-BSA。此前类似策略在 256×256、50 步及不同提示词条件下的成绩，不能直接迁移到本轮 576×384、10 步用例；本次迁移也不是旧环境与全部底层实现的逐项等价复现。

**音频定向探测与未完成实验。** 为减少全局 AdaLN 调整对视频的影响，进一步尝试仅调整音频输出 AdaLN。现有运行先遇到归档权限问题，修正后又以退出码 134 异常终止，未形成可用于验收的视频和音频评分。因此该方案应记为“未完成验证”，不能描述为数值效果较差或已经解决音频问题。

上述探索表明，交接顺序与局部数值精度需要分开诊断。交接同步在当前用例中恢复了主要生成一致性；后续精度修改则未呈现单调收益。保留 H3-BSA 的依据，是已有候选中它的视频 SSIM 对两条参考均最高，而不是证明其他方案普遍无效或所有误差已经消除。后续仍需对复杂图片、提示词、步数与分辨率开展受控泛化实验，并联合评估视频、音频与性能。

### 3.5 一致性的含义、代价与边界

官方比较函数先解码视频，再对最长边不超过 512 像素的灰度帧计算 SSIM，最后取帧平均。93%–95% 的 SSIM 表示该指标下的结构相似程度，不表示逐像素相同，也不代表音频一致。B-pytorch 与 B-legacy 之间的历史平均 SSIM 本身为 94.231%；因此，参考路径之间也存在差异，不能将任何一条参考路径默认视作另一条路径的逐位等价实现。

设备同步会限制跨流重叠，可能增加等待。当前方案优先验证交接正确性，尚无同条件端到端加速结论。后续可在明确生产流与消费流的前提下，研究事件依赖、局部同步及其性能模型，以减少不必要的等待。

当前海光 MHA 内部含 ATen 运算，视频和音频 VAE 使用官方 Diffusers/PyTorch。H3-BSA 不读取参考视频来生成结果，不替换参考条件或 latent，也不修改权重。现有结果尚不能据此证明全部算子满足赛事要求或已经解决全部数值差异。

## 4. 实验设置与结果

### 4.1 实验设置

任务为首帧条件视频与音频生成（FL2VA），三条路径使用相同权重、输入图像、提示词与随机种子。生成分辨率为 576×384，共 124 帧、24 fps，采样步数为 10，seed 为 42。B-pytorch 使用官方 Diffusers pipeline 与 group offload；B-legacy 使用 InfiniLM 的 Legacy 分支；H3-BSA 使用 H3-BSA 同步入口。完整参数见 [case.json](manifests/case.json)。

### 4.2 实测耗时

以下为历史服务器一次 H3-BSA运行的阶段耗时，配置为八卡、576×384、124 帧、10 steps；尚无重复运行统计。

```text
模型加载                            59.35 秒
条件编码器计算                       8.87 秒
H3 Transformer 计算                 15.80 秒（9 次调用）
视频和音频解码                       8.46 秒
生成流水线总计                      36.49 秒
端到端总计                          97.86 秒
```

模型加载为四个创建/加载阶段之和；解码为视频与音频解码之和。生成流水线包含上述计算、解码及其他处理，不能重复相加；端到端包含加载、生成和媒体封装，不包含编译、依赖下载及作业启动。原始阶段值及日志哈希见[耗时记录](results/h3-bsa-timing.json)。

### 4.3 生成一致性

以下为本次直接运行未修改的 `examples/minimax_h3_compare.py`、比较已保存视频所得到的**完整控制台输出**。两次运行均正常退出，仅进行视频比较，没有重新生成视频。代码将参考端标签固定打印为 `PyTorch`，所以 B-legacy 对照中的标签也保持原样。

**B-pytorch 对照**（[原始日志](results/h3-bsa-vs-pytorch-compare.log)）：

```text

[task] reference_fl2va: using existing result: /home/jizhenwei/Downloads/competition/minimax-h3-submission/private/comparison-inputs/pytorch.mp4

[task] infinilm_fl2va: using existing result: /home/jizhenwei/Downloads/competition/minimax-h3-submission/private/comparison-inputs/h3-bsa.mp4

[compare] FL2VA: decoding outputs

[result] MiniMax-H3 video comparison

  FL2VA
    Size:         PASS (PyTorch 576x384, InfiniLM 576x384)
    Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)
    Average SSIM: 93.089%
```

**B-legacy 对照**（[原始日志](results/h3-bsa-vs-legacy-compare.log)）：

```text

[task] reference_fl2va: using existing result: /home/jizhenwei/Downloads/competition/minimax-h3-submission/private/comparison-inputs/legacy.mp4

[task] infinilm_fl2va: using existing result: /home/jizhenwei/Downloads/competition/minimax-h3-submission/private/comparison-inputs/h3-bsa.mp4

[compare] FL2VA: decoding outputs

[result] MiniMax-H3 video comparison

  FL2VA
    Size:         PASS (PyTorch 576x384, InfiniLM 576x384)
    Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)
    Average SSIM: 95.154%
```

当前只报告这个 FL2VA 用例，REF2VA 尚无对应验收结果。两种参考结果本身也有差异，不能混用。音频对 B-legacy 的原始 PCM 余弦为 **0.8834**，尚未达到项目约定的 0.95 目标。结果与视频哈希见[结果清单](results/historical_summary.json)。

## 5. 完成情况

### 5.1 已实现功能

1. **跨框架张量交接同步与生成一致性修复。** 实现 PyTorch → InfiniCore 张量交接前的设备同步，保留 InfiniLM 条件编码器与 H3 Transformer 原生推理入口；在既有 FL2VA 用例中，相对 B-pytorch 和 B-legacy 的平均视频 SSIM 分别达到 93.089% 和 95.154%。
2. **海光环境下的 Flash Attention 链接与加载适配。** 修复厂商 cp311 wheel 与独立 Flash 库布局的兼容问题，支持按实际扩展库路径链接和显式加载，为原生推理链路提供依赖适配。
3. **生成链路与对照评估流程。** 提供 FL2VA 视频与音频生成、B-pytorch 和 B-legacy 对照入口，以及官方视频 SSIM 比较和音频 PCM 一致性评估脚本，保留完整输出与实验记录。
4. **源码交付与复现材料整理。** 完成服务器源码归档、本地源码恢复与逐文件校验，将源码、构建说明、实验结果和目录导航集中到本项目目录。

### 5.2 继续改进事项

1. **H3-BSA 的泛化性研究。** 面向更复杂的输入图像与提示词，系统考察不同采样步数和生成分辨率下的生成一致性与稳定性，评估方法在不同任务条件下的适用范围。
2. **异构环境下的性能上限探测与性能建模。** 面向单卡及多卡异构环境，研究计算、访存、跨设备通信与同步开销，探测理论性能上限并建立性能模型，为后续算子优化与编译优化提供依据。

### 5.3 适用范围与复现状态

H3-BSA 修正跨框架异步交接顺序，现有实验支持特定 FL2VA 用例下的生成一致性结论。当前服务器源码包含该实验之后的精度修改，不能仅凭本次源码同步将历史成绩视为当前源码的复测结果。干净环境构建、当前源码的完整生成复测、REF2VA 以及音频目标验收仍需完成。第三方源码和厂商依赖的使用边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 6. 源码组织与文件说明

项目根目录为 `/home/jizhenwei/Downloads/competition/minimax-h3-submission`。以下展开项目维护的目录与文件；第三方源码仅展开主要子目录，其完整内容保留在相应目录中。

```text
minimax-h3-submission/
├── README.md                         # 方法、构建、使用、实验结果与文件导航
├── THIRD_PARTY_NOTICES.md             # 第三方来源、许可证与分发范围
├── .gitignore                        # Git 忽略规则；不等于交付文件清单
├── sources/                          # 实际用于构建的完整源码
│   ├── InfiniLM/
│   │   ├── csrc/                     # 原生推理引擎和模型 C++ 实现
│   │   ├── python/infinilm/          # Python 包与原生模块接口
│   │   ├── examples/                 # 官方入口，包括 minimax_h3.py
│   │   ├── third_party/              # 随源码保留的第三方依赖
│   │   └── xmake.lua                 # InfiniLM 构建定义
│   ├── InfiniCore/
│   │   ├── src/                      # 算子、运行时、通信与框架适配实现
│   │   ├── include/                  # C/C++ 头文件
│   │   ├── python/                   # Python 绑定及包代码
│   │   ├── xmake/                    # 平台构建配置，含海光 Flash 链接逻辑
│   │   ├── submodules/               # 源码树保留的子模块目录
│   │   ├── third_party/              # 第三方依赖
│   │   └── xmake.lua                 # InfiniCore 构建定义
│   ├── ninetoothed/
│   │   ├── src/                      # 九齿算子生成工具源码
│   │   ├── examples/                 # 算子示例
│   │   └── tests/                    # 上游测试
│   └── diffusers/
│       ├── src/diffusers/            # B-pytorch pipeline、模型及 VAE
│       ├── examples/                 # 上游示例
│       ├── docs/                     # 上游文档
│       └── tests/                    # 上游测试
├── scripts/                          # 本项目的准备、运行与评估入口
│   ├── prepare_sources.py            # 可选：联网获取固定上游源码
│   ├── check_environment.py          # 检查输入图像、模型路径和八卡环境
│   ├── run.sh                        # 统一运行 infini / pytorch / legacy
│   ├── run_h3_bsa.py               # H3-BSA 的 H3-BSA 张量交接同步入口
│   ├── compare.sh                    # 调用官方视频比较脚本
│   └── compare_audio.py              # 音频 PCM 对齐条件下的一致性评估
├── references/                       # B-pytorch 参考入口
│   ├── upstream_fl2va_example.py      # 固定版本的官方 FL2VA 示例
│   └── pytorch_fl2va_group_offload.py # 本实验配置与 group offload 入口
├── patches/                          # 可审阅的修改差异
│   ├── infinicore-flash-layout.patch # 厂商 Flash 库布局兼容补丁
│   ├── pytorch-example-configuration.patch # 官方示例与实验配置的差异
│   └── h3-bsa-launcher-portability.patch # H3-BSA 入口的路径适配差异
├── manifests/                        # 输入、版本、环境及校验记录，详见下表
├── results/                          # 历史实验的结构化结果
│   ├── exploration/                  # 其他候选方案的原始评分、失败记录及说明
│   ├── h3-bsa-timing.json             # 阶段耗时与日志来源
│   ├── h3-bsa-vs-pytorch-compare.log   # 官方脚本相对 PyTorch 的完整控制台输出
│   ├── h3-bsa-vs-legacy-compare.log    # 官方脚本相对 Legacy 的完整控制台输出
│   ├── comparison-run.json           # 本次比较的命令、环境和输入哈希
│   ├── before-bridge-sync-original.json # 同步前的原始评分
│   ├── h3-bsa-generation-original.log # 历史生成和同步诊断原始日志
│   ├── h3-bsa-video-comparison-original.json # 历史比较结果 JSON
│   ├── official-video-comparison-original.json # 官方函数完整评分报告
│   ├── historical_summary.json        # 历史 SSIM、媒体信息和产物哈希
│   └── h3-bsa-audio-vs-legacy.json        # H3-BSA 相对 B-legacy 的音频指标
├── licenses/                         # 第三方许可证副本
│   ├── InfiniCore-LICENSE
│   └── diffusers-LICENSE
```

`sources/` 是编译入口；`scripts/` 是运行和评估入口；`results/` 保留指标与方法分析所依据的原始记录。权重、赛事图像、厂商 Flash wheel 和 DTK 工具链需要另外准备。

原始比较日志中的 `private/comparison-inputs/` 是当次评分使用的历史路径。日志内容保持原样；该临时目录现已移出提交目录，重新生成与评分请使用第 1、2 节的 `runs/` 路径。

### 6.1 说明文件的作用

| 文件 | 内容与用途 |
|---|---|
| [README.md](README.md) | 项目主入口；先完成第 1 节构建，再运行第 2 节对照实验，按第 4 节理解指标。 |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 说明上游来源、许可证与本地材料的分发范围。 |
| 各 `sources/*/README.md` | 各上游项目自身的构建、接口与使用说明；本实验配置以本项目 README 为入口。 |

### 6.2 清单文件的作用

| `manifests/` 中的文件 | 含义 |
|---|---|
| `case.json` | 输入图片 SHA256、prompt、seed、分辨率、帧数与步数。 |
| `sources.json` | 原始上游仓库地址与固定 commit；不描述服务器工作区的全部后续修改。 |
| `server-source-files.json` | 服务器导出时的逐文件 SHA256 与原始路径；包含已移出提交目录的安装态副本记录。 |
| `server-source-snapshot.json` | 本次服务器源码归档哈希、导出时间、文件数量及排除项。 |
| `local-source-provenance.json` | 本地补入的完整 Diffusers 源码来源与目录内容哈希。 |
| `vendor-environment.json` | 厂商 Python、Torch、Flash 版本与 wheel 校验信息。 |
| `server_installed.freeze.txt` | 服务器已安装 Python 包记录，不是完整的依赖安装锁。 |
| `server_build_manifest.json` | 历史构建环境记录，不代表当前快照已经重建通过。 |
| `server_crosscheck.json` | 先前材料与服务器文件的交叉核对记录。 |
| `flash-layout-hashes.json` | Flash 布局修复相关文件的历史哈希。 |
| `pytorch-example-source.json` | 官方 PyTorch 示例的来源与版本定位。 |
| `file-sha256.json` | 本项目说明、脚本与结果文件的当前内容校验；第三方源码另见源码清单。 |

### 6.3 运行生成的目录

构建后，`build/` 保存 xmake 配置与缓存，`install/` 保存私有安装产物。推理后，`runs/` 保存视频、日志、环境记录和比较结果；这些目录不属于原始源码。`.git/` 为版本管理元数据，`__pycache__/` 为 Python 自动生成缓存。

本地 `sources/` 被 Git 忽略。交付给测试人员时应复制整个 `minimax-h3-submission/` 目录；仅执行 `git clone` 不会自动带上这些本地源码。备份、源码压缩包和临时材料不属于当前交付内容。
