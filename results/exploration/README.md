# 探索实验原始记录

本目录保留原始文件内容，不改写内部实验编号和媒体文件名。项目方法名称统一为 H3-BSA。

| 文件 | 作用 |
|---|---|
| attention-and-vae-video-original.json | 同步前、交接同步、VAE 参数精度及 DiT 注意力精度候选的官方视频评分。 |
| adaln-video-original.json | AdaLN 运算顺序候选的官方视频评分。 |
| combined-precision-video-original.json | 完整精度组合成功生成后的官方视频评分。 |
| audio-candidates-original.json | 各候选相对 Legacy 的原始 PCM 余弦及 RMSE。 |
| audio-adaln-archive-failure-original.json | 音频定向候选的归档权限失败记录。 |
| audio-adaln-runtime-failure-original.json | 音频定向候选后续退出码 134 的运行失败记录，无有效评分。 |

README 的候选方案表是这些记录的汇总。局部调整均保留交接同步；VAE 候选并非严格单因素完整 FP32 解码实验。失败记录不用于评价生成质量。
