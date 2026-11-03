# DSV4 算子下沉：CANN 9.1 / 9.2 输入差异清单

核对日期：2026-11-03。

目标：由 sgl-kernel-npu 接收稳定的输入语义，在 kernel 层适配 CANN 9.1 / 9.2，使 SGLang 不需要判断 CANN 版本或拼装底层 ABI。

本清单覆盖 Kernel #939 当前计划下沉的 **13 个公共 API**，重点记录参数顺序、必需/可选、默认值、shape、dtype、layout、缓存编码和硬件支持域。本文不修改代码，不将静态核对描述为 NPU 执行验证。

## 1. 基线与阅读约定

| 对象 | 固定基线 |
| --- | --- |
| Kernel #939 | `b4f7723cfebcf881631098f317eea19ac9f4290b`；公共入口已按 attention / kvcacheio / norm / rope / moe 分类 |
| CANN ops-transformer 9.1.0 | `6f0d3ec620b338cab4f2d00322ff2e0d141419d2` |
| CANN ops-transformer 9.2.0 | `7bc7c0412cafb501b4b7bcfb3dbef90985016308` |

**必须区分两类比较：**

- **正式接口版本比较**：9.1 和 9.2 都有对应 Torch 扩展时，直接比较 schema 和约束。
- **旧接口迁移比较**：9.1 目标环境只有 experimental / recipe / sgl-kernel-npu 绑定时，与 9.2 正式接口比较；必须写明旧入口来源，不能把它概括成“CANN 9.1 官方签名”。

“schema 相同”仅表示本次提取的 Torch 签名一致，不代表对应 CANN 安装包已包含内核，也不代表各硬件上的输入值域完全相同。“未找到正式入口”表示在本次核对的 9.1 目录和 Torch 扩展清单中未找到，不能据此推断用户部署环境中没有另外安装的 recipe/custom 包。

9.1 源码已经有 mixed-quant MLA、sparse MLA、quant indexer、mHC 的正式扩展原型。不要把 original/compressed/residual lengths 等参数全部描述成 9.2 新增。

## 2. 13 个 API 的总表

| # | 当前公共 API 所属类别 / 名称 | 9.1 输入基线 | 9.2 正式输入差异 | kernel 层适配重点 | 证据状态 |
| --- | --- | --- | --- | --- | --- |
| 1 | attention / sparse_flash_mla_metadata | 已有正式扩展 | Torch schema 相同 | 固定默认值、输入设备契约；旧 sharedkv metadata 单列适配 | schema 已确认一致 |
| 2 | attention / sparse_flash_mla | 已有正式扩展 | Torch schema 相同 | 稳定长度、mask、layout、索引语义 | schema 已确认一致 |
| 3 | attention / mixed_quant_sparse_flash_mla_metadata | 已有正式扩展 | Torch schema 相同 | quant_mode 与实际 KV 格式配对；硬件能力检查 | schema 已确认一致，支持域需逐环境验证 |
| 4 | attention / mixed_quant_sparse_flash_mla | quant_mode 为 keyword-only，默认 None；cmp_ratio=None | quant_mode 移至 q 后且必传；cmp_ratio=1 | 公共 API 显式定义参数，底层优先用关键字；区分不同 KV 编码 | 签名差异已确认 |
| 5 | attention / quant_lightning_indexer_metadata | 已有正式扩展 | Torch schema 相同 | 区分原始长度和压缩长度；规范量化模式 | schema 已确认一致 |
| 6 | attention / quant_lightning_indexer | 已有正式扩展 | Torch schema 相同 | query/key/scale/topk 的统一契约；旧实验接口单列映射 | schema 已确认一致 |
| 7 | kvcacheio / kv_compress_epilog | 未找到同名正式扩展；当前旧路径是 sgl-kernel-npu 的 npu 绑定 | cache/x/slot_mapping + 字符串 quant_mode / round_scale / x_scale | 缓存字节格式、枚举、skip sentinel、默认值必须一起适配 | 9.2 签名及旧 native 绑定已确认；9.1 部署包待核对 |
| 8 | kvcacheio / indexer_quant_cache | 未找到同名正式扩展；旧调用为 recipe indexer_compress_epilog | cache/cache_scale/x/slot_mapping，增加量化属性 | 参数名、scale dtype/shape、量化模式 | 9.2 已确认；旧 recipe 完整 schema 待取证 |
| 9 | norm / mhc_pre → mhc_pre_sinkhorn | 已有正式扩展；9.1 文档不支持 A5 | Torch schema 相同；9.2 增加 A5 的 3D/4D 支持域 | 统一外部 TND 输入；按代际适配 rank/维度约束 | schema 和文档差异已确认 |
| 10 | norm / mhc_post | hRes 必须为 Tensor | hRes 变为 Tensor?；仅 A5 支持 None | 可空能力检查；公共 comb 与底层 hRes 的对应 | schema 和硬件约束差异已确认 |
| 11 | rope / inplace_partial_rotary_mul | 未找到同名正式扩展；旧调用为 recipe custom 绑定 | x/r1/r2，rotary_mode="interleave"，partial_slice=[0,0] | 模式枚举、切片默认值、四维输入和广播 | 9.2 已确认；旧 recipe 完整 schema 待取证 |
| 12 | attention / compressor | experimental/native 融合接口包含 norm_weight、rope_sin/cos 等 | 正式 compressor 不再接收这些后处理输入；cmp_ratio 为第 6 个必需参数 | 先确定公共语义，再组合核心压缩 + RMSNorm/RoPE；不能直接替换绑定 | 明确存在计算边界及输入集合差异 |
| 13 | moe / moe_gating_top_k | CANN V1 文档不含 hash 两个输入；normType=2 支持域较窄 | V2 增加 inputIds/tid2eid；9.2 更新 normType=2 支持说明 | torch_npu schema 能力检查；hash 与 scoring 语义不能仅改参数名 | CANN 文档差异已确认；torch_npu 版本绑定待核对 |

## 3. Attention / metadata 的公共输入集合

下面是 #1～#4 共有的正式接口输入语义，**9.1 扩展中已经存在**：

| 输入组 | 参数 | 统一含义 |
| --- | --- | --- |
| Query | q（计算算子） | TND 为 `[Tq,Nq,D]`，BSND 为 `[B,Sq,Nq,D]`；metadata 另传 num_heads_q / num_heads_kv / head_dim |
| KV | ori_kv / cmp_kv（计算算子） | 原始与压缩 KV；是否量化及具体字段编码由格式契约决定 |
| 分页 | ori_block_table / cmp_block_table（计算算子） | 每请求的分页映射；PA_BBND 的缓存形状为 `[pages,page_size,Nkv,packed_dim]` |
| 累计边界 | cu_seqlens_q / cu_seqlens_ori_kv / cu_seqlens_cmp_kv | 各请求的前缀和；不要与每请求有效长度混用 |
| 有效长度 | seqused_q / seqused_ori_kv / seqused_cmp_kv | 各请求实际参与计算的 token 数 |
| 压缩尾部 | cmp_residual_kv | 与 seqused_cmp_kv、cmp_ratio 一起表达压缩前长度 |
| 显式稀疏 | ori_sparse_indices / cmp_sparse_indices（计算算子） | 稀疏位置列表；逻辑位置或物理 offset 由已支持的 topk_value_mode 决定 |
| 稀疏有效数量 | ori_topk_length / cmp_topk_length | 每行实际参与的索引数量，区别于索引张量的 padded 宽度 |
| Mask | ori_mask_mode / cmp_mask_mode / ori_win_left / ori_win_right | 分别表达原始与压缩 KV 的可见范围 |
| Layout | layout_q / layout_kv | 明确传递；不能仅把旧 PA_ND 字符串改成 PA_BBND 就认为缓存兼容 |

对 9.1 只有旧 sharedkv 接口的部署环境，可能需要把共享原始长度 L 转成 `seqused_ori_kv=L`、`seqused_cmp_kv=L//r`、`cmp_residual_kv=L%r`。这是**旧接口到正式语义的适配**，不是所有 9.1→9.2 正式调用都需要执行的转换。不要对已经传入压缩长度的参数再除一次 r。

### 3.1 #1 sparse_flash_mla_metadata

- 两版必需标量均为：`num_heads_q, num_heads_kv, head_dim`。
- `*` 后的可选 Tensor、batch/max length、topk、mask、layout、has_ori_kv/has_cmp_kv 的 schema 本次比较相同。
- 多个标量在 Torch schema 中写成 `None`，实际 metadata Python 包装会规范化，例如 batch/max/topk 为 0、cmp_ratio 为 1、mask 为 0、窗口为 -1、layout 为 BSND、has_* 为 True。公共 API 不宜依赖这种隐式处理。
- 旧 DSpark host metadata 使用 CPU 长度镜像；正式 metadata 的输入/输出设备应以实际 backend 明确约定。不能根据同名参数就默认 host 和 NPU Tensor 可互换，也不要为 host 路径在 graph 中临时 `.cpu()`。

### 3.2 #2 sparse_flash_mla

- 两版正式 schema 相同：`q` 为位置输入，其余为 keyword-only。
- 两版共同默认：`softmax_scale=1.0`、`cmp_ratio=1`、`ori_mask_mode=0`、`cmp_mask_mode=0`、窗口 `-1/-1`、layout `BSND/BSND`、`topk_value_mode=1`。
- SGLang 的普通滑窗语义应显式传 `ori_mask_mode=4` 与窗口；不要误以为正式算子默认就是滑窗。
- DSpark 需要显式原始稀疏列表与有效数量。旧实现是否支持这些输入需按对应 backend 验证；9.1 旧量化实验文档中的 ori_sparse_indices 曾标为预留，不可据此承诺 DSpark 等价。

来源：[9.1 sparse MLA 扩展](https://gitcode.com/cann/ops-transformer/blob/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/torch_extension/cann_ops_transformer/ops/sparse_flash_mla.py)、[9.2 sparse MLA 扩展](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/attention/sparse_flash_mla/torch_extension/sparse_flash_mla.py)。

### 3.3 #3 mixed_quant_sparse_flash_mla_metadata

- 两版必需标量均为：`num_heads_q, num_heads_kv, head_dim, quant_mode`；其余 schema 一致。
- 量化模式、rope_head_dim、layout 与 reader 的实际缓存格式必须一致；不能仅探测到函数存在就使用任意模式。
- 9.2 文档的 A5 路径为 quant_mode 1/2；A2/A3 的 TQ4 是 quant_mode=3 的另一种输入组合，不能套用 A5 mode 1 的缓存描述。
- metadata 是后续计算算子的输入：9.1/9.2 本次版本均生成 1024 个 int32；当前 master 已是 2048。大小相同也不证明内容兼容，必须由配套 backend 生成。这个差异不应通过 SGLang 手工 pad/trim 屏蔽。

### 3.4 #4 mixed_quant_sparse_flash_mla

| 输入差异 | 9.1 正式扩展 | 9.2 正式扩展 | 适配要求 |
| --- | --- | --- | --- |
| quant_mode 的位置 | `q, *, ..., quant_mode=None` | `q, quant_mode, *, ...` | 公共 API 将其定义为显式必需参数；向下用关键字传递可以兼容两种 schema |
| cmp_ratio 默认值 | None | 1 | kernel 层固定默认 1 或要求显式提供；不要向旧 C++ 整型参数透传 None |
| rope_head_dim / softmax_scale | schema 默认 None | schema 仍默认 None | DSV4 路径明确传 64 与模型实际 scale；不要假设 9.2 已补齐全部默认值 |
| 长度与 sparse 参数 | 已有 original/compressed/residual、topk lengths | 同组参数存在 | 本组不是新增输入集合；需要核对的是运行支持域 |
| key_dtype / value_dtype | 已有可选参数 | 仍保留 | 没有证据时不要为包装自行发明默认 dtype 或推断所有 dtype 可用 |

KV 物理编码必须单列：

| 所属路径 | reader 参数 / 输入缓存格式 | 说明 |
| --- | --- | --- |
| 9.1 旧 experimental KvQuantSparseAttnSharedkv | kv_quant_mode=1、tile_size=64、PA_ND；文档为 640B 行，RoPE BF16 + NoPE FP8 + 7 个 E8M0 scale 后填充 | 这是旧实验接口，不是所有 9.1 正式 mixed_quant 的统一格式 |
| 9.2 mixed_quant，mode 1 | 608B：64 个 BF16 RoPE + 448 个 FP8 NoPE + 7 个 BF16 scale + padding | #939 当前描述符采用此格式 |
| 9.2 mixed_quant，mode 2 | 584B，E8M0 scale；PA 页内数据区与 scale 区有专门组织约定 | 不能仅改变 row_alignment 或 mode 数字复用 mode 1 缓存 |
| 9.2 mixed_quant，mode 3，A2/A3 | q 为 FP16/BF16；ori_kv 为相同 dtype 的非量化 512 维；cmp_kv 为 UINT8、258B TQ4；TND/PA_BBND | 支持域与 mode 1/2 不同；文档中禁传部分 seqused/sparse 辅助输入，不能通用地原样透传全部 kwargs |

来源：[9.1 mixed-quant 扩展](https://gitcode.com/cann/ops-transformer/blob/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/torch_extension/cann_ops_transformer/ops/mixed_quant_sparse_flash_mla.py)、[9.1 旧实验格式](https://gitcode.com/cann/ops-transformer/blob/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/kv_quant_sparse_attn_sharedkv/README.md)、[9.2 mixed-quant 扩展](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/attention/mixed_quant_sparse_flash_mla/torch_extension/mixed_quant_sparse_flash_mla.py)、[9.2 输入约束](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/attention/mixed_quant_sparse_flash_mla/README.md)。

## 4. Indexer 两个输入接口

### 4.1 #5 quant_lightning_indexer_metadata

两版正式 schema 相同：

```python
quant_lightning_indexer_metadata(
    num_heads_q, num_heads_k, head_dim, topk, quant_mode,
    *, cu_seqlens_q=None, cu_seqlens_k=None,
    seqused_q=None, seqused_k=None, cmp_residual_k=None,
    batch_size=None, max_seqlen_q=None, max_seqlen_k=None,
    layout_q=None, layout_k=None, mask_mode=None, cmp_ratio=None,
)
```

9.2 源码位于 quant_lightning_indexer_v2 目录，但 Python 名称仍是 quant_lightning_indexer_metadata。不能因为目录有 v2 就认为 9.1 正式 schema 缺少这组输入。

### 4.2 #6 quant_lightning_indexer

两版共同的位置输入：

```python
query, key, weights, query_dequant_scale, key_dequant_scale, topk, quant_mode
```

共同 keyword-only 输入：cu_seqlens_q/k、seqused_q/k、cmp_residual_k、block_table、output_idx_offset、metadata；默认 max_seqlen_q=-1、layout_q/k="BSND"、mask_mode=0、cmp_ratio=1、return_value=0。本次 schema 比较一致。

若对接旧实验 `npu_quant_lightning_indexer`，需在 kernel 层处理以下迁移：

| 旧调用语义 | 正式接口语义 |
| --- | --- |
| actual_seq_lengths_query / actual_seq_lengths_key | 明确区分累计 query 边界 cu_seqlens_q 与压缩 key 有效长度 seqused_k |
| 单个原始 KV 长度 L | 对 C4 路径得到 seqused_k=L//4、cmp_residual_k=L%4；若已是压缩长度则不能再除 |
| layout_query / layout_key，例如 PA_BSND | layout_q / layout_k，例如 PA_BBND；缓存实际组织需配套验证 |
| sparse_count / sparse_mode | topk / mask_mode，不能忽略值域与 mask 语义核对 |
| query_quant_mode / key_quant_mode | 正式 quant_mode 编码；按实际量化 dtype 与 scale 格式映射，不能只按名字替换 |

来源：[9.1 正式 indexer 扩展](https://gitcode.com/cann/ops-transformer/blob/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/torch_extension/cann_ops_transformer/ops/quant_lightning_indexer.py)、[9.2 正式 indexer 扩展](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/attention/quant_lightning_indexer_v2/torch_extension/quant_lightning_indexer.py)。

## 5. Cache writer 两个接口

### 5.1 #7 kv_compress_epilog

9.1 中未找到同名正式 Torch 扩展。当前旧路径能核实的是 **sgl-kernel-npu 自带的 torch.ops.npu 绑定**，不能把它称为 9.1 官方扩展：

```python
# 旧 native 绑定
kv_compress_epilog(
    kv_compress_cache, x, slot_mapping,
    quant_group_size, quant_mode, round_scale_flag, layout,
)

# 9.2 正式扩展
kv_compress_epilog(
    cache, x, slot_mapping, *,
    quant_group_size=64, quant_mode="fp8_e8m0",
    round_scale=True, x_scale=1.0,
)
```

| 差异 | 旧 native / 旧调用 | 9.2 正式接口 | 适配要求 |
| --- | --- | --- | --- |
| cache 输入名与存储视图 | kv_compress_cache；旧调用常 reshape 后传 FP8 存储 | cache；#939 传原缓存的 UINT8 字节视图 | 不转换数据，只在物理编码一致时做 view；不能用 view 修复编码差异 |
| quant_mode | 整数枚举 | 对外字符串 fp8_bf16 / fp8_e8m0 / hifloat8_fp4 | 按量化语义建立映射，不能复用相同整数猜测含义 |
| round_scale_flag | 旧参数名 | round_scale=True | 在 kernel 层重命名并固定默认 |
| layout | 必传旧属性 | 正式 Torch schema 不再接收 | 由目标格式描述与适配器决定，不能继续裸透传 |
| x_scale | 旧 schema 无此输入 | 新增，默认 1.0；正式文档中主要用于 HiFloat8 静态路径 | 不支持该能力的旧后端应校验，不能静默丢弃有意义的非默认值 |
| slot_mapping | 需按旧实现与框架实际约定核对 | -1 跳过，0 是有效槽位；支持 INT32/INT64 | sentinel 要与分配、padding、图缓冲区同时统一 |

**当前 #939 的默认值风险**：公共入口未显式指定模式时，生产实现沿用 CANN 默认 fp8_e8m0，参考实现默认 fp8_bf16；当前 608B 格式应显式使用 fp8_bf16。此处需在公共输入契约统一，不能只依赖 SGLang 某一个调用点传对。

来源：[旧 native schema](https://github.com/sgl-project/sgl-kernel-npu/blob/b4f7723cfebcf881631098f317eea19ac9f4290b/csrc/pytorch_extensions.cpp)、[9.2 Torch schema](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/attention/kv_compress_epilog/torch_extension/kv_compress_epilog.py)、[9.2 writer 约束](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/attention/kv_compress_epilog/README.md)。

### 5.2 #8 indexer_quant_cache

9.1 中未找到同名正式扩展。旧 recipe 调用点提供 `indexer_compress_cache, indexer_compress_scale, x, slot_mapping`；这是调用点证据，**尚未取得旧 recipe 安装包的完整 schema**，不能据此确认其他参数均不存在。

9.2 正式签名：

```python
indexer_quant_cache(
    cache, cache_scale, x, slot_mapping, *,
    quant_mode="fp8", round_scale=True, x_scale=1.0,
)
```

- 参数名由 indexer_compress_cache / indexer_compress_scale 统一到 cache / cache_scale。
- x 为 `[tokens,D]` 的 FP16/BF16；slot_mapping 为 `[tokens]` INT32；-1 跳过。
- fp8 为逐 token 动态量化，scale 存储为 FLOAT32；mxfp8 / mxfp4 是另一组 scale 和分组契约，不能沿用同一 scale buffer。
- round_scale 与 x_scale 有模式相关语义。9.1 后端不具备这些功能时，需要拒绝不支持的模式或在 kernel 层实现等价路径。

来源：[9.2 Torch schema](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/attention/indexer_quant_cache/torch_extension/indexer_quant_cache.py)、[9.2 dtype / scale 约束](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/attention/indexer_quant_cache/README.md)、[旧调用迁移 diff](https://github.com/sgl-project/sglang/pull/47438/files)。

## 6. mHC 两个接口

### 6.1 #9 mhc_pre → mhc_pre_sinkhorn

两版正式 Torch schema 相同，9 个输入均无默认值：

```python
mhc_pre_sinkhorn(x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, outFlag)
```

公共输入名的映射也可共用：fn→phi、scale→alpha、base→bias、hc_mult→hcMult、sinkhorn_iters→numIters、hc_eps→hcEps、rms_norm_eps→normEps；推理路径传 outFlag=False。

| 输入支持域 | 9.1 文档 | 9.2 文档 | 适配要求 |
| --- | --- | --- | --- |
| A5 | 该版本 README 标记不支持 | 支持 | 不能用“函数已注册”替代硬件能力判断 |
| A2/A3 的 x | 旧扩展按四维索引构造 meta；文档要求末维 128 对齐 | 明确为四维 BSND；c 128 对齐且在给定范围 | 外部统一 TND `[T,n,c]`，必要时 view 为 `[T,1,n,c]` |
| A5 的 x | 无正式支持承诺 | 可三维 TND 或四维 BSND；c 仅 4096/7168 | 同一个公共 rank 不代表允许任意 hidden size |
| hcMult / numIters | 文档仅支持 4 / 20 | 仍仅支持 4 / 20 | 在 kernel 层统一校验，不能将参考实现支持域当生产支持域 |
| 参数值域 | 按旧支持域 | A5 另有限定 phi 范围及 eps 稳定性说明 | 版本/代际相关限制应在能力描述中体现 |

来源：[9.1 schema](https://gitcode.com/cann/ops-transformer/blob/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/torch_extension/cann_ops_transformer/ops/mhc_pre_sinkhorn.py)、[9.1 支持域](https://gitcode.com/cann/ops-transformer/blob/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/mhc/mhc_pre_sinkhorn/README.md)、[9.2 schema](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/mhc/mhc_pre_sinkhorn/torch_extension/mhc_pre_sinkhorn.py)、[9.2 支持域](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/mhc/mhc_pre_sinkhorn/README.md)。

### 6.2 #10 mhc_post

```python
# 9.1
mhc_post(Tensor x, Tensor hRes, Tensor hOut, Tensor hPost)
# 9.2
mhc_post(Tensor x, Tensor? hRes, Tensor hOut, Tensor hPost)
```

- 明确变化：hRes 从必需 Tensor 变为可空 Tensor。**可空不等于可以省略这个位置参数**；正式 schema 仍需提供 hRes 或 None。
- 9.2 仅 A5 允许 hRes=None，此时退化为直接残差连接；A2/A3 仍要求 hRes。
- 公共接口 `mhc_post(x, residual, post, comb)` 对应底层 `(residual, comb, x, post)`；这个顺序转换在两版都需要，不是新版本才新增的变化。
- 最小公共支持域可以要求 comb 非空；如果公共 API 要支持 None，需给 9.1 实现等价语义并验证，或明确拒绝该组合。

来源：[9.1 schema](https://gitcode.com/cann/ops-transformer/blob/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/torch_extension/cann_ops_transformer/ops/mhc_post.py)、[9.2 schema](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/mhc/mhc_post/torch_extension/mhc_post.py)、[9.2 可空限制](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/mhc/mhc_post/README.md)。

## 7. #11 inplace_partial_rotary_mul

9.1 未找到同名正式 Torch 扩展；此前 SGLang 调用 recipe 的 `torch.ops.custom.inplace_partial_rotary_mul`。旧调用具有 x/cos/sin 及 rotary_mode、partial_slice，但旧安装包完整 schema 和枚举定义仍需取证。

9.2 正式签名：

```python
inplace_partial_rotary_mul(
    x, r1, r2, *, rotary_mode="interleave", partial_slice=[0, 0],
)
# r1=cos，r2=sin；x 原地修改
```

| 输入项 | 9.2 要求 | kernel 层应屏蔽的细节 |
| --- | --- | --- |
| rotary_mode | Torch 层字符串 interleave；底层文档对应整数 1 | recipe 若使用不同枚举，应按数学模式映射，而非强制沿用整数 |
| partial_slice | `[start,end)`；默认 `[0,0]` 表示零长度旋转、直接返回 | DSV4 要旋转整段时必须明确 `[0,rope_dim]`，不能把默认值理解为全量旋转 |
| x | 四维 BSND、连续；FP16/BF16/FP32 | 公共 TND 输入由 kernel 包装 view，保持原地别名语义 |
| cos/sin | 同样 dtype 支持，尾维等于 slice 长度，并满足代际相关广播约束 | 统一输入 rank/广播；不能无条件支持任意 expanded 非连续视图 |

来源：[9.2 schema](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/posembedding/inplace_partial_rotary_mul/torch_extension/inplace_partial_rotary_mul.py)、[9.2 shape / 广播约束](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/posembedding/inplace_partial_rotary_mul/README.md)。

## 8. #12 compressor：输入集合和计算边界都不同

### 8.1 9.1 experimental / 当前 native 融合接口

9.1 experimental 文档描述的 compressor 包含压缩后的 RMSNorm 与 RoPE。当前 sgl-kernel-npu 的 npu 绑定为：

```python
compressor(
    x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
    state_block_table=None, cu_seqlens=None, seqused=None, start_pos=None,
    rope_head_dim=64, cmp_ratio=4, coff=1, norm_eps=1e-6,
    rotary_mode=1, cache_mode=1, state_cache_stride_dim0=0,
)
```

最后的 stride 参数来自当前 sgl-kernel-npu 绑定，不能推定 9.1 experimental 安装包也具备。9.1 文档还限定 A3 不支持 cache_mode=2。

### 8.2 9.2 正式 compressor

```python
compressor(
    x, wkv, wgate, state_cache, ape, cmp_ratio, *,
    state_block_table=None, cu_seqlens=None, seqused=None, start_pos=None,
    coff=1, cache_mode=1,
)
```

| 输入项 | 旧融合路径 | 9.2 正式路径 | 适配影响 |
| --- | --- | --- | --- |
| 第 6 个位置参数 | norm_weight | cmp_ratio，必传 | 不能直接换函数名沿用位置参数列表 |
| norm_weight / norm_eps | 接收，用于压缩后 RMSNorm | 不接收 | 公共 API 若保留融合语义，9.2 backend 需组合对应后处理 |
| rope_sin / rope_cos / rope_head_dim / rotary_mode | 接收，用于压缩后 RoPE | 不接收 | 不能简单删除 kwargs；否则计算语义改变 |
| state_cache_stride_dim0 | 当前 native 扩展有 | 正式 Torch schema 无 | cache view / stride 契约需单独处理，不能裸透传 |
| state_cache | FP32 `[blocks,block_size,2*coff*D]` | 同类状态输入 | 相同 shape 不自动保证 cache_mode 与 block table 语义一致 |
| state_block_table | 旧文档有 0 跳过及 A3 cycle 限制 | cache_mode=1 为二维页表；=2 为每请求一维 bank；cycle 下 0 有效 | zero sentinel、cycle 和 explicit 模式要分别校验 |

**当前 #939 尚未切换到 9.2 的这个正式入口**：其生产实现仍调用 `torch.ops.npu.compressor`，输入含 norm_weight。实现名虽然叫 cann_ops_transformer，实际绑定仍是 native npu；双版本适配时必须明确选择的是“融合 compressor”还是“核心 compressor + 后处理”。

来源：[9.1 experimental 输入](https://gitcode.com/cann/ops-transformer/blob/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/experimental/attention/compressor/README.md)、[当前 native schema](https://github.com/sgl-project/sgl-kernel-npu/blob/b4f7723cfebcf881631098f317eea19ac9f4290b/csrc/pytorch_extensions.cpp#L120)、[9.2 正式 schema](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/attention/compressor/torch_extension/compressor.py)、[9.2 状态约束](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/attention/compressor/README.md)、[#939 当前实现](https://github.com/sgl-project/sgl-kernel-npu/blob/b4f7723cfebcf881631098f317eea19ac9f4290b/python/sgl_kernel_npu/sgl_kernel_npu/operators/attention/compressor.py)。

## 9. #13 moe_gating_top_k

公共入口使用 torch_npu backend，不能只对比 cann_ops_transformer 包就得出 Python 输入兼容性结论。CANN 版本和 torch_npu/op-plugin 的具体构建组合都需要记录。

| 输入 / 语义 | 9.1 CANN 文档 | 9.2 CANN 文档 | kernel 层要求 |
| --- | --- | --- | --- |
| 基础输入 | x、bias、k、kGroup、groupCount、groupSelectMode、renorm、normType、outFlag、routedScalingFactor、eps | 基础集合保留 | 统一公共 snake_case 参数，并核对安装的 torch.ops.npu schema |
| Hash 输入 | 本次读取的 V1 文档未列 inputIds/tid2eid；未找到同路径 V2 文档 | V2 增加 inputIdsOptional、tid2eidOptional | 两者配套、按能力选择；不能向旧 schema 无条件传这两个 kwargs |
| inputIds | V1 未列 | `[tokens]`，INT32/INT64；值范围对应 tid2eid 行数 | 校验 shape 与值域 |
| tid2eid | V1 未列 | `[numKeys,k]`，INT32/INT64 | 与 k、输入索引组合一致；hash 模式另有 k≤64 和分组限制 |
| normType=2 | 文档写 Softplus，且仅 A5 支持 | 写 SqrtSoftplus；A2/A3 要求 groupCount=1 | 文档存在数学表述差异，尚未通过旧 kernel 数值验证；不能直接断言“实际计算从 Softplus 改成 SqrtSoftplus” |
| bias 与权重来源 | 有 correction bias 输入 | bias 用于选择，权重来自未加 bias 的归一化得分 | 公共参考与生产 backend 应核对相同数学语义 |

当前 #939 在 input_ids/tid2eid 均为空时不向 torch_npu 传这两个 kwargs，有利于保留基础 TopK 调用；但非空时仍需做实际 schema/能力判断。不能将“基础模式可用”推广为“9.1 支持全部 hash 与 normType=2 组合”。

来源：[9.1 V1 文档](https://gitcode.com/cann/ops-transformer/blob/6f0d3ec620b338cab4f2d00322ff2e0d141419d2/moe/moe_gating_top_k/docs/aclnnMoeGatingTopK.md)、[9.2 V2 文档](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/moe/moe_gating_top_k/docs/aclnnMoeGatingTopKV2.md)、[9.2 模式约束](https://gitcode.com/cann/ops-transformer/blob/7bc7c0412cafb501b4b7bcfb3dbef90985016308/moe/moe_gating_top_k/README.md)、[#939 torch_npu 适配器](https://github.com/sgl-project/sgl-kernel-npu/blob/b4f7723cfebcf881631098f317eea19ac9f4290b/python/sgl_kernel_npu/sgl_kernel_npu/implementation/torch_npu/moe/moe_gating_top_k.py)。

## 10. 本清单对应的适配任务

| 优先级 | 任务 | 涉及 API |
| --- | --- | --- |
| P0 | 明确 compressor 公共计算边界，禁止把带 norm/rope 的旧列表直接传给正式核心算子 | #12 |
| P0 | 定义配套缓存格式描述：字段顺序、row/page bytes、scale dtype、reader/writer 模式、skip sentinel | #3/#4/#7/#8 |
| P0 | 固定 kv writer 的公共默认量化模式，确保所有实现一致 | #7 |
| P1 | 显式声明 mixed_quant 的必需参数和默认值，按名称适配两个 schema | #4 |
| P1 | 处理 hRes 可空、mHC rank 与 A5 支持域 | #9/#10 |
| P1 | 单独处理 MoE hash/V2 与 scoring 支持域，不按 CANN 版本号猜 torch_npu schema | #13 |
| P1 | 为旧 host metadata、旧稀疏索引和正式 metadata 建立输入设备与配套 backend 契约 | #1～#6 |
| P2 | 取得目标 9.1 机器 recipe 的完整 schema，补全 indexer_compress_epilog 与 partial RoPE 的严格逐参数映射 | #8/#11 |

### 仍需从目标环境补齐的证据

1. CANN、torch_npu、cann_ops_transformer、recipe/custom、sgl-kernel-npu 的实际包版本与构建来源。
2. 9.1 recipe 两个算子的完整 schema，以及 MoE 绑定实际支持的参数。
3. 9.1 正式扩展是否具备配套的 aclnn 符号与 kernel，尤其 A5 mixed-quant / mHC。
4. CPU host metadata 与 NPU metadata 的输入 device 要求；各量化模式的读写格式及图模式行为。

在上述证据齐备前，“未确认”不能改写成“无差异”或“已兼容”。本文已确认的 schema 相同项无需人为拆成两份实现；实际能力不同的部分应在 kernel 初始化/准备阶段选择，而不是让 SGLang 判断版本。
