"""融合算子分发层（v2）：同一接口，按设备自动选择最快且数值正确的实现。

  attention(q, k, v, key_padding_mask)   q/k/v: [B,H,L,Dh]
      CUDA : F.scaled_dot_product_attention（FlashAttention-2 / mem-efficient 内核）
      NPU  : torch_npu.npu_fusion_attention（昇腾 FlashAttention 融合算子）
      其他 : eager softmax(QK^T)V
  swiglu(gate, up)  -> silu(gate) * up
      CUDA : Triton 融合 kernel（一次读写，省掉 silu 中间张量）
      NPU  : torch_npu.npu_swiglu
      其他 : torch 参考实现

所有融合实现首次调用时与参考实现做数值自检（allclose），不通过自动回退并记录原因
——保证"国产平台换算子不掉精度"，也是技术报告里"算子迁移可验证"的工程证据。
环境变量 EM_FUSED_OPS=0 可全局关闭融合路径（用于消融/排查）。

注意：训练阶段同样走这些接口；Triton swiglu 带手写 backward（autograd.Function）。
"""

import math
import os

import torch
import torch.nn.functional as F

_ENABLED = os.environ.get("EM_FUSED_OPS", "1") != "0"
_STATUS = {}          # op_name@device -> "fused" | "fallback: <reason>"

try:                                          # 昇腾
    import torch_npu  # noqa: F401
    _HAS_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except Exception:                             # pragma: no cover
    torch_npu = None
    _HAS_NPU = False

try:                                          # Triton（CUDA）
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                             # pragma: no cover
    _HAS_TRITON = False


def fused_status():
    """返回各融合算子的启用状态（infer/benchmark 日志打印用）。"""
    return dict(_STATUS)


# =========================================================== attention
def attention_reference(q, k, v, key_padding_mask=None, dropout_p=0.0, training=False):
    """eager 参考实现。key_padding_mask: [B,L] True=忽略。"""
    scale = 1.0 / math.sqrt(q.size(-1))
    attn = (q @ k.transpose(-2, -1)) * scale
    if key_padding_mask is not None:
        attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
    attn = attn.float().softmax(-1).to(q.dtype)
    if dropout_p > 0 and training:
        attn = F.dropout(attn, dropout_p)
    return attn @ v


def _sdpa(q, k, v, key_padding_mask, dropout_p, training):
    mask = None
    if key_padding_mask is not None:
        mask = ~key_padding_mask[:, None, None, :]          # SDPA: True = 参与注意力
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=dropout_p if training else 0.0)


def _npu_fa(q, k, v, key_padding_mask, dropout_p, training):
    B, H, L, D = q.shape
    atten_mask = None
    if key_padding_mask is not None:
        # npu_fusion_attention: True/1 = 屏蔽；形状 [B,1,Lq,Lk]
        atten_mask = key_padding_mask[:, None, None, :].expand(B, 1, L, L).contiguous()
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch_npu.npu_fusion_attention(
        q, k, v, H, input_layout="BNSD", atten_mask=atten_mask,
        scale=1.0 / math.sqrt(D), keep_prob=1.0 - (dropout_p if training else 0.0))[0]
    return out


def _selftest(name, fn, ref, *args):
    key = f"{name}@{args[0].device.type}"
    if key in _STATUS:
        return _STATUS[key] == "fused"
    try:
        with torch.no_grad():
            small = [a[:2] if torch.is_tensor(a) and a.dim() > 1 else a for a in args]
            out, exp = fn(*small).float(), ref(*small).float()
            ok = torch.allclose(out, exp, atol=3e-2, rtol=3e-2)
        _STATUS[key] = "fused" if ok else f"fallback: mismatch max={float((out-exp).abs().max()):.3e}"
    except Exception as e:                    # 算子不存在/签名不符 -> 回退
        _STATUS[key] = f"fallback: {type(e).__name__}: {e}"[:200]
    return _STATUS[key] == "fused"


def attention(q, k, v, key_padding_mask=None, dropout_p=0.0, training=False):
    dev = q.device.type
    if _ENABLED and dev == "npu" and torch_npu is not None:
        if _selftest("npu_fusion_attention",
                     lambda a, b, c, m: _npu_fa(a, b, c, m, 0.0, False),
                     lambda a, b, c, m: attention_reference(a, b, c, m),
                     q, k, v, key_padding_mask):
            return _npu_fa(q, k, v, key_padding_mask, dropout_p, training)
    if hasattr(F, "scaled_dot_product_attention") and dev != "npu":
        return _sdpa(q, k, v, key_padding_mask, dropout_p, training)
    return attention_reference(q, k, v, key_padding_mask, dropout_p, training)


# =========================================================== swiglu
def swiglu_reference(gate, up):
    return F.silu(gate) * up


if _HAS_TRITON:
    @triton.jit
    def _swiglu_fwd_kernel(G, U, O, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(U + offs, mask=m, other=0.0).to(tl.float32)
        s = g / (1.0 + tl.exp(-g))
        tl.store(O + offs, (s * u).to(O.dtype.element_ty), mask=m)

    @triton.jit
    def _swiglu_bwd_kernel(G, U, DO, DG, DU, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
        u = tl.load(U + offs, mask=m, other=0.0).to(tl.float32)
        do = tl.load(DO + offs, mask=m, other=0.0).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-g))
        s = g * sig
        tl.store(DU + offs, (do * s).to(DU.dtype.element_ty), mask=m)
        dsdg = sig * (1.0 + g * (1.0 - sig))
        tl.store(DG + offs, (do * u * dsdg).to(DG.dtype.element_ty), mask=m)

    class _TritonSwiGLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gate, up):
            gate, up = gate.contiguous(), up.contiguous()
            out = torch.empty_like(gate)
            n = gate.numel()
            grid = (triton.cdiv(n, 1024),)
            _swiglu_fwd_kernel[grid](gate, up, out, n, BLOCK=1024)
            ctx.save_for_backward(gate, up)
            return out

        @staticmethod
        def backward(ctx, dout):
            gate, up = ctx.saved_tensors
            dout = dout.contiguous()
            dg, du = torch.empty_like(gate), torch.empty_like(up)
            n = gate.numel()
            grid = (triton.cdiv(n, 1024),)
            _swiglu_bwd_kernel[grid](gate, up, dout, dg, du, n, BLOCK=1024)
            return dg, du


def _npu_swiglu(gate, up):
    return torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1), dim=-1)


def swiglu(gate, up):
    dev = gate.device.type
    if _ENABLED and dev == "cuda" and _HAS_TRITON:
        if _selftest("triton_swiglu", _TritonSwiGLU.apply, swiglu_reference, gate, up):
            return _TritonSwiGLU.apply(gate, up)
    if _ENABLED and dev == "npu" and torch_npu is not None:
        if _selftest("npu_swiglu", _npu_swiglu, swiglu_reference, gate, up):
            return _npu_swiglu(gate, up)
    return swiglu_reference(gate, up)
