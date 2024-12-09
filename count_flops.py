# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import pickle
import re
import warnings
import contextlib
import collections
import click
import numpy as np
import torch
import dnnlib

warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')

#----------------------------------------------------------------------------

@contextlib.contextmanager
def hook_torch_ops():
    conv2d_orig = torch.nn.functional.conv2d
    conv_transpose2d_orig = torch.nn.functional.conv_transpose2d
    einsum_orig = torch.einsum
    scaled_dot_product_attention_orig = torch.nn.functional.scaled_dot_product_attention

    def conv2d_hook(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        padding = tuple(padding) if isinstance(padding, (list, tuple)) else (padding,)
        stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,)
        dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,)
        return conv2d_orig(input, weight, bias, stride, padding, dilation, groups)
    def conv_transpose2d_hook(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        padding = tuple(padding) if isinstance(padding, (list, tuple)) else (padding,)
        return conv_transpose2d_orig(input, weight, bias=bias, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)
    def einsum_hook(eq, *ops):
        if eq in ['nhcq,nhck->nhqk', 'ncq,nck->nqk', 'b h d e, b h d n -> b h e n']:
            return ops[0].mT @ ops[1]
        if eq in ['nhqk,nhck->nhcq', 'nqk,nck->ncq']:
            return ops[1] @ ops[0].mT
        if eq in ['b h d n, b h e n -> b h d e', 'b h i d, b h j d -> b h i j']:
            return ops[0] @ ops[1].mT
        if eq in ['b h i j, b h j d -> b h i d']:
            return ops[0] @ ops[1]
        if eq == 'nhwpqc->nchpwq':
            return torch.permute(ops[0], (0,5,1,3,2,4))
        raise ValueError(f'Unsupported einsum "{eq}"')
    def scaled_dot_product_attention_hook(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False):
        assert dropout_p == 0.0, "unimplemented"
        assert is_causal == False, "unimplemented"
        assert attn_mask is None, "unimplemented"
        attn_weight = torch.softmax((query @ key.transpose(-2, -1) / np.sqrt(query.size(-1))), dim=-1)
        return attn_weight @ value

    torch.nn.functional.conv2d = conv2d_hook
    torch.nn.functional.conv_transpose2d = conv_transpose2d_hook
    torch.einsum = einsum_hook
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention_hook

    yield

    torch.nn.functional.conv2d = conv2d_orig
    torch.nn.functional.conv_transpose2d = conv_transpose2d_orig
    torch.einsum = einsum_orig
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention_orig

#----------------------------------------------------------------------------

def count_flops(
    net,                                # Path, URL, or torch.nn.Module.
    net_args    = None,                 # Positional arguments for the network. None = select automatically.
    net_kwargs  = {},                   # Keyword arguments for the network.
    verbose     = True,                 # Enable status prints?
    device      = torch.device('cuda'), # Which compute device to use.
):
    if isinstance(net, str):
        print(f'Loading network from {net} ...')
        with dnnlib.util.open_url(net, verbose=verbose) as f:
            net = pickle.load(f)['ema']
    net.to(device).requires_grad_(False)

    if net_args is None:
        x = torch.zeros([1, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        sigma = torch.zeros([1], device=device)
        labels = torch.zeros([1, net.label_dim], device=device)
        net_args = (x, sigma, labels)

    if verbose:
        print('Tracing...')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.cuda.reset_peak_memory_stats(device)
    mem_before_trace = torch.cuda.max_memory_allocated(device)
    with hook_torch_ops():
        warnings.filterwarnings('ignore', 'Converting a tensor to a Python integer might cause the trace to be incorrect.')
        warnings.filterwarnings('ignore', 'Converting a tensor to a Python boolean might cause the trace to be incorrect.')
        def trace_fn():
            return net(*net_args, **net_kwargs)
        trace = torch.jit.trace(trace_fn, (), check_trace=False)
    mem_after_trace = torch.cuda.max_memory_allocated(device)

    def list_nodes(graph):
        nodes = []
        for node in graph.nodes():
            if node.kind() == 'prim::PythonOp':
                nodes += list_nodes(node.g('Subgraph'))
            else:
                nodes += [node]
        return nodes
    nodes = list_nodes(trace.graph)

    ops = {
        'prim::Constant':           lambda *args: 0,
        'prim::NumToTensor':        lambda *args: 0,
        'prim::ListConstruct':      lambda *args: 0,
        'prim::ListUnpack':         lambda *args: 0,
        'profiler::_record_function_exit': lambda *args: 0,
        'profiler::_record_function_enter_new': lambda *args: 0,
        'aten::to':                 lambda *args: 0,
        'aten::zeros':              lambda *args: 0,
        'aten::ones':               lambda *args: 0,
        'aten::full':               lambda *args: 0,
        'aten::ones_like':          lambda *args: 0,
        'aten::reshape':            lambda *args: 0,
        'aten::flatten':            lambda *args: 0,
        'aten::slice':              lambda *args: 0,
        'aten::contiguous':         lambda *args: 0,
        'aten::t':                  lambda *args: 0,
        'aten::mT':                 lambda *args: 0,
        'aten::unsqueeze':          lambda *args: 0,
        'aten::unbind':             lambda *args: 0,
        'aten::tile':               lambda *args: 0,
        'aten::chunk':              lambda *args: 0,
        'aten::cat':                lambda *args: 0,
        'aten::lift_fresh':         lambda *args: 0,
        'aten::size':               lambda *args: 0,
        'aten::Int':                lambda *args: 0,
        'aten::detach':             lambda *args: 0,
        'aten::resolve_conj':       lambda *args: 0,
        'aten::resolve_neg':        lambda *args: 0,
        'aten::dropout':            lambda *args: 0,
        'aten::arange':             lambda *args: 0,
        'aten::ScalarImplicit':     lambda *args: 0,
        'aten::view':               lambda *args: 0,
        'aten::upsample_nearest2d': lambda *args: 0,
        'aten::embedding':          lambda *args: 0,
        'aten::split':              lambda *args: 0,
        'aten::permute':            lambda *args: 0,
        'aten::transpose':          lambda *args: 0,
        'aten::empty':              lambda *args: 0,
        'aten::numel':              lambda *args: 0,
        'aten::expand':             lambda *args: 0,
        'aten::expand_as':          lambda *args: 0,
        'aten::eye':                lambda *args: 0,
        'aten::repeat':             lambda *args: 0,
        'aten::select':             lambda *args: 0,
        'aten::squeeze':            lambda *args: 0,
        'aten::copy_':              lambda *args: 0,
        'aten::pad':                lambda *args: 0,
        'aten::flip':               lambda *args: 0,
        'aten::clone':              lambda *args: 0,
        'aten::gather':             lambda *args: 0,
        'prim::TupleConstruct':     lambda *args: 0,
        'aten::add':                lambda out, *args: out,
        'aten::add_':               lambda out, *args: out,
        'aten::sub':                lambda out, *args: out,
        'aten::rsub':               lambda out, *args: out,
        'aten::exp':                lambda out, *args: out,
        'aten::mul':                lambda out, *args: out,
        'aten::mul_':               lambda out, *args: out,
        'aten::div':                lambda out, *args: out,
        'aten::addcmul':            lambda out, *args: out,
        'aten::lerp':               lambda out, *args: out + [2],
        'aten::floor_divide':       lambda out, *args: out,
        'aten::pow':                lambda out, *args: out,
        'aten::reciprocal':         lambda out, *args: out,
        'aten::square':             lambda out, *args: out,
        'aten::sqrt':               lambda out, *args: out,
        'aten::rsqrt':              lambda out, *args: out,
        'aten::sin':                lambda out, *args: out,
        'aten::cos':                lambda out, *args: out,
        'aten::log':                lambda out, *args: out,
        'aten::eq':                 lambda out, *args: out,
        'aten::gt':                 lambda out, *args: out,
        'aten::neg':                lambda out, *args: out,
        'aten::mean':               lambda out, *args: out,
        'aten::sum':                lambda out, *args: out,
        'aten::clamp_min':          lambda out, *args: out,
        'aten::linalg_vector_norm': lambda out, a, *args: a,
        'aten::leaky_relu':         lambda out, *args: out + [2],
        'aten::silu':               lambda out, *args: out + [3],
        'aten::sigmoid':            lambda out, *args: out + [3],
        'aten::gelu':               lambda out, *args: out + [14], # https://arxiv.org/abs/2210.13452
        'aten::softmax':            lambda out, *args: out + [3],
        'aten::group_norm':         lambda out, *args: out + [4],
        'aten::layer_norm':         lambda out, *args: out + [4],
        'aten::upsample_bilinear2d':lambda out, *args: out + [4],
        'aten::outer':              lambda out, *args: out,
        'aten::matmul':             lambda out, a, *args: out + a[-1:],
        'aten::addmm':              lambda out, inp, a, *args: out + [a[-1] + 1],
        'aten::linear':             lambda out, a, *args: out + a[-1:],
        'aten::_convolution':       lambda out, x, w, *args: out + w[1:],
        'aten::clamp_':             lambda out, *args: out + [2],
        'aten::clamp':              lambda out, *args: out + [2],
        'aten::avg_pool2d':         lambda out, a, *args: a,
        'aten::affine_grid_generator': lambda out, a, *args: out,
    }

    res = dnnlib.EasyDict()
    res.op_count = collections.defaultdict(int)
    res.op_flops = collections.defaultdict(np.float64)

    for node in nodes:
        if node.kind() not in ops:
            print(f'Unknown node kind "{node.kind()}"\n{node.schema()}')
            for o in node.outputs():
                print(str(o))
            print('\n')
            continue
        args = []
        for arg in list(node.outputs()) + list(node.inputs()):
            if m := re.search(r':\s*(Float|Double|Half|Long)\(([\d\s,]*)', str(arg)):
                shape = m.group(2).strip(', ')
                args.append([] if shape == '' else [int(x) for x in shape.split(',')])
            elif re.search(r':\s*(float|int|bool|str|NoneType|int\[\]|Tensor\[\]|Device|Scalar|__torch__\.torch\.classes\.profiler\._RecordFunction)\s*=', str(arg)):
                args.append([])
            else:
                print(f'Unknown input specification:\n{arg}\n')
                args.append([])
        flops = np.prod(np.float64(ops[node.kind()](*args)))
        label = node.kind() if flops else 'Other'
        res.op_count[label] += 1
        res.op_flops[label] += flops

    res.total_params = sum(x.numel() for x in net.parameters())
    res.total_flops = sum(res.op_flops.values())
    res.mem_per_sample = mem_after_trace - mem_before_trace
    return res

#----------------------------------------------------------------------------

def print_flops(res):
    print()
    print(f'{"Parameters":<12s}{res.total_params/1e6:<8.2f}M')
    print(f'{"Compute":<12s}{res.total_flops/1e9:<8.2f}Gflops/sample')
    print(f'{"GPU memory":<12s}{res.mem_per_sample/2**30:<8.2f}GB/sample')
    print()
    print(f'{"Op":<26s}{"Count":<7s}{"Gflops":s}')
    for op, flops in sorted(res.op_flops.items(), key=lambda x: -x[1]):
        print(f'{op:<26s}{res.op_count[op]:<7d}{flops/1e9:.2f}')
    print(f'{"Total":<26s}{sum(res.op_count.values()):<7d}{sum(res.op_flops.values())/1e9:.2f}')
    print()

#----------------------------------------------------------------------------

@click.command()
@click.argument('net', metavar='PKL|URL', type=str)
def cli(net):
    """Compute model flops.

    Example:

    \b
    python count_flops.py https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.130.pkl
    """
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise click.ClickException('Distributed execution is not supported')
    print_flops(count_flops(net))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cli()

#----------------------------------------------------------------------------
