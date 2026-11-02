#!/usr/bin/env python3
"""Generate deterministic, intentionally small ONNX operator-coverage fixtures.

Uses only the Python standard library. These are synthetic test graphs, not trained SR models.
"""
import argparse
import pathlib
import struct


def varint(value):
    value = int(value)
    if value < 0:
        value &= (1 << 64) - 1
    out = bytearray()
    while value > 127:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def field(number, wire, data):
    return varint((number << 3) | wire) + data


def integer(number, value):
    return field(number, 0, varint(value))


def payload(number, value):
    if isinstance(value, str):
        value = value.encode('utf-8')
    return field(number, 2, varint(len(value)) + value)


def dims_shape(shape):
    return b''.join(payload(1, integer(1, size)) for size in shape)


def value_info(name, shape):
    tensor_type = integer(1, 1) + payload(2, dims_shape(shape))
    return payload(1, name) + payload(2, payload(1, tensor_type))


def tensor(name, dims, values, dtype=1):
    raw = (struct.pack('<' + 'f' * len(values), *values) if dtype == 1 else
           struct.pack('<' + 'q' * len(values), *values))
    return (b''.join(integer(1, d) for d in dims) + integer(2, dtype) +
            payload(8, name) + payload(9, raw))


def attribute_ints(name, values):
    return payload(1, name) + b''.join(integer(8, v) for v in values) + integer(20, 7)


def attribute_int(name, value):
    return payload(1, name) + integer(3, value) + integer(20, 2)


def node(op, inputs, outputs, attributes=()):
    return (b''.join(payload(1, name) for name in inputs) +
            b''.join(payload(2, name) for name in outputs) + payload(4, op) +
            b''.join(payload(5, attr) for attr in attributes))


def identity_conv_weights(channels):
    return [float(o == i) for o in range(channels) for i in range(channels)]


def model(kind, nodes, initializers):
    graph = (b''.join(payload(1, n) for n in nodes) + payload(2, kind) +
             b''.join(payload(5, t) for t in initializers) +
             payload(11, value_info('image', [1, 3, 8, 8])) +
             payload(12, value_info('output', [1, 3, 16, 16])))
    return (integer(1, 7) + payload(2, 'TexUpSR') +
            payload(6, 'Synthetic operator-coverage fixture; not a trained super-resolution model') +
            payload(7, graph) + payload(8, integer(2, 13)))


def resize(source, target):
    return node('Resize', [source, '', 'resize_scales'], [target])


def make_cnn():
    init = [tensor('cnn_w', [3, 3, 1, 1], identity_conv_weights(3)),
            tensor('resize_scales', [4], [1., 1., 2., 2.])]
    nodes = [node('Conv', ['image', 'cnn_w'], ['conv']),
             node('Relu', ['conv'], ['activated']), resize('activated', 'output')]
    return model('cnn', nodes, init)


def make_rrdb_style():
    # Two compact residual dense units; this is RRDB-style operator coverage,
    # not a faithful/trained ESRGAN RRDB generator.
    init = [tensor('dense_w', [2, 3, 1, 1], [.125] * 6),
            tensor('merge_w', [3, 5, 1, 1], [.05] * 15),
            tensor('resize_scales', [4], [1., 1., 2., 2.])]
    nodes = []
    source = 'image'
    for block in range(2):
        nodes.extend([
            node('Conv', [source, 'dense_w'], [f'dense_{block}']),
            node('LeakyRelu', [f'dense_{block}'], [f'activated_{block}']),
            node('Concat', [source, f'activated_{block}'], [f'joined_{block}'],
                 [attribute_int('axis', 1)]),
            node('Conv', [f'joined_{block}', 'merge_w'], [f'merged_{block}']),
            node('Add', [source, f'merged_{block}'], [f'residual_{block}'])])
        source = f'residual_{block}'
    nodes.append(resize(source, 'output'))
    return model('rrdb_style', nodes, init)


def make_attention():
    init = [tensor('flat_shape', [3], [1, 64, 3], 7),
            tensor('image_shape', [4], [1, 8, 8, 3], 7),
            tensor('proj_w', [3, 3], identity_conv_weights(3)),
            tensor('atten_scale', [1], [0.125]),
            tensor('resize_scales', [4], [1., 1., 2., 2.])]
    nodes = [
        node('Transpose', ['image'], ['nhwc'], [attribute_ints('perm', [0, 2, 3, 1])]),
        node('Reshape', ['nhwc', 'flat_shape'], ['tokens']),
        node('MatMul', ['tokens', 'proj_w'], ['queries']),
        node('Transpose', ['queries'], ['keys'], [attribute_ints('perm', [0, 2, 1])]),
        node('MatMul', ['queries', 'keys'], ['scores']),
        node('Mul', ['scores', 'atten_scale'], ['scaled']),
        node('Softmax', ['scaled'], ['attention'], [attribute_int('axis', -1)]),
        node('MatMul', ['attention', 'tokens'], ['attended']),
        node('Reshape', ['attended', 'image_shape'], ['attended_image']),
        node('Transpose', ['attended_image'], ['nchw'], [attribute_ints('perm', [0, 3, 1, 2])]),
        node('Add', ['image', 'nchw'], ['combined']),
        resize('combined', 'output')]
    return model('attention', nodes, init)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='tests/data/m4')
    args = parser.parse_args()
    directory = pathlib.Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, maker in [('cnn_2x.onnx', make_cnn),
                        ('rrdb_style_2x.onnx', make_rrdb_style),
                        ('attention_2x.onnx', make_attention)]:
        output = directory / name
        output.write_bytes(maker())
        print(f'{output}: {output.stat().st_size} bytes')


if __name__ == '__main__':
    main()
