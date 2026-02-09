import paddle

import pytest

from paddle_geometric.nn import RGCNConv


def get_available_devices():
    """获取可用的设备列表"""
    devices = ['cpu']
    if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        devices.append('gpu')
    return devices


def skip_if_cuda_error():
    try:
        import paddle
        paddle.set_device('cpu')
        return False
    except Exception as e:
        if 'cuda' in str(e).lower() or 'CUDA' in str(e):
            pytest.skip(f"CUDA 兼容性问题: {e}")
        return False


@pytest.fixture(autouse=True)
def setup_cpu_mode():
    """自动设置 CPU 模式"""
    skip_if_cuda_error()



@pytest.mark.parametrize('device', get_available_devices())
def test_rgcn_conv(device):
    paddle.set_device(device)

    x = paddle.randn([4, 16])
    edge_index = paddle.to_tensor([[0, 1, 2, 3], [1, 0, 1, 1]])
    edge_type = paddle.to_tensor([0, 1, 0, 1])

    conv = RGCNConv(16, 32, num_relations=2)
    assert str(conv) == 'RGCNConv(16, 32, num_relations=2)'

    out = conv(x, edge_index, edge_type)
    assert out.shape == [4, 32]

    # Test with None input (trainable embeddings)
    conv = RGCNConv(4, 32, num_relations=2, num_bases=2)
    out = conv(None, edge_index, edge_type)
    assert out.shape == [4, 32]

    # Test bipartite message passing:
    x1 = paddle.randn([4, 4])
    x2 = paddle.randn([2, 16])
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 2, 3], [0, 0, 1, 0, 1, 1]])
    edge_type = paddle.to_tensor([0, 1, 1, 0, 0, 1])

    conv = RGCNConv((4, 16), 32, num_relations=2)
    assert str(conv) == 'RGCNConv((4, 16), 32, num_relations=2)'

    out = conv((x1, x2), edge_index, edge_type)
    assert out.shape == [2, 32]