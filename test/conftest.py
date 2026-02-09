"""
pytest配置文件 - 用于处理CUDA兼容性问题
"""
import pytest
import warnings


def pytest_configure(config):
    """pytest配置钩子"""
    # 在测试开始前设置CPU模式
    try:
        import paddle
        paddle.set_device('cpu')
    except Exception as e:
        warnings.warn(f"Failed to set CPU device: {e}")


@pytest.fixture(scope="session", autouse=True)
def setup_cpu_mode():
    try:
        import paddle
        paddle.set_device('cpu')
    except Exception as e:
        pytest.skip(f"Cannot initialize Paddle in CPU mode: {e}")
    yield