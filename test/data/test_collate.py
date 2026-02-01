import pytest
import paddle

from paddle_geometric.data import Data, HeteroData, Batch
from paddle_geometric.data.collate import collate


def test_collate_basic():
    """Test basic collate functionality."""
    data_list = [
        Data(x=paddle.randn((4, 8)), edge_index=paddle.to_tensor([[0, 1], [1, 2]])),
        Data(x=paddle.randn((5, 8)), edge_index=paddle.to_tensor([[0, 1, 2], [1, 2, 3]])),
    ]

    out, slice_dict, inc_dict = collate(Data, data_list)

    assert out.x.shape == (9, 8)
    assert out.edge_index.tolist() == [[0, 1, 4, 5], [1, 2, 5, 6]]
    assert slice_dict['x'].tolist() == [0, 4, 9]
    assert slice_dict['edge_index'].tolist() == [0, 2, 5]


def test_collate_with_batch():
    """Test collate with batch vector."""
    data_list = [
        Data(x=paddle.randn((4, 8)), edge_index=paddle.to_tensor([[0, 1], [1, 2]])),
        Data(x=paddle.randn((5, 8)), edge_index=paddle.to_tensor([[0, 1, 2], [1, 2, 3]])),
    ]

    out, slice_dict, inc_dict = collate(Data, data_list, add_batch=True)

    assert out.batch.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert out.ptr.tolist() == [0, 4, 9]


def test_collate_with_follow_batch():
    """Test collate with follow_batch parameter."""
    data_list = [
        Data(x=paddle.randn((4, 8)), edge_index=paddle.to_tensor([[0, 1], [1, 2]])),
        Data(x=paddle.randn((5, 8)), edge_index=paddle.to_tensor([[0, 1, 2], [1, 2, 3]])),
    ]

    out, slice_dict, inc_dict = collate(Data, data_list, follow_batch=['x'])

    assert 'x_batch' in out
    assert out.x_batch.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert 'x_ptr' in out
    assert out.x_ptr.tolist() == [0, 4, 9]


def test_collate_with_exclude_keys():
    """Test collate with exclude_keys parameter."""
    data_list = [
        Data(x=paddle.randn((4, 8)), y=paddle.to_tensor([0, 1, 0, 1])),
        Data(x=paddle.randn((5, 8)), y=paddle.to_tensor([1, 0, 1, 0, 1])),
    ]

    out, slice_dict, inc_dict = collate(Data, data_list, exclude_keys=['y'])

    assert 'y' not in out
    assert 'x' in out


def test_collate_without_increment():
    """Test collate without increment."""
    data_list = [
        Data(x=paddle.randn((4, 8)), edge_index=paddle.to_tensor([[0, 1], [1, 2]])),
        Data(x=paddle.randn((5, 8)), edge_index=paddle.to_tensor([[0, 1, 2], [1, 2, 3]])),
    ]

    out, slice_dict, inc_dict = collate(Data, data_list, increment=False)

    # Without increment, edge_index should not be shifted
    assert out.edge_index.tolist() == [[0, 1, 0, 1, 2], [1, 2, 1, 2, 3]]


def test_collate_heterogeneous():
    """Test collate with heterogeneous data."""
    data1 = HeteroData()
    data1['user'].x = paddle.randn((4, 8))
    data1['user', 'buys', 'item'].edge_index = paddle.to_tensor([[0, 1], [0, 1]])

    data2 = HeteroData()
    data2['user'].x = paddle.randn((3, 8))
    data2['user', 'buys', 'item'].edge_index = paddle.to_tensor([[0], [0]])

    data_list = [data1, data2]
    out, slice_dict, inc_dict = collate(HeteroData, data_list)

    assert out['user'].x.shape == (7, 8)
    assert out['user', 'buys', 'item'].edge_index.tolist() == [[0, 1, 4], [0, 1, 0]]


def test_collate_with_dict_values():
    """Test collate with dictionary values."""
    class MyData(Data):
        pass

    data_list = [
        MyData(x={'a': paddle.randn((4, 2)), 'b': paddle.randn((4, 3))}),
        MyData(x={'a': paddle.randn((5, 2)), 'b': paddle.randn((5, 3))}),
    ]

    out, slice_dict, inc_dict = collate(MyData, data_list)

    assert out.x['a'].shape == (9, 2)
    assert out.x['b'].shape == (9, 3)


def test_collate_with_list_values():
    """Test collate with list values."""
    class MyData(Data):
        pass

    data_list = [
        MyData(x=[paddle.randn((4, 2)), paddle.randn((4, 3))]),
        MyData(x=[paddle.randn((5, 2)), paddle.randn((5, 3))]),
    ]

    out, slice_dict, inc_dict = collate(MyData, data_list)

    assert len(out.x) == 2
    assert out.x[0].shape == (9, 2)
    assert out.x[1].shape == (9, 3)


def test_collate_with_scalar_values():
    """Test collate with scalar values."""
    data_list = [
        Data(x=paddle.randn((4, 8)), y=0),
        Data(x=paddle.randn((5, 8)), y=1),
    ]

    out, slice_dict, inc_dict = collate(Data, data_list)

    assert out.x.shape == (9, 8)
    assert out.y.tolist() == [0, 1]


def test_collate_with_custom_cat_dim():
    """Test collate with custom __cat_dim__."""
    class MyData(Data):
        def __cat_dim__(self, key, value, *args, **kwargs):
            if key == 'foo':
                return None
            else:
                return super().__cat_dim__(key, value, *args, **kwargs)

    x = paddle.to_tensor([1, 2, 3], dtype=paddle.float32)
    foo = paddle.randn((4,))
    y = paddle.to_tensor(1)

    data = MyData(x=x, foo=foo, y=y)
    data_list = [data, data]

    out, slice_dict, inc_dict = collate(MyData, data_list)

    # foo should be stacked with new dimension
    assert out.foo.shape == (2, 4)
    # x and y should be concatenated
    assert out.x.tolist() == [1, 2, 3, 1, 2, 3]
    assert out.y.tolist() == [1, 1]


def test_collate_with_new_dimension():
    """Test collate with new dimension for some attributes."""
    class MyData(Data):
        def __cat_dim__(self, key, value, *args, **kwargs):
            if key == 'foo':
                return None
            else:
                return super().__cat_dim__(key, value, *args, **kwargs)

    x = paddle.to_tensor([1, 2, 3], dtype=paddle.float32)
    foo = paddle.randn((4,))
    y = paddle.to_tensor(1)

    data = MyData(x=x, foo=foo, y=y)
    data_list = [data, data]

    out, slice_dict, inc_dict = collate(MyData, data_list)

    assert len(out) == 3
    assert out.x.tolist() == [1, 2, 3, 1, 2, 3]
    assert out.foo.shape == (2, 4)
    assert out.y.tolist() == [1, 1]


def test_batch_from_data_list():
    """Test Batch.from_data_list method."""
    data_list = [
        Data(x=paddle.randn((4, 8)), edge_index=paddle.to_tensor([[0, 1], [1, 2]])),
        Data(x=paddle.randn((5, 8)), edge_index=paddle.to_tensor([[0, 1, 2], [1, 2, 3]])),
    ]

    batch = Batch.from_data_list(data_list)

    assert batch.x.shape == (9, 8)
    assert batch.batch.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert batch.num_graphs == 2