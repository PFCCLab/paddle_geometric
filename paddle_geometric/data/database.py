import io
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from paddle_geometric.utils.mixin import CastMixin
import paddle
from paddle import Tensor
from tqdm import tqdm
import pickle  # Used for serializing and deserializing complex objects

from paddle_geometric import EdgeIndex, Index
from paddle_geometric.edge_index import SortOrder


@dataclass
class TensorInfo(CastMixin):
    """Describes the type information of a tensor, including data type, size,
    whether it's an index or an edge index."""
    dtype: paddle.dtype
    size: Tuple[int, ...] = (-1,)
    is_index: bool = False
    is_edge_index: bool = False

    def __post_init__(self) -> None:
        # A tensor cannot be both an index and an edge index simultaneously
        if self.is_index and self.is_edge_index:
            raise ValueError("Tensor cannot be both 'Index' and 'EdgeIndex' at the same time.")
        if self.is_index:
            self.size = (-1,)  # Dynamic size for index tensors
        if self.is_edge_index:
            self.size = (2, -1)  # Edge indices are two-dimensional


def maybe_cast_to_tensor_info(value: Any) -> Union[Any, TensorInfo]:
    """Converts input to TensorInfo if it meets the criteria."""
    if not isinstance(value, dict):
        return value
    if len(value) < 1 or len(value) > 3:
        return value
    if 'dtype' not in value:
        return value
    valid_keys = {'dtype', 'size', 'is_index', 'is_edge_index'}
    if len(set(value.keys()) | valid_keys) != len(valid_keys):
        return value
    return TensorInfo.cast(value)


Schema = Union[Any, Dict[str, Any], Tuple[Any], List[Any]]

SORT_ORDER_TO_INDEX: Dict[Optional[SortOrder], int] = {
    None: -1,
    SortOrder.ROW: 0,
    SortOrder.COL: 1,
}
INDEX_TO_SORT_ORDER = {v: k for k, v in SORT_ORDER_TO_INDEX.items()}


class Database(ABC):
    """Abstract base class for a database that supports inserting and retrieving data.

    A database acts as an index-based key-value store for tensors and other custom data.
    """
    def __init__(self, schema: Schema = object) -> None:
        schema_dict = self._to_dict(schema)
        self.schema: Dict[Union[str, int], Any] = schema_dict

    @abstractmethod
    def connect(self) -> None:
        """Connects to the database.
        Databases will automatically connect on instantiation.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Closes the connection to the database."""
        raise NotImplementedError

    @abstractmethod
    def insert(self, index: int, data: Any) -> None:
        """Insert data at a specified index."""
        raise NotImplementedError

    def multi_insert(
        self,
        indices: Union[Sequence[int], Tensor, slice, range],
        data_list: Sequence[Any],
        batch_size: Optional[int] = None,
        log: bool = False,
    ) -> None:
        """Insert multiple data entries at specified indices."""
        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)

        length = min(len(indices), len(data_list))
        batch_size = length if batch_size is None else batch_size

        if log and length > batch_size:
            desc = f'Insert {length} entries'
            offsets = tqdm(range(0, length, batch_size), desc=desc)
        else:
            offsets = range(0, length, batch_size)

        for start in offsets:
            self._multi_insert(
                indices[start:start + batch_size],
                data_list[start:start + batch_size],
            )

    def _multi_insert(
        self,
        indices: Union[Sequence[int], Tensor, range],
        data_list: Sequence[Any],
    ) -> None:
        """Internal method for batch insertion."""
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        for index, data in zip(indices, data_list):
            self.insert(index, data)

    @abstractmethod
    def get(self, index: int) -> Any:
        """Retrieve data from a specified index."""
        raise NotImplementedError

    def multi_get(
        self,
        indices: Union[Sequence[int], Tensor, slice, range],
        batch_size: Optional[int] = None,
    ) -> List[Any]:
        """Retrieve data from multiple indices."""
        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)

        length = len(indices)
        batch_size = length if batch_size is None else batch_size

        data_list: List[Any] = []
        for start in range(0, length, batch_size):
            chunk_indices = indices[start:start + batch_size]
            data_list.extend(self._multi_get(chunk_indices))
        return data_list

    def _multi_get(self, indices: Union[Sequence[int], Tensor]) -> List[Any]:
        """Internal method for batch retrieval."""
        if isinstance(indices, Tensor):
            indices = indices.tolist()
        return [self.get(index) for index in indices]

    @staticmethod
    def _to_dict(value: Any) -> Dict[Union[str, int], Any]:
        """Convert the input value to a dictionary."""
        if isinstance(value, dict):
            return value
        if isinstance(value, (tuple, list)):
            return {i: v for i, v in enumerate(value)}
        return {0: value}

    def slice_to_range(self, indices: slice) -> range:
        """Convert a slice object into a range object."""
        start = indices.start or 0
        stop = indices.stop or len(self)
        step = indices.step or 1
        return range(start, stop, step)

    def __len__(self) -> int:
        """Return the number of entries in the database."""
        raise NotImplementedError

    def __getitem__(
        self,
        key: Union[int, Sequence[int], Tensor, slice, range],
    ) -> Union[Any, List[Any]]:
        """Retrieve data using index or slice."""
        if isinstance(key, int):
            return self.get(key)
        return self.multi_get(key)

    def __setitem__(
        self,
        key: Union[int, Sequence[int], Tensor, slice, range],
        value: Union[Any, Sequence[Any]],
    ) -> None:
        """Insert data using index or slice."""
        if isinstance(key, int):
            self.insert(key, value)
        else:
            self.multi_insert(key, value)

    def __repr__(self) -> str:
        try:
            return f"{self.__class__.__name__}({len(self)})"
        except NotImplementedError:
            return f"{self.__class__.__name__}()"


class SQLiteDatabase(Database):
    """An SQLite-based key-value database implementation.

    Uses SQLite to store tensors and other data types.
    """
    def __init__(self, path: str, name: str, schema: Schema = object) -> None:
        super().__init__(schema)

        warnings.filterwarnings('ignore', '.*given buffer is not writable.*')

        import sqlite3
        self.path = path
        self.name = name
        self._connection: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None
        self.connect()

        # Create the table (if it does not exist) by mapping the Python schema
        # to the corresponding SQL schema:
        sql_schema = ',\n'.join([
            f'  {col_name} {self._to_sql_type(type_info)}'
            for col_name, type_info in zip(self._col_names, self.schema.values())
        ])
        query = (f'CREATE TABLE IF NOT EXISTS {self.name} (\n'
                 f'  id INTEGER PRIMARY KEY,\n'
                 f'{sql_schema}\n'
                 f')')
        self.cursor.execute(query)

    def connect(self) -> None:
        """Connect to the SQLite database."""
        import sqlite3
        self._connection = sqlite3.connect(self.path)
        self._cursor = self._connection.cursor()

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.commit()
            self._connection.close()
            self._connection = None
            self._cursor = None

    @property
    def connection(self) -> Any:
        """Return the database connection object."""
        if self._connection is None:
            raise RuntimeError("No open database connection")
        return self._connection

    @property
    def cursor(self) -> Any:
        """Return the database cursor object."""
        if self._cursor is None:
            raise RuntimeError("No open database connection")
        return self._cursor

    def insert(self, index: int, data: Any) -> None:
        """Insert a single data entry."""
        query = (f'INSERT INTO {self.name} '
                 f'(id, {self._joined_col_names}) '
                 f'VALUES (?, {self._dummies})')
        self.cursor.execute(query, (index, *self._serialize(data)))
        self.connection.commit()

    def _multi_insert(
        self,
        indices: Union[Sequence[int], Tensor, range],
        data_list: Sequence[Any],
    ) -> None:
        if isinstance(indices, Tensor):
            indices = indices.tolist()

        data_list = [(index, *self._serialize(data))
                     for index, data in zip(indices, data_list)]

        query = (f'INSERT INTO {self.name} '
                 f'(id, {self._joined_col_names}) '
                 f'VALUES (?, {self._dummies})')
        self.cursor.executemany(query, data_list)
        self.connection.commit()

    def get(self, index: int) -> Any:
        """Retrieve a single data entry."""
        query = (f'SELECT {self._joined_col_names} FROM {self.name} '
                 f'WHERE id = ?')
        self.cursor.execute(query, (index,))
        row = self.cursor.fetchone()
        if row is None:
            raise KeyError(f"Index {index} not found in database")
        return self._deserialize(row)

    def multi_get(
        self,
        indices: Union[Sequence[int], Tensor, slice, range],
        batch_size: Optional[int] = None,
    ) -> List[Any]:
        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)
        elif isinstance(indices, Tensor):
            indices = indices.tolist()

        join_table_name = f'{self.name}__join'
        query = (f'CREATE TEMP TABLE {join_table_name} (\n'
                 f'  id INTEGER,\n'
                 f'  row_id INTEGER\n'
                 f')')
        self.cursor.execute(query)

        query = f'INSERT INTO {join_table_name} (id, row_id) VALUES (?, ?)'
        self.cursor.executemany(query, zip(indices, range(len(indices))))
        self.connection.commit()

        query = (f'SELECT {self._joined_col_names} '
                 f'FROM {self.name} INNER JOIN {join_table_name} '
                 f'ON {self.name}.id = {join_table_name}.id '
                 f'ORDER BY {join_table_name}.row_id')
        self.cursor.execute(query)

        if batch_size is None:
            data_list = self.cursor.fetchall()
        else:
            data_list = []
            while True:
                chunk_list = self.cursor.fetchmany(size=batch_size)
                if len(chunk_list) == 0:
                    break
                data_list.extend(chunk_list)

        query = f'DROP TABLE {join_table_name}'
        self.cursor.execute(query)

        return [self._deserialize(data) for data in data_list]

    def __len__(self) -> int:
        """Get the total number of entries in the database."""
        query = f"SELECT COUNT(*) FROM {self.name}"
        self.cursor.execute(query)
        return self.cursor.fetchone()[0]

    # Helper functions ########################################################

    @cached_property
    def _col_names(self) -> List[str]:
        return [f'COL_{key}' for key in self.schema.keys()]

    @cached_property
    def _joined_col_names(self) -> str:
        return ', '.join(self._col_names)

    @cached_property
    def _dummies(self) -> str:
        return ', '.join(['?'] * len(self.schema.keys()))

    def _to_sql_type(self, type_info: Any) -> str:
        if type_info == int:
            return 'INTEGER NOT NULL'
        if type_info == float:
            return 'FLOAT'
        if type_info == str:
            return 'TEXT NOT NULL'
        return 'BLOB NOT NULL'

    def _serialize(self, data: Any) -> List[bytes]:
        """Serialize data into a byte stream."""
        # Handle both dict-like data and single tensor/data
        if isinstance(data, dict):
            return [pickle.dumps(data.get(key)) for key in self.schema.keys()]
        elif len(self.schema) == 1 and 0 in self.schema:
            # Single object schema: {0: type}, data is a single value (e.g., Tensor)
            return [pickle.dumps(data)]
        else:
            # Fallback: try to access as dict or use data directly
            return [
                pickle.dumps(data.get(key) if hasattr(data, 'get') else data)
                for key in self.schema.keys()
            ]

    def _deserialize(self, row: Tuple[bytes]) -> Any:
        """Deserialize a byte stream into original data."""
        result = {
            key: pickle.loads(value)
            for key, value in zip(self.schema.keys(), row)
        }

        # If schema has only one key (0), return the single value instead of dict
        if len(result) == 1 and 0 in result:
            return result[0]
        return result


class RocksDatabase(Database):
    """A RocksDB-based key-value database implementation.

    Uses RocksDB to store tensors and other data types.
    """
    def __init__(self, path: str, schema: Schema = object) -> None:
        super().__init__(schema)
        import rocksdict

        self.path = path
        self._db: Optional[rocksdict.Rdict] = None
        self.connect()

    def connect(self) -> None:
        """Connect to the RocksDB database."""
        import rocksdict
        self._db = rocksdict.Rdict(self.path, options=rocksdict.Options(raw_mode=True))

    def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def db(self) -> Any:
        """Return the database object."""
        if self._db is None:
            raise RuntimeError("No open database connection")
        return self._db

    @staticmethod
    def to_key(index: int) -> bytes:
        """Convert an integer index to bytes."""
        return index.to_bytes(8, byteorder="big", signed=True)

    def insert(self, index: int, data: Any) -> None:
        """Insert a single data entry."""
        self.db[self.to_key(index)] = self._serialize(data)

    def get(self, index: int) -> Any:
        """Retrieve a single data entry."""
        return self._deserialize(self.db[self.to_key(index)])

    def _multi_get(self, indices: Union[Sequence[int], Tensor]) -> List[Any]:
        """RocksDB 批量 key 查询"""
        if isinstance(indices, Tensor):
            indices = indices.tolist()

        # rocksdict.Rdict 逐个获取 key
        return [self._deserialize(self.db[self.to_key(index)])
                for index in indices]

    def _serialize(self, data: Any) -> bytes:
        """Serialize data into a byte stream."""
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        return buffer.getvalue()

    def _deserialize(self, row: bytes) -> Any:
        """Deserialize a byte stream into original data."""
        return pickle.loads(row)
