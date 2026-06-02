#!/usr/bin/env python3
import warnings
import struct
import numpy as np
import re
import lzf

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


def parse_ascii_pc_data(f, dtype, metadata):
    """ Use numpy to parse ascii pointcloud data.
    """
    return np.loadtxt(f, dtype=dtype, delimiter=' ')


def parse_binary_pc_data(f, dtype, metadata):
    rowstep = metadata['points'] * dtype.itemsize
    # for some reason pcl adds empty space at the end of files
    buf = f.read(rowstep)
    return np.frombuffer(buf, dtype=dtype)


def parse_binary_compressed_pc_data(f, dtype, metadata):
    """ Parse lzf-compressed data.
    Format is undocumented but seems to be:
    - compressed size of data (uint32)
    - uncompressed size of data (uint32)
    - compressed data
    - junk
    """
    fmt = 'II'
    compressed_size, uncompressed_size = \
        struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    compressed_data = f.read(compressed_size)
    # TODO what to use as second argument? if buf is None
    # (compressed > uncompressed)
    # should we read buf as raw binary?
    buf = lzf.decompress(compressed_data, uncompressed_size)
    if len(buf) != uncompressed_size:
        raise IOError('Error decompressing data')
    # the data is stored field-by-field
    pc_data = np.zeros(metadata['width'], dtype=dtype)
    ix = 0
    for dti in range(len(dtype)):
        dt = dtype[dti]
        bytes = dt.itemsize * metadata['width']
        column = np.frombuffer(buf[ix:(ix + bytes)], dtype=dt)
        pc_data[dtype.names[dti]] = column
        ix += bytes
    return pc_data


def _default_size_for_pcd_type(type_char):
    """PCD 头缺少 SIZE 时，按 TYPE 字母推断常见字节数。"""
    t = str(type_char).strip().upper()
    if t == 'F':
        return 4
    if t == 'U':
        return 1
    if t == 'I':
        return 4
    return 4


def _normalize_pcd_metadata(metadata):
    """补齐 PCD 头缺省字段，并将 map 转为 list（Python 3）。"""
    if 'fields' not in metadata:
        raise ValueError("PCD 头缺少 FIELDS")

    n = len(metadata['fields'])
    if 'type' not in metadata:
        metadata['type'] = ['F'] * n
    else:
        metadata['type'] = list(metadata['type'])

    if 'size' not in metadata:
        metadata['size'] = [_default_size_for_pcd_type(t) for t in metadata['type']]
    else:
        metadata['size'] = [int(x) for x in metadata['size']]

    if 'count' not in metadata:
        metadata['count'] = [1] * n
    else:
        metadata['count'] = [int(x) for x in metadata['count']]

    if len(metadata['type']) != n or len(metadata['size']) != n or len(metadata['count']) != n:
        raise ValueError(
            "PCD 头 FIELDS/TYPE/SIZE/COUNT 数量不一致: "
            "fields={}, type={}, size={}, count={}".format(
                n, len(metadata['type']), len(metadata['size']), len(metadata['count'])
            )
        )

    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    else:
        metadata['viewpoint'] = [float(x) for x in metadata['viewpoint']]

    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def parse_header(lines):
    """ Parse header of PCD files.
    """
    metadata = {}
    for ln in lines:
        ln = ln.strip()
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match(r'(\w+)\s+(.+)', ln)
        if not match:
            warnings.warn("warning: can't understand line: %s" % ln)
            continue
        key, value = match.group(1).lower(), match.group(2).strip()
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = [int(x) for x in value.split()]
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = [float(x) for x in value.split()]
        elif key == 'data':
            metadata[key] = value.strip().lower()
    return _normalize_pcd_metadata(metadata)


def _build_dtype(metadata):
    """ Build numpy structured array dtype from pcl metadata.
    Note that fields with count > 1 are 'flattened' by creating multiple
    single-count fields.
    *TODO* allow 'proper' multi-count fields.
    """
    fieldnames = []
    typenames = []
    idx = 0
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        # if f=='_':
        #     continue
        idx += 1
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            if f == '_':
                fieldnames.extend(['%s%s_%04d_%d' % (f, c, i, idx) for i in range(c)])
                typenames.extend([np_type] * c)
            else:
                fieldnames.extend(['%s%s_%04d' % (f, c, i) for i in range(c)])
                typenames.extend([np_type] * c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def points_from_fileobj(f):
    """ Parse pointcloud coming from file object f
    """
    header = []
    while True:
        ln = f.readline().strip()
        ln_str = ln.decode(encoding='utf-8', errors='strict')
        header.append(ln_str)
        if ln_str.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        raise ValueError('DATA field is neither "ascii" or "binary" or\
                "binary_compressed"')
    return pc_data, metadata

def get_metadata_from_pcd_file(file_path):
    with open(file_path, 'rb') as f:
        header = []
        while True:
            ln = f.readline().strip()
            ln_str = ln.decode(encoding='utf-8', errors='strict')
            header.append(ln_str)
            if ln_str.startswith('DATA'):
                metadata = parse_header(header)
                break
    return metadata

def get_points_from_pcd_file(file_path, num_features=None):
    points, metadata = load_structured_points(file_path)

    fields = [item for item in metadata['fields'] if item != "_"]
    if num_features is None:
        num_features = len(fields)
    points_ = points[fields[0]].reshape(-1, 1)

    i = 0
    for field in fields[1:num_features]:
        points_ = np.hstack((points_, points[field].reshape(-1, 1)))
        i +=1

    return points_, points, metadata['fields']


def load_structured_points(file_path):
    with open(file_path, 'rb') as f:
        points, metadata = points_from_fileobj(f)
    return points, metadata


def save_structured_points(structured, metadata, save_path):
    """将 structured 点云写回 PCD，保留 metadata 中的字段与 DATA 格式。"""
    metadata = _normalize_pcd_metadata(dict(metadata))
    structured = np.ascontiguousarray(structured)
    point_num = int(structured.shape[0])
    metadata["points"] = point_num
    metadata["width"] = point_num
    metadata["height"] = 1

    fields_str = " ".join(metadata["fields"])
    size_str = " ".join(map(str, metadata["size"]))
    type_str = " ".join(metadata["type"])
    count_str = " ".join(map(str, metadata["count"]))
    data_type = metadata["data"]
    viewpoint = metadata.get("viewpoint", [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    vp_str = " ".join(map(str, viewpoint))
    version = metadata.get("version", ".7")

    with open(save_path, "wb") as pcd_file:
        pcd_file.write("# .PCD v0.7 - Point Cloud Data file format\n".encode("utf-8"))
        pcd_file.write("VERSION {}\n".format(version).encode("utf-8"))
        pcd_file.write("FIELDS {}\n".format(fields_str).encode("utf-8"))
        pcd_file.write("SIZE {}\n".format(size_str).encode("utf-8"))
        pcd_file.write("TYPE {}\n".format(type_str).encode("utf-8"))
        pcd_file.write("COUNT {}\n".format(count_str).encode("utf-8"))
        pcd_file.write("WIDTH {}\n".format(point_num).encode("utf-8"))
        pcd_file.write("HEIGHT 1\n".encode("utf-8"))
        pcd_file.write("VIEWPOINT {}\n".format(vp_str).encode("utf-8"))
        pcd_file.write("POINTS {}\n".format(point_num).encode("utf-8"))
        pcd_file.write("DATA {}\n".format(data_type).encode("utf-8"))

        if data_type == "ascii":
            valid_fields = [f for f in metadata["fields"] if f != "_"]
            for i in range(point_num):
                row = [structured[f][i] for f in valid_fields]
                pcd_file.write((" ".join(map(str, row)) + "\n").encode("utf-8"))
        elif data_type == "binary":
            pcd_file.write(structured.tobytes(order="C"))
        elif data_type == "binary_compressed":
            uncompressed_lst = []
            for fieldname in structured.dtype.names:
                column = np.ascontiguousarray(structured[fieldname]).tobytes()
                uncompressed_lst.append(column)
            uncompressed = b"".join(uncompressed_lst)
            uncompressed_size = len(uncompressed)
            buf = lzf.compress(uncompressed)
            if buf is None:
                buf = uncompressed
                compressed_size = uncompressed_size
            else:
                compressed_size = len(buf)
            pcd_file.write(struct.pack("II", compressed_size, uncompressed_size))
            pcd_file.write(buf)
        else:
            raise ValueError(
                'DATA 须为 "ascii"、"binary" 或 "binary_compressed"，当前为 {}'.format(
                    data_type
                )
            )
