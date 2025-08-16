#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹存储和管理系统

基于H5MD (HDF5 for Molecular Dynamics) 标准的轨迹存储系统。
提供高效的数据压缩、并行I/O支持和丰富的元数据管理功能。

主要特性：
- HDF5格式的高效存储
- 支持大规模轨迹数据
- 灵活的压缩选项
- 增量式写入支持
- 丰富的元数据记录

Author: Gilbert Young
Created: 2025-08-15
"""

import h5py
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class TrajectoryWriter:
    """
    H5MD格式轨迹文件写入器
    
    遵循H5MD v1.1规范，提供高效的分子动力学轨迹存储。
    
    Parameters
    ----------
    filename : str
        输出文件名（.h5或.hdf5扩展名）
    mode : str
        文件打开模式：'w'(覆盖), 'a'(追加), 'x'(创建新文件)
    compression : str, optional
        压缩算法：'gzip', 'lzf', 'szip'等
    compression_opts : int, optional
        压缩级别（gzip: 1-9）
    chunk_size : int, optional
        块大小，影响I/O性能
    
    Examples
    --------
    >>> writer = TrajectoryWriter('trajectory.h5', compression='gzip')
    >>> writer.initialize(n_atoms=100, n_frames_estimate=1000)
    >>> writer.write_frame(positions, box, time=0.0, step=0)
    >>> writer.close()
    """
    
    def __init__(
        self,
        filename: str,
        mode: str = 'w',
        compression: Optional[str] = 'gzip',
        compression_opts: Optional[int] = 4,
        chunk_size: int = 100
    ):
        self.filename = Path(filename)
        self.mode = mode
        self.compression = compression
        self.compression_opts = compression_opts
        self.chunk_size = chunk_size
        
        self.file = None
        self.n_atoms = None
        self.n_frames = 0
        self.initialized = False
        
        # H5MD组
        self.h5md_group = None
        self.particles_group = None
        self.observables_group = None
        self.parameters_group = None
        
        # 数据集
        self.position_dataset = None
        self.box_dataset = None
        self.time_dataset = None
        self.step_dataset = None
        
        logger.info(f"创建轨迹写入器: {filename}")
        
    def open(self):
        """打开HDF5文件"""
        os.makedirs(self.filename.parent, exist_ok=True)
        self.file = h5py.File(self.filename, self.mode)
        logger.debug(f"打开HDF5文件: {self.filename} (mode={self.mode})")
        
    def initialize(
        self,
        n_atoms: int,
        n_frames_estimate: int = 1000,
        atom_types: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化H5MD文件结构
        
        Parameters
        ----------
        n_atoms : int
            原子数量
        n_frames_estimate : int
            预估帧数（用于优化存储）
        atom_types : list, optional
            原子类型列表
        metadata : dict, optional
            额外的元数据
        """
        if self.file is None:
            self.open()
            
        self.n_atoms = n_atoms
        
        # 创建H5MD根组
        self.h5md_group = self.file.create_group('h5md')
        self.h5md_group.attrs['version'] = np.array([1, 1])
        
        # 创建作者信息
        author_group = self.h5md_group.create_group('author')
        author_group.attrs['name'] = 'ThermoElasticSim'
        author_group.attrs['email'] = 'thermoelasticsim@example.com'
        
        # 创建创建者信息
        creator_group = self.h5md_group.create_group('creator')
        creator_group.attrs['name'] = 'ThermoElasticSim'
        creator_group.attrs['version'] = '4.0.0'
        
        # 创建主要数据组
        self.particles_group = self.file.create_group('particles')
        self.observables_group = self.file.create_group('observables')
        self.parameters_group = self.file.create_group('parameters')
        
        # 创建原子组
        all_atoms = self.particles_group.create_group('all')
        
        # 存储原子类型（如果提供）
        if atom_types:
            species_data = [t.encode('utf-8') for t in atom_types]
            all_atoms.create_dataset(
                'species',
                data=species_data,
                dtype=h5py.string_dtype(encoding='utf-8')
            )
        
        # 创建位置数据集（可扩展）
        position_group = all_atoms.create_group('position')
        
        # 优化的块形状：(chunk_size, n_atoms, 3)
        chunk_shape = (min(self.chunk_size, n_frames_estimate), n_atoms, 3)
        
        self.position_dataset = position_group.create_dataset(
            'value',
            shape=(0, n_atoms, 3),
            maxshape=(None, n_atoms, 3),
            dtype=np.float32,
            chunks=chunk_shape,
            compression=self.compression,
            compression_opts=self.compression_opts
        )
        self.position_dataset.attrs['unit'] = 'Angstrom'
        
        # 时间数据集
        self.time_dataset = position_group.create_dataset(
            'time',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(chunk_shape[0],)
        )
        self.time_dataset.attrs['unit'] = 'ps'
        
        # 步数数据集
        self.step_dataset = position_group.create_dataset(
            'step',
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(chunk_shape[0],)
        )
        
        # 创建盒子数据集
        box_group = all_atoms.create_group('box')
        box_group.attrs['dimension'] = 3
        box_group.attrs['boundary'] = ['periodic', 'periodic', 'periodic']
        
        edges_group = box_group.create_group('edges')
        self.box_dataset = edges_group.create_dataset(
            'value',
            shape=(0, 3, 3),
            maxshape=(None, 3, 3),
            dtype=np.float32,
            chunks=(chunk_shape[0], 3, 3),
            compression=self.compression,
            compression_opts=self.compression_opts
        )
        self.box_dataset.attrs['unit'] = 'Angstrom'
        
        # 存储元数据
        if metadata:
            for key, value in metadata.items():
                self.parameters_group.attrs[key] = value
                
        self.parameters_group.attrs['created'] = datetime.now().isoformat()
        self.parameters_group.attrs['n_atoms'] = n_atoms
        
        self.initialized = True
        logger.info(f"初始化H5MD文件结构完成: {n_atoms}原子")
        
    def write_frame(
        self,
        positions: np.ndarray,
        box: Optional[np.ndarray] = None,
        time: Optional[float] = None,
        step: Optional[int] = None,
        velocities: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
        stress: Optional[np.ndarray] = None,
        energy: Optional[float] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        写入单帧数据
        
        Parameters
        ----------
        positions : np.ndarray
            原子位置 (n_atoms, 3)
        box : np.ndarray, optional
            晶格矢量 (3, 3)
        time : float, optional
            时间（ps）
        step : int, optional
            MD步数
        velocities : np.ndarray, optional
            速度 (n_atoms, 3)
        forces : np.ndarray, optional
            力 (n_atoms, 3)
        stress : np.ndarray, optional
            应力张量 (3, 3)
        energy : float, optional
            总能量
        temperature : float, optional
            温度
        **kwargs
            其他观测量
        """
        if not self.initialized:
            raise RuntimeError("必须先调用initialize()初始化文件结构")
            
        # 扩展数据集
        frame_idx = self.n_frames
        self.position_dataset.resize((frame_idx + 1, self.n_atoms, 3))
        self.position_dataset[frame_idx] = positions.astype(np.float32)
        
        if box is not None:
            self.box_dataset.resize((frame_idx + 1, 3, 3))
            self.box_dataset[frame_idx] = box.astype(np.float32)
            
        if time is not None:
            self.time_dataset.resize((frame_idx + 1,))
            self.time_dataset[frame_idx] = time
            
        if step is not None:
            self.step_dataset.resize((frame_idx + 1,))
            self.step_dataset[frame_idx] = step
            
        # 写入可选数据
        if velocities is not None:
            self._write_optional_data('velocities', velocities, frame_idx, unit='Angstrom/ps')
            
        if forces is not None:
            self._write_optional_data('forces', forces, frame_idx, unit='eV/Angstrom')
            
        # 写入观测量
        if stress is not None:
            self._write_observable('stress', stress, frame_idx, unit='GPa')
            
        if energy is not None:
            self._write_observable('energy', energy, frame_idx, unit='eV')
            
        if temperature is not None:
            self._write_observable('temperature', temperature, frame_idx, unit='K')
            
        # 写入额外观测量
        for key, value in kwargs.items():
            self._write_observable(key, value, frame_idx)
            
        self.n_frames += 1
        
        # 定期刷新缓冲区
        if self.n_frames % self.chunk_size == 0:
            self.file.flush()
            logger.debug(f"刷新缓冲区，已写入{self.n_frames}帧")
            
    def _write_optional_data(self, name: str, data: np.ndarray, frame_idx: int, unit: str = ''):
        """写入可选的粒子数据"""
        all_atoms = self.particles_group['all']
        
        if name not in all_atoms:
            group = all_atoms.create_group(name)
            chunk_shape = (min(self.chunk_size, 1000), self.n_atoms, 3)
            dataset = group.create_dataset(
                'value',
                shape=(0, self.n_atoms, 3),
                maxshape=(None, self.n_atoms, 3),
                dtype=np.float32,
                chunks=chunk_shape,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            if unit:
                dataset.attrs['unit'] = unit
        else:
            dataset = all_atoms[name]['value']
            
        dataset.resize((frame_idx + 1, self.n_atoms, 3))
        dataset[frame_idx] = data.astype(np.float32)
        
    def _write_observable(self, name: str, data: Union[float, np.ndarray, str], frame_idx: int, unit: str = ''):
        """写入观测量数据"""
        if name not in self.observables_group:
            if isinstance(data, str):
                # 处理字符串数据
                shape = (0,)
                maxshape = (None,)
                chunk_shape = (self.chunk_size,)
                dtype = h5py.string_dtype(encoding='utf-8')
            elif isinstance(data, (int, float)):
                shape = (0,)
                maxshape = (None,)
                chunk_shape = (self.chunk_size,)
                dtype = np.float64
            else:
                data = np.asarray(data)
                shape = (0,) + data.shape
                maxshape = (None,) + data.shape
                chunk_shape = (min(self.chunk_size, 1000),) + data.shape
                dtype = data.dtype
                
            dataset = self.observables_group.create_dataset(
                name,
                shape=shape,
                maxshape=maxshape,
                dtype=dtype,
                chunks=chunk_shape,
                compression=self.compression if len(shape) > 1 else None
            )
            if unit:
                dataset.attrs['unit'] = unit
        else:
            dataset = self.observables_group[name]
            
        if isinstance(data, str):
            dataset.resize((frame_idx + 1,))
            dataset[frame_idx] = data
        elif isinstance(data, (int, float)):
            dataset.resize((frame_idx + 1,))
            dataset[frame_idx] = data
        else:
            new_shape = (frame_idx + 1,) + data.shape
            dataset.resize(new_shape)
            dataset[frame_idx] = data
            
    def write_metadata(self, metadata: Dict[str, Any]):
        """
        写入额外的元数据
        
        Parameters
        ----------
        metadata : dict
            元数据字典
        """
        for key, value in metadata.items():
            self.parameters_group.attrs[key] = value
            
    def close(self):
        """关闭文件"""
        if self.file:
            self.file.attrs['n_frames'] = self.n_frames
            self.file.flush()
            self.file.close()
            logger.info(f"关闭轨迹文件，共写入{self.n_frames}帧")
            

class TrajectoryReader:
    """
    H5MD格式轨迹文件读取器
    
    提供高效的轨迹数据读取和分析功能。
    
    Parameters
    ----------
    filename : str
        输入文件名
    mode : str
        读取模式：'r'(只读), 'r+'(读写)
    cache_size : int
        内存缓存帧数
    
    Examples
    --------
    >>> reader = TrajectoryReader('trajectory.h5')
    >>> reader.open()
    >>> for frame_idx in range(reader.n_frames):
    ...     frame = reader.read_frame(frame_idx)
    ...     process_frame(frame)
    >>> reader.close()
    """
    
    def __init__(self, filename: str, mode: str = 'r', cache_size: int = 100):
        self.filename = Path(filename)
        self.mode = mode
        self.cache_size = cache_size
        
        self.file = None
        self.n_atoms = None
        self.n_frames = None
        
        # 缓存
        self._cache = {}
        self._cache_range = None
        
        logger.info(f"创建轨迹读取器: {filename}")
        
    def open(self):
        """打开HDF5文件"""
        if not self.filename.exists():
            raise FileNotFoundError(f"轨迹文件不存在: {self.filename}")
            
        self.file = h5py.File(self.filename, self.mode)
        
        # 读取基本信息
        if 'parameters' in self.file:
            self.n_atoms = self.file['parameters'].attrs.get('n_atoms')
            
        if 'particles/all/position/value' in self.file:
            pos_data = self.file['particles/all/position/value']
            self.n_frames = pos_data.shape[0]
            if self.n_atoms is None:
                self.n_atoms = pos_data.shape[1]
                
        logger.info(f"打开轨迹文件: {self.n_frames}帧, {self.n_atoms}原子")
        
    def read_frame(self, frame_idx: int) -> Dict[str, Any]:
        """
        读取指定帧
        
        Parameters
        ----------
        frame_idx : int
            帧索引
            
        Returns
        -------
        dict
            帧数据字典
        """
        if frame_idx < 0 or frame_idx >= self.n_frames:
            raise IndexError(f"帧索引超出范围: {frame_idx}")
            
        frame_data = {}
        
        # 读取位置
        positions = self.file['particles/all/position/value'][frame_idx]
        frame_data['positions'] = positions
        
        # 读取时间和步数
        if 'particles/all/position/time' in self.file:
            time_dataset = self.file['particles/all/position/time']
            if len(time_dataset) > frame_idx:
                frame_data['time'] = time_dataset[frame_idx]
            
        if 'particles/all/position/step' in self.file:
            step_dataset = self.file['particles/all/position/step']
            if len(step_dataset) > frame_idx:
                frame_data['step'] = step_dataset[frame_idx]
            
        # 读取盒子
        if 'particles/all/box/edges/value' in self.file:
            frame_data['box'] = self.file['particles/all/box/edges/value'][frame_idx]
            
        # 读取速度和力（如果存在）
        if 'particles/all/velocities/value' in self.file:
            frame_data['velocities'] = self.file['particles/all/velocities/value'][frame_idx]
            
        if 'particles/all/forces/value' in self.file:
            frame_data['forces'] = self.file['particles/all/forces/value'][frame_idx]
            
        # 读取观测量
        if 'observables' in self.file:
            for key in self.file['observables'].keys():
                frame_data[key] = self.file['observables'][key][frame_idx]
                
        return frame_data
        
    def read_frames(self, start: int = 0, stop: Optional[int] = None, stride: int = 1) -> List[Dict[str, Any]]:
        """
        批量读取多帧
        
        Parameters
        ----------
        start : int
            起始帧
        stop : int, optional
            结束帧（不包含）
        stride : int
            步长
            
        Returns
        -------
        list
            帧数据列表
        """
        if stop is None:
            stop = self.n_frames
            
        frames = []
        for idx in range(start, stop, stride):
            frames.append(self.read_frame(idx))
            
        return frames
        
    def get_metadata(self) -> Dict[str, Any]:
        """获取元数据"""
        metadata = {}
        
        if 'h5md' in self.file:
            h5md = self.file['h5md']
            metadata['h5md_version'] = h5md.attrs.get('version')
            
            if 'author' in h5md:
                metadata['author'] = dict(h5md['author'].attrs)
                
            if 'creator' in h5md:
                metadata['creator'] = dict(h5md['creator'].attrs)
                
        if 'parameters' in self.file:
            metadata['parameters'] = dict(self.file['parameters'].attrs)
            
        return metadata
        
    def get_trajectory_info(self) -> Dict[str, Any]:
        """获取轨迹概要信息"""
        info = {
            'n_frames': self.n_frames,
            'n_atoms': self.n_atoms,
            'filename': str(self.filename),
            'file_size': self.filename.stat().st_size / (1024**2),  # MB
        }
        
        # 获取可用数据类型
        available_data = []
        
        if 'particles/all/position/value' in self.file:
            available_data.append('positions')
            
        if 'particles/all/velocities/value' in self.file:
            available_data.append('velocities')
            
        if 'particles/all/forces/value' in self.file:
            available_data.append('forces')
            
        if 'particles/all/box/edges/value' in self.file:
            available_data.append('box')
            
        if 'observables' in self.file:
            available_data.extend(self.file['observables'].keys())
            
        info['available_data'] = available_data
        
        # 获取时间范围
        if 'particles/all/position/time' in self.file:
            times = self.file['particles/all/position/time']
            if len(times) > 0:
                info['time_range'] = (times[0], times[-1])
                info['time_unit'] = times.attrs.get('unit', 'unknown')
            else:
                info['time_range'] = (0, 0)
                info['time_unit'] = 'unknown'
            
        return info
        
    def close(self):
        """关闭文件"""
        if self.file:
            self.file.close()
            logger.info("关闭轨迹文件")


# 便捷函数
def save_trajectory(
    filename: str,
    positions_list: List[np.ndarray],
    boxes_list: Optional[List[np.ndarray]] = None,
    times: Optional[List[float]] = None,
    **kwargs
) -> None:
    """
    便捷函数：保存轨迹数据
    
    Parameters
    ----------
    filename : str
        输出文件名
    positions_list : list
        位置列表
    boxes_list : list, optional
        晶格矢量列表
    times : list, optional
        时间列表
    **kwargs
        其他数据
    """
    writer = TrajectoryWriter(filename)
    
    n_atoms = positions_list[0].shape[0]
    n_frames = len(positions_list)
    
    writer.initialize(n_atoms, n_frames)
    
    for i, positions in enumerate(positions_list):
        box = boxes_list[i] if boxes_list else None
        time = times[i] if times else i * 1.0
        
        frame_kwargs = {}
        for key, values in kwargs.items():
            if isinstance(values, list) and len(values) > i:
                frame_kwargs[key] = values[i]
                
        writer.write_frame(positions, box=box, time=time, step=i, **frame_kwargs)
        
    writer.close()
    

def load_trajectory(filename: str, start: int = 0, stop: Optional[int] = None, stride: int = 1) -> Dict[str, Any]:
    """
    便捷函数：加载轨迹数据
    
    Parameters
    ----------
    filename : str
        输入文件名
    start : int
        起始帧
    stop : int, optional
        结束帧
    stride : int
        步长
        
    Returns
    -------
    dict
        轨迹数据字典
    """
    reader = TrajectoryReader(filename)
    reader.open()
    
    info = reader.get_trajectory_info()
    frames = reader.read_frames(start, stop, stride)
    metadata = reader.get_metadata()
    
    reader.close()
    
    return {
        'frames': frames,
        'info': info,
        'metadata': metadata
    }