"""
FLQ (Federated Low-bitwidth Quantization) 算法实现
基于论文《Federated Optimal Framework with Low-bitwidth Quantization for Distribution System》
"""

import numpy as np
import torch
import torch.nn as nn

def flq_sign_quantization(gradients):
    """
    FLQ符号量化（二进制梯度量化）
    基于论文公式 (15): G_B^m(θ^k) = |∇f_m(θ^k)|_l * sign(∇f_m(θ^k))
    
    Args:
        gradients: numpy数组，梯度向量
        
    Returns:
        quantized: 量化后的梯度
        scale_factor: 缩放因子
        logical_bits: 理论比特数（每参数1位）
    """
    # 计算符号
    signs = np.sign(gradients)
    
    # 计算缩放因子（使用平均绝对值，如论文公式(8)）
    scale_factor = np.mean(np.abs(gradients))
    
    # 量化：符号 × 缩放因子
    quantized = signs * scale_factor
    
    # 理论比特数：每个参数1位
    logical_bits = len(gradients)
    
    return quantized, scale_factor, logical_bits

def flq_8bit_quantization(gradients):
    """
    FLQ 8位量化，基于论文的Lloyd-Max量化原理
    
    Args:
        gradients: numpy数组，梯度向量
        
    Returns:
        quantized: 量化后的梯度
        scale_factor: 缩放因子
        zero_point: 零点偏移
        logical_bits: 理论比特数（每参数8位）
    """
    # 找到梯度的最大绝对值
    max_abs = np.max(np.abs(gradients))
    
    if max_abs == 0:
        return gradients, 1.0, 0, 8 * len(gradients)
    
    # 对称量化到[-127, 127]范围（8位有符号整数）
    scale_factor = max_abs / 127.0
    
    # 量化到int8
    quantized_int8 = np.clip(np.round(gradients / scale_factor), -128, 127)
    
    # 反量化回FP32
    quantized = quantized_int8 * scale_factor
    
    # 理论比特数：每个参数8位
    logical_bits = 8 * len(gradients)
    
    return quantized, scale_factor, 0, logical_bits

def flq_4bit_quantization(gradients):
    """
    FLQ 4位量化，基于论文的均匀量化原理
    
    Args:
        gradients: numpy数组，梯度向量
        
    Returns:
        quantized: 量化后的梯度
        scale_factor: 缩放因子
        zero_point: 零点偏移
        logical_bits: 理论比特数（每参数4位）
    """
    # 找到梯度的最大绝对值
    max_abs = np.max(np.abs(gradients))
    
    if max_abs == 0:
        return gradients, 1.0, 0, 4 * len(gradients)
    
    # 对称量化到[-7, 7]范围（4位有符号整数）
    scale_factor = max_abs / 7.0
    
    # 量化到int4（-8到7）
    quantized_int4 = np.clip(np.round(gradients / scale_factor), -8, 7)
    
    # 反量化回FP32
    quantized = quantized_int4 * scale_factor
    
    # 理论比特数：每个参数4位
    logical_bits = 4 * len(gradients)
    
    return quantized, scale_factor, 0, logical_bits

def flq_binary_gradient_quantization(gradients, reference_gradients=None):
    """
    论文中的二进制梯度量化算法（第IV节）
    基于公式 (16) 的量化方法
    
    Args:
        gradients: 当前梯度
        reference_gradients: 参考梯度（上一轮的量化梯度）
        
    Returns:
        quantized: 量化后的梯度
        logical_bits: 理论比特数
    """
    if reference_gradients is None:
        # 如果没有参考梯度，使用符号量化
        return flq_sign_quantization(gradients)
    
    # 计算梯度创新（当前梯度与参考梯度的差值）
    gradient_innovation = gradients - reference_gradients
    
    # 计算量化半径
    R_k = np.max(np.abs(gradient_innovation))
    
    if R_k == 0:
        return reference_gradients, 1 * len(gradients)
    
    # 使用1位量化（二进制）
    # 量化粒度：2^(-b+1)，其中b=1
    quantization_granularity = 1.0  # 2^(-1+1) = 1
    
    # 公式 (16) 的简化版本：量化到{-1, 1}
    quantized_innovation = np.sign(gradient_innovation)
    
    # 反量化
    quantized = reference_gradients + quantized_innovation * R_k
    
    # 理论比特数：每个参数1位
    logical_bits = 1 * len(gradients)
    
    return quantized, R_k, logical_bits

def flq_adaptive_quantization(gradients, target_compression_ratio=8):
    """
    自适应FLQ量化，根据目标压缩比选择量化方案
    
    Args:
        gradients: 梯度向量
        target_compression_ratio: 目标压缩比（相对于32位FP32）
        
    Returns:
        quantized: 量化后的梯度
        scale_factor: 缩放因子
        bits_per_param: 每参数比特数
        logical_bits: 总理论比特数
    """
    if target_compression_ratio >= 32:
        # 符号量化（1位）
        return flq_sign_quantization(gradients)
    elif target_compression_ratio >= 4:
        # 8位量化
        quantized, scale, _, logical_bits = flq_8bit_quantization(gradients)
        return quantized, scale, 8, logical_bits
    elif target_compression_ratio >= 2:
        # 4位量化
        quantized, scale, _, logical_bits = flq_4bit_quantization(gradients)
        return quantized, scale, 4, logical_bits
    else:
        # 8位量化作为默认
        quantized, scale, _, logical_bits = flq_8bit_quantization(gradients)
        return quantized, scale, 8, logical_bits

class FLQQuantizer:
    """FLQ量化器类，支持多种量化模式"""
    
    def __init__(self, mode="sign", reference_gradients=None):
        """
        初始化FLQ量化器
        
        Args:
            mode: 量化模式 ("sign", "8bit", "4bit", "binary", "adaptive")
            reference_gradients: 参考梯度（用于二进制量化）
        """
        self.mode = mode
        self.reference_gradients = reference_gradients
        self.quantization_history = []
    
    def quantize(self, gradients):
        """
        量化梯度
        
        Args:
            gradients: numpy数组，梯度向量
            
        Returns:
            quantized: 量化后的梯度
            info: 量化信息字典
        """
        if self.mode == "sign":
            quantized, scale, logical_bits = flq_sign_quantization(gradients)
            info = {
                "scale_factor": scale,
                "bits_per_param": 1,
                "logical_bits": logical_bits,
                "compression_ratio": 32.0
            }
        elif self.mode == "8bit":
            quantized, scale, zero_point, logical_bits = flq_8bit_quantization(gradients)
            info = {
                "scale_factor": scale,
                "zero_point": zero_point,
                "bits_per_param": 8,
                "logical_bits": logical_bits,
                "compression_ratio": 4.0
            }
        elif self.mode == "4bit":
            quantized, scale, zero_point, logical_bits = flq_4bit_quantization(gradients)
            info = {
                "scale_factor": scale,
                "zero_point": zero_point,
                "bits_per_param": 4,
                "logical_bits": logical_bits,
                "compression_ratio": 8.0
            }
        elif self.mode == "binary":
            quantized, scale, logical_bits = flq_binary_gradient_quantization(gradients, self.reference_gradients)
            info = {
                "scale_factor": scale,
                "bits_per_param": 1,
                "logical_bits": logical_bits,
                "compression_ratio": 32.0,
                "reference_updated": True
            }
            self.reference_gradients = quantized  # 更新参考梯度
        elif self.mode == "adaptive":
            quantized, scale, bits_per_param, logical_bits = flq_adaptive_quantization(gradients)
            compression_ratio = 32.0 / bits_per_param
            info = {
                "scale_factor": scale,
                "bits_per_param": bits_per_param,
                "logical_bits": logical_bits,
                "compression_ratio": compression_ratio
            }
        else:
            raise ValueError(f"Unknown quantization mode: {self.mode}")
        
        # 记录量化历史
        self.quantization_history.append({
            "original_norm": np.linalg.norm(gradients),
            "quantized_norm": np.linalg.norm(quantized),
            "quantization_error": np.linalg.norm(gradients - quantized),
            "info": info
        })
        
        return quantized, info
    
    def get_quantization_stats(self):
        """获取量化统计信息"""
        if not self.quantization_history:
            return None
        
        stats = {
            "total_rounds": len(self.quantization_history),
            "avg_compression_ratio": np.mean([h["info"]["compression_ratio"] for h in self.quantization_history]),
            "avg_quantization_error": np.mean([h["quantization_error"] for h in self.quantization_history]),
            "total_bits_saved": sum([h["info"]["logical_bits"] * (32 - h["info"]["bits_per_param"]) // 32 
                                   for h in self.quantization_history])
        }
        
        return stats