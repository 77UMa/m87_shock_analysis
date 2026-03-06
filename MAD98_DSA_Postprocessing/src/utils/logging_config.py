#!/usr/bin/env python3
"""
日志配置模块

该模块提供统一的日志配置和管理功能，支持：
- 文件日志记录
- 控制台日志输出
- 日志级别控制
- 运行时间和状态记录
"""

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir="logs", log_level=logging.INFO, log_name=None, logger_name='DSAWorkflow'):
    """
    设置日志配置

    Args:
        log_dir: 日志目录
        log_level: 日志级别
        log_name: 日志文件名（如果不提供，则使用时间戳）
        logger_name: logger 对象的名称（用于区分不同模块）

    Returns:
        logger: 配置好的日志对象
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 如果没有提供日志文件名，使用时间戳
    if log_name is None:
        log_name = f"dsa_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    log_file = os.path.join(log_dir, log_name)

    # 创建日志对象
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # 清除之前的处理器（避免重复）
    logger.handlers.clear()

    # 创建文件处理器（支持日志轮转，避免单个文件过大）
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # 创建日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志对象
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, log_file


def log_start(logger, snapshot_name, config=None):
    """
    记录任务开始

    Args:
        logger: 日志对象
        snapshot_name: 快照文件名
        config: 配置参数（可选）
    """
    logger.info(f"{'='*60}")
    logger.info(f"开始处理: {snapshot_name}")
    logger.info(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if config:
        logger.info(f"配置参数:")
        for key, value in config.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for k, v in value.items():
                    logger.info(f"    {k}: {v}")
            else:
                logger.info(f"  {key}: {value}")

    logger.info(f"{'='*60}")


def log_finish(logger, snapshot_name, output_file, processing_time):
    """
    记录任务完成

    Args:
        logger: 日志对象
        snapshot_name: 快照文件名
        output_file: 输出文件路径
        processing_time: 处理时间（秒）
    """
    logger.info(f"{'='*60}")
    logger.info(f"完成处理: {snapshot_name}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"处理用时: {processing_time:.2f} 秒")
    logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*60}")


def log_error(logger, snapshot_name, error_msg, exc_info=True):
    """
    记录错误

    Args:
        logger: 日志对象
        snapshot_name: 快照文件名
        error_msg: 错误消息
        exc_info: 是否包含异常信息
    """
    logger.error(f"{'='*60}")
    logger.error(f"处理失败: {snapshot_name}")
    logger.error(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.error(f"错误信息: {error_msg}", exc_info=exc_info)
    logger.error(f"{'='*60}")


def log_summary(logger, total_files, successful_files, failed_files, total_time):
    """
    记录处理摘要

    Args:
        logger: 日志对象
        total_files: 总文件数
        successful_files: 成功文件数
        failed_files: 失败文件数
        total_time: 总处理时间（秒）
    """
    logger.info(f"{'='*60}")
    logger.info(f"处理摘要")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"成功文件数: {successful_files}")
    logger.info(f"失败文件数: {failed_files}")
    logger.info(f"成功率: {successful_files/total_files*100:.1f}%")
    logger.info(f"总处理时间: {total_time:.2f} 秒")
    logger.info(f"平均处理时间: {total_time/total_files:.2f} 秒/文件")
    logger.info(f"{'='*60}")