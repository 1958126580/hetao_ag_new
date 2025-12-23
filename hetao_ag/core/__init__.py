# -*- coding: utf-8 -*-
"""
hetao_ag.core - 核心功能模块
=============================

提供智慧农牧业库的基础组件：单位系统、配置管理、日志工具和通用函数。

本模块是整个库的基础设施层,提供:
    - 严格的SI单位系统,防止单位混淆错误
    - 灵活的配置管理,支持多种配置源
    - 结构化日志记录,便于调试和监控
    - 常用工具函数,提高开发效率

模块组成:
    - units: SI单位系统和物理量类
    - config: 配置管理器
    - logger: 日志工具
    - utils: 通用工具函数

设计原则:
    - 类型安全: 完整的类型注解
    - 防御性编程: 全面的输入验证
    - 性能优化: 关键路径优化
    - 文档完善: 详细的中文文档

作者: Hetao College
版本: 1.0.0
许可证: MIT
"""

from .units import (
    Unit,
    Quantity,
    Dimension,
    DimensionError,
    meters,
    kilometers,
    hectares,
    celsius,
    kilopascals,
    ds_per_m,
    megajoules_per_m2,
    mm_per_day,
)

from .config import (
    ConfigManager,
    ConfigError,
    create_default_config,
)

from .logger import (
    Logger,
    get_logger,
    ColoredFormatter,
)

from .utils import (
    safe_divide,
    clamp,
    linear_interpolate,
    array_interpolate,
    day_of_year,
    degrees_to_radians,
    radians_to_degrees,
    moving_average,
    normalize,
    rmse,
    mae,
    r_squared,
    ensure_path,
    ensure_directory,
    validate_model,
    ValidationResult,
    Timer,
)

__version__ = "1.0.0"
__author__ = "Hetao College"
__license__ = "MIT"

__all__ = [
    # 单位系统
    "Unit",
    "Quantity",
    "Dimension",
    "DimensionError",
    "meters",
    "kilometers",
    "hectares",
    "celsius",
    "kilopascals",
    "ds_per_m",
    "megajoules_per_m2",
    "mm_per_day",
    # 配置管理
    "ConfigManager",
    "ConfigError",
    "create_default_config",
    # 日志工具
    "Logger",
    "get_logger",
    "ColoredFormatter",
    # 通用工具
    "safe_divide",
    "clamp",
    "linear_interpolate",
    "array_interpolate",
    "day_of_year",
    "degrees_to_radians",
    "radians_to_degrees",
    "moving_average",
    "normalize",
    "rmse",
    "mae",
    "r_squared",
    "ensure_path",
    "ensure_directory",
    "validate_model",
    "ValidationResult",
    "Timer",
]
