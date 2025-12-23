# 河套智慧农牧业库 (hetao_ag)

<p align="center">
  <strong>面向智慧农业和畜牧管理的综合Python库</strong>
</p>

<p align="center">
  <a href="#功能特性">功能特性</a> |
  <a href="#安装指南">安装指南</a> |
  <a href="#快速开始">快速开始</a> |
  <a href="#模块说明">模块说明</a> |
  <a href="#文档">文档</a> |
  <a href="#示例">示例</a>
</p>

---

## 项目概述

**hetao_ag** 是一个专为智慧农业和畜牧管理设计的综合 Python 库,特别针对河套灌区等干旱半干旱地区。本库提供了涵盖土壤建模、水循环管理、作物生长模拟、畜牧监测、遥感分析和农场优化的集成解决方案。

### 核心亮点

- **科学严谨**: 基于国际公认模型(FAO-56、Maas-Hoffman、van Genuchten)
- **SI 单位系统**: 自动单位转换,防止单位混淆错误
- **模块化设计**: 各模块可独立使用或组合使用
- **中文优先文档**: 完整的中文注释和文档
- **生产就绪**: 全面的测试覆盖和实际验证

---

## 功能特性

| 模块          | 功能                              | 应用场景                 |
| ------------- | --------------------------------- | ------------------------ |
| **core**      | 单位系统、配置管理、日志记录      | 科学计算标准化、实验追踪 |
| **soil**      | 土壤水分/盐分建模、传感器校准     | 土壤墒情监测、盐碱地管理 |
| **water**     | FAO-56 蒸散发、水量平衡、灌溉调度 | 精准灌溉、水资源管理     |
| **crop**      | 作物生长模拟、胁迫响应、物候跟踪  | 产量预测、种植决策       |
| **livestock** | 动物检测、行为分类、健康监测      | 智慧牧场、疫病预警       |
| **space**     | 遥感指数、影像处理、物候分类      | 作物监测、植被制图       |
| **opt**       | 线性规划、遗传算法、场景分析      | 资源优化、决策支持       |

---

## 安装指南

### 系统要求

- Python 3.10 或更高版本
- 操作系统: Windows / Linux / macOS

### 基础安装

```bash
# 克隆仓库
git clone https://github.com/1958126580/hetao_ag_new.git
cd hetao_ag_new

# 安装基础版本(仅需numpy)
pip install -e .
```

### 完整安装

```bash
# 安装所有可选依赖
pip install -e ".[full]"
```

### 模块化安装

```bash
# 遥感支持(rasterio等)
pip install -e ".[space]"

# 畜牧AI支持(torch、opencv)
pip install -e ".[livestock]"

# 优化支持(pulp、scipy)
pip install -e ".[opt]"
```

### 验证安装

```python
import hetao_ag as hag
print(hag.__version__)  # 输出: 1.0.0
```

---

## 快速开始

### 5 分钟入门教程

```python
import hetao_ag as hag
import numpy as np

# ============================================
# 1. 计算参考蒸散发
# ============================================
from hetao_ag.water import eto_penman_monteith, WeatherData

weather = WeatherData(
    t_mean=25.0, t_max=32.0, t_min=18.0,
    rh=55.0, u2=2.0, rs=22.0,
    elevation=1050, latitude=40.8, doy=180
)
et0 = eto_penman_monteith(weather)
print(f"参考蒸散发(ET0): {et0:.2f} mm/day")

# ============================================
# 2. 土壤水分模拟
# ============================================
from hetao_ag.soil import SoilMoistureModel, SoilType

soil = SoilMoistureModel(
    field_capacity=0.32,
    wilting_point=0.12,
    initial_moisture=0.25,
    soil_type=SoilType.LOAM
)

# 模拟日水量平衡
result = soil.step_day(rain_mm=15, irrigation_mm=0, et_mm=5)
print(f"土壤含水量: {result['moisture']:.3f}")
print(f"水分胁迫因子: {soil.stress_factor:.3f}")

# ============================================
# 3. 作物盐分胁迫评估
# ============================================
from hetao_ag.crop import yield_reduction_salinity_crop, classify_salt_tolerance

ECe = 6.0  # 土壤电导率(dS/m)
rel_yield = yield_reduction_salinity_crop(ECe, crop="wheat")
tolerance = classify_salt_tolerance("wheat")

print(f"小麦在ECe={ECe}时的相对产量: {rel_yield*100:.1f}%")
print(f"小麦耐盐性: {tolerance}")

# ============================================
# 4. 遥感植被指数
# ============================================
from hetao_ag.space import compute_ndvi, classify_vegetation_health

red = np.array([[120, 130], [110, 90]], dtype=np.uint16)
nir = np.array([[200, 210], [180, 160]], dtype=np.uint16)

ndvi = compute_ndvi(red, nir)
print(f"NDVI平均值: {ndvi.mean():.3f}")
print(f"植被状况: {classify_vegetation_health(ndvi.mean())}")

# ============================================
# 5. 灌溉调度
# ============================================
from hetao_ag.water import IrrigationScheduler, ScheduleType

scheduler = IrrigationScheduler(
    method=ScheduleType.SOIL_MOISTURE,
    trigger_threshold=0.5
)

recommendation = scheduler.recommend_by_moisture(
    current_moisture=0.18,
    field_capacity=0.32,
    wilting_point=0.12
)

if recommendation.should_irrigate:
    print(f"建议灌溉: {recommendation.amount_mm:.1f} mm")
    print(f"原因: {recommendation.reason}")
    print(f"紧急程度: {recommendation.urgency}")
```

---

## 模块说明

### 核心模块 (`hetao_ag.core`)

库的基础设施,提供必要的核心功能。

#### 单位系统

```python
from hetao_ag.core import Quantity, Unit, meters, hectares, celsius

# 创建物理量
length = meters(100)
area = hectares(50)
temp = celsius(25)

# 单位转换
length_km = length.to(Unit.KILOMETER)
print(length_km)  # 0.1 km

# 算术运算
total = meters(100) + Quantity(0.5, Unit.KILOMETER)
print(total)  # 600 m
```

#### 配置管理

```python
from hetao_ag.core import ConfigManager, create_default_config

config = ConfigManager(defaults=create_default_config())

# 嵌套键访问
fc = config.get("soil.field_capacity")
et_method = config.get("water.et_method", default="penman_monteith")

# 动态配置
config.set("irrigation.efficiency", 0.90)
```

#### 日志记录

```python
from hetao_ag.core import get_logger

logger = get_logger("experiment")
logger.info("开始模拟", soil_ec=4.5, irrigation=True)
logger.log_experiment_start("产量预测", parameters={"model": "CropModel"})
```

---

### 土壤模块 (`hetao_ag.soil`)

全面的土壤水分和盐分建模。

#### 土壤水分模型

```python
from hetao_ag.soil import SoilMoistureModel, SoilType

model = SoilMoistureModel(
    field_capacity=0.32,
    wilting_point=0.12,
    initial_moisture=0.25,
    soil_type=SoilType.LOAM
)

# 多日模拟
for day in range(10):
    result = model.step_day(rain_mm=5 if day % 3 == 0 else 0, et_mm=4)
    print(f"第{day+1}天: 含水量={result['moisture']:.3f}")

# 关键属性
print(f"当前胁迫因子: {model.stress_factor:.2f}")
print(f"需灌溉量: {model.irrigation_need_mm:.1f} mm")
```

#### 盐分模型

```python
from hetao_ag.soil import SalinityModel, classify_soil_salinity

model = SalinityModel(initial_ECe=4.0, root_depth_m=0.3)

# 含盐水灌溉
model.irrigate(amount_mm=60, ec_water=1.5)
print(f"灌溉后EC: {model.ECe:.2f} dS/m")

# 淋洗
model.leach(drainage_mm=40)
print(f"淋洗后EC: {model.ECe:.2f} dS/m")
print(f"盐分等级: {classify_soil_salinity(model.ECe)}")
```

#### 传感器校准

```python
from hetao_ag.soil import SensorCalibrator
import numpy as np

calibrator = SensorCalibrator()

# 使用田间数据校准
raw_readings = np.array([300, 450, 600, 750])
ground_truth = np.array([0.10, 0.20, 0.30, 0.40])

result = calibrator.auto_calibrate(raw_readings, ground_truth)
print(f"最佳方法: {result.method.value}, R²={result.r_squared:.4f}")

# 应用校准
calibrated = result.apply(500)
print(f"原始值500 -> 校准值{calibrated:.3f}")
```

---

### 水循环模块 (`hetao_ag.water`)

符合 FAO-56 标准的水循环管理。

#### 蒸散发计算

```python
from hetao_ag.water import (
    eto_penman_monteith, eto_hargreaves,
    WeatherData, crop_coefficient, etc_crop,
    extraterrestrial_radiation
)

# 完整气象数据 - Penman-Monteith方法
weather = WeatherData(
    t_mean=25, t_max=32, t_min=18,
    rh=55, u2=2.0, rs=22,
    elevation=1050, latitude=40.8, doy=180
)
et0_pm = eto_penman_monteith(weather)

# 有限数据 - Hargreaves方法
Ra = extraterrestrial_radiation(latitude=40.8, doy=180)
et0_hg = eto_hargreaves(25, 32, 18, Ra)

print(f"Penman-Monteith ET0: {et0_pm:.2f} mm/day")
print(f"Hargreaves ET0: {et0_hg:.2f} mm/day")

# 作物蒸散发
kc = crop_coefficient("mid", "wheat")
etc = etc_crop(et0_pm, kc)
print(f"小麦ETc(中期): {etc:.2f} mm/day")
```

#### 水量平衡

```python
from hetao_ag.water import WaterBalance

wb = WaterBalance(
    initial_storage_mm=80,
    max_storage_mm=120,
    min_storage_mm=40
)

# 10天模拟
weather_data = [
    {"precip": 0, "et": 5}, {"precip": 15, "et": 4},
    {"precip": 0, "et": 6}, {"precip": 0, "et": 5},
    {"precip": 0, "et": 5}, {"precip": 30, "et": 3},
    {"precip": 0, "et": 4}, {"precip": 0, "et": 6},
    {"precip": 0, "et": 5}, {"precip": 0, "et": 5},
]

for i, w in enumerate(weather_data, 1):
    record = wb.step_day(precip_mm=w["precip"], et_mm=w["et"])
    print(f"第{i}天: 储水量={record.soil_moisture:.1f}mm")

# 汇总
summary = wb.get_summary()
print(f"\n总降水量: {summary['total_precipitation_mm']:.1f} mm")
print(f"总蒸散发: {summary['total_et_mm']:.1f} mm")
```

---

### 作物模块 (`hetao_ag.crop`)

作物生长模拟和胁迫分析。

#### 胁迫响应模型

```python
from hetao_ag.crop import (
    yield_reduction_salinity_crop,
    water_stress_from_moisture,
    combined_stress_factor,
    yield_with_stress,
    CROP_SALT_TOLERANCE
)

# 盐分胁迫
ECe = 8.0
for crop in ["wheat", "maize", "cotton", "barley"]:
    rel_yield = yield_reduction_salinity_crop(ECe, crop)
    print(f"{crop}: ECe={ECe} dS/m时产量{rel_yield*100:.1f}%")

# 水分胁迫
ks_water = water_stress_from_moisture(
    soil_moisture=0.18,
    field_capacity=0.32,
    wilting_point=0.12
)

# 组合胁迫
ks_combined = combined_stress_factor(ks_water, rel_yield)
actual_yield = yield_with_stress(
    potential_yield=6000,  # kg/ha
    water_stress=ks_water,
    salinity_stress=rel_yield
)
print(f"\n实际产量: {actual_yield:.0f} kg/ha")
```

#### 物候期跟踪

```python
from hetao_ag.crop import PhenologyTracker, GrowthStage

tracker = PhenologyTracker("wheat")

# 累积生长度日
import numpy as np
np.random.seed(42)

for day in range(100):
    t_max = 25 + 8 * np.sin(day / 100 * np.pi) + np.random.randn()
    t_min = t_max - 10
    tracker.accumulate_gdd(t_max, t_min)

    if day % 20 == 0:
        print(f"第{day}天: GDD={tracker.accumulated_gdd:.0f}, "
              f"阶段={tracker.current_stage.value}")

print(f"\n成熟进度: {tracker.progress_to_maturity()*100:.1f}%")
print(f"当前Kc: {tracker.get_kc_for_stage()}")
```

#### 完整作物模型

```python
from hetao_ag.crop import CropModel

model = CropModel("wheat")

# 生长季模拟
for day in range(120):
    result = model.update_daily(
        t_max=25, t_min=15, et=5,
        soil_moisture=0.25,
        field_capacity=0.32,
        wilting_point=0.12,
        ECe=3.0
    )

print(f"最终生物量: {model.accumulated_biomass:.0f} kg/ha")
print(f"估算产量: {model.estimate_yield():.0f} kg/ha")
print(f"平均胁迫: {np.mean(model.stress_history):.3f}")
```

---

### 畜牧模块 (`hetao_ag.livestock`)

智能畜牧监测和健康管理。

#### 动物检测

```python
from hetao_ag.livestock import AnimalDetector

detector = AnimalDetector(
    model_name="yolov5s",
    confidence_threshold=0.5
)

# 检测图像中的动物
detections = detector.detect("farm_image.jpg")
for det in detections:
    print(f"检测到{det.label},位置{det.bbox},置信度={det.confidence:.2f}")

# 按物种计数
counts = detector.count_animals("farm_image.jpg")
print(f"动物数量: {counts}")
```

#### 行为分类

```python
from hetao_ag.livestock import BehaviorClassifier, AnimalBehavior

classifier = BehaviorClassifier()

# 根据运动特征分类
behavior = classifier.classify_from_motion(
    motion_magnitude=0.3,
    head_position="down"
)
print(f"检测到的行为: {behavior.value}")
```

#### 健康监测

```python
from hetao_ag.livestock import HealthMonitor, HerdHealthMonitor

# 个体监测
monitor = HealthMonitor("cow_001")

# 建立基线(7天正常数据)
import numpy as np
for _ in range(7):
    monitor.update_activity(100 + np.random.randn() * 5)
    monitor.update_feeding_time(240 + np.random.randn() * 10)

# 检测异常
monitor.update_activity(60)  # 活动量降低
monitor.update_temperature(40.2)  # 发烧

alerts = monitor.check_health()
for alert in alerts:
    print(f"[{alert.severity.value}] {alert.alert_type.value}: {alert.message}")

# 群体监测
herd = HerdHealthMonitor()
for i in range(10):
    herd.add_animal(f"cow_{i:03d}")

summary = herd.get_summary()
print(f"群体健康摘要: {summary}")
```

---

### 遥感模块 (`hetao_ag.space`)

遥感和植被分析。

#### 光谱指数

```python
from hetao_ag.space import (
    compute_ndvi, compute_savi, compute_evi,
    compute_lswi, compute_ndwi,
    classify_vegetation_health
)
import numpy as np

# 多光谱数据样本
blue = np.random.randint(50, 150, (100, 100), dtype=np.uint16)
green = np.random.randint(80, 180, (100, 100), dtype=np.uint16)
red = np.random.randint(100, 200, (100, 100), dtype=np.uint16)
nir = np.random.randint(150, 250, (100, 100), dtype=np.uint16)
swir = np.random.randint(100, 200, (100, 100), dtype=np.uint16)

# 计算指数
ndvi = compute_ndvi(red, nir)
savi = compute_savi(red, nir, L=0.5)
evi = compute_evi(blue, red, nir)
lswi = compute_lswi(nir, swir)

print(f"NDVI范围: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
print(f"SAVI平均: {savi.mean():.3f}")
print(f"EVI平均: {evi.mean():.3f}")
```

#### 物候分类

```python
from hetao_ag.space import PhenologyClassifier, temporal_smoothing

# 时间序列NDVI数据(12个月,100x100像素)
ndvi_series = np.random.rand(12, 100, 100) * 0.5 + 0.2

# 平滑时间序列
smoothed = temporal_smoothing(ndvi_series, window=3)

# 作物分类
classifier = PhenologyClassifier(smoothed)
crop_map = classifier.classify_crops(n_classes=3)

# 提取像素特征
features = classifier.extract_features(50, 50)
print(f"峰值NDVI: {features.peak_value:.3f}")
print(f"峰值时间: 第{features.peak_time + 1}月")
print(f"生长季长度: {features.end_of_season - features.start_of_season}个月")
```

---

### 优化模块 (`hetao_ag.opt`)

资源优化和决策支持。

#### 线性规划

```python
from hetao_ag.opt import LinearOptimizer, optimize_crop_mix

# 定义作物经济参数
crops = [
    {"name": "wheat", "profit_per_ha": 500, "water_per_ha": 3000},
    {"name": "maize", "profit_per_ha": 600, "water_per_ha": 5000},
    {"name": "alfalfa", "profit_per_ha": 400, "water_per_ha": 2000},
]

# 优化配置
solution = optimize_crop_mix(
    crops,
    total_land=100,  # 公顷
    total_water=300000  # 立方米
)

print("最优作物组合:")
for crop, area in solution.items():
    print(f"  {crop}: {area:.1f} ha")
```

#### 遗传算法

```python
from hetao_ag.opt import GeneticOptimizer, GAConfig

# 定义优化问题
def my_fitness(x):
    # 最大化利润同时最小化用水
    profit = x[0] * 500 + x[1] * 600  # 小麦、玉米面积
    water = x[0] * 3000 + x[1] * 5000
    water_penalty = max(0, water - 300000) * 0.01
    return profit - water_penalty

optimizer = GeneticOptimizer(
    fitness_func=my_fitness,
    n_vars=2,
    bounds=[(0, 100), (0, 100)],
    config=GAConfig(
        population_size=50,
        generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
)

result = optimizer.optimize()
print(f"最优解: 小麦={result.best_solution[0]:.1f}ha, "
      f"玉米={result.best_solution[1]:.1f}ha")
print(f"最佳适应度: {result.best_fitness:.0f}")
```

#### 场景分析

```python
from hetao_ag.opt import ScenarioEvaluator

crop_params = {
    "wheat": {
        "yield_kg_ha": 6000,
        "water_need_mm": 400,
        "price_per_kg": 0.8,
        "cost_per_ha": 1200
    },
    "maize": {
        "yield_kg_ha": 10000,
        "water_need_mm": 600,
        "price_per_kg": 0.6,
        "cost_per_ha": 1500
    }
}

evaluator = ScenarioEvaluator(crop_params, total_land=100, total_water=500000)

# 评估不同场景
s1 = evaluator.evaluate_scenario("全部小麦", {"wheat": 100}, 450)
s2 = evaluator.evaluate_scenario("全部玉米", {"maize": 100}, 650)
s3 = evaluator.evaluate_scenario("混合种植", {"wheat": 50, "maize": 50}, 500)

# 对比
comparison = evaluator.compare_scenarios()
print(f"最佳利润场景: {comparison['best_profit'].name}")
print(f"最佳水效场景: {comparison['best_water_efficiency'].name}")
```

---

## 文档

详细文档位于 `docs/` 目录:

- **[API 参考](docs/API_REFERENCE.md)**: 所有模块的完整 API 文档
- **[用户手册](docs/USER_MANUAL.md)**: 包含示例的综合使用指南

### 获取帮助

```python
# 查看函数文档
from hetao_ag.water import eto_penman_monteith
help(eto_penman_monteith)

# 列出模块内容
from hetao_ag import soil
print(dir(soil))
```

---

## 示例

`examples/` 目录中提供了示例脚本:

| 脚本                   | 说明                 |
| ---------------------- | -------------------- |
| `demo.py`              | 所有模块的综合演示   |
| `example_core.py`      | 单位系统、配置和日志 |
| `example_soil.py`      | 土壤水分和盐分建模   |
| `example_water.py`     | 蒸散发和灌溉         |
| `example_crop.py`      | 作物生长和胁迫模拟   |
| `example_livestock.py` | 动物检测和健康监测   |
| `example_space.py`     | 遥感指数和分类       |
| `example_opt.py`       | 优化和场景分析       |

### 运行示例

```bash
# 运行所有示例
python examples/demo.py

# 运行特定模块示例
python examples/example_soil.py
python examples/example_water.py
```

---

## 最佳实践

### 1. 始终使用单位系统

```python
# 推荐
from hetao_ag.core import meters, hectares
area = hectares(50)

# 不推荐
area = 50  # 单位不明确
```

### 2. 配置优于硬编码

```python
# 推荐
config = ConfigManager(config_file="farm_config.yaml")
fc = config.get("soil.field_capacity")

# 不推荐
fc = 0.32  # 硬编码值难以维护
```

### 3. 记录实验参数

```python
logger = get_logger("experiment")
logger.log_experiment_start(
    "灌溉优化",
    parameters={"method": "deficit", "target_etc": 0.8},
    random_seed=42
)
```

### 4. 验证模型结果

```python
from hetao_ag.core import validate_model
result = validate_model(observed, predicted)
print(f"RMSE: {result.rmse:.4f}, R²: {result.r_squared:.4f}")
```

---

## 贡献

欢迎贡献!请随时提交问题和拉取请求。

---

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。

---

## 引用

如果您在研究中使用 hetao_ag,请引用:

```bibtex
@software{hetao_ag,
  author = {Hetao College},
  title = {hetao_ag: 智慧农业Python库},
  version = {1.0.0},
  year = {2024},
  url = {https://github.com/1958126580/hetao_ag_new}
}
```

---

## 联系方式

- **作者**: Hetao College
- **邮箱**: 1958126580@qq.com
- **仓库**: https://github.com/1958126580/hetao_ag_new

---

<p align="center">
  <strong>河套智慧农牧业库 - 让农业更智能</strong>
</p>
