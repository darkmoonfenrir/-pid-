# -pid-
加热炉是过程控制中常见的对象，控制目标是将炉温稳定在设定值附近。传统系统辨识方法通常需预设模型结构，并结合输入输出数据进行参数拟合，其对数据量的依赖较低，这在数据获取受限的工业环境中是一大优势。然而，在控制器设计阶段，传统PID参数整定主要依赖经验法则或手动试凑，往往需要多次实验调整，不仅效率低下，也难以在复杂工况下实现理想的控制性能。该项目使用多种经典辨识方法，从加热炉的加热功率与温度响应数据中辨识出系统模型。随后，引入粒子群优化智能优化算法，对PID控制器的比例系数、积分时间和微分时间进行全局搜索优化，目标是最小化超调、稳态误差与调节时间等关键控制指标。训练优化过程中，控制精度与鲁棒性不断提升，显著提高系统性能与参数整定效率。

two_point$smith.py用于初步估计参数
在此基础上通过compare.py，与数据集进行比较，手动粗调参数
然后LM.py用于使用通过非线性最小二乘拟合进行进一步的传递函数评估
particle.py用于多次调整参数进行粒子群优化，试图求出可行的pid解
PSO使用粒子群优化，对于求得的优秀解进行在此优化
score用于打分
