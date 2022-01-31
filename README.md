# Machine Learning

一些基础机器学习算法实现

主要参考

> 李锐, 李鹏, 曲亚东, 等. 机器学习实战[M]. BEIJING BOOK CO. INC., 2013.

使用纯python3实现，没有使用相关的框架，修改，补充，完善了参考资料中的相关细节，所有实验数据都在data文件夹下；

代码和原理均以Jupyter Lab的形式展示，方便阅读和学习；

## 主要内容

- Numpy Test
    - 关于Numpy array和matrix的简单示例
- Decision Tree
    - 决策树（ID3）
    - Matplotlib形象化决策树
    - 测试及存储分类器
    - e.g. 使用决策树预测隐形眼镜类型
    - 总结
- Naive Bayes
    - 描述及特点
    - 使用NB进行文本分类
    - 使用NB过滤垃圾邮件
    - 使用NB获取不同类型信息的用词倾向
    - 总结
- KNN
    - 描述及特点
    - 核心原理
    - 算法流程
    - 实战
        - e.g.1简单实例
        - e.g.2 KNN改进约会网站配对
        - e.g.3 KNN手写数字识别
- Logistic Regression
  - 描述及特点
  - 理论基础
  - 基于最优化方法获取最佳回归系数
    - 批次梯度上升
    - 随机梯度上升
  - e.g. 从疝气病症预测病马死亡率
  - 总结
- SVM
    - simpleSMO
        - 理论基础
        - SMO（Sequential Minimal Optimization）
    - PlattSMO
        - Platt SMO
        - 应用核函数解决非线性可分问题
        - 手写数字识别
        - 总结
- AdaBoost集成多个弱分类器提高分类性能
    - 集成学习
    - AdaBoost
    - AdaBoost示例
- LinearRegression
    - 线性回归
    - 局部加权线性回归
    - 缩减系数辅助“理解”数据
- RegressionTree
    - 复杂数据局部性建模
    - 连续和离散特征的树的构建
    - CART算法用于回归
    - 树剪枝
    - 模型树
    - 树回归和标准回归比较
- KMeanClustering
    - K-Mean clustering
    - Bisecting K-Mean算法
    - 示例:对地图上的点聚类
- Apriori
    - 关联分析
    - Apriori
    - 从频繁项集中挖掘关联规则
    - e.g. 发现毒蘑菇的相似特征
- FP-growth
    - FP-growth
    - 从FP树中挖掘频繁项集
    - e.g. 从新闻网站点击流中挖掘
- PCA
    - PCA
    - e.g.
- SVD
    - SVD
    - 基于协同过滤的推荐引擎
    - e.g.
        - 餐馆菜肴推荐引擎
        - 基于SVD的图像压缩