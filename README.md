# Bank-Marketing-and-Business-Intelligence
商务智能实训展示，产品预测

## 文件目录

* data ：存放数据集，与预处理后的excel文件
* DV：存放工程文件，注明功能与版本号
* img：存放效果图片，在目录下的README注明图片说明

## TODO

1. 数据预处理：csv转换xls，注意没有值的列
2. 特征分析与可视化
3. 机器学习模型训练
4. 训练结果可视化
5. PPT

## 学习与参考

* ![img](https://www.oracle.com/webfolder/s/analytic-store/i/sample.png)**Machine Learning Approach to Chronic Kidney Analysis** 肾病预测，参考其绘图类型　<https://www.oracle.com/solutions/business-analytics/data-visualization/examples.html>
* **DV Workshop - Basics of Training & Applying Predictive Models w/ DV**　如何用data-visualization使用机器学习模型　<https://www.oracle.com/solutions/business-analytics/data-visualization/tutorials.html>
* **Bank-Marketing**　github开源python 分析过程 <https://github.com/kunalBhashkar/Bank-Marketing-Data-Set-Classification>

1. 

## 数据集

http://archive.ics.uci.edu/ml/datasets/Bank+Marketing

预测用户是否接受某产品的推销

### 说明

特征数量:17

数据集上传日期:2012-2-14

分类:二分类

领域:商业

1）bank-additional-full.csv: 包含所有数据（41188）和20个输入，并按日期排序（从2008年5月到2010年11月） 
2）bank-additional.csv: 抽取其中10％的示例（4119），从1）中随机选择，以及20个输入。
3）bank-full.csv: 包含所有示例和17个输入，按日期排序
4）bank.csv: 具有10％的示例和17个输入，从(3)中随机选择

### 特征信息

**输入变量**：

**客户自身因素**

| 编号 | 特征      | 中文名   | 类别 | 说明                                                         |
| ---- | --------- | -------- | ---- | ------------------------------------------------------------ |
| 1    | age       | 年龄     | 数值 |                                                              |
| 2    | job       | 工作     | 分类 | '管理员'，'蓝领'，'企业家'，'女佣'，'管理'，'退休' ，'自雇人士'，'服务'，'学生'，'技师'，'失业'，'未知' |
| 3    | marital   | 婚姻     | 分类 | '离婚'，'已婚'，'单身'，'未知'                               |
| 4    | education | 教育     | 分类 | 'basic.4y'，'basic.6y'，'basic.9y'，'high.school'，'illiterate'，'professional.course '，'university.degree'，'未知' |
| 5    | default   | 违约记录 | 分类 | '不'，''是'，'未知'                                          |
| 6    | housing   | 住房贷款 | 分类 | '不'，'是'，'未知'                                           |
| 7    | loan      | 贷款     | 分类 | '不'，'是'，'未知'                                           |



**当前广告推销相关因素**

| 编号 | 特征        | 中文名         | 类别 | 说明                                                         |
| ---- | ----------- | -------------- | ---- | ------------------------------------------------------------ |
| 8    | contact     | 联系人沟通类型 | 分类 | '移动电话'，'固定电话'                                       |
| 9    | month       | 联系的月份     | 分类 |                                                              |
| 10   | day_of_week | 联系是星期几   | 分类 |                                                              |
| 11   | duration    | 联系持续时间   | 数值 | **重要说明：此属性高度影响输出目标**（如果打算采用现实的预测模型，应将其丢弃) |



**其他因素**

| 编号 | 特征     | 中文名                     | 类别 | 说明                     |
| ---- | -------- | -------------------------- | ---- | ------------------------ |
| 12   | campaign | 和此客户执行的联系次数     | 数值 | 包括最后一次联系         |
| 13   | pdays    | 上次和这个客户联系隔了几天 | 数值 | 999表示客户之前没联系过  |
| 14   | previos  | 之前执行的联系人数量       | 数值 |                          |
| 15   | poutcome | 上一次营销的结果           | 分类 | '失败'，'不存在'，'成功' |



**社会和经济因素** 

| 编号 | 特征           | 中文名                    | 类别 | 说明                     |
| ---- | -------------- | ------------------------- | ---- | ------------------------ |
| 16   | emp.var.rate   | 就业变化率 - 季度指标     | 数值 |        |
| 17   | cons.price.idx | 消费者价格指数 - 月度指标 | 数值 |   |
| 18   | cons.conf.idx  | 消费者信心指数 - 月度指标 | 数值 |                          |
| 19   | euribor3m      | 3个月费率 - 每日指标      | 数值 | |
| 20   | nr.employed    | 员工人数 - 季度指标       | 数值 |                          |



**输出变量**:

| 编号 | 特征 | 中文名                 |
| ---- | ---- | ---------------------- |
| 21   | y    | 客户是否订购了定期存款 |