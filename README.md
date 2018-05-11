###策略：
物以类聚，人以群分
作品分类：分K1类 belongs to [10, 100] (IMDB电影分类标签)
- 输入特征：人脸 + 文字 --> 图像
- 算法：K-means, autoencoder
- 输出model：<cate_id, cate_features, [photo_id: set]>

人分类：分K2类，K2  belongs to [2, 2^K1/9] (三教九流)
- 输入example：<user_id, [interest_on_cate_id: fixed length]>
- 算法：K-means
- 输出model: <pop_id, pop_features, [user_id: set]>, pop: population, normalization and cold-starting

预测：
- 为用户、作品归类：<user_id, photo_id> -> <pop_id, cate_id>
- 计算点击率：P{click} = pop_features[cate_id]