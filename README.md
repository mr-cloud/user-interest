###策略：
物以类聚，人以群分
作品分类：分K1类 belongs to [10, 100] (IMDB电影分类标签: 12, Youtube: 18)
- 输入特征：人脸 + 文字 --> 图像
    - face features. <num_face, face_occu, gender_pref ([0, 1] 0 means all female, 1 means all male), age, looking    
- 算法：K-means, autoencoder
- 输出model：<cate_id, cate_features, [photo_id: set]>

人分类：分K2类，K2  belongs to [K1, min(1000, C(K1, 2))] (三教九流, 千人千面)
- 输入example：<user_id, [interest_on_cate_id: fixed length]>
- 算法：K-means
- 输出model: <pop_id, pop_features, [user_id: set]>, pop: population, normalization and cold-starting

预测：
- 为用户、作品归类：<user_id, photo_id> -> <pop_id, cate_id>
- 计算点击率：P{click} = pop_features[cate_id] mapped into [0, 1]
- 输出结果：<user_id, photo_id, click_probability>
