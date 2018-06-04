## 数据探查

### 数据源与隐藏规律
- train_interaction.txt
    - 2.0E7
    - 其中用户数为1.5E4
- train_face.txt
    - 1.6E6
    - 不是所有作品都有人脸数据
- train_text.txt
    - 4.3E6
    - 每个单词已编码
    - 所有的作品都有封面数据
- test_interaction.txt
    - 3.3E6
    - 待预测的用户都有交互历史
    - 用户数同*train_interaction.txt*

##策略：
###物以类聚，人以群分
作品分类：分K1类 belongs to [10, 100] (IMDB电影分类标签: 12, Youtube: 18)
- 输入特征：人脸 --> 文字 --> 图像
    - face features. <num_face, face_occu, gender_pref ([0, 1] 0 means all female, 1 means all male), age, looking
    - text features. a)Embedding b)TF-IDF
    - hot features. <show, click, like, follow>    
- 算法：K-means, autoencoder
- 输出model：<cate_id, cate_features, [photo_id: set]>  
<=> KMeans with <example_id, label> 

人分类：分K2类，K2  belongs to [K1, max(K1 + 1, min(1000, C(K1, 2)))] (三教九流, 千人千面) **考虑到用户数量少，所以可以不对用户分类**
- 输入example：<user_id, [interest_on_cate_id: fixed length]>
- 算法：K-means
- 输出model: <pop_id, pop_features, [user_id: set]>, pop: population, normalization with **MinMaxScaler** and cold-starting  
<=> KMeans with <example_id, label> 

预测：
- 为用户、作品归类：<user_id, photo_id> -> <pop_id, cate_id>
- 计算点击率：P{click} = pop_features[cate_id] mapped into [0, 1]
- 输出结果：<user_id, photo_id, click_probability>


###监督学习 v1.0.0
什么样的用户面对什么样的作品，出现点击或者不点击: <user_features, photo_features, click ({0 ,1})>
- 输入样例：历史交互数据
<user_id, photo_id, click> -> <user_id_count, user_click_oof, user_play_time_oof, duration_time, time, click>
- 扩展feature
    - 用户相关
        - user_id_count: 用户在训练集中出现的交互次数
        - user_click_oof: click per show. 每次展现，用户平均点击动作的发生频率
        - user_play_time_oof: play time per click. 每次点击后，播放的时长
    - 作品相关
        - duration_time
    - 交互相关
        - time: timestamp
    

- 模型：LR
- 预测：LR.predict_proba(X_test)[:, 1]

###监督学习 v2.0.0
- 扩展作品相关feature
    <user_click_oof, user_play_time_oof, duration_time, time, num_face, face_occu, gender_pref, age, looking, topic, click>
- 模型使用集成学习算法 + GridSearch调优


### 文字信息处理
- 常用汉语词汇量50000