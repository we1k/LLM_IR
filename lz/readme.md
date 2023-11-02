数据来源

* 74.57.json 来自于 threshold-50_temperature0.5_top_p0.3_answer 的 related_str，这个是直接用langchain的retrieve，k更大
* 76.75.json 来自于 threshold-40_temperature0.5_top_p0.6_answer 的 related_str，这个是手写的，上下文句子更多


建议是

* 不管header匹配的如何，都加上sentence（header匹配的好，可以k为1；不好可以为4）这样子
