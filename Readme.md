# 项目的由来
1、分类、抽取、序列标注、生成任务是自然语言处理的四大经典任务，其中，分类、抽取任务，可以使用规则进行快速实现。而对于生成而言，则与统计深度学习关系较为密切。  
2、当前，GPT系列，自动文本生成、文本图像生成，图像文本生成等魔幻主义大作频频上演。  
3、目前开源的seq2seq模型项目晦涩难度，不利于阅读与入门。  
受此三个现实背景，也正好在接触生成这个任务，特做此项目。  


# 项目的构成
项目场景：该项目以自动对诗为使用场景，即用户给定上一句，要求模型给出下一句，是个较理想的生成例子。  
项目代码结构：  
    data.txt:为训练数据，此处使用的是对联诗句数据  
    seq2seq_predict.py:使用seq2seq模型进行下一句生成的脚本  
    seq2seq_train.py:使用seq2seq模型进行生成的脚本  
    model/:  
        config.txt:预训练时形成的一些关键参数，如最大长度等，字数等。  
        input_vocab.pkl:输入语句的字符索引  
        output_vovab.pkl:输出语句的字符索引，此处将输入和输出进行区分成两个vocab，可以用于不同语种翻译等场景，如果不需要也可以合成一个。  
        s2s_model.h5:模型名称  
    image:  
        lstm_seq2seq_model.png:序列生成模型网路结构图  

# 项目的思想：
采用character字级别，通过搭建lstm-encoder和lstm-decoder进行seq2seq生成任务。  

![image](https://github.com/liuhuanyong/KerasSeq2seqGeneration/blob/master/image/lstm_seq2seq_model.png)

# 项目的使用：
1、python seq2seq_train.py,进行模型训练。  
2、python seq2seq_predict.py,进行模型测试。  

# 项目的总结：

1，本项目完成了一个基于keras实现的自动对诗文本生成功能。  
2，这是一个较为简单的入门级项目，欢迎补充。 

# 关于作者：
如有自然语言处理、知识图谱、事理图谱、社会计算、语言资源建设等问题或合作，可联系我：
1、我的自然语言处理开源项目：https://liuhuanyong.github.io
2、我的csdn技术博客：https://blog.csdn.net/lhy2014
3、我的联系方式: 刘焕勇，中国科学院软件研究所，lhy_in_blcu@126.com.
4、我的共享知识库项目：刘焕勇，事理类知识库数据集，http://www.openkg.cn/organization/datahorizon.
5、我的工业项目：刘焕勇，以事理为核心的金融情报探索：https://datahorizon.cn.
