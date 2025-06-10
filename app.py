import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
from surprise import Dataset, Reader

@st.cache_resource
def load_trained_model():
    """加载实验七训练好的模型"""
    try:
        model_path = 'surprise_model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        st.success(f"模型 '{model_path}' 加载成功！")
        return model
    except FileNotFoundError:
        st.error(f"错误: 模型文件 '{model_path}' 未找到。请确保已运行 train_model.py。")
        return None
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

@st.cache_data
def load_experiment_data():
    """加载实验七的数据文件"""
    try:
        ratings_df = pd.read_csv('processed_ratings.csv')
        # 由于Excel文件没有列名，我们需要指定header=None并手动设置列名
        jokes_df = pd.read_excel('Dataset4JokeSet.xlsx', header=None)
        jokes_df.columns = ['joke_text'] # 假设笑话文本在第一列
        jokes_df['joke_id'] = jokes_df.index + 1 # 创建基于索引的joke_id，从1开始
        jokes_df = jokes_df[['joke_id', 'joke_text']] # 确保列顺序和名称正确
        st.success("实验数据加载成功！")
        return ratings_df, jokes_df
    except FileNotFoundError as e:
        st.error(f"错误: 必需的数据文件未找到: {e}。请确保 'processed_ratings.csv' 和 'Dataset4JokeSet.xlsx' 存在。")
        return None, None
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None, None

def display_random_jokes():
    """显示随机笑话并收集评分"""
    ratings_df, jokes_df = load_experiment_data()
    if jokes_df is None:
        st.warning("无法加载笑话数据，请检查数据文件。")
        return

    if 'selected_jokes' not in st.session_state:
        joke_ids = jokes_df['joke_id'].tolist()
        st.session_state.selected_jokes = random.sample(joke_ids, 3)
        st.session_state.user_ratings = {}
    
    st.header("请为以下笑话评分")
    
    cols = st.columns(3)
    
    for i, joke_id in enumerate(st.session_state.selected_jokes):
        with cols[i]:
            joke_text = jokes_df[jokes_df['joke_id'] == joke_id]['joke_text'].iloc[0]
            st.subheader(f"笑话 {i+1}")
            st.write(joke_text)
            
            current_rating = st.session_state.user_ratings.get(joke_id, 0.0)
            st.session_state.user_ratings[joke_id] = st.slider(
                f"请为笑话 {joke_id} 评分 (当前: {current_rating:.1f})",
                min_value=-10.0, max_value=10.0, value=current_rating, step=0.5,
                key=f"rating_joke_{joke_id}"
            )

def generate_recommendations_with_model(user_ratings, model, model_type, jokes_df):
    """
    使用训练好的模型生成推荐
    """
    recommendations = []
    if model_type == 'surprise':
        # 创建新用户ID，为未评分笑话生成预测
        # 假设新用户的ID是一个未曾出现的整数，例如 9999
        new_user_id = 9999

        # 获取所有笑话ID
        all_joke_ids = jokes_df['joke_id'].unique()

        # 获取用户已评分的笑话ID
        rated_joke_ids = set(user_ratings.keys())

        # 找出用户未评分的笑话ID
        unrated_joke_ids = [joke_id for joke_id in all_joke_ids if joke_id not in rated_joke_ids]

        predictions = []
        for joke_id in unrated_joke_ids:
            # 预测用户对未评分笑话的评分
            predicted_rating = model.predict(new_user_id, joke_id).est
            predictions.append((joke_id, predicted_rating))

        # 排序并返回top-5推荐
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_5_recommendations = predictions[:5]
        
        # 结合笑话文本
        for joke_id, predicted_rating in top_5_recommendations:
            joke_text = jokes_df[jokes_df['joke_id'] == joke_id]['joke_text'].iloc[0]
            recommendations.append((joke_id, joke_text, predicted_rating))

    elif model_type == 'fastai':
        # TODO: 使用FastAI模型预测
        # 处理输入格式，调用模型预测
        pass
    
    # TODO: 排序并返回top-5推荐
    return recommendations

def display_recommendations():
    """显示推荐结果"""
    if not st.session_state.recommendations_generated:
        # 显示生成推荐的按钮和提示
        if st.button("获取个性化推荐"):
            if len(st.session_state.user_ratings) >= 3:
                model = load_trained_model()
                ratings_df, jokes_df = load_experiment_data()

                if model is None or jokes_df is None:
                    return

                model_type = detect_model_type(model)

                if model_type == 'unknown':
                    st.error("无法识别模型类型，无法生成推荐。")
                    return
                
                st.session_state.recommendations = generate_recommendations_with_model(
                    st.session_state.user_ratings, model, model_type, jokes_df
                )
                # 在生成推荐时初始化 rec_ratings
                if st.session_state.recommendations:
                    st.session_state.rec_ratings = {joke_id: 0.0 for joke_id, _, _ in st.session_state.recommendations}
                st.session_state.recommendations_generated = True # 标记推荐已生成
                st.rerun() # 重新运行以立即显示推荐
            else:
                st.warning("请先为至少3个笑话评分")
        else:
            st.info("请点击'获取个性化推荐'按钮生成推荐。")

    else: # recommendations_generated is True, 显示推荐和滑块
        if st.session_state.recommendations: # 确保推荐数据存在
            st.subheader("为您推荐的笑话：")
            
            for i, (joke_id, joke_text, predicted_rating) in enumerate(st.session_state.recommendations):
                st.write(f"**推荐笑话 {i+1} (预测评分: {predicted_rating:.2f})**")
                st.write(joke_text)
                
                current_rec_rating = st.session_state.rec_ratings.get(joke_id, 0.0)
                st.session_state.rec_ratings[joke_id] = st.slider(
                    f"您对推荐笑话 {joke_id} 的评分",
                    min_value=-10.0, max_value=10.0, value=current_rec_rating, step=0.5,
                    key=f"rec_rating_joke_{joke_id}"
                )
        else:
            st.error("未能生成推荐数据，请尝试重新开始或联系支持。") # 异常情况处理

def calculate_satisfaction(rec_ratings, rating_range):
    """
    计算用户满意度
    """
    if not rec_ratings:
        return 0
    
    min_rating, max_rating = rating_range
    
    avg_rating = sum(rec_ratings) / len(rec_ratings)
    # 归一化到0-100%
    satisfaction = ((avg_rating - min_rating) / (max_rating - min_rating)) * 100
    return max(0, satisfaction)

def display_satisfaction():
    """显示满意度计算结果"""
    if st.button("计算推荐满意度"):
        if 'rec_ratings' in st.session_state and st.session_state.rec_ratings:
            rec_ratings_list = list(st.session_state.rec_ratings.values())
            satisfaction = calculate_satisfaction(rec_ratings_list, (-10, 10))
            st.subheader("推荐满意度评估：")
            st.info(f"您对推荐的笑话的平均满意度为: {satisfaction:.2f}%")
            st.write("这表示您对系统推荐的笑话的喜爱程度。")
        else:
            st.warning("请先为推荐的笑话评分才能计算满意度。")

def detect_model_type(model):
    """检测模型类型"""
    # 根据模型属性判断类型
    if hasattr(model, 'predict') and 'surprise' in str(type(model)).lower():
        return 'surprise'
    # TODO: 如果未来有FastAI模型，可以添加类似的检测
    # elif hasattr(model, 'get_preds') and 'fastai' in str(type(model)).lower():
    #     return 'fastai'
    else:
        return 'unknown'
    
def prepare_user_data_for_model(user_ratings, model_type):
    """为模型准备用户数据"""
    if model_type == 'surprise':
        # 转换为Surprise格式
        pass
    elif model_type == 'fastai':
        # 转换为FastAI格式
        pass

# 应用状态管理
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = False
    st.session_state.model_loaded = False
    st.session_state.data_loaded = False
    st.session_state.current_step = 1
    st.session_state.user_ratings = {}
    st.session_state.rec_ratings = {}
    st.session_state.recommendations = []
    st.session_state.recommendations_generated = False # 新增：跟踪推荐是否已生成

def main():
    st.set_page_config(page_title="笑话推荐系统", layout="wide")
    st.title("基于协同过滤的笑话推荐系统")
    
    # 侧边栏显示应用信息
    with st.sidebar:
        st.header("应用信息")
        st.subheader("模型状态")
        model = load_trained_model()
        if model:
            st.session_state.model_loaded = True
            st.success("模型已加载")
        else:
            st.session_state.model_loaded = False
            st.error("模型未加载")
        
        st.subheader("数据状态")
        ratings_df, jokes_df = load_experiment_data()
        if ratings_df is not None and jokes_df is not None:
            st.session_state.data_loaded = True
            st.success("数据已加载")
            st.write(f"评分数据: {len(ratings_df)} 条")
            st.write(f"笑话数据: {len(jokes_df)} 条")
        else:
            st.session_state.data_loaded = False
            st.error("数据未加载")
        
        st.subheader("当前步骤")
        st.write(f"步骤 {st.session_state.current_step}")

    # 主要功能区域
    if st.session_state.current_step == 1:
        st.header("步骤 1: 加载模型和数据")
        if st.session_state.model_loaded and st.session_state.data_loaded:
            st.success("模型和数据已成功加载，可以开始评分。")
            if st.button("进入步骤 2: 笑话评分"):
                st.session_state.current_step = 2
                st.rerun()
        else:
            st.warning("请确保模型和数据文件存在，并已成功加载。")
            
    elif st.session_state.current_step == 2:
        st.header("步骤 2: 为笑话评分")
        display_random_jokes()
        if st.button("提交评分并进入步骤 3: 获取推荐"):
            if len(st.session_state.user_ratings) >= 3:
                st.session_state.current_step = 3
                st.rerun()
            else:
                st.warning("请至少为3个笑话评分才能继续。")

    elif st.session_state.current_step == 3:
        st.header("步骤 3: 获取个性化推荐")
        display_recommendations()
        # 只有在推荐已经生成并存在时才显示"进入步骤 4"按钮
        if st.session_state.recommendations_generated and st.session_state.recommendations:
            if st.button("进入步骤 4: 满意度评估"):
                st.session_state.current_step = 4
                st.rerun()

    elif st.session_state.current_step == 4:
        st.header("步骤 4: 推荐满意度评估")
        display_satisfaction()
        if st.button("重新开始"):
            st.session_state.clear()
            st.rerun()

    # TODO: 实现步骤化的用户界面
    # 1. 模型加载状态
    # 2. 随机笑话评分
    # 3. 推荐生成
    # 4. 满意度计算

if __name__ == "__main__":
    # 添加在重新开始时重置 recommendations_generated 的逻辑
    if 'app_initialized' not in st.session_state or st.session_state.current_step == 1:
        st.session_state.recommendations_generated = False
    main()



