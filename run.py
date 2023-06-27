# 2023.0613 @Brian

import os
import glob
import argparse
import pickle
import re
import random
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import fontManager
import cv2
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import math
from dtw import *
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import MinMaxScaler


def init_seed(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def load_subject_keypoints(subject, annot_df):

    subject_annot = annot_df.loc[annot_df['subject'] == subject]

    subject_kp_filepath = f"common/pose3d/output/{subject}/keypoints_3d_mhformer.npz"
    assert os.path.exists(subject_kp_filepath), f"Subject {subject} 3D keypoints file doesn't exist!"

    subject_keypoints = np.load(subject_kp_filepath, encoding='latin1', allow_pickle=True)["reconstruction"]
    subject_trimmed_kp = subject_keypoints[int(subject_annot['start']):int(subject_annot['end'])]

    print(subject_trimmed_kp.shape)

    subject_video_filepath = f"input/{subject}.mp4"
    subject_video_fps = cv2.VideoCapture(subject_video_filepath).get(cv2.CAP_PROP_FPS)

    return subject_trimmed_kp, subject_video_fps


def distance(point_1, point_2):

    length = math.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2 + (point_1[2]-point_2[2])**2)
    return length


def calculateAngle(point_1, point_2, point_3):

    a = math.sqrt((point_2[0]-point_3[0])**2 + (point_2[1]-point_3[1])**2 + (point_2[2]-point_3[2])**2)
    b = math.sqrt((point_1[0]-point_3[0])**2 + (point_1[1]-point_3[1])**2 + (point_1[2]-point_3[2])**2)
    c = math.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2 + (point_1[2]-point_2[2])**2)

    if (-2*a*c) == 0:
        B = math.degrees(math.acos((b*b-a*a-c*c) / (-2*a*c+1)))
    elif ((b*b-a*a-c*c) / (-2*a*c)) >= 1 or ((b*b-a*a-c*c) / (-2*a*c)) <= -1:
        B = math.degrees(math.acos(1))
    else:
        B = math.degrees(round(math.acos((b*b-a*a-c*c) / (-2*a*c)), 3))

    return B


def calc_vector_proj(u, n):
    '''Project vector u on Plane P. 
       (https://www.geeksforgeeks.org/vector-projection-using-python/)
    '''

    # Vector u 
    u = np.array([u[0], u[1], u[2]])       
    
    # Vector n: n is orthogonal vector to Plane P
    n = np.array([n[0], n[1], n[2]])       
    
    # Finding norm of the vector n 
    n_norm = np.sqrt(sum(n**2))    
    
    # Apply the formula as mentioned above
    # for projecting a vector onto the orthogonal vector n
    # find dot product using np.dot()
    proj_of_u_on_n = (np.dot(u, n)/n_norm**2)*n
    
    # Subtract proj_of_u_on_n from u: this is the projection of u on Plane P
    result = u - proj_of_u_on_n
    # print("Projection of Vector u on Plane P is: ", result)

    return result


def pos_neg(x):

    return -1 if x < 0 else 1


def get_curves(s1_time, s2_time, s1_feature, s2_feature):
    '''
    Get the interpolated curves of the original data.
    '''

    s1_xlim = np.arange(0, len(s1_time), 1)
    s1_x_curve = CubicSpline(s1_xlim, s1_time, bc_type='natural')
    s1_y_curve = CubicSpline(s1_xlim, s1_feature, bc_type='natural')

    s2_xlim = np.arange(0, len(s2_time), 1)
    s2_x_curve = CubicSpline(s2_xlim, s2_time, bc_type='natural')
    s2_y_curve = CubicSpline(s2_xlim, s2_feature, bc_type='natural')

    max_x, min_x = max(max(s1_time), max(s2_time)),  min(min(s1_time), min(s2_time))
    max_y, min_y = max(max(s1_feature), max(s2_feature)),  min(min(s1_feature), min(s2_feature))

    return s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y


def plot_subject_seperate(feature_name, xlabel, ylabel, s1_time, s2_time, s1_feature, s2_feature):

    f = plt.figure()
    plt.xlim([min(min(s1_time), min(s2_time)), max(max(s1_time), max(s2_time))])
    plt.ylim([min(min(s1_feature), min(s2_feature)), max(max(s1_feature), max(s2_feature))])
    plt.plot(s1_time, s1_feature, label=args.subject1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    ax = plt.gca()
    ax.get_legend().legendHandles[0].set_color("#1f77b4")
    ax.get_lines()[0].set_color("#1f77b4")
    # plt.show()
    
    f.savefig(f"./output/{TIMESTAMP}/{feature_name}_s1_{TIMESTAMP[:-1]}")

    f = plt.figure()
    plt.xlim([min(min(s1_time), min(s2_time)), max(max(s1_time), max(s2_time))])
    plt.ylim([min(min(s1_feature), min(s2_feature)), max(max(s1_feature), max(s2_feature))])
    plt.plot(s2_time, s2_feature, label=args.subject2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    ax = plt.gca()
    ax.get_legend().legendHandles[0].set_color("#ff7f0e")
    ax.get_lines()[0].set_color("#ff7f0e")
    # plt.show()

    f.savefig(f"./output/{TIMESTAMP}/{feature_name}_s2_{TIMESTAMP[:-1]}")

    plt.close(f)


def plot_subject_concatenate(feature_name, xlabel, ylabel, s1_time, s2_time, s1_feature, s2_feature):

    ## Plot Subjects together without 1d-rescale
    
    # f = plt.figure()
    # plt.plot(s1_time, s1_feature, label=args.subject1)
    # plt.plot(s2_time, s2_feature, label=args.subject2)
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.legend()
    # plt.show()
    
    # f.savefig(f"./output/{TIMESTAMP}/{feature_name}_{TIMESTAMP[:-1]}")

    ## Plot Subjects together with 1d-rescale

    s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y = get_curves(s1_time, s2_time, s1_feature, s2_feature)

    factor = max_x / min(max(s1_time), max(s2_time))
    factor_s1, factor_s2 = (1, factor) if max(s1_time) > max(s2_time) else (factor, 1)

    f = plt.figure()
    plt.plot(factor_s1 * s1_x_curve(s1_xlim), s1_y_curve(s1_xlim), label=args.subject1)
    plt.plot(factor_s2 * s2_x_curve(s2_xlim), s2_y_curve(s2_xlim), label=args.subject2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    # plt.show()

    f.savefig(f"./output/{TIMESTAMP}/{feature_name}_{TIMESTAMP[:-1]}")

    plt.close(f)


def euclidean_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):
    '''
    https://tech.gorilla.co/how-can-we-quantify-similarity-between-time-series-ed1d0b633ca0
    '''

    s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y = get_curves(s1_time, s2_time, s1_feature, s2_feature)
    s1, s2, s_max = s1_y_curve(np.linspace(0, len(s1_time), 1000)), s2_y_curve(np.linspace(0, len(s2_time), 1000)), np.random.uniform(min_y, max_y, size=1000).reshape(-1, 1)

    subject_distance = np.sqrt(np.sum((s1 - s2) ** 2))
    max_distance = np.sqrt(np.sum((s1 - s_max) ** 2))
    min_distance = 0

    similarity = (subject_distance / (max_distance - min_distance)) * 100
    similarity = min(max((100 - similarity), 0), 100)

    print(f"<Euclidean> Subject_distance: {subject_distance}, Max_distance: {max_distance}, Min_distance: {min_distance}")

    return similarity


def pearsonCorr_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):

    s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y = get_curves(s1_time, s2_time, s1_feature, s2_feature)
    s1, s2 = s1_y_curve(np.linspace(0, len(s1_time), 1000)), s2_y_curve(np.linspace(0, len(s2_time), 1000))

    a_diff = s1 - np.mean(s1)
    p_diff = s2 - np.mean(s2)
    numerator = np.sum(a_diff * p_diff)
    denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(p_diff ** 2))
    subject_corr = numerator / denominator

    # print(f"Subject_Correlation: {subject_corr}")

    return min(max(subject_corr, 0) * 100, 100)


def dtw_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):
    '''
    In time series analysis, dynamic time warping (DTW) is an algorithm for measuring similarity between 
    two temporal sequences, which may vary in speed. 
    (https://dynamictimewarping.github.io/python/)
    '''

    s1_x_curve, s1_y_curve, s1_xlim, s2_x_curve, s2_y_curve, s2_xlim, max_x, min_x, max_y, min_y = get_curves(s1_time, s2_time, s1_feature, s2_feature)
    s1, s2, s_max = s1_y_curve(np.linspace(0, len(s1_time), 1000)), s2_y_curve(np.linspace(0, len(s2_time), 1000)), np.random.uniform(min_y, max_y, size=1000).reshape(-1, 1)

    alignment_threeway = dtw(s1, s2, keep_internals=True)    
    alignment_twoway = dtw(s1, s2, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c"))
    alignment_threeway.plot(type="threeway")
    alignment_twoway.plot(type="twoway",offset=-2).figure.savefig(f"./output/{TIMESTAMP}/{feature_name}_similarity_{TIMESTAMP[:-1]}")
    # plt.show()
    
    subject_distance, min_distance, max_distance = alignment_twoway.distance, 0, dtw(s1, s_max, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")).distance
    similarity = (subject_distance / (max_distance - min_distance)) * 100
    similarity = min(max((100 - similarity), 0), 100)

    print(f"<DTW> Subject_distance: {subject_distance}, Max_distance: {max_distance}, Min_distance: {min_distance}")

    return similarity


def meanStd_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):

    mean_standard, mean_learner = np.mean(s1_feature), np.mean(s2_feature)
    std_standard, std_learner  = np.std(s1_feature, ddof=0), np.std(s2_feature, ddof=1)

    if mean_learner < 160:
        mean = max((1 - (abs(mean_learner-mean_standard) / 180)) * 30, 0)
        std  = max((1 - (abs(std_learner-std_standard) / std_standard)) * 0, 0)
        bonus = 70
    else:
        mean = max((1 - (abs(mean_learner-mean_standard) / 180)) * 20, 0)
        std  = min(max((1 - (abs(std_learner-std_standard) / std_standard)) * 80, 0), 50)
        bonus = 0
        
    similarity = int(round(mean + std + bonus))

    return similarity


def similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature):

    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((s1_feature, s2_feature), axis=0).reshape(-1, 1))
    s1_feature_scaled = scaler.transform(s1_feature.reshape(-1, 1))
    s2_feature_scaled = scaler.transform(s2_feature.reshape(-1, 1))
    s1_feature, s2_feature = s1_feature_scaled, s2_feature_scaled
    print(s1_feature_scaled.shape, s2_feature_scaled.shape)

    old_similarity = meanStd_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature)
    euclidean_similarity = euclidean_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature)
    pearsonCorr_similarity = pearsonCorr_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature)
    dtw_similarity = dtw_similarity_function(feature_name, s1_time, s2_time, s1_feature, s2_feature)

    if pearsonCorr_similarity == 100 and dtw_similarity == 100:
        similarity = 100
    elif pearsonCorr_similarity == 100 and dtw_similarity != 100:
        similarity = pearsonCorr_similarity - 0.2 * dtw_similarity
    elif pearsonCorr_similarity != 100 and dtw_similarity == 100:
        similarity = dtw_similarity - 0.2 * pearsonCorr_similarity
    elif 60 < pearsonCorr_similarity < 80 and 50 < dtw_similarity < 70:
        similarity = 0.6 * pearsonCorr_similarity + 0.4 * dtw_similarity + 15
    elif 60 < pearsonCorr_similarity < 80 and dtw_similarity < 50:
        similarity = 0.9 * pearsonCorr_similarity + 0.1 * dtw_similarity + 15
    elif 60 < pearsonCorr_similarity < 80 and 70 < dtw_similarity:
        similarity = 0.1 * pearsonCorr_similarity + 0.9 * dtw_similarity + 10
    elif pearsonCorr_similarity < 60 and 50 < dtw_similarity < 70:
        similarity = 0.4 * pearsonCorr_similarity + 0.6 * dtw_similarity + 25
    elif 80 < pearsonCorr_similarity and 50 < dtw_similarity < 70:
        similarity = 0.9 * pearsonCorr_similarity + 0.1 * dtw_similarity + 10
    else:
        similarity = max(pearsonCorr_similarity, dtw_similarity)

    similarity = min(similarity, 100)
    
    print("Old similarity: ", old_similarity)
    print('Euclidean similarity:', euclidean_similarity)
    print('Pearson Correlation similarity', pearsonCorr_similarity)
    print('DTW similarity:', dtw_similarity)
    print('similarity:', similarity, end="\n\n")

    with open(f'output/{TIMESTAMP}/{feature_name}_eval_{TIMESTAMP[:-1]}.txt', 'w') as f:
        
        f.writelines(f'<{feature_name}>\n')
        f.writelines(f'Old similarity: {old_similarity}\n')
        f.writelines(f'Euclidean similarity: {euclidean_similarity}\n')
        f.writelines(f'Pearson Correlation similarity: {pearsonCorr_similarity}\n')
        f.writelines(f'DTW similarity: {dtw_similarity}\n')
        f.writelines(f'similarity: {similarity}\n')

    return similarity


def evaluate_arm_wave_ang(s1_walk_kp, s2_walk_kp, s1_video_fps, s2_video_fps):

    print("Arm Relative Waving Angle >>>>>")

    ## Reference Arm Position (Initial Position)

    s1_init_wrist_coord = s1_walk_kp[0][h36m_skeleton["l_wrist"]]
    s2_init_wrist_coord = s2_walk_kp[0][h36m_skeleton["l_wrist"]]

    ## Calculate Subject Arm Relative Waving Angle

    s1_arm_wave_ang = np.array([calculateAngle(f[h36m_skeleton["l_wrist"]], f[h36m_skeleton["l_shoulder"]], s1_init_wrist_coord) for f in s1_walk_kp])
    s2_arm_wave_ang = np.array([calculateAngle(f[h36m_skeleton["l_wrist"]], f[h36m_skeleton["l_shoulder"]], s2_init_wrist_coord) for f in s2_walk_kp])
    
    s1_time = [(i / s1_video_fps) for i in range(len(s1_walk_kp))]
    s2_time = [(i / s2_video_fps) for i in range(len(s2_walk_kp))]   

    ## Plot Subjects Arm Relative Waving Angle
    
    plot_subject_seperate('arm_wave_ang', 'sec', 'degree', s1_time, s2_time, s1_arm_wave_ang, s2_arm_wave_ang)
    plot_subject_concatenate('arm_wave_ang', 'sec', 'degree', s1_time, s2_time, s1_arm_wave_ang, s2_arm_wave_ang)

    ## Calculate Subject Arm Relative Waving Angles Similarities (https://dynamictimewarping.github.io/python/)

    similarity = similarity_function('arm_wave_ang', s1_time, s2_time, s1_arm_wave_ang, s2_arm_wave_ang)

    return similarity


def evaluate_step_length(s1_walk_kp, s2_walk_kp, s1_video_fps, s2_video_fps):

    print("Walk Step Length >>>>>")

    ## Calculate Subject Walk Step Length

    s1_step_length = np.array([distance(f[h36m_skeleton["l_foot"]], f[h36m_skeleton["r_foot"]]) for f in s1_walk_kp])
    s2_step_length = np.array([distance(f[h36m_skeleton["l_foot"]], f[h36m_skeleton["r_foot"]]) for f in s2_walk_kp])
    
    s1_time = [(i / s1_video_fps) for i in range(len(s1_walk_kp))]
    s2_time = [(i / s2_video_fps) for i in range(len(s2_walk_kp))]   

    ## Plot Subjects Walk Step Length
    
    plot_subject_seperate('step_length', 'sec', 'normalized_length', s1_time, s2_time, s1_step_length, s2_step_length)
    plot_subject_concatenate('step_length', 'sec', 'normalized_length', s1_time, s2_time, s1_step_length, s2_step_length)

    ## Calculate Subject Walk Step Length Similarities (https://dynamictimewarping.github.io/python/)

    similarity = similarity_function('step_length', s1_time, s2_time, s1_step_length, s2_step_length)

    return similarity


def evaluate_body_fl_ang(s1_walk_kp, s2_walk_kp, s1_video_fps, s2_video_fps):

    print("Body Foward Lean Angle >>>>>")

    ## Calculate Subject Body Foward Lean Angle

    s1_body_fl_ang = np.array([calculateAngle(f[h36m_skeleton["throat"]], f[h36m_skeleton["spine"]], f[h36m_skeleton["hip"]]) for f in s1_walk_kp])
    s2_body_fl_ang = np.array([calculateAngle(f[h36m_skeleton["throat"]], f[h36m_skeleton["spine"]], f[h36m_skeleton["hip"]]) for f in s2_walk_kp])
    
    s1_time = [(i / s1_video_fps) for i in range(len(s1_walk_kp))]
    s2_time = [(i / s2_video_fps) for i in range(len(s2_walk_kp))]   

    ## Plot Subjects Body Foward Lean Angle
    
    plot_subject_seperate('body_fl_ang', 'sec', 'degree', s1_time, s2_time, s1_body_fl_ang, s2_body_fl_ang)
    plot_subject_concatenate('body_fl_ang', 'sec', 'degree', s1_time, s2_time, s1_body_fl_ang, s2_body_fl_ang)

    ## Calculate Subject Body Foward Lean Angles Similarities (https://dynamictimewarping.github.io/python/)

    similarity = similarity_function('body_fl_ang', s1_time, s2_time, s1_body_fl_ang, s2_body_fl_ang)

    return similarity


def evaluate_body_lrl_ang(s1_walk_kp, s2_walk_kp, s1_video_fps, s2_video_fps, degree):

    print("Body Left-Right Directional Lean Angle >>>>>")

    ## Configuring Vectors

    # The orthogonal vector of the xy-plane. (a vector that points out from the floor)
    v = (0, 0, 1)

    # The orthogonal vector of the xz-plane + 30 degree right. (a vector that points out from the subject body)
    n = (np.cos(degree * np.pi / 180), np.sin(degree * np.pi / 180), 0)

    ## Calculate Subject Body Left-Right Directional Lean Angle 

    s1_body_lrl_ang = np.array([calculateAngle(f[h36m_skeleton["hip"]] + calc_vector_proj(f[h36m_skeleton["throat"]] - f[h36m_skeleton["hip"]], n), 
                                               f[h36m_skeleton["hip"]], 
                                               f[h36m_skeleton["hip"]] + v) * pos_neg(calc_vector_proj(f[h36m_skeleton["throat"]] - f[h36m_skeleton["hip"]], n)[1])
                                for f in s1_walk_kp])
    s2_body_lrl_ang = np.array([calculateAngle(f[h36m_skeleton["hip"]] + calc_vector_proj(f[h36m_skeleton["throat"]] - f[h36m_skeleton["hip"]], n), 
                                               f[h36m_skeleton["hip"]], 
                                               f[h36m_skeleton["hip"]] + v) * pos_neg(calc_vector_proj(f[h36m_skeleton["throat"]] - f[h36m_skeleton["hip"]], n)[1])
                                for f in s2_walk_kp])
    
    s1_time = [(i / s1_video_fps) for i in range(len(s1_walk_kp))]
    s2_time = [(i / s2_video_fps) for i in range(len(s2_walk_kp))]   

    ## Plot Subjects Body Left-Right Directional Lean Angle
    
    plot_subject_seperate('body_lrl_ang', 'sec', 'degree', s1_time, s2_time, s1_body_lrl_ang, s2_body_lrl_ang)
    plot_subject_concatenate('body_lrl_ang', 'sec', 'degree', s1_time, s2_time, s1_body_lrl_ang, s2_body_lrl_ang)

    ## Calculate Subject Body Left-Right Directional Lean Angle Similarities (https://dynamictimewarping.github.io/python/)

    similarity = similarity_function('body_lrl_ang', s1_time, s2_time, s1_body_lrl_ang, s2_body_lrl_ang)

    return similarity


# Define Configurations
SEED = 0
SOURCE_FOLDER = "input\\"
TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())

# Argument Parser
parser = argparse.ArgumentParser(description='main')
parser.add_argument('--subject1', required=True, type=str, help="Subject 1 3D filename.")
parser.add_argument('--subject2', required=True, type=str, help="Subject 2 3D filename.")
parser.add_argument('--mode', required=True, type=str, help="Front or Side input video.")
args = parser.parse_args()


if __name__ == "__main__":

    ## Set up random seed on everything

    init_seed(SEED)


    ## Load Annotation file

    subjects_annot_filepath = f"annotation/walk_analysis.csv"
    assert os.path.exists(subjects_annot_filepath), "Subjects annotation file doesn't exist!"
    annot_df = pd.read_csv(subjects_annot_filepath, encoding='utf8')


    ## Load subject strokes keypoints and Get subject video information

    h36m_skeleton = {
        "head": 10, "neck": 9, "throat": 8, "spine": 7, "hip": 0,
        "r_shoulder": 14, "r_elbow": 15, "r_wrist": 16, "l_shoulder": 11, "l_elbow": 12, "l_wrist": 13,
        "r_hip": 1, "r_knee": 2, "r_foot": 3, "l_hip": 4, "l_knee": 5, "l_foot": 6
        }
    
    s1_walk_kp, s1_video_fps = load_subject_keypoints(args.subject1, annot_df)
    s2_walk_kp, s2_video_fps = load_subject_keypoints(args.subject2, annot_df)
    print()


    ## Walk Analysis

    os.makedirs(f'output/{TIMESTAMP}')

    if args.mode == "side":

        # 1. Evaluate Arm Relative Waving Angle
        arm_wave_ang_similarity = evaluate_arm_wave_ang(s1_walk_kp, s2_walk_kp, s1_video_fps, s2_video_fps)

        # 2. Evaluate Walk Step Length
        step_length_similarity = evaluate_step_length(s1_walk_kp, s2_walk_kp, s1_video_fps, s2_video_fps)

        # 3. Evaluate Body Foward Lean Angle
        body_fl_ang_similarity = evaluate_body_fl_ang(s1_walk_kp, s2_walk_kp, s1_video_fps, s2_video_fps)

        # Export the Analysis Results

        with open(f'output/{TIMESTAMP}/evalAll_{TIMESTAMP[:-1]}.txt', 'w') as f:
            
            f.writelines(f'{args.subject1} & {args.subject2}\n')
            f.writelines(f'arm_wave_ang_similarity: {arm_wave_ang_similarity}\n')
            f.writelines(f'step_length_similarity: {step_length_similarity}\n')
            f.writelines(f'body_fl_ang_similarity: {body_fl_ang_similarity}\n')

    elif args.mode == "front":

        # 4. Evaluate Body Left-Right Lean Angle
        body_lrl_ang_similarity = evaluate_body_lrl_ang(s1_walk_kp, s2_walk_kp, s1_video_fps, s2_video_fps, 60)


        # Export the Analysis Results

        with open(f'output/{TIMESTAMP}/evalAll_{TIMESTAMP[:-1]}.txt', 'w') as f:
            
            f.writelines(f'{args.subject1} & {args.subject2}\n')
            f.writelines(f'body_lrl_ang_similarity: {body_lrl_ang_similarity}\n')




