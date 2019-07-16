import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
    
def visualize_matching(word2id, matching_scores, context_tokens, qn_tokens, true_ans_start, true_ans_end, pred_ans_start, pred_ans_end, true_answer, pred_answer, MPM_dim=10):
    """
    Generate heat map of multipespective matching scores for one example.

    Inputs:
      word2id: dictionary mapping word (string) to word id (int)
      context_tokens, qn_tokens: lists of strings, no padding.
        Note these do *not* contain UNKs.
      true_ans_start, true_ans_end, pred_ans_start, pred_ans_end: ints
      true_answer, pred_answer: strings
    """
    
    # matching_scores: shape (context_len, 8*MPM_dim)
    # matching_scores = matching_scores.T # shape (8*MPM_dim, context_len)
    window = matching_scores[pred_ans_start-10:pred_ans_end+10,:] # window around predicted answer
    candidates = context_tokens[pred_ans_start-10:pred_ans_end+10]
    
    #truth_window = matching_scores[true_ans_start:true_ans_end+1, :]
    #truth = context_tokens[true_ans_start:true_ans_end+1]
    
    #combined_scores = np.vstack((window, candidates))
    #combined_words = 

    perspectives = ['full_match', 'maxpool_match', 'attentive_match', 'max_attentive_match']

    for i in range(4): # heat map for 4 perspectives
        plt.figure()
        ax = sns.heatmap(window[:,i*MPM_dim: (i+1)*MPM_dim])
        ax.set_yticklabels(candidates, rotation=0)
        ax.set_xlabel('Forward: ' + perspectives[i])
        ax.set_title('True answer: ' + true_answer + ", predicted answer: " + pred_answer)
        fig = ax.get_figure()
        fig.savefig(str(i) + '.png')

    for i in range(4,8): # heat map for 4 perspectives
        plt.figure()
        ax = sns.heatmap(window[:, i*MPM_dim: (i+1)*MPM_dim])
        ax.set_yticklabels(candidates, rotation=0)
        ax.set_xlabel('Backward: ' + perspectives[i-4])
        ax.set_title('True answer: ' + true_answer + ", predicted answer: " + pred_answer)
        fig = ax.get_figure()
        fig.savefig(str(i) + '.png')
