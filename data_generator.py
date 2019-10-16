from __future__ import print_function

import numpy as np
from python_speech_features import logfbank
import pickle as pickle_method
# 这两个包用于语音端点检测
import vad_ex
import webrtcvad
import tensorflow as tf
import re
import random
import os


def data_iterator(in_dir, data_type, segment_length=1.6, num_spk_per_batch=55, num_utt_per_batch=10):

    def generator(list, num_elements):
        """这个方法挺无聊的。。。就是为了随机生成随机的speaker"""
        # python gets list arg as reference
        batch = list[:num_elements]
        del list[:num_elements]
        list += batch
        return batch

    def is_invalid_spk(spk_id):
        # check if each speaker has more than at least self.hparams.num_utt_per_batch utterances
        """这个方法是为了检测每个speaker是否拥有num_utt_per_batch条声音，为了减少不必要的计算。。。干脆默认所有的speaker都满足好了"""
        spk_utt = [1 for pickle in pickles if re.search(spk_id+'_', pickle)]
        num_utt = sum(spk_utt)
        if num_utt < num_utt_per_batch:
            return True
        else:
            return False

    pickles = os.listdir(in_dir + "/" + data_type)
    """speaker的list"""
    spk_names = list(set([pickle.split("_")[0] for pickle in pickles]))

    """序列长度，帧数"""
    num_frame = int(segment_length * 100)
    spk_batch = generator(spk_names, num_spk_per_batch)

    target_batch = [spk for spk in range(num_spk_per_batch) for _ in range(num_utt_per_batch)]
    in_batch = []

    for spk_id in spk_batch:
        speaker_pickle_files_list = [file_name for file_name in
                                     os.listdir(in_dir + "/" + data_type) if
                                     re.search(spk_id, file_name) is not None]
        num_pickle_per_speaker = len(speaker_pickle_files_list)

        utt_idx_list = random.sample(range(num_pickle_per_speaker), k=num_utt_per_batch)

        for utt_idx in utt_idx_list:
            utt_pickle = speaker_pickle_files_list[utt_idx]
            utt_path = in_dir + "/" + data_type + "/" + utt_pickle
            with open(utt_path, "rb") as f:
                load_dict = pickle_method.load(f)
                total_logmel_feats = load_dict["LogMel_Features"]

            start_idx = random.randrange(0, total_logmel_feats.shape[0] - num_frame)
            logmel_feats = total_logmel_feats[start_idx:start_idx + num_frame, :]
            in_batch.append(logmel_feats)

        in_batch = np.asarray(in_batch)  # num spk * num utt, log mel
        target_batch = np.asarray(target_batch)  # spkid lables

        return in_batch, target_batch


def tf_data_iterator():


    pass


if __name__ == "__main__":

    in_data_path = "dataset/librispeech/libritest"
    data_type = ""

    data_iterator(in_dir=in_data_path, data_type=data_type)

    pass