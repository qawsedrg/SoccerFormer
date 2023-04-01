import os
import json

import torch
from tqdm import tqdm
import numpy as np

from modeling.meta_archs import *

from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2
from SoccerNet.Downloader import getListGames

from torch.utils.data import Dataset,DataLoader

class SoccerNetTrainDataset(Dataset):
    def __init__(self,
                 feat_file_address,
                 features="npy",
                 ):
        self.game_id=[]
        self.game_feats=[]
        self.half=[]
        # must load two halfs of the same game in a consecutive order
        for file in sorted(os.listdir(feat_file_address)):
            if file.split('.')[-1] == features:
                feature = np.load(os.path.join(feat_file_address, file))
                self.game_id.append(file)
                self.game_feats.append(torch.from_numpy(feature).permute((1, 0)))
                self.half.append(file.split('.')[-2])

    def __getitem__(self, index):
        return {"video_id": self.game_id[index],
                "feats": self.game_feats[index],
                "half": self.half[index]}

    def __len__(self):
        return len(self.game_feats)

fps=2
UrlLocal="./data"

if True:
    data = SoccerNetTrainDataset(UrlLocal)
    # batch_size must be 2, since a game has two 2 parts
    test_loader = DataLoader(data, batch_size=2, collate_fn=lambda x: x)

    device = torch.device("cuda:0")

    # choose one or several models
    models = [torch.load("./models/model_512_{:}.pt".format(i), map_location=device) for i in [1]]

    for (i, video_list) in tqdm(enumerate(test_loader),total=len(test_loader)):
        predict_dict = dict()
        predictions = []
        for video in video_list:
            segs = []
            labels = []
            scoress = []
            for model in models:
                model.eval()
                result = model([video],thresh=0.1)
                seg = result[0]["segments"].cpu().long().numpy()
                label = result[0]["labels"].cpu().numpy()
                scores = result[0]["scores"].cpu().numpy()
                segs.append(seg)
                labels.append(label)
                scoress.append(scores)

            seg = np.concatenate(segs)
            label = np.concatenate(labels)
            scores = np.concatenate(scoress)
            bins = np.bincount(seg)
            bins_sorted = np.sort(bins)

            predict_dict["UrlLocal"] = UrlLocal

            acceptable_idx = np.argsort(bins)[bins_sorted >= max(bins_sorted[int(19 * len(bins) / 20)], 1)]
            acceptable_idx=np.sort(acceptable_idx)

            max_interval=2
            counter=0
            while counter<len(acceptable_idx):
                idx=acceptable_idx[counter]
                prediction_dict=dict()
                label_bin=np.array([],dtype=int)
                pred_seg = []
                scores_bin=np.array([])
                while np.sum(acceptable_idx[counter:]<=idx+max_interval)>=1:
                    label_bin = np.concatenate((label_bin,label[seg == acceptable_idx[counter]]))
                    pred_seg.extend([acceptable_idx[counter]]*len(label[seg == acceptable_idx[counter]]))
                    scores_bin = np.concatenate((scores_bin, scores[seg == acceptable_idx[counter]]))
                    idx = acceptable_idx[counter]
                    counter+=1
                label_pre = np.argmax(np.bincount(label_bin))
                mask=label_pre==label_bin
                scores_pred=np.mean(scores_bin[mask])
                frame=np.mean(pred_seg)
                prediction_dict["gameTime"] = "{} - {}:{}".format(video["half"],int((frame // fps)) // 60, int((frame // fps)) % 60 if (frame // fps) % 60>=10 else "0{}".format(int((frame // fps) % 60)))
                prediction_dict["label"] = dict(zip(EVENT_DICTIONARY_V2.values(), EVENT_DICTIONARY_V2.keys()))[label_pre]
                prediction_dict["position"] = str(int(frame/fps*1000))
                prediction_dict["confidence"] = str(scores_pred)
                predictions.append(prediction_dict)
        predict_dict["predictions"]=predictions
        if not os.path.exists(predict_dict["UrlLocal"]):
            os.makedirs(predict_dict["UrlLocal"])
        with open("{:}/{:}.results_spotting.json".format(predict_dict["UrlLocal"],'.'.join(video["video_id"].split('.')[:-2])), "w") as f:
            json.dump(predict_dict, f, indent=2)