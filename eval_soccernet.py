import os
import json

from tqdm import tqdm

from modeling.meta_archs import *

from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2
from SoccerNet.Downloader import getListGames

from torch.utils.data import Dataset,DataLoader

class SoccerNetTrainDataset(Dataset):
    def __init__(self,
                 feat_file_address,
                 features="ResNET_TF2_PCA512.npy",
                 status="train",
                 ):
        self.path = feat_file_address
        self.features = features
        self.listGames = getListGames(status)
        self.game_id = list()
        self.game_feats = list()
        for game in tqdm(self.listGames):
            id1 = os.path.join(self.path, game, "1_" + self.features)
            feat_half1 = torch.from_numpy(np.load(id1)).permute((1, 0))
            id2 = os.path.join(self.path, game, "2_" + self.features)
            feat_half2 = torch.from_numpy(np.load(id2)).permute((1, 0))

            self.game_id.append(id1)
            self.game_id.append(id2)
            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)

    def __getitem__(self, index):
        return {"video_id": self.game_id[index],
                "feats": self.game_feats[index]}

    def __len__(self):
        return len(self.game_feats)

if True:
    data = SoccerNetTrainDataset("./SoccerNet", status="valid")
    # batch_size must be 2, since a game has two 2 parts
    test_loader = DataLoader(data, batch_size=2, collate_fn=lambda x: x)

    device=torch.device("cuda:0")

    # choose one or several models
    models=[torch.load("./models/model_512_{:}.pt".format(i), map_location=device) for i in [1]]

    for (i, video_list) in tqdm(enumerate(test_loader),total=len(test_loader)):
        predict_dict = dict()
        predictions = []
        for video in video_list:
            segs = []
            labels = []
            scoress = []
            for model in models:
                model.eval()
                result = model([video])
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

            half = video["video_id"].split("/")[-1][0]
            predict_dict["UrlLocal"] = "/".join(video["video_id"].split("/")[2:5])

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
                # fps=2
                prediction_dict["gameTime"] = "{} - {}:{}".format(half,int((frame // 2)) // 60, int((frame // 2)) % 60 if (frame // 2) % 60>=10 else "0{}".format(int((frame // 2) % 60)))
                prediction_dict["label"] = dict(zip(EVENT_DICTIONARY_V2.values(), EVENT_DICTIONARY_V2.keys()))[label_pre]
                prediction_dict["position"] = str(int(frame/2*1000))
                prediction_dict["half"] = half
                prediction_dict["confidence"] = str(scores_pred)
                predictions.append(prediction_dict)
        predict_dict["predictions"]=predictions
        if not os.path.exists("./predictions/"+predict_dict["UrlLocal"]):
            os.makedirs("./predictions/"+predict_dict["UrlLocal"])
        with open("./predictions/{}/results_spotting.json".format(predict_dict["UrlLocal"]), "w") as f:
            json.dump(predict_dict, f, indent=2)

from SoccerNet.Evaluation.ActionSpotting import evaluate
results = evaluate(SoccerNet_path="./SoccerNet", Predictions_path="./predictions",
                   split="valid", version=2, prediction_file="results_spotting.json", metric="loose")
print("loose Average mAP: ", results["a_mAP"])
print("loose Average mAP per class: ", results["a_mAP_per_class"])
print("loose Average mAP visible: ", results["a_mAP_visible"])
print("loose Average mAP visible per class: ", results["a_mAP_per_class_visible"])
print("loose Average mAP unshown: ", results["a_mAP_unshown"])
print("loose Average mAP unshown per class: ", results["a_mAP_per_class_unshown"])