import argparse
from pathlib import Path
from fastai import basic_train as fa_train
from fastai import basic_data as fa_bsdata
from fastai.vision import image as fa_image
from fastai.vision import data as fa_data
import pandas as pd
import json

import warnings
warnings.filterwarnings("ignore")

fish_model = '/scratch/genomics/triznam/fastai_inference/Peru_Fish.pkl'
classes = ['Ancistrus', 'Apistogramma', 'Astyanax', 'Bario', 'Bryconops', 'Bujurquina', 
           'Bunocephalus', 'Characidium', 'Charax', 'Copella', 'Corydoras', 'Creagrutus', 
           'Curimata', 'Doras', 'Erythrinus', 'Gasteropelecus', 'Gymnotus', 'Hemigrammus', 
           'Hyphessobrycon', 'Moenkhausia', 'Otocinclus', 'Oxyropsis', 'Phenacogaster', 
           'Pimelodella', 'Prochilodus', 'Pygocentrus', 'Pyrrhulina', 'Rineloricaria', 'Sorubim', 
           'Tatia', 'Tetragonopterus', 'Tyttocharax']
parent_path = Path(fish_model).parent

ap = argparse.ArgumentParser()
ap.add_argument('-i', "--imagepath", 
                help="direct path of image to classify")
ap.add_argument('-d', "--imagedir", 
                help="path for directory containing images to classify")
ap.add_argument("-o", "--outfile",
                help="file path for classification tsv output")
ap.add_argument("-k", "--topk", type=int,
                help="top k classifications to return per image")
args = ap.parse_args()

if args.topk is not None:
    topk = int(args.topk)
else:
    topk = 5

def tensor_to_topk(prediction_tensor, class_list, k):
    prediction_tuples = [(class_list[i], float(conf) * 100) for i, conf in enumerate(list(prediction_tensor))]
    top_k = sorted(prediction_tuples, key = lambda x: x[1], reverse=True)[:k]
    return top_k

#Load model and test against single image
if args.imagepath is not None:
    fish_learner = fa_train.load_learner(parent_path, fish_model)
    test_image = fa_image.open_image(args.imagepath)
    prediction = fish_learner.predict(test_image)
    topk_preds = tensor_to_topk(prediction[2], classes, topk)
    print(json.dumps(topk_preds, indent=2))

elif args.imagedir is not None:
    image_list = fa_data.ImageList.from_folder(args.imagedir)
    fish_learner = fa_train.load_learner(parent_path, fish_model,
                                        test=image_list)
    preds,_ = fish_learner.get_preds(ds_type=fa_bsdata.DatasetType.Test)
    file_names = [image.name for image in image_list.items]
    prediction_results = []
    for i, file_name in enumerate(file_names):
        prediction_result = {'file_name': file_name}
        topk_preds = tensor_to_topk(preds[i], classes, topk)
        for k, pred in enumerate(topk_preds):
            pred_class, pred_conf = pred
            class_key = 'class_' + str(k + 1)
            conf_key = 'conf_' + str(k + 1)
            prediction_result[class_key] = pred_class
            prediction_result[conf_key] = pred_conf
        prediction_results.append(prediction_result)
    prediction_df = pd.DataFrame(prediction_results)
    prediction_df.to_csv(args.outfile,
                         index=False, sep='\t')


else:
    print('You need to either select imagepath or imagedir')
