import pandas as pd
from tqdm import tqdm
import json

def main():
    shots_path = "../resources/sfdad/shots.csv"
    output_path = "../resources/annotations/sfdad_anno.csv"
    charbank_path = "../resources/charbanks/sfdad_charbank_empty.json"

    shots = pd.read_csv(shots_path)
    video_ids = shots.video_id.unique()

    # video_id,shot_id,start,end,duration,text,bboxes,pred_ids
    text = ""
    bboxes = "[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]"
    pred_ids = {}
    
    results = []
    charbank = {}
    for video_id in tqdm(video_ids, total=len(video_ids)):
        charbank[video_id] = []
        video_shots = shots[(shots.video_id == video_id)]
        for _, row in video_shots.iterrows():
            shot_id = row['shot_id']
            start = row['Start Time (seconds)']
            end = row['End Time (seconds)']
            duration = row['Length (seconds)']
            results.append({
                'video_id': video_id,
                'shot_id': shot_id,
                'start': start,
                'end': end,
                'duration': duration,
                'text': text,
                'bboxes': bboxes,
                'pred_ids': pred_ids,
            })

    results = pd.DataFrame(results)
    results.to_csv(output_path, index=False)

    json.dump(charbank, open(charbank_path, 'w'), indent=4)

if __name__ == "__main__":
    main()
