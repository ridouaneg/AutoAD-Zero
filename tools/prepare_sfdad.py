import pandas as pd
from tqdm import tqdm
import json

def main():
    shots_path = "../resources/sfdad/shots.csv"
    output_path = "../resources/annotations/sfdad_full_anno.csv"
    charbank_path = "../resources/charbanks/sfdad_full_charbank_empty.json"
    with_subs = False

    shots_df = pd.read_csv(shots_path)

    video_ids = shots_df.video_id.unique()

    # video_id,shot_id,start,end,duration,text,bboxes,pred_ids
    text = ""
    bboxes = "[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]"
    pred_ids = {}

    annotations = []
    charbank = {}

    for video_id in tqdm(video_ids, total=len(video_ids)):
        charbank[video_id] = []
        video_shots = shots_df[(shots_df.video_id == video_id)]
        for _, row in video_shots.iterrows():
            shot_id = row['shot_id']
            start = row['Start Time (seconds)']
            end = row['End Time (seconds)']
            duration = row['Length (seconds)']

            if with_subs:
                #subs_in_shot = subs_df[
                #    (subs_df['video_id'] == video_id) & 
                #    (subs_df['start'] < end) & 
                #    (subs_df['end'] > start)
                #].sort_values(by='start')
                pass
            else:
                subs_in_shot = pd.DataFrame()

            if subs_in_shot.empty:
                annotations.append({
                    'video_id': video_id,
                    'shot_id': shot_id,
                    'start': start,
                    'end': end,
                    'duration': duration,
                    'text': "",
                    'bboxes': "[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]",
                    'pred_ids': "{}"
                })
            else:
                # Track the current time within the shot
                current_time = start
        
                # Iterate over the subtitles and capture gaps between them
                for sub_index, sub in subs_in_shot.iterrows():
                    if sub['start'] > current_time:
                        # Add an annotation for the gap before this subtitle
                        annotations.append({
                            'video_id': video_id,
                            'shot_id': shot_id,
                            'start': current_time,
                            'end': sub['start'],
                            'duration': sub['start'] - current_time,
                            'text': "",
                            'bboxes': "[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]",
                            'pred_ids': "{}"
                        })
            
                    # Update the current time to the end of the subtitle
                    current_time = sub['end']
        
                # If there's still time left after the last subtitle, annotate that
                if current_time < end:
                    annotations.append({
                        'video_id': video_id,
                        'shot_id': shot_id,
                        'start': current_time,
                        'end': end,
                        'duration': end - current_time,
                        'text': "",
                        'bboxes': "[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]",
                        'pred_ids': "{}"
                    })
    
        results = pd.DataFrame(annotations)
        results.to_csv(output_path, index=False)

    json.dump(charbank, open(charbank_path, 'w'), indent=4)

if __name__ == "__main__":
    main()
