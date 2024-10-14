import argparse
import os
import sys

def main(args):
    # Dynamically set environment variables based on args.model_path
    model_base_path = os.path.dirname(args.model_path)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_CACHE'] = model_base_path
    #os.environ['TRANSFORMERS_CACHE'] = "/home/ridouane/weights/cache_dir"
    #os.environ['TRANSFORMERS_CACHE'] = "/lustre/fswork/projects/rech/kcn/ucm72yx/weights"

    # Dynamically set sys.path
    sys.path.append(os.path.join(model_base_path, "VideoLLaMA2"))
    #sys.path.append("/home/ridouane/weights/cache_dir/VideoLLaMA2")
    #sys.path.append("/lustre/fswork/projects/rech/kcn/ucm72yx/weights/VideoLLaMA2")

    import torch
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # Import local modules after updating sys.path
    from videollama2.model.builder import load_pretrained_model
    from promptloader import get_general_prompt
    from dataloader import MADEval_FrameLoader, TVAD_FrameLoader, CMDAD_FrameLoader, SFDAD_FrameLoader

    # initialize VideoLLaMA2
    model_path = args.model_path
    model_name = args.model_path.split("/")[-1]
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
    model = model.cuda()

    # formulate text prompt template
    general_prompt = get_general_prompt(args.prompt_idx)
    
    # build dataloader
    if args.dataset == "tvad":
        D = TVAD_FrameLoader
        video_type = "TV series"
    elif args.dataset == "cmdad":
        D = CMDAD_FrameLoader
        video_type = "movie"
    elif args.dataset == "madeval":
        D = MADEval_FrameLoader
        video_type = "movie"
    elif args.dataset == "sfdad":
        D = SFDAD_FrameLoader
        video_type = "movie"
    else:
        print("Check dataset name")
        sys.exit()

    anno_df = pd.read_csv(args.anno_path)
    anno_df.sort_values(by='shot_id', inplace=True)

    if args.iteration > 0:
        anno_df = anno_df.iloc[args.samples_per_job * args.iteration: args.samples_per_job * (args.iteration + 1)]

    ad_dataset = D(anno_df=anno_df, tokenizer=tokenizer, processor=processor, general_prompt=general_prompt, video_type = video_type,
                                anno_path=args.anno_path, charbank_path=args.charbank_path, video_dir=args.video_dir,
                                label_type=args.label_type, label_width=args.label_width, label_alpha=args.label_alpha)
              
    loader = torch.utils.data.DataLoader(ad_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                            collate_fn=ad_dataset.collate_fn, shuffle=False, pin_memory=True)
    
    output_dir = f"{args.output_dir}/{args.dataset}_ads"
    os.makedirs(output_dir, exist_ok=True)

    start_sec = []
    end_sec = []
    start_sec_ = []
    end_sec_ = []
    text_gt = []
    text_gen = []
    vids = []
    for idx, input_data in tqdm(enumerate(loader), total=len(loader), desc='EVAL'): 
        videos = [video.cuda() for video in input_data["video"]]
        modal_list = ['video'] * len(videos)
        input_ids = input_data["input_id"]

        # left padding for batch inference
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x.flip(dims=[0]) for x in input_ids],
            batch_first=True,
            padding_value=tokenizer.pad_token_id).flip(dims=[1]).cuda()
        attention_mask=input_ids.ne(tokenizer.pad_token_id).cuda()
        
        # repeat experiments max_exp
        redo_exp = True
        counter = 0
        while counter < args.max_exp and redo_exp:
            # if counter >= 1:
            #     print("Redoing experiment due to over-length (unstable) outputs")
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images_or_videos=videos,
                    modal_list=modal_list,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [output.replace("\n", " ") for output in outputs]
            redo_exp = False
            counter += 1
            for output in outputs:
                if len(output) > 2000: # if output length > 2000, assume it is unstable and redo the experiment
                    redo_exp = True
                
        vids.extend(input_data["imdbid"])
        text_gt.extend(input_data["gt_text"])
        text_gen.extend(outputs) 
        start_sec.extend(input_data["start"])
        end_sec.extend(input_data["end"])
        start_sec_.extend(input_data["start_"])
        end_sec_.extend(input_data["end_"])

        output_df = pd.DataFrame.from_records({'vid': vids, 'start': start_sec, 'end': end_sec, 'start_': start_sec_, 'end_': end_sec_, 'text_gt': text_gt, 'text_gen': text_gen})
        output_df.to_csv(os.path.join(output_dir, f'stage1_{args.save_prefix}-{args.label_type}-{args.prompt_idx}_it{args.iteration}.csv'))

    output_df = pd.DataFrame.from_records({'vid': vids, 'start': start_sec, 'end': end_sec, 'start_': start_sec_, 'end_': end_sec_, 'text_gt': text_gt, 'text_gen': text_gen})
    output_df.to_csv(os.path.join(output_dir, f'stage1_{args.save_prefix}-{args.label_type}-{args.prompt_idx}_it{args.iteration}.csv'))
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('llama')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--dataset', default="tvad", type=str)
    parser.add_argument('--prompt_idx', default=0, type=int)
    parser.add_argument('--save_prefix', default="", type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--anno_path', default=None, type=str)
    parser.add_argument('--charbank_path', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--video_dir', default=None, type=str)
    parser.add_argument('--label_type', default="circles", type=str)
    parser.add_argument('--label_width', default=10, type=int, help='label_width, 10 in a canvas 1000')
    parser.add_argument('--label_alpha', default=0.8, type=float)
    parser.add_argument('-j', '--num_workers', default=8, type=int, help='init mode')
    parser.add_argument('--max_exp', default=5, type=int, help='maximum number of repeating experiments')
    parser.add_argument('--iteration', default=-1, type=int, help='iteration')
    parser.add_argument('--samples_per_job', default=8192, type=int, help='number of subsamples')
    args = parser.parse_args()

    main(args)
