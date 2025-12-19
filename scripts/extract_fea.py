import argparse
import os
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from RL.setting import *
from RL.Fea_extracter import *


POS_PUNCTUATION = [".", "?", "...", "!", "+/", "+/?", "" "...?", ",", "-", "+\"/.", "+...", "++/.", "+/."]

'''
extract features from the generation csv file; OR
append features to the generated file: assuming the preprocessed content
'''

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Extract feature from the generations')

    parser.add_argument('--base_dir', type=str, default=f'{ROOT}/dataset/eval/gen/seq2seq',
                        help='Path to the generation file')
    parser.add_argument('--output_dir', type=str, default=f'{ROOT}/dataset/eval/gen/seq2seq/example',
                        help='Directory to save results')
    parser.add_argument('--history_num', type=int, default=4,
                        help='the number of prior turns in multi-turn generations')
    parser.add_argument('--speaker_lst', type=list, default=['CHI','ADULT'],
                        help='the tested speaker roles')
    parser.add_argument('--model_lst', type=list, default=['qwen2.5','mistral'],
                        help='the tested single-turn models')
    parser.add_argument('--gen_type', type=str, default='single',
                        help='different types of generation: single; multi; all; skip')
    parser.add_argument('--context_len', type=str, default=4,
                        help='context_len of the multi-turn generations')
    parser.add_argument('--word_info_path', type=str, default=f'{ROOT}/dataset/eval/gen/seq2seq/material/word_info.csv',
                        help='Directory to word info')
    parser.add_argument('--func_info_path', type=str, default=f'{ROOT}/dataset/eval/gen/seq2seq/material/func_info.csv',
                        help='Directory to func info')
    parser.add_argument('--sent_model_path', type=str, default='paraphrase-MiniLM-L6-v2',
                        help='Name or path to the sentence model')
    parser.add_argument('--fea_set', type=list, default=['word','syn','align','div','repe'],   #['word','syn','align','div'],
                        help='Name of the feature set to be extracted; keys of the dict; new combination should be put as key-value pairs')
    parser.add_argument('--model_path', type=str, default= f'{ROOT}/model/CRF_SA',
                        help='Path to the SA model')
    parser.add_argument('--debug', action="store_true",
                        help='whether to debug the code with 4 rows')
    return parser.parse_args(argv)

#TODO: reduce the number of arguments for multi-turn gen

role_dict = {'CHI':'ADULT', 'ADULT':'CHI'}
target_POS = ['VERB', 'NOUN','PROPN', 'ADV', 'ADJ','PRON','INTJ']
content_POS = ['VERB', 'NOUN', 'ADV', 'ADJ','PROPN']
fea_dict = {'word': ['freq','conc','word_len','sent_len',"ttr","non_word_type_rate","non_word_rate","distinct_2","distinct_3"],
            'align': ['lemma_align', 'dep_align', 'sem_align'],
            'repe': ['lemma_repe', 'dep_repe', 'sem_repe'],
            'div': ['lemma_div','dep_div','sem_div'],
            'syn':['tree_depth', 'clause', 'POS_count', 'pp_den','lex_den', 'func_den','func_den_new'],
            'fea':['func_den_new'],
            'SA':['speech_act'],
            'semEnt':['sem_ent']
            }


def apply_feature_extractor(
    df: pd.DataFrame,
    content_header: str,
    speaker_header: str,
    feature_extractor,
    features: list,
    content_POS=content_POS,
    target_POS=target_POS,
    func_header='word',
    SA_annotator=None,
    SA_processor=None
) -> pd.DataFrame:
    """
    Apply feature extractor on the dialogue level.
    Returns a DataFrame with extracted features from `df`.
    """
    # Initialize lists
    sentences = df[content_header]
    doc_lst, lemma_lst, dep_graphs = feature_extractor.extract_doc(sentences, features)
    word_vectors = feature_extractor.extract_vec(sentences, features)
    frame_lst = []

    # Compute turn-level features
    if any(f in features for f in fea_dict['word']) or any(f in features for f in fea_dict['syn']):
        turn_features = [item for item in features if item in set(fea_dict['word']) or item in set(fea_dict['syn'])]
        results = []
        for idx in tqdm(range(len(sentences))):  # Handle potential NaN values
            target_sent = sentences.iloc[idx]
            doc = doc_lst[idx] if idx < len(doc_lst) else ''
            lemma = lemma_lst[idx] if idx < len(lemma_lst) else ''
            results.append(
                feature_extractor.get_turn_fea(
                    target_sent, doc, lemma, turn_features, content_POS, target_POS, func_header
                )
            )
        # Concatenate turn-level features
        turn_fea = pd.concat(results, ignore_index=True)
        turn_fea = turn_fea.reset_index(drop=True)
        frame_lst.append(turn_fea)

    # Compute alignment features
    if any(f in features for f in fea_dict['align']) or any(f in features for f in fea_dict['repe']):
        align_fea = feature_extractor.get_align_fea(features, lemma_lst, dep_graphs, word_vectors)
        align_fea = pd.DataFrame(align_fea)
        align_fea = align_fea.reset_index(drop=True)
        frame_lst.append(align_fea)

    # Compute diversity features
    if any(f in features for f in fea_dict['div']):
        div_fea_dict = {}
        # Ensure DataFrame indices are sequential
        df = df.reset_index(drop=True)
        # Group by the specified column
        df_grouped = df.groupby(speaker_header)

        for speaker, df_group in df_grouped:
            indices = df_group.index.tolist()  # Get indices for the grouped DataFrame
            div_fea_dict[speaker] = feature_extractor.get_div_fea(
                features, indices, lemma_lst, dep_graphs, word_vectors
            )
        # Convert diversity features to DataFrame
        feature_df = pd.DataFrame.from_dict(div_fea_dict, orient='index')
        # Merge features with the main DataFrame
        merged_df = df.merge(feature_df, left_on=speaker_header, right_index=True, how='left')
        # Extract only the new columns added from feature_df
        div_fea = merged_df[feature_df.columns]
        div_fea = div_fea.reset_index(drop=True)
        frame_lst.append(div_fea)

    # Annotate speech acts on the dialogue level
    if 'SA' in features:
        # convert the format to prepare for the SA extraction
        gen = SA_processor.convert_format(df, 'cleaned')
        sa_fea = SA_annotator.annotate(gen)[['speech_act']]
        frame_lst.append(sa_fea)
    # concatenate fea
    if len(frame_lst) > 0:
        fea_frame = pd.concat(frame_lst,axis=1)
    else:
        fea_frame = pd.DataFrame()
    return fea_frame, word_vectors



def main(argv):
    # Args parser
    args = parse_args(argv)
    word_info = pd.read_csv(args.word_info_path)
    func_info = pd.read_csv(args.func_info_path)
    sent_model = SentenceTransformer(args.sent_model_path)

    ##########################
    # Load files
    ##########################

    # Initialize the DataMerger class
    data_merger = DataMerger(base_dir=args.base_dir, role_dict=role_dict)

    # load files
    if args.gen_type == 'single':
        gen = data_merger.concat_single_turn(model_lst=args.model_lst, speaker_lst=args.speaker_lst,history_num=args.history_num)

    elif args.gen_type == 'multi':
        gen = data_merger.concat_multi_turn(history_num=args.history_num, context_len=args.context_len)

    elif args.gen_type == 'all':
        data_single = data_merger.concat_single_turn(model_lst=args.model_lst, speaker_lst=args.speaker_lst,history_num=args.history_num)
        data_multi = data_merger.concat_multi_turn(history_num=args.history_num, context_len=args.context_len)
        gen = pd.concat([data_single, data_multi])

    else:
        if args.base_dir.endswith('xlsx'):
            print('Loading excel file')
            print(args.base_dir)
            gen = pd.read_excel(args.base_dir)
        elif args.base_dir.endswith('csv'):
            gen = pd.read_csv(args.base_dir)
    print('Finished loading the file')


    # make sure that there is no additional index columns
    gen = gen.loc[:, "history":]
    if args.debug:
        gen = gen.head(200)
        print('#####################################')
        print('Entering the debugging mode!!!')

    # clean the gen column
    if args.gen_type == 'skip':
        print('Text preprocessing done, skipping cleaning the generations')
        if args.debug:
            filename_start = args.base_dir.split('/')[-1].split('.')[0]
            filename_end = args.base_dir.split('/')[-1].split('.')[-1]
            filename = f'{filename_start}_debug.{filename_end}'
        else:
            filename = args.base_dir.split('/')[-1]

    elif args.gen_type == 'other':
        gen['cleaned'] = gen['content'].apply(preprocess)
        print('Finished cleaning the generations')
        if args.debug:
            filename_start = args.base_dir.split('/')[-1].split('.')[0]
            filename_end = args.base_dir.split('/')[-1].split('.')[-1]
            filename = f'{filename_start}_debug.{filename_end}'
        else:
            filename = args.base_dir.split('/')[-1]

    else:
        gen['cleaned'] = gen['content'].apply(preprocess)
        print('Finished cleaning the generations')
        if args.debug:
            filename = f'example_{args.gen_type}_debug.xlsx'
        else:
            filename = f'example_{args.gen_type}.xlsx'

    if len(args.fea_set) == 0:
        print('Not extracting features')
        # save the results to the output dir
        gen.to_excel(f"{args.output_dir}/{filename}", index=False)
        print(f'Saving results to {args.output_dir}/{filename}')

    else:
        ##############################
        # Initialize fea extractors
        ##############################
        # get the feature list from the fea_dict
        fea_lst = [fea_dict.get(key) for key in args.fea_set]
        features = [x for xs in fea_lst for x in xs]

        # initialize fea extractor
        feature_extractor = FeatureExtractor(word_info,func_info, embedding_model=sent_model)

        annotator = None
        processor = None
        SemEnt_extractor = None
        if 'SA' in features:
            annotator = Annotator(
                model_path=args.model_path,
                use_bi_grams=True,
                use_pos=True,
                use_past=True,
                use_repetitions=True
            )
            processor = POSTagProcessor(POS_PUNCTUATION)


        ##############################
        # Extract features
        ##############################

        col = ['history','gen_type', 'model', 'test_role','month']
        grouped = gen.groupby(col)
        frame_all = pd.DataFrame()
        for group, group_frame in tqdm(grouped):
            # extract features for each conversation
            if args.debug:
                group_frame = group_frame.head(100)
                print('#####################################')
                print('Entering the debugging mode!!!')

            frame_lst = [group_frame]

            ########################################
            # Turn and dialogue level features
            ########################################
            word_vector_lst = []
            fea_frame = pd.DataFrame()
            file_frame_grouped = group_frame.groupby(['path'])  # group by the file names
            for file, file_frame in file_frame_grouped:
                # apply fea extractor on the dialogue level
                frame,word_vector = apply_feature_extractor(file_frame, 'cleaned', 'speaker',
                                    feature_extractor,features=features,SA_annotator=annotator,SA_processor=processor)
                word_vector_lst.extend(word_vector)
                fea_frame = pd.concat([fea_frame, frame])

            frame_lst.append(fea_frame)

            ##############################
            # Model-level features
            ##############################
            # Compute SemEnt fea
            if 'sem_ent' in features:
                semEnt_dict = {}
                # Ensure DataFrame indices are sequential
                group_frame = group_frame.reset_index(drop=True)
                # Group by the specified column
                df_grouped = group_frame.groupby('speaker')
                for speaker, df_group in df_grouped:
                    indices = df_group.index.tolist()  # Get indices for the grouped DataFrame
                    # intialize the class
                    semEnt_analyzer = SemEnt(word_vector_lst,indices=indices)
                    semEnt_dict[speaker] = semEnt_analyzer.compute_diversity()
                # Convert diversity features to DataFrame
                feature_df = pd.DataFrame.from_dict(semEnt_dict, orient='index')
                # set the header
                feature_df.columns = ["sem_ent", "k"]
                # Merge features with the main DataFrame
                merged_df = group_frame.merge(feature_df, left_on='speaker', right_index=True, how='left')
                # Extract only the new columns added from feature_df
                semEnt_fea = merged_df[feature_df.columns]
                semEnt_fea = semEnt_fea.reset_index(drop=True)
                frame_lst.append(semEnt_fea)
            # concatenate fea
            for i, frame in enumerate(frame_lst):
                frame.index = pd.RangeIndex(len(frame))
            # concatenate all the features
            result_frame = pd.concat(frame_lst, axis=1)
            # concatenate all the generations
            frame_all = pd.concat([frame_all,result_frame])

        if 'POS_count' in frame_all.columns:
            frame_all = frame_all.drop(columns=['POS_count'])

        # save the results to output dir
        os.makedirs(args.output_dir, exist_ok=True)
        if filename.endswith('.csv'):
            frame_all.to_csv(f"{args.output_dir}/{filename}", index=False)
        elif filename.endswith('.xlsx'):
            frame_all.to_excel(f"{args.output_dir}/{filename}", index=False)
        print(f'Saving results to {args.output_dir}/{filename}')


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

