import json

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(path, data):
    with open(path, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')

def fix_raw_pr_data(pr_data_path):
    pr_data = load_jsonl(pr_data_path)
    new_pr_data = []
    for pr in pr_data:
        if 'timeline' in pr:
            for event in pr['timeline']:
                if event['type'] == 'commit':
                    author_date = event['author_date']
        new_pr_data.append(pr)
    # save_jsonl(pr_data_path, new_pr_data)

def fix_pr_data(pr_data_path):
    map_keys = {
        'Change Introduction': 'change_introduction',
        'Change Type': 'change_type',
        'Discussion Evidence': 'discussion_evidence',
        'Severity Assessment': 'severity_assessment',
        'Resolving Information': 'resolving_information',
        'Change Introduction.Code Snippet': 'change_introduction.code_snippet',
        'Change Introduction.Commit SHA': 'change_introduction.commit_sha',
        'Discussion Evidence.Discussion Summary': 'discussion_evidence.discussion_summary',
        'Discussion Evidence.First Mention Timestamp': 'discussion_evidence.first_mention_timestamp',
        'Discussion Evidence.Original Reviewer Comment': 'discussion_evidence.original_reviewer_comment',
        'Severity Assessment.Severity Level': 'severity_assessment.severity_level',
        'Severity Assessment.Justification': 'severity_assessment.justification',
        'Resolving Information.Code Snippet': 'resolving_information.code_snippet',
        'Resolving Information.Commit SHA': 'resolving_information.commit_sha',
        'Resolving Information.Resolution Explanation': 'resolving_information.resolution_explanation',
    }
    pr_data = load_jsonl(pr_data_path)
    new_pr_data = []
    for pr in pr_data:
        new_defects = []
        if 'defects' not in pr:
            new_pr_data.append(pr)
            continue
        for defect in pr['defects']:
            if 'change_introduction' in defect:
                new_defects.append(defect)
                continue
            
            new_defect = {}
            for key, value in map_keys.items():
                if key in defect:
                    new_defect[value] = defect[key]
                elif '.' in key:
                    key1, key2 = key.split('.')
                    value1, value2 = value.split('.')
                    if key1 in defect:
                        new_defect[value1][value2] = defect[key1][key2]
                        del new_defect[value1][key2]
            new_defects.append(new_defect)
        pr['defects'] = new_defects
        new_pr_data.append(pr)
    save_jsonl(pr_data_path, new_pr_data)


def fix_pr_timeline(pr_data_path):
    pr_data = load_jsonl(pr_data_path)
    timeline_data_path = "/SWRBench/data/prs_raw/astropy__astropy__prs.jsonl"
    timeline_data = load_jsonl(timeline_data_path)
    timeline_data_dict = {f['number']: f for f in timeline_data}
    for pr in pr_data:
        if pr['pr_number'] not in timeline_data_dict:
            import pdb; pdb.set_trace()
        pr['timeline'] = timeline_data_dict[pr['pr_number']]['timeline']
    save_jsonl(pr_data_path, pr_data)

def compare_prompt(file_path1, file_path2):
    data1 = load_jsonl(file_path1)
    data2 = load_jsonl(file_path2)
    data2_dict = {f['instance_id']: f for f in data2}
    for i in range(len(data1)):
        if data1[i]['instance_id'] not in data2_dict:
            continue
        assert data1[i]['system_prompt'] == data2_dict[data1[i]['instance_id']]['system_prompt']
        assert data1[i]['user_prompt'] == data2_dict[data1[i]['instance_id']]['user_prompt']

    print("Prompt is the same")
    
if __name__ == "__main__":
    # /SWRBench/data/prs_raw/astropy__astropy__prs.jsonl
    # fix_raw_pr_data("/SWRBench/data/prs_raw/astropy__astropy__prs.jsonl")
    # fix_pr_data("/SWRBench/data/astropy__astropy/collect_pr_run_0/analysis_results.jsonl")
    # fix_pr_data("/SWRBench/data/astropy__astropy/collect_pr_run_1/analysis_results.jsonl")
    # fix_pr_data("/SWRBench/data/astropy__astropy/collect_pr_run_2/analysis_results.jsonl")
    
    # fix_pr_timeline("/SWRBench/data/astropy__astropy/collect_pr_run_0/analysis_results.jsonl")
    # fix_pr_timeline("/SWRBench/data/astropy__astropy/collect_pr_run_1/analysis_results.jsonl")
    # fix_pr_timeline("/SWRBench/data/astropy__astropy/collect_pr_run_2/analysis_results.jsonl")
    
    compare_prompt(
        "/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview/generation.jsonl", 
        "/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-flash-preview/generation.jsonl"
    )
    
    compare_prompt(
        "/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview/generation.jsonl", 
        "/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/claude-4-opus/generation.jsonl"
    )
    
    compare_prompt(
        "/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/gemini-2.5-pro-preview/generation.jsonl", 
        "/SWRBench/logs/swr_datasets_0520_d5c5/PR-Agent/claude-4-sonnet/generation.jsonl"
    )