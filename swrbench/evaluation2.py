def create_round2_prompt(item):
    instance = item['instance']
    pred = item['pred']
    pr_title = instance["pr_title"]
    pr_statement = instance["pr_statement"]
    pr_timeline = instance["pr_timeline"]
    pred_review = pred["review"]
    
    defects_description = []
    for i, defect in enumerate(instance['defects']):
        defects_description.append(
            f"<Ground Truth Defect GT-{i+1}>\n"
            f"Defect Type: {defect['defect_type']} {DEFECT_TYPE_TEXT_MAP[defect['defect_type']]}\n"
            f"Defect Description: {defect['defect_discussion']['description']}\n"
            f"Defect Code Snippet: {defect['defect_introducing_commit']['code']}\n"
            f"</Ground Truth Defect GT-{i+1}>\n"
        )   
    
    # Extract ground truth review content from timeline
    ground_truth_reviews = []
    for item in pr_timeline:
        if item['type'] == 'description':
            continue  
        elif item['type'] == 'comment':
            ground_truth_reviews.append(f"<Start of Comment>\nTime: {item['created_at']}\nAuthor: {item['user']}\nComment: {item['body']}\n<End of Comment>")
        elif item['type'] == 'review_comment':
            reply_comments = ""
            for comment in item['reply']:
                reply_comments += f"<Start of Sub Review Comment>\nTime: {comment['created_at']}\nAuthor: {comment['user']}\nComment: {comment['body']}\n<End of Sub Review Comment>\n"
            ground_truth_reviews.append(f"<Start of Review Comment>\n<Start of Related Diff Hunk>\nFile: {item['path']}\n{item['diff_hunk']}\n<End of Related Diff Hunk>\n{reply_comments}<End of Review Comment>")
        elif item['type'] == 'commit':            
            ground_truth_reviews.append(f"<Start of Commit>\nTime: {item['date']}\nSHA: {item['sha']}\nAuthor: {item['author']}\nMessage: {item['message']}\n<End of Commit>")
        elif item['type'] == 'review':
            ground_truth_reviews.append(f"<Start of Review>\nTime: {item['created_at']}\nAuthor: {item['user']}\nReview: {item['body']}\n<End of Review>")
    
    assert len(ground_truth_reviews) > 0, "No ground truth review content available."
    
    ground_truth_reviews = "\n".join(ground_truth_reviews)
    
    return (
        "You are an objective Evaluation Assistant. Your task is to evaluate a predicted code review based on how accurately and usefully it identifies the actual defects present in a pull request, comparing it against a ground truth list of defects.\n"
        "**CONTEXT:**\n"
        "<Ground Truth Defects>\n"
        "This section lists the actual defects confirmed to be introduced by the pull request. Each defect has an ID (e.g., GT-1), a type, and a description.\n"
        f"{ground_truth_reviews.strip()}\n"
        "</Ground Truth Defects>\n"
        "**EVALUATION TASK:**\n"
        "**Match Ground Truth Defects to Predicted Defects:**\n"
        "*   For *each* defect listed in `<Ground Truth Defects>` (e.g., `GT-1`, `GT-2`, etc.): \n"
        "    *   Determine if any of the predicted defects identified in Step 1 (`PRED-1`, `PRED-2`, etc.) successfully hit this ground truth defect. \n"
        "    *   A `Hit` occurs if a predicted defect *substantially and correctly* identifies the ground truth defect. This typically corresponds to a usefulness score of 4 or higher for the hitting predicted defect, with higher scores indicating a better hit. \n"
        "    *   If multiple predicted defects seem relevant, choose the one that provides the *most accurate and complete* description of the ground truth defect as the primary hit. \n"
        "    *   Record `Hit: YES` or `Hit: NO`. \n"
        "    *   If `Hit: YES`, record the ID (e.g., `PRED-1`) of the predicted defect that hit it in the `Hit by:` field. If `Hit: NO`, leave `Hit by: N/A`. \n"
        "**OUTPUT FORMAT:**\n"
        "Present your evaluation strictly in the following format. Do not add any introductory or concluding remarks outside this structure.\n"
        "**EVALUATION RESULT:**\n"
        "**Ground Truth Defect Coverage:**\n"
        "<Ground Truth Defect GT-1>\n"
        "Defect Type: [Copy from the corresponding Ground Truth Defect input]\n"
        "Description: [Copy from the corresponding Ground Truth Defect input]\n"
        "Hit: [YES or NO]\n"
        "Hit by: [PRED-ID if Hit is YES, otherwise N/A]\n"
        "</Ground Truth Defect GT-1>\n"
        "<Ground Truth Defect GT-2>\n"
        "Defect Type: [Copy from the corresponding Ground Truth Defect input]\n"
        "Description: [Copy from the corresponding Ground Truth Defect input]\n"
        "Hit: [YES or NO]\n"
        "Hit by: [PRED-ID if Hit is YES, otherwise N/A]\n"
        "</Ground Truth Defect GT-2>\n"
        "[... Continue for all Ground Truth Defects listed in the input ...]\n"
        "**END OF EVALUATION RESULT**\n"
    )

def create_defect_round1_prompt(item):
    instance = item['instance']
    pred = item['pred']
    pr_title = instance["pr_title"]
    pr_statement = instance["pr_statement"]
    pr_timeline = instance["pr_timeline"]
    pred_review = pred["review"]
    
    defects_description = []
    for i, defect in enumerate(instance['defects']):
        defects_description.append(
            f"<Ground Truth Defect GT-{i+1}>\n"
            f"Defect Type: {defect['defect_type']} {DEFECT_TYPE_TEXT_MAP[defect['defect_type']]}\n"
            f"Defect Description: {defect['defect_discussion']['description']}\n"
            f"Defect Code Snippet: {defect['defect_introducing_commit']['code']}\n"
            f"</Ground Truth Defect GT-{i+1}>\n"
        )   
    
    # Extract ground truth review content from timeline
    ground_truth_reviews = []
    for item in pr_timeline:
        if item['type'] == 'description':
            continue  
        elif item['type'] == 'comment':
            ground_truth_reviews.append(f"<Start of Comment>\nTime: {item['created_at']}\nAuthor: {item['user']}\nComment: {item['body']}\n<End of Comment>")
        elif item['type'] == 'review_comment':
            reply_comments = ""
            for comment in item['reply']:
                reply_comments += f"<Start of Sub Review Comment>\nTime: {comment['created_at']}\nAuthor: {comment['user']}\nComment: {comment['body']}\n<End of Sub Review Comment>\n"
            ground_truth_reviews.append(f"<Start of Review Comment>\n<Start of Related Diff Hunk>\nFile: {item['path']}\n{item['diff_hunk']}\n<End of Related Diff Hunk>\n{reply_comments}<End of Review Comment>")
        elif item['type'] == 'commit':            
            ground_truth_reviews.append(f"<Start of Commit>\nTime: {item['date']}\nSHA: {item['sha']}\nAuthor: {item['author']}\nMessage: {item['message']}\n<End of Commit>")
        elif item['type'] == 'review':
            ground_truth_reviews.append(f"<Start of Review>\nTime: {item['created_at']}\nAuthor: {item['user']}\nReview: {item['body']}\n<End of Review>")
    
    assert len(ground_truth_reviews) > 0, "No ground truth review content available."
    
    ground_truth_reviews = "\n".join(ground_truth_reviews)
    
    return (
        "You are an objective Evaluation Assistant. Your task is to evaluate a predicted code review based on how accurately and usefully it identifies the actual defects present in a pull request, comparing it against a ground truth list of defects.\n"
        "**CONTEXT:**\n"
        "<Pull Request Details>\n"
        f"- **Pull Request Title**: {pr_title}\n"
        f"- **Pull Request Description**: {pr_statement}\n"
        f"- **Pull Request Review Content**: \n{ground_truth_reviews}\n"
        "</Pull Request Details>\n"
        "<Ground Truth Defects>\n"
        "This section lists the actual defects confirmed to be introduced by the pull request. Each defect has an ID (e.g., GT-1), a type, and a description.\n"
        f"{ground_truth_reviews.strip()}\n"
        "</Ground Truth Defects>\n"
        "<Predicted Review>\n"
        "This is the review result generated by the system being evaluated.\n"
        f"{pred_review.strip()}\n"
        "</Predicted Review>\n"
        "**EVALUATION TASK:**\n"
        "Follow these steps precisely:\n"
        "1.  **Identify Predicted Defects:**\n"
        "    *   Carefully read the `<Predicted Review>`. \n"
        "    *   Identify each distinct potential defect or issue mentioned. \n"
        "    *   Assign a unique sequential ID to each identified issue, starting from `PRED-1`, `PRED-2`, etc. \n"
        "    *   Extract the core description of the issue from the `<Predicted Review>`. \n"
        "2.  **Assess Usefulness of Predicted Defects:**\n"
        "    *   For *each* defect identified in Step 1 (`PRED-1`, `PRED-2`, etc.), evaluate its usefulness based on how well it corresponds to any *actual* defect described in `<Ground Truth Defects>` or if it provides other value (e.g., identifies a real but minor issue not in the ground truth, stylistic suggestion vs. functional defect). \n"
        "    *   Use the following 0-10 scale: \n"
        "        *   10: Precisely identifies a ground truth defect with a clear and accurate explanation. \n"
        "        *   7-9: Correctly identifies the core of a ground truth defect but might lack some detail or precision in the explanation. \n"
        "        *   4-6: Partially identifies aspects of a ground truth defect, or identifies a valid but less critical issue not listed in the ground truth. The comment is relevant but may not pinpoint the main problem effectively. \n"
        "        *   1-3: Mentions something tangentially related to a ground truth defect or a code element, but misses the core issue or is very vague. Might identify a trivial non-issue (e.g., minor style nitpick presented as a defect). \n"
        "        *   0: Completely irrelevant, incorrect, hallucinates an issue, or significantly misinterprets the code/defect. \n"
        "**OUTPUT FORMAT:**\n"
        "Present your evaluation strictly in the following format. Do not add any introductory or concluding remarks outside this structure.\n"
        "**EVALUATION RESULT:**\n"
        "**1. Analysis of Predicted Review Defects:**\n"
        "<Predicted Defect PRED-1>\n"
        "Description: [Extract the description of the first defect identified from the <Predicted Review>]\n"
        "Usefulness: [Assign the usefulness score (0-10) based on the criteria]\n"
        "</Predicted Defect PRED-1>\n"
        "<Predicted Defect PRED-2>\n"
        "Description: [Extract the description of the second defect identified from the <Predicted Review>]\n"
        "Usefulness: [Assign the usefulness score (0-10) based on the criteria]\n"
        "</Predicted Defect PRED-2>\n"
        "[... Continue for all identified predicted defects ...]\n"
        "**END OF EVALUATION RESULT**\n"
    )

def create_clean_round1_prompt(item):
    instance = item['instance']
    pred = item['pred']
    pr_title = instance["pr_title"]
    pr_statement = instance["pr_statement"]
    # For clean PRs, we don't need the detailed timeline, just the context that it's clean.
    pred_review = pred["review"]

    ground_truth_context = "The pull request is confirmed to be good and clean; no actual defects were introduced."

    return (
        "You are an objective Evaluation Assistant. Your task is to evaluate a predicted code review for a pull request that is known to be **clean** (contains no defects).\n"
        "**CONTEXT:**\n"
        "<Pull Request Details>\n"
        f"- **Pull Request Title**: {pr_title}\n"
        f"- **Pull Request Description**: {pr_statement}\n"
        f"- **Ground Truth Assessment**: {ground_truth_context}\n"
        "</Pull Request Details>\n"
        "<Predicted Review>\n"
        "This is the review result generated by the system being evaluated.\n"
        f"{pred_review.strip()}\n"
        "</Predicted Review>\n"
        "**EVALUATION TASK:**\n"
        "Since the Ground Truth is that the PR is clean, any issue raised in the `<Predicted Review>` is a **False Positive**. Your task is to identify these false positives and assess their severity.\n"
        "Follow these steps precisely:\n"
        "1.  **Identify Potential Issues Raised:**\n"
        "    *   Carefully read the `<Predicted Review>`. \n"
        "    *   Does the review correctly state the PR is good/clean, or does it raise potential issues/defects? \n"
        "    *   If it raises issues, identify each distinct potential issue mentioned. \n"
        "    *   Assign a unique sequential ID to each identified issue, starting from `PRED-FP-1`, `PRED-FP-2`, etc. (FP for False Positive). \n"
        "    *   Extract the core description of the issue from the `<Predicted Review>`. \n"
        "    *   If the review *correctly* states the PR is clean and raises *no* issues, proceed directly to the output format and indicate 'No False Positives Identified'.\n"
        "2.  **Assess False Positive Severity:**\n"
        "    *   For *each* issue identified in Step 1 (`PRED-FP-1`, `PRED-FP-2`, etc.), evaluate its severity as a false positive using the following 0-10 scale: \n"
        "        *   0: Not a defect, just a general approval statement (e.g., \"Looks good\", \"No issues found\"). This is the ideal outcome.\n"
        "        *   1-3: Minor, potentially subjective style suggestion or nitpick presented inappropriately as a necessary change or defect. Low impact, unlikely to mislead significantly.\n"
        "        *   4-6: Suggests a non-existent issue based on a plausible misunderstanding of the code or requirements, or flags a very minor best practice deviation as a defect. Moderately misleading.\n"
        "        *   7-9: Fabricates a significant functional, logical, or performance issue that does not exist. High potential to mislead the developer and waste time.\n"
        "        *   10: Hallucinates a critical security vulnerability, data corruption risk, or major architectural flaw where none exists. Critical impact.\n"
        "**OUTPUT FORMAT:**\n"
        "Present your evaluation strictly in the following format. Do not add any introductory or concluding remarks outside this structure.\n"
        "**EVALUATION RESULT:**\n"
        "**Analysis of Predicted Review for Clean PR:**\n"
        "[If no false positives were identified, state: No False Positives Identified.]\n"
        "[Otherwise, list each identified false positive:]\n"
        "<Predicted False Positive PRED-FP-1>\n"
        "Description: [Extract the description of the first false positive identified from the <Predicted Review>]\n"
        "Severity: [Assign the false positive severity score (0-10) based on the criteria]\n"
        "</Predicted False Positive PRED-FP-1>\n"
        "<Predicted False Positive PRED-FP-2>\n"
        "Description: [Extract the description of the second false positive identified from the <Predicted Review>]\n"
        "Severity: [Assign the false positive severity score (0-10) based on the criteria]\n"
        "</Predicted False Positive PRED-FP-2>\n"
        "[... Continue for all identified predicted false positives ...]\n"
        "**END OF EVALUATION RESULT**\n"
    )


def evaluate_multi(args, item, logger):
    e1 = evaluate_one(args, item, logger)
    e2 = evaluate_one(args, item, logger)
    e3 = evaluate_one(args, item, logger)
    
    # Create a copy of gt_defects from e1
    gt_defects = {}
    for gt_id, gt_info in e1['gt_defects'].items():
        gt_defects[gt_id] = gt_info.copy()
    
    # Check if each defect is hit in all three evaluations
    hit_count = 0
    for gt_id in gt_defects:
        is_hit_in_all = (
            e1['gt_defects'].get(gt_id, {}).get('hit', '').lower() == 'yes' and
            e2['gt_defects'].get(gt_id, {}).get('hit', '').lower() == 'yes' and
            e3['gt_defects'].get(gt_id, {}).get('hit', '').lower() == 'yes'
        )
        
        if is_hit_in_all:
            gt_defects[gt_id]['hit'] = 'YES'
            hit_count += 1
        else:
            gt_defects[gt_id]['hit'] = 'NO'
            gt_defects[gt_id]['hit_by'] = 'N/A'
    
    result = {
        "instance_id": item['instance']['instance_id'],
        "hit": hit_count,
        "total": e1['total'],
        "gt_defects": gt_defects,
        "pred_defects": e1['pred_defects'],
        "raw_results": [e1, e2, e3]
    }
    return result