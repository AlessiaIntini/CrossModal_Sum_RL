"""Prompts of every task: frame selection (V2V), summary (V2T) and both (V2VT).

The selection rules are shared by SFT, DPO, GRPO, inference and zero-shot: trained
checkpoints depend on the exact wording, so it must not change.
"""

SELECTION_SYSTEM_PROMPT = (
    "You are a video summarization assistant. Select the frames that best "
    "summarize the important events in the video."
)
SUMMARY_STYLES = ("reference", "concise")
SUMMARY_WORD_RANGE = (42, 50)
SYSTEM_PROMPTS = {
    "concise": "You are a video summarization assistant. Select representative frames and write one concise summary.",
    "reference": "You are a video summarization assistant. Select representative frames and write a summary of the video.",
}
INDEX_PHRASES = {"numbered": ", numbered from", "indexed": " indexed from"}
ZERO_SHOT_INDEX_PHRASE = INDEX_PHRASES["numbered"]
SUMMARY_TIMELINE_END = "sampled at one frame per second. "
TIMESTAMP_SENTENCE = "(timestamp i). "
MMSS_NOTE = (
    "The video timestamps are shown as MM:SS; convert them to seconds, "
    "so 1:05 is frame 65, not frame 105. "
)


def build_timeline_intro(count, index_phrase=", numbered from", unit="frame"):
    # Opening sentence about the timeline; index_phrase is ignored on the pair timeline.
    if unit == "pair":
        return (
            f"The timeline contains {count} candidate segments, numbered from 0 "
            f"to {count - 1}. Each segment covers two consecutive seconds of the "
            "video. Select the segments that form the best extractive video "
            "summary.\n\n"
        )
    return (
        f"The timeline contains {count} candidate frames{index_phrase} 0 to "
        f"{count - 1}, sampled at one frame per second. Select the frames that "
        "form the best extractive video summary.\n\n"
    )


def build_selection_requirements(unit="frame"):
    # Target share of the timeline and shape of the selection.
    item = "segments" if unit == "pair" else "frames"
    return (
        "Selection requirements:\n"
        f"- Select only {item} containing important, representative events.\n"
        "- Aim to select approximately 20% of the timeline; normally keep the "
        f"selection between 15% and 25% of all {item}.\n"
        f"- Prefer a small number of coherent temporal segments over many isolated {item}.\n"
        "- Use at most 12 intervals.\n\n"
    )


def build_interval_rules(count, unit="frame"):
    # Validity rules of the output intervals, also enforced by the GRPO reward.
    item = "segment" if unit == "pair" else "frame"
    return (
        f"- Every index must be an integer between 0 and {count - 1}, inclusive.\n"
        "- Intervals must be sorted chronologically.\n"
        "- Intervals must not overlap.\n"
        "- Merge overlapping or consecutive intervals into one maximal interval.\n"
        "- Therefore, for consecutive output intervals [a,b] and [c,d], c must "
        "be greater than b + 1.\n"
        f"- Represent an isolated selected {item} i as [i,i].\n"
    )


def build_selection_prompt(count, index_phrase=", numbered from", unit="frame"):
    # V2V prompt: a JSON array of intervals over a timeline of `count` indices.
    item = "segment" if unit == "pair" else "frame"
    return (
        build_timeline_intro(count, index_phrase, unit)
        + build_selection_requirements(unit)

        + "Output requirements:\n"
        "- Return exactly one valid JSON array of inclusive integer intervals:\n"
        f"  [[start_{item},end_{item}],...]\n"
        + build_interval_rules(count, unit)
        + f"- Return [] only if no {item} can be selected.\n"
        "- Do not return timestamps, seconds, objects, strings, decimal numbers, "
        "Markdown, code fences, comments, or explanatory text.\n\n"

        "Valid example: [[0,2],[5,6]]\n"
        "Invalid example: [[0,2],[2,4]] because the intervals overlap.\n"
        "Invalid example: [[0,2],[3,4]] because consecutive intervals must be "
        "merged into [[0,4]].\n\n"

        "Return only the JSON array."
    )


def summary_requirement(summary_style, word_range=None):
    # (instruction, illustrative example) of the requested summary style.
    if summary_style == "concise":
        return ("Write exactly one concise summary of the video",
                "A short one-sentence description of the video.")
    low, high = word_range or SUMMARY_WORD_RANGE
    return (f"Write a summary of 3 to 4 sentences, about {low} to {high} words in "
            "total, describing the main events of the video in the order they happen",
            "First sentence describing the setting and the people. Second sentence "
            "describing the main action. Third sentence describing how the video ends.")


def build_joint_prompt(count, index_phrase=" indexed from", summary_style="concise", unit="frame",
                       word_range=None):
    # V2VT prompt: one JSON object with selected_frames and summary (SFT V2VT and GRPO).
    requirement, example = summary_requirement(summary_style, word_range)
    item = "segment" if unit == "pair" else "frame"
    return (
        build_timeline_intro(count, index_phrase, unit)
        + build_selection_requirements(unit)

        + "Output requirements:\n"
        '- Return exactly one valid JSON object in the form:\n'
        f'  {{"selected_frames":[[start_{item},end_{item}],...],"summary":"..."}}\n'
        + build_interval_rules(count, unit)
        + f'- {requirement} in the "summary" field.\n'
        "- Do not return timestamps, seconds, decimal numbers, numeric strings, "
        f"per-{item} captions, Markdown, code fences, explanations, or additional "
        "fields.\n"
        "- Output raw JSON only, with no surrounding text.\n\n"

        f"Selecting substantially more {item}s than the target share above is "
        "progressively penalized, the more so the further you go beyond it. Do "
        "not select the whole timeline.\n\n"

        f'Valid example: {{"selected_frames":[[2,3],[10,14]],"summary":"{example}"}}\n'
        'Invalid example: {"selected_frames":[[0,2],[2,4]],"summary":"..."} '
        "because the intervals overlap.\n"
        'Invalid example: {"selected_frames":[[0,2],[3,4]],"summary":"..."} '
        "because consecutive intervals must be merged into [[0,4]].\n\n"

        "Return only the JSON object."
    )


def build_summary_prompt(count, summary_style="concise", unit="frame", word_range=None):
    # V2T prompt: one JSON object with only the summary.
    requirement, example = summary_requirement(summary_style, word_range)
    # The concise V2T summary is a single sentence.
    if summary_style == "concise":
        requirement, example = "Write exactly one grammatical sentence", "A short one-sentence description of the video."
    timeline = (
        f"The timeline contains {count} candidate segments, numbered from 0 to "
        f"{count - 1}; each segment covers two consecutive seconds of the video. "
        if unit == "pair" else
        f"The timeline contains {count} candidate frames, numbered from 0 "
        f"to {count - 1}, sampled at one frame per second. "
    )
    return (
        timeline +
        f"Only a {'concise ' if summary_style == 'concise' else ''}final summary is required, not the selected frames. "
        "The summary should capture the essential events and content of the "
        "video accurately. Do not include frame indices, timestamps, or any "
        f"other details. {requirement}. Do not use "
        "additional text, Markdown, code fences, or extra JSON fields. "
        'Return exactly one valid JSON object in the form {"summary":"..."}. '
        f'Example (illustrative only): {{"summary":"{example}"}}'
    )


def subsampled_note(count, shown_frames):
    # Sentence for models that see only a uniform subset of the frames; empty when they see all of
    # them.
    if shown_frames is None or shown_frames >= count:
        return ""
    return (
        f"You are shown {shown_frames} of the {count} frames, sampled "
        "uniformly and each labeled with its second. "
    )


def zero_shot_anchor(count, ratio=0.20, shown_frames=None):
    # Zero-shot additions after the timeline intro: the link between indices and seconds, and the
    # frame budget.
    target = max(1, round(ratio * count))
    subsampled = subsampled_note(count, shown_frames)
    if subsampled:
        subsampled += (
            f"Still select intervals over all frames 0 to {count - 1}, "
            "including those that are not shown.\n"
        )
    return (
        "Frame i is the frame shown at second i of the video (timestamp i). "
        + subsampled
        + f"Aim for about {target} of the {count} frames in total.\n\n"
    )


def build_zero_shot_selection_prompt(count, ratio=0.20, shown_frames=None):
    # Zero-shot V2V prompt: the training rules plus the zero-shot anchor.
    return (
        build_timeline_intro(count, ZERO_SHOT_INDEX_PHRASE)
        + zero_shot_anchor(count, ratio, shown_frames)
        + build_selection_requirements()
        + "Output requirements:\n"
        "- Return exactly one valid JSON array of inclusive integer intervals:\n"
        "  [[start_frame,end_frame],...]\n"
        + build_interval_rules(count)
        + "- Return [] only if no frame can be selected.\n"
        "- Do not return timestamps, seconds, objects, strings, decimal numbers, "
        "Markdown, code fences, comments, or explanatory text.\n\n"
        "Valid example: [[0,2],[5,6]]\n"
        "Invalid example: [[0,2],[2,4]] because the intervals overlap.\n"
        "Invalid example: [[0,2],[3,4]] because consecutive intervals must be "
        "merged into [[0,4]].\n\n"
        "Return only the JSON array."
    )


def build_zero_shot_summary_prompt(count, summary_style="reference", shown_frames=None):
    # Zero-shot V2T prompt: the frame-timeline summary prompt plus the subsampling note.
    prompt = build_summary_prompt(count, summary_style, "frame")
    return prompt.replace(
        SUMMARY_TIMELINE_END, SUMMARY_TIMELINE_END + subsampled_note(count, shown_frames), 1
    )


def build_zero_shot_joint_prompt(count, ratio=0.20, shown_frames=None, summary_style="reference"):
    # Zero-shot V2VT prompt: the joint prompt with the zero-shot anchor inserted after the intro.
    intro = build_timeline_intro(count, ZERO_SHOT_INDEX_PHRASE)
    joint = build_joint_prompt(count, index_phrase=ZERO_SHOT_INDEX_PHRASE, summary_style=summary_style)
    if not joint.startswith(intro):
        raise ValueError("the joint prompt does not start with the timeline intro")
    return intro + zero_shot_anchor(count, ratio, shown_frames) + joint[len(intro):]


def add_mmss_note(prompt):
    # Models that see an MP4 read the timestamps as MM:SS: they are asked to convert them to
    # seconds.
    if TIMESTAMP_SENTENCE not in prompt:
        raise ValueError("prompt without the timestamp sentence")
    return prompt.replace(TIMESTAMP_SENTENCE, TIMESTAMP_SENTENCE + MMSS_NOTE, 1)
