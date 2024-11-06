from .logging_setup import logger
from whisperx.utils import get_writer
from .utils import remove_files
import copy
# subtitles

def process_subtitles(
    deep_copied_result,
    align_language,
    result_diarize,
    output_format_subtitle,
    TRANSLATE_AUDIO_TO,
):
    name_ori = "sub_ori."
    name_tra = "sub_tra."
    remove_files(
        [name_ori + output_format_subtitle, name_tra + output_format_subtitle]
    )

    writer = get_writer(output_format_subtitle, output_dir=".")
    word_options = {
        "highlight_words": False,
        "max_line_count": None,
        "max_line_width": None,
    }

    # original lang
    subs_copy_result = copy.deepcopy(deep_copied_result)
    subs_copy_result["language"] = (
        "zh" if align_language == "zh-TW" else align_language
    )
    for segment in subs_copy_result["segments"]:
        segment.pop("speaker", None)

    try:
        writer(
            subs_copy_result,
            name_ori[:-1] + ".mp3",
            word_options,
        )
    except Exception as error:
        logger.error(str(error))
        if str(error) == "list indices must be integers or slices, not str":
            logger.error(
                "Related to poor word segmentation"
                " in segments after alignment."
            )
        subs_copy_result["segments"][0].pop("words")
        writer(
            subs_copy_result,
            name_ori[:-1] + ".mp3",
            word_options,
        )

    # translated lang
    subs_tra_copy_result = copy.deepcopy(result_diarize)
    subs_tra_copy_result["language"] = (
        "ja" if TRANSLATE_AUDIO_TO in ["ja", "zh", "zh-TW"] else align_language
    )
    subs_tra_copy_result.pop("word_segments", None)
    for segment in subs_tra_copy_result["segments"]:
        for key in ["speaker", "chars", "words"]:
            segment.pop(key, None)

    writer(
        subs_tra_copy_result,
        name_tra[:-1] + ".mp3",
        word_options,
    )

    return name_tra + output_format_subtitle


def linguistic_level_segments(
    result_base,
    linguistic_unit="word",  # word or char
):
    linguistic_unit = linguistic_unit[:4]
    linguistic_unit_key = linguistic_unit + "s"
    result = copy.deepcopy(result_base)

    if linguistic_unit_key not in result["segments"][0].keys():
        raise ValueError("No alignment detected, can't process")

    segments_by_unit = []
    for segment in result["segments"]:
        segment_units = segment[linguistic_unit_key]
        # segment_speaker = segment.get("speaker", "SPEAKER_00")

        for unit in segment_units:

            text = unit[linguistic_unit]

            if "start" in unit.keys():
                segments_by_unit.append(
                    {
                        "start": unit["start"],
                        "end": unit["end"],
                        "text": text,
                        # "speaker": segment_speaker,
                    }
                    )
            elif not segments_by_unit:
                pass
            else:
                segments_by_unit[-1]["text"] += text

    return {"segments": segments_by_unit}


def break_aling_segments(
    result: dict,
    break_characters: str = "",  # ":|,|.|"
):
    result_align = copy.deepcopy(result)

    break_characters_list = break_characters.split("|")
    break_characters_list = [i for i in break_characters_list if i != '']

    if not break_characters_list:
        logger.info("No valid break characters were specified.")
        return result

    logger.info(f"Redivide text segments by: {str(break_characters_list)}")

    # create new with filters
    normal = []

    def process_chars(chars, letter_new_start, num, text):
        start_key, end_key = "start", "end"
        start_value = end_value = None

        for char in chars:
            if start_key in char:
                start_value = char[start_key]
                break

        for char in reversed(chars):
            if end_key in char:
                end_value = char[end_key]
                break

        if not start_value or not end_value:
            raise Exception(
                f"Unable to obtain a valid timestamp for chars: {str(chars)}"
            )

        return {
            "start": start_value,
            "end": end_value,
            "text": text,
            "words": chars,
        }

    for i, segment in enumerate(result_align['segments']):

        logger.debug(f"- Process segment: {i}, text: {segment['text']}")
        # start = segment['start']
        letter_new_start = 0
        for num, char in enumerate(segment['chars']):

            if char["char"] is None:
                continue

            # if "start" in char:
            #     start = char["start"]

            # if "end" in char:
            #     end = char["end"]

            # Break by character
            if char['char'] in break_characters_list:

                text = segment['text'][letter_new_start:num+1]

                logger.debug(
                    f"Break in: {char['char']}, position: {num}, text: {text}"
                )

                chars = segment['chars'][letter_new_start:num+1]

                if not text:
                    logger.debug("No text")
                    continue

                if num == 0 and not text.strip():
                    logger.debug("blank space in start")
                    continue

                if len(text) == 1:
                    logger.debug(f"Short char append, num: {num}")
                    normal[-1]["text"] += text
                    normal[-1]["words"].append(chars)
                    continue

                # logger.debug(chars)
                normal_dict = process_chars(chars, letter_new_start, num, text)

                letter_new_start = num+1

                normal.append(normal_dict)

            # If we reach the end of the segment, add the last part of chars.
            if num == len(segment["chars"]) - 1:

                text = segment['text'][letter_new_start:num+1]

                # If remain text len is not default len text
                if num not in [len(text)-1, len(text)] and text:
                    logger.debug(f'Remaining text: {text}')

                if not text:
                    logger.debug("No remaining text.")
                    continue

                if len(text) == 1:
                    logger.debug(f"Short char append, num: {num}")
                    normal[-1]["text"] += text
                    normal[-1]["words"].append(chars)
                    continue

                chars = segment['chars'][letter_new_start:num+1]

                normal_dict = process_chars(chars, letter_new_start, num, text)

                letter_new_start = num+1

                normal.append(normal_dict)

    # Rename char to word
    for item in normal:
        words_list = item['words']
        for word_item in words_list:
            if 'char' in word_item:
                word_item['word'] = word_item.pop('char')

    # Convert to dict default
    break_segments = {"segments": normal}

    msg_count = (
        f"Segment count before: {len(result['segments'])}, "
        f"after: {len(break_segments['segments'])}."
    )
    logger.info(msg_count)

    return break_segments
