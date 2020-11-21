def clean_ingredient_name(ingredient_name, normalization_fixes):
    words = ingredient_name.split('_')
    cleaned_words = []
    for word in words:
        if word in normalization_fixes:
            cleaned_words.append(normalization_fixes[word])
        else:
            cleaned_words.append(word)

    return ' '.join(cleaned_words)


def clean_substitutes(subtitutes, normalization_fixes):
    cleaned_subtitutes = []
    for subtitute in subtitutes:
        cleaned_subtitutes.append(clean_ingredient_name(subtitute, normalization_fixes))

    return cleaned_subtitutes
