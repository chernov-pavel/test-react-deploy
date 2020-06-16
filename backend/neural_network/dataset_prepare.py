import codecs
import glob

import regex


def clear_text(text):
    text = regex.sub("(?s){{.+?}}", "", text)
    text = regex.sub("(?s){.+?}", "", text)
    text = regex.sub("(?s)\[\[([^]]+\|)", "", text)
    text = regex.sub("(?s)\[\[([^]]+\:.+?]])", "", text)
    text = regex.sub("[']{5}", "", text)
    text = regex.sub("[']{3}", "", text)
    text = regex.sub("[']{2}", "", text)
    text = regex.sub(u"[^ \r\n\p{Cyrillic}\d\-'.?!]", " ", text)
    text = text.lower()
    text = regex.sub("[ ]{2,}", " ", text)
    text = regex.sub("([.!?]+$)", r" \1", text)
    text = regex.sub("!..", "", text)
    text = regex.sub("[-]+", "", text)
    text = regex.sub("^[ ]+\n$", "", text)
    text = regex.sub("[.]", "", text)
    text = regex.sub("\r\n", " ", text)
    text = regex.sub("\n", " ", text)
    text_length = 0
    while(text_length != len(text)):
        text_length = len(text)
        text = regex.sub("  ", " ", text)

    return text


if __name__ == '__main__':
    paths = glob.glob('data\\raw\\*.txt')
    for path in paths:
        input_file = codecs.open(path, 'r', 'utf_8_sig')
        text_file = input_file.read()
        text = clear_text(text_file)
        output_file = open('data\\dataset.txt', 'a')
        output_file.write(text)
        output_file.close()