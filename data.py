import xml.etree.ElementTree as ET
import tensorflow as tf
import csv
import os

t = ET.parse("data/xml/a01-000u.xml").getroot()


def processXmlRoot(t):

    docDims = [int(t.attrib["height"]), int(t.attrib["width"])]

    results = []

    for line in t[1]:
        words = list(filter(lambda e: e.tag == "word", line))
        # line is [[x, y], [x + w, y + h]]
        x = int(words[0][0].attrib["x"])
        y = docDims[0]
        w = int(words[-1][-1].attrib["x"]) + \
            int(words[-1][-1].attrib["width"]) - x
        h = 0
        for word in words:
            for cmp in word:
                word_y = int(cmp.attrib["y"])
                word_h = int(cmp.attrib["height"])
                y = min(word_y, y)
                h = max(h, word_y + word_h - y)
        results.append([x, y, w, h])
    return results


def writeCsv(lists, path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for l in lists:
            writer.writerow(l)


def readCsv(path):
    with open(path) as csv_file:
        reader = csv.reader(csv_file)
        results = []
        for row in reader:
            new_row = list(map(int, row))
            results.append(new_row)
        return results


writeCsv(processXmlRoot(t), "data/csv/a01-000u.csv")
print(readCsv("data/csv/a01-000u.csv"))
