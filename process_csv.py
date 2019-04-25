import xml.etree.ElementTree as ET
import csv
import os


def processXmlRoot(t):

    docDims = [int(t.attrib["height"]), int(t.attrib["width"])]

    results = []

    for line in t[1]:
        words = list(filter(lambda e: e.tag == "word" and len(e) > 0, line))
        # print(words[-1].children)
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


if __name__ == "__main__":
    filenames = os.listdir("data/xml")

    for filename in filenames:
        t = ET.parse(f"data/xml/{filename}").getroot()
        print(filename)
        csv_filename = filename.replace(".xml", ".csv")
        writeCsv(processXmlRoot(t), f"data/csv/{csv_filename}")
