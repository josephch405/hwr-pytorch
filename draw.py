from PIL import Image, ImageDraw
from data import readCsv

boxes = readCsv("data/csv/l04-136.csv")

im = Image.open("data/forms/l04-136.png")

draw = ImageDraw.Draw(im)

for box in boxes:
    draw.rectangle([box[0], box[1], box[0]+box[2],
                    box[1]+box[3]], outline="red")
del draw

im.save("out.png", "PNG")
