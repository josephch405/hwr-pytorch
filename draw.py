from PIL import Image, ImageDraw
from data import readCsv

boxes = readCsv("data/csv/a01-000u.csv")

im = Image.open("data/forms/a01-000u.png")

draw = ImageDraw.Draw(im)

for box in boxes:
    draw.rectangle([box[0], box[1], box[0]+box[2],
                    box[1]+box[3]], outline="red")
del draw

im.save("data/forms/out.png", "PNG")
