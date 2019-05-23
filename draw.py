from PIL import Image, ImageDraw
import data
import numpy as np

# boxes = readCsv("data/csv/l04-136.csv")

# im = Image.open("data/forms/l04-136.png")

# draw = ImageDraw.Draw(im)

# for box in boxes:
#     draw.rectangle([box[0], box[1], box[0]+box[2],
#                     box[1]+box[3]], outline="red")
# del draw

# im.save("out.png", "PNG")

transforms = data.MakeSquareAndTargets()

dataset = data.HWRSegmentationDataset("data/csv", "data/forms", transforms)

im_i = 15

img_np = dataset[im_i]['image'][0]
tgt = dataset[im_i]['target']

img_pil = Image.fromarray(np.uint8(img_np.numpy()*255))
draw = ImageDraw.Draw(img_pil)

image_dim = 512
grid_width = image_dim / 32

for y_row in range(tgt.shape[1]):
    # Row / y axis
    for x_col in range(tgt.shape[2]):
        tup = tgt[:, y_row, x_col]  # [offset_x, offset_y, w, h, valid]
        if tup[4] > 0:
            print(tup)
            c_x = grid_width * (x_col + 0.5 + tup[0])
            c_y = grid_width * (y_row + 0.5 + tup[1])
            w = tup[2] * image_dim
            h = tup[3] * image_dim
            draw.rectangle([c_x - w / 2, c_y - h / 2, c_x + w / 2,
                            c_y + h / 2], outline="red")

img_pil.save("out.png", "PNG")
