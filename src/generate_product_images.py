
import os
import math
import pandas as pd

from PIL import Image, ImageDraw, ImageFont


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
IMAGE_DIR = os.path.join(BASE_DIR, "assets", "products")

os.makedirs(IMAGE_DIR, exist_ok=True)

products_path = os.path.join(
    DATASET_DIR, "products.csv"
)

products = pd.read_csv(products_path)


# --------------------------------------------------
# IMAGE SETTINGS
# --------------------------------------------------

WIDTH = 700
HEIGHT = 520

category_colors = {
    "Footwear": (225, 238, 255),
    "Men's Clothing": (232, 225, 255),
    "Women's Clothing": (255, 229, 240),
    "Accessories": (255, 242, 215),
    "Electronics": (220, 245, 244),
    "Beauty": (255, 228, 225),
    "Home & Kitchen": (230, 240, 224),
    "Sports": (255, 235, 210)
}

category_symbols = {
    "Footwear": "SHOES",
    "Men's Clothing": "FASHION",
    "Women's Clothing": "STYLE",
    "Accessories": "ACCESSORY",
    "Electronics": "GADGET",
    "Beauty": "BEAUTY",
    "Home & Kitchen": "HOME",
    "Sports": "SPORTS"
}


# --------------------------------------------------
# FONT LOADER
# --------------------------------------------------

def load_font(size, bold=False):

    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold
        else "C:/Windows/Fonts/arial.ttf",

        "C:/Windows/Fonts/calibrib.ttf" if bold
        else "C:/Windows/Fonts/calibri.ttf"
    ]

    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass

    return ImageFont.load_default()


# --------------------------------------------------
# DRAW PRODUCT ILLUSTRATION
# --------------------------------------------------

def draw_product_image(product):

    category = product["category"]

    background = category_colors.get(
        category, (235, 235, 245)
    )

    image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        background
    )

    draw = ImageDraw.Draw(image)

    # Main white product panel
    draw.rounded_rectangle(
        (35, 30, 665, 490),
        radius=30,
        fill=(255, 255, 255)
    )

    # Decorative circles
    draw.ellipse(
        (470, 65, 620, 215),
        fill=background
    )

    draw.ellipse(
        (80, 320, 180, 420),
        fill=background
    )

    # Product illustration area
    center_x = WIDTH // 2
    center_y = 230

    category = str(category)

    # Draw a distinct simple illustration by category.
    if category == "Footwear":

        # Stylized sneaker silhouette
        draw.rounded_rectangle(
            (190, 180, 500, 300),
            radius=55,
            fill=(55, 75, 120)
        )

        draw.polygon(
            [(190, 255), (120, 300), (510, 300),
             (540, 270), (470, 250)],
            fill=(75, 95, 140)
        )

        for x in range(270, 430, 45):
            draw.line(
                (x, 210, x + 25, 245),
                fill=(255, 255, 255),
                width=8
            )

        draw.line(
            (150, 310, 530, 310),
            fill=(40, 45, 60),
            width=8
        )

    elif category in ["Men's Clothing", "Women's Clothing"]:

        # Stylized shirt
        draw.polygon(
            [
                (270, 150), (210, 190), (150, 280),
                (220, 320), (245, 275),
                (245, 365), (455, 365),
                (455, 275), (480, 320),
                (550, 280), (490, 190),
                (430, 150), (390, 190),
                (310, 190)
            ],
            fill=(95, 110, 175)
        )

        draw.line(
            (310, 190, 350, 235, 390, 190),
            fill=(255, 255, 255),
            width=8
        )

        draw.line(
            (350, 235, 350, 350),
            fill=(255, 255, 255),
            width=5
        )

    elif category == "Accessories":

        # Stylized wristwatch
        draw.rounded_rectangle(
            (295, 105, 405, 355),
            radius=30,
            fill=(45, 55, 80)
        )

        draw.rounded_rectangle(
            (255, 175, 445, 300),
            radius=45,
            fill=(185, 205, 225),
            outline=(65, 80, 110),
            width=10
        )

        draw.ellipse(
            (290, 205, 410, 325),
            fill=(45, 60, 90)
        )

        draw.line(
            (350, 265, 350, 220),
            fill=(255, 255, 255),
            width=7
        )

        draw.line(
            (350, 265, 390, 285),
            fill=(255, 255, 255),
            width=7
        )

    elif category == "Electronics":

        # Stylized wireless earbuds
        draw.rounded_rectangle(
            (230, 245, 470, 355),
            radius=35,
            fill=(55, 65, 85)
        )

        draw.rounded_rectangle(
            (270, 170, 330, 285),
            radius=25,
            fill=(80, 100, 140)
        )

        draw.rounded_rectangle(
            (370, 170, 430, 285),
            radius=25,
            fill=(80, 100, 140)
        )

        draw.ellipse(
            (280, 155, 320, 205),
            fill=(80, 100, 140)
        )

        draw.ellipse(
            (380, 155, 420, 205),
            fill=(80, 100, 140)
        )

    elif category == "Beauty":

        # Stylized beauty bottle
        draw.rounded_rectangle(
            (275, 160, 425, 380),
            radius=30,
            fill=(230, 135, 160)
        )

        draw.rectangle(
            (320, 105, 380, 165),
            fill=(65, 65, 85)
        )

        draw.rounded_rectangle(
            (305, 235, 395, 300),
            radius=12,
            fill=(255, 240, 240)
        )

        draw.line(
            (350, 180, 350, 220),
            fill=(255, 255, 255),
            width=7
        )

    elif category == "Home & Kitchen":

        # Stylized water bottle
        draw.rounded_rectangle(
            (285, 135, 415, 380),
            radius=35,
            fill=(85, 150, 175)
        )

        draw.rectangle(
            (315, 95, 385, 145),
            fill=(55, 85, 105)
        )

        draw.rounded_rectangle(
            (305, 225, 395, 280),
            radius=12,
            fill=(235, 245, 250)
        )

        draw.text(
            (325, 240),
            "H2O",
            font=load_font(25, True),
            fill=(45, 85, 115)
        )

    elif category == "Sports":

        # Stylized sports ball
        draw.ellipse(
            (230, 130, 470, 370),
            fill=(235, 145, 65),
            outline=(70, 70, 80),
            width=9
        )

        draw.arc(
            (245, 145, 455, 355),
            30, 210,
            fill=(255, 255, 255),
            width=8
        )

        draw.arc(
            (245, 145, 455, 355),
            210, 390,
            fill=(255, 255, 255),
            width=8
        )

        draw.line(
            (250, 250, 450, 250),
            fill=(255, 255, 255),
            width=7
        )

    # Product text
    product_name = str(product["product_name"])
    brand = str(product["brand"])
    price = float(product["price"])
    product_id = str(product["product_id"])

    draw.text(
        (65, 390),
        brand,
        font=load_font(21, True),
        fill=(90, 90, 110)
    )

    draw.text(
        (65, 425),
        product_name[:55],
        font=load_font(19, True),
        fill=(35, 40, 65)
    )

    draw.text(
        (520, 425),
        f"Rs.{price:.0f}",
        font=load_font(19, True),
        fill=(25, 115, 80)
    )

    return image


# --------------------------------------------------
# GENERATE ALL 36 IMAGES
# --------------------------------------------------

print("\nGenerating product illustrations...")
print("-" * 50)

image_paths = []

for _, product in products.iterrows():

    product_id = product["product_id"]

    image = draw_product_image(product)

    filename = f"{product_id}.jpg"

    output_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image.save(
        output_path,
        quality=95
    )

    image_paths.append(
        f"assets/products/{filename}"
    )

    print(
        f"Created: {filename} | "
        f"{product['product_name']}"
    )


# --------------------------------------------------
# UPDATE PRODUCT CSV
# --------------------------------------------------

products["image_url"] = image_paths

products.to_csv(
    products_path,
    index=False
)

print("\n" + "=" * 50)
print("PRODUCT IMAGE GENERATION COMPLETED")
print("=" * 50)

print("Total products:", len(products))
print("Total images:", len(image_paths))
print("Image folder:", IMAGE_DIR)

print("\nAll product images generated successfully!")
