import numpy as np
import cv2
from PIL import Image, ImageOps

from paddleocr import PaddleOCR


class PaddleEngine:

    def __init__(self):

        self.ocr = PaddleOCR(
            lang="fr",
            use_angle_cls=True,
            use_gpu=False
        )


    def extract(self, image_input):
        """
        Lance l'OCR sur l'image.

        Args:
            image_input: Chemin du fichier (str/Path) ou image BGR déjà
                         prétraitée (np.ndarray). Si un tableau numpy est
                         fourni, on l'utilise directement sans rechargement.

        Returns:
            Liste de dicts {"text", "confidence", "box"}.
        """
        if isinstance(image_input, np.ndarray):
            image = image_input
        else:
            image = self.load_image(image_input)

        result = self.ocr.ocr(
            image,
            cls=True
        )

        ocr_data = []

        for line in result:

            for item in line:

                box = item[0]

                text = item[1][0]

                confidence = item[1][1]

                ocr_data.append({

                    "text": text,

                    "confidence": confidence,

                    "box": box

                })

        return ocr_data


    def load_image(self, image_path):
        """
        Charge l'image en appliquant la rotation EXIF si elle est
        présente.

        Les photos prises avec un téléphone stockent souvent les
        pixels "à plat" avec un tag EXIF Orientation qui indique
        comment l'afficher (ex: tourné de 90°/180°/270°). PaddleOCR
        (via cv2) ignore ce tag et lit les pixels bruts : si on lui
        passe directement le chemin du fichier, il peut donc "voir"
        une image tournée alors qu'elle s'affiche correctement dans
        n'importe quelle visionneuse d'images.

        Résultat concret observé : texte lu à l'envers/de travers,
        ordre de tri haut->bas complètement inversé, et un OCR qui
        produit du charabia sur une partie des zones de texte.

        En appliquant ImageOps.exif_transpose ici, on garantit que
        PaddleOCR reçoit toujours l'image dans la bonne orientation,
        peu importe le tag EXIF du fichier source.
        """

        pil_image = Image.open(image_path)

        pil_image = ImageOps.exif_transpose(pil_image)

        rgb_array = np.array(pil_image.convert("RGB"))

        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        return bgr_array