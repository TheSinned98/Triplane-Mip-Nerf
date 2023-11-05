import cv2
import glob
import numpy as np
import re
from natsort import natsorted

# Pfad zu den Ordnern
folder1 = './data/nerf_synthetic/lego_mip/test/*'
folder2 = './logs/blender_paper_lego_laplace/testset_200000/*'

# Liste der Bilder in jedem Ordner
images1 = glob.glob(folder1)
images2 = glob.glob(folder2)

# Filtern Sie die Bilder im ersten Ordner, um Dateien auszuschließen, die die Wörter "normal" oder "depth" enthalten
images1 = [img for img in images1 if not re.search(r'(normal|depth)', img)]

# Wählen Sie nur jedes achte Bild aus
images1 = images1[::8]

# Sortieren Sie die Listen natürlich
images1 = natsorted(images1)
images2 = natsorted(images2)

# Überprüfen Sie, ob beide Ordner die gleiche Anzahl von Bildern enthalten
if len(images1) != len(images2):
    print("Die Ordner enthalten nicht die gleiche Anzahl von Bildern.")
else:
    psnr_values = []
    with open('psnr_results_lego_laplace.txt', 'w') as f:
        # Durchlaufen Sie jedes Bildpaar
        for img1_path, img2_path in zip(images1, images2):
            # Bilder laden
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            # Skalieren Sie das Bild aus Ordner 1 auf die Hälfte seiner Größe
            img1 = cv2.resize(img1, (img1.shape[1]//2, img1.shape[0]//2))

            # PSNR berechnen
            psnr = cv2.PSNR(img1, img2)
            psnr_values.append(psnr)

            # Ergebnis in die Datei schreiben
            f.write(f'PSNR für {img1_path} und {img2_path}: {psnr}\n')

        # Durchschnittlichen PSNR und Verlust berechnen
        avg_psnr = np.mean(psnr_values)
        avg_loss = 100 - avg_psnr

        # Durchschnittswerte in die Datei schreiben
        f.write(f'\nDurchschnittlicher PSNR: {avg_psnr}\n')
        f.write(f'Durchschnittlicher Verlust: {avg_loss}\n')