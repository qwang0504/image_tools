from image_tools import CloneTool
from PyQt5.QtWidgets import QApplication
import numpy as np

#image = 255*np.ones((512,512,3), dtype=np.uint8)
image = np.load('/media/martin/DATA/Mecp2/processed/2024_09_25_04_MeCP2_fish1_chunk_001.npy')

app = QApplication([])
window = CloneTool(image)
window.show()
app.exec()