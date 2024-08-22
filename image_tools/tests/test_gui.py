from image_tools import FishInfo
from PyQt5.QtWidgets import QApplication
import numpy as np

image = np.zeros((512,512,3), dtype=np.uint8)

app = QApplication([])
window = FishInfo(image)
window.show()
app.exec()