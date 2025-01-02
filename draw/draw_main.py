import tkinter as tk
from PIL import Image, ImageDraw

# Dimensioni della tela originale
CANVAS_SIZE = 280  # 280x280 pixel
# Dimensioni dell'immagine ridimensionata
IMG_SIZE = 28  # 28x28 pixel

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Disegna e salva")

        # Creazione della tela
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack()

        # Variabili per il disegno
        self.old_x = None
        self.old_y = None
        self.line_width = 8
        self.color = 'black'

        # Immagine per il disegno
        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Collegare gli eventi del mouse
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # Pulsante per salvare
        self.save_button = tk.Button(root, text="Salva come 28x28", command=self.save_image)
        self.save_button.pack()

    def paint(self, event):
        if self.old_x and self.old_y:
            # Disegna sulla tela
            self.canvas.create_line(
                self.old_x, self.old_y, event.x, event.y,
                width=self.line_width, fill=self.color, capstyle=tk.ROUND, smooth=tk.TRUE
            )
            # Disegna sull'immagine
            self.draw.line(
                [self.old_x, self.old_y, event.x, event.y],
                fill=self.color, width=self.line_width
            )
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x = None
        self.old_y = None

    def save_image(self):
        # Ridimensiona l'immagine a 28x28
        resized_image = self.image.resize((IMG_SIZE, IMG_SIZE))
        # Salva l'immagine
        resized_image.save("drawing_28x28.png")
        print("Immagine salvata come drawing_28x28.png")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()