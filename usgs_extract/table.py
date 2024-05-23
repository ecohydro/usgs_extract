# AUTOGENERATED! DO NOT EDIT! File to edit: ../03_table.ipynb.

# %% auto 0
__all__ = ['Table']

# %% ../03_table.ipynb 3
## put all import statements here

# %% ../03_table.ipynb 5
class Table():
    """ 
    A class to represent a table. 
    """
    def __init__(self, image):
        self.image = image 
    
    def show(self, scale=0.4):
        """
        Display the image of the table.
        """
        width, height = self.image.size
        display(self.image.resize((int(0.4*width), (int(0.4*height)))))
    
    
        

# %% ../03_table.ipynb 6
## put functions and classes here
