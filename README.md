# FRCNN2
INSTRUCTIONS FOR USE:
1. Install Docker for Desktop [here][link]
2. Download this repository using the "Code" dropdown
3. Extract the contents of the zipfile to your desired folder
4. Navigate to this folder using the command line and run the command - "docker build -t frcnn2image ." (This may take a bit)
5. Still in the command line, type "docker compose up". A jupyter notebook environment should appear. If it does not, try pasting one of the links provided in the command line into your browser of choice
6. If you have images annotated from the previous version skip to 7. Otherwise, move the images you would like to use for training into the 'annotations' folder and open the 'annotate' notebook
7. Open the FRCNN2 notebook, all following instructions can be found there
8. Once you are finished using the notebook, make sure to type the command "docker compose down" into the command line while in the FRCNN folder. This will ensure that the program is fully closed and won't waste your computer's resources.


[link]: https://www.docker.com/products/docker-desktop/
