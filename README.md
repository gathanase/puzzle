Inspired by "How to solve a white puzzle" (https://www.youtube.com/watch?v=WsPHBD5NsS0)

Puzzle pieces are provided by https://www.reddit.com/r/StuffMadeHere/comments/zaq7f7/puzzle_piece_raw_image_dataset/

There is a much more challenging dataset available at: https://www.kaggle.com/datasets/etaifour/jigsawpuzzle

# Open in Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gathanase/puzzle/blob/main/solve144.ipynb)

# Roadmap
* read image DONE
* compute contours DONE
* compute pieces DONE
  * assert size and number of pieces?
* compute 4 corners DONE
  * assert corners?
* compute 4 edge contour DONE
* compute 4 edge features DONE
  * assert number of corner&border&regular pieces DONE
* compute puzzle size DONE
* assemble border pieces DONE
  * assert the border
* assemble inner pieces TODO
