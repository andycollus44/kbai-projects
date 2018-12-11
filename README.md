# [CS-7637] Knowledge-based Artificial Intelligence: Projects

This repository contains my implementation of an AI agent that solves Raven's Progressive Matrices test.

## Overview

The agent was developed over three different submissions each solving an increasingly challenging set of tests, as follows:

1. **Submission 1:** Set B of 2x2 matrices.
2. **Submission 2:** Set C of 3x3 matrices.
3. **Submission 3:** Sets D and E of 3x3 matrices.

Each set is comprised of **48 problems** divided equally into the following four categories:

* **Basic:** Provided set of Raven's-like problems.
* **Challenge:** Provided set of more difficult Raven's-like problems.
* **Test:** Hidden set of Raven's-like problems for evaluation.
* **Ravens:** Hidden set of the real Raven's problems.

The agent was implemented using Python, NumPy and Pillow. No other image processing library was allowed.

## How It Works

My final agent works in two phases, each leveraging the Generate & Test technique. First, it attempts to solve the problem visually via affine transformations of the images without any knowledge representation. It has a list of the different transformations it can try and applies them all to the given problem for each of the different axes, i.e. row-wise and column-wise, to generate the expected answer. It then iterates through each of the available candidates to test which one best matches the expected answer.

To find this best match, it uses the Normalized Root Mean Square Error as a similarity measure based on the pixel-by-pixel difference between the images, producing a score between 0.0 (no match) and 1.0 (perfect match). However, my agent only trusts answers with a minimum similarity measure of 0.8 or 0.9 (depending on the transformation) or more. Once it has found all the answers based on each of the transformations and axes, it simply takes the one with the highest score.

Now, if the visual-only phase does not yield an answer, then my agent goes into the second phase and attempts to solve the problem semantically via Semantic Networks. In order to build semantic relationships, my agent performs two crucial pre-processing steps: shape extraction and shape classification. To extract shapes, I implemented the contour-following algorithm as proposed by Jonghoon, et al. (2016) that considers a lot of different cases, in particular corners, in order to extract close to 100 percent of the points that form the contour of an object. Having the contour of each of the shapes inside an image, other attributes are computed like its approximate area, size, whether it is filled or not, etc.

In my current design, the shape extractor merges similar shapes that are too close to each other to handle the fact that the contour-following algorithm will find two contours in shapes with thick outlines (one external and one internal). However, this caused some false negatives in some of the problems; to mitigate this, I also implemented the Connected Components algorithm (n.d.) with the idea of using it as a way of extracting only bounding boxes for simple purposes like counting shapes which does not require the handling of two closely-related contours.

On the other hand, to classify shapes, I implemented the $1 Recognizer, as proposed by Wobbrock, Wilson and Li (2007). This is a template-based algorithm that is significantly more accurate since it uses previously recorded shapes as templates to match a given shape to. It is also rotation, translation and scale invariant. However, for this algorithm to work, I had to extract shapes from the problem sets and save their contours as templates.

Once all the shapes from each of the images of the problem have been extracted, my agent then starts generating semantics between them to represent knowledge. As with the first phase, my agent also has a list of relationships it can attempt to build for a particular problem. Because this whole process can be very computationally expensive, my agent tries to build one relationship at a time row-wise and column-wise, and then tests each answer to see which one complies with the expected semantics. Moreover, because there is no image similarity measure here, the agent only trusts the answer if it is the same for both axes. If the answers differ, then it continues to attempt the next relationship until an answer is found or all relationships have been exhausted.

Finally, if an answer is not found in any of the phases, the agent skips the problem.

## Performance

My agent obtained a final accuracy of ~69% (133 out of 192 problems) which I find very satisfying. The complete breakdown of correct answers is shown in the following table:

| Problem Set | Basic | Test | Ravens | Challenge | Total   |
|:-----------:|:-----:|:----:|:------:|:---------:|:-------:|
|      B      |   12  |  12  |   11   |     10    |   45    |
|      C      |   11  |   7  |    4   |     6     |   28    |
|      D      |   11  |   7  |    2   |     6     |   26    | 
|      E      |   12  |  11  |    6   |     5     |   34    |

## Learn More

To learn more about each iteration of the agent, refer to each of the complete reports in the `Reports/` directory.

## References

1.	Connected-component labeling (n.d.) In Wikipedia. Retrieved from https://en.wikipedia.org/wiki/Connected-component_labeling 
2.	Jonghoon, S., et al. (2016). Fast Contour-Tracing Algorithm Based on a Pixel-Following Method for Image Sensors. In Sensors. Basel, Switzerland.
3.	Wobbrock, J. O., Wilson, A.D., Li, Y. (2007). Gestures without Libraries, Toolkits or Training: A $1 Recognizer for User Interface Prototypes. In Proceedings of the 20th annual ACM symposium on User interface software and technology. Newport, Rhode Island, USA.


