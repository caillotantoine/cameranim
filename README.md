# CAMERANIM

Cameranim is a small renderrer aiming to show how is it like to move in a pointcloud for a camera. This tool help to generate gif images and two types of camera are supported: perspective and equidistant.

# Example

## Perspective camera (60º FOV)

For a translation colinear with optical axis, it is hard to see a lot of change. Hints are faint, it is not easy to estimate the quantity of movement precisely.

![Translation in the same direction with optical axis.](persp_translation_x.gif)

It is a much better with a translation orthogonal to the optical axis, the quantity of movement is clearer.

![Translation in an orthogonal direction of optical axis.](persp_translation_y.gif)

But, when it comes to a rotation, it is hard to distinguish it with the previous translation.

![Rotation around the vertical axis](persp_rotation_z.gif)

## Equidistant camera (210º)
Here, the side of the image give a better perception of the translation colinear with the optical axis.

![Colinearº](equidist_translation_x.gif)

The rotation and the translation have two very different look.
![translation on horizontal axis.](equidist_translation_y.gif)
![rotation](equidist_rotation_z.gif)